"""
Industrial Analytics — Spark Structured Streaming
==================================================
Reads from Kafka topic `industrial-telemetry`, processes in real-time,
and writes results to SQL Server.

Write targets
  dbo.telemetry_raw         raw event log            (append every 10 s)
  dbo.asset_minute_fact     1-min window aggregates  (append every 60 s)
  dbo.asset_minute_features engineered ML features   (append every 60 s)
  dbo.asset_anomaly_events  threshold-based alerts   (append, deduped by unique index)
  dbo.asset_risk_minute     per-minute risk scores   (append)
  dbo.asset_risk_current    latest risk per asset    (MERGE every 60 s)

Design note — recommendations (dbo.asset_recommendations)
  Recommendations require cooldown logic, acknowledgment tracking, and
  business rules that are stateful across sessions.  These remain the
  responsibility of the .NET recommendation worker.  Spark feeds it by
  keeping asset_risk_current and asset_anomaly_events up to date.
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, from_json, to_timestamp,
    avg, count, max as spark_max,
    when, lit, window,
)
from pyspark.sql.types import (
    StructType, StructField,
    StringType, DoubleType, IntegerType,
)
import json
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("industrial-streaming")


# ── Configuration ──────────────────────────────────────────────────────────────

KAFKA_SERVERS  = "kafka:29092"       # internal Docker listener
KAFKA_TOPIC    = "industrial-telemetry"
KAFKA_GROUP    = "spark-industrial-analytics"

JDBC_URL = (
    "jdbc:sqlserver://sqlserver:1433;"
    "databaseName=Industrail_AI;"
    "encrypt=true;"
    "trustServerCertificate=true;"
)
JDBC_USER     = "sa"
JDBC_PASSWORD = "IndustrialAI#2026"
JDBC_DRIVER   = "com.microsoft.sqlserver.jdbc.SQLServerDriver"

CHECKPOINT_DIR = "/opt/spark/checkpoints"

JDBC_OPTS = {
    "url":      JDBC_URL,
    "user":     JDBC_USER,
    "password": JDBC_PASSWORD,
    "driver":   JDBC_DRIVER,
}

# Anomaly thresholds — mirrors .NET HybridAnomalyDetector thresholds.
# key      = DataFrame column name after 1-min aggregation
# tag      = original Kafka tag name (written to asset_anomaly_events.signal)
# type     = anomaly_type string
# low      = True if the violation is "value < threshold" (LOW_FLOW, LOW_PRESSURE)
THRESHOLDS = {
    "avg_temp_c":       {"tag": "temp_bearing_c",         "type": "HIGH_TEMP",    "low": False, "warn": 75.0, "crit": 88.0},
    "avg_vib_mm_s":     {"tag": "vib_rms_mm_s",           "type": "HIGH_VIB",     "low": False, "warn":  4.0, "crit":  7.0},
    "avg_current_a":    {"tag": "motor_current_a",        "type": "HIGH_CURRENT", "low": False, "warn": 25.0, "crit": 30.0},
    "avg_flow_m3h":     {"tag": "flow_rate_m3_h",         "type": "LOW_FLOW",     "low": True,  "warn": 85.0, "crit": 70.0},
    "avg_pressure_bar": {"tag": "discharge_pressure_bar", "type": "LOW_PRESSURE", "low": True,  "warn":  6.5, "crit":  5.5},
}


# ── Telemetry JSON schema  (matches TelemetryEvent produced by .NET) ──────────

TELEMETRY_SCHEMA = StructType([
    StructField("asset_id",   StringType(),  True),
    StructField("asset_type", StringType(),  True),
    StructField("ts",         StringType(),  True),   # ISO-8601 Z string
    StructField("tag",        StringType(),  True),
    StructField("value",      DoubleType(),  True),
    StructField("quality",    IntegerType(), True),
])


# ── Low-level JDBC helpers ────────────────────────────────────────────────────

def jdbc_write(df, table: str, mode: str = "append") -> None:
    """Append (or overwrite) a DataFrame into a SQL Server table via JDBC."""
    (
        df.write
        .format("jdbc")
        .option("dbtable", table)
        .options(**JDBC_OPTS)
        .mode(mode)
        .save()
    )


def jdbc_execute(spark: SparkSession, sql: str) -> None:
    """
    Execute arbitrary DDL/DML (e.g. MERGE) against SQL Server using py4j.
    The MSSQL JDBC driver is already on Spark's classpath (via --packages),
    so we load it with the current thread's context classloader and invoke
    DriverManager directly — no extra Python packages required.
    """
    jvm = spark.sparkContext._jvm
    cl  = jvm.java.lang.Thread.currentThread().getContextClassLoader()
    jvm.java.lang.Class.forName(JDBC_DRIVER, True, cl)

    info = jvm.java.util.Properties()
    info.setProperty("user", JDBC_USER)
    info.setProperty("password", JDBC_PASSWORD)

    conn = jvm.java.sql.DriverManager.getConnection(JDBC_URL, info)
    try:
        conn.setAutoCommit(True)
        conn.createStatement().executeUpdate(sql)
    finally:
        conn.close()


def _suppress_dup(exc: Exception, table: str, batch_id: int) -> None:
    """
    Re-raise unless it's a SQL Server duplicate-key error (2627 / 2601).
    Those happen only on Spark checkpoint replay — safe to skip.
    """
    msg = str(exc)
    if "2627" in msg or "2601" in msg:
        log.warning(f"Duplicate key in {table} batch {batch_id} — skipped (checkpoint replay)")
    else:
        raise exc


# ── SparkSession ──────────────────────────────────────────────────────────────

def build_spark() -> SparkSession:
    return (
        SparkSession.builder
        .appName("IndustrialTelemetryStreaming")
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.streaming.stopGracefullyOnShutdown", "true")
        .getOrCreate()
    )


# ── Kafka ingestion & parsing ─────────────────────────────────────────────────

def read_kafka(spark: SparkSession):
    return (
        spark.readStream
        .format("kafka")
        .option("kafka.bootstrap.servers", KAFKA_SERVERS)
        .option("subscribe", KAFKA_TOPIC)
        .option("startingOffsets", "latest")
        .option("failOnDataLoss", "false")
        .option("kafka.group.id", KAFKA_GROUP)
        .load()
    )


def parse_events(raw_df):
    """
    Deserialise Kafka value (JSON string) → typed DataFrame.
    Drops:
      - rows where ts cannot be parsed
      - rows with OPC UA quality < 192 (Good) — e.g. Uncertain (64)
    """
    return (
        raw_df
        .select(
            from_json(col("value").cast("string"), TELEMETRY_SCHEMA).alias("e"),
            col("offset").alias("kafka_offset"),
            col("partition").alias("kafka_partition"),
        )
        .select("e.*", "kafka_offset", "kafka_partition")
        .withColumn("ts", to_timestamp(col("ts"), "yyyy-MM-dd'T'HH:mm:ss.SSS'Z'"))
        .filter(col("ts").isNotNull())
        .filter(col("asset_id").isNotNull())
        .filter(col("quality") == 192)
    )


# ── Stream 1 — Raw telemetry → dbo.telemetry_raw ─────────────────────────────

def write_raw_batch(batch_df, batch_id):
    if batch_df.isEmpty():
        return
    cnt = batch_df.count()
    log.info(f"[raw] batch={batch_id}  rows={cnt}")
    jdbc_write(
        batch_df.select(
            "asset_id", "asset_type", "ts", "tag",
            "value", "quality", "kafka_offset", "kafka_partition",
        ),
        "dbo.telemetry_raw",
    )


# ── Stream 2 — 1-minute windowed analytics ────────────────────────────────────

def spread_tags(df):
    """
    Create nullable per-tag columns so a single groupBy can avg across
    all tags per asset-minute without a pivot (which needs a fixed schema).
    """
    def tag_val(name):
        return when(col("tag") == name, col("value"))

    return (
        df
        .withColumn("temp_c",    tag_val("temp_bearing_c"))
        .withColumn("vib_mm_s",  tag_val("vib_rms_mm_s"))
        .withColumn("current_a", tag_val("motor_current_a"))
        .withColumn("flow_m3h",  tag_val("flow_rate_m3_h"))
        .withColumn("pres_bar",  tag_val("discharge_pressure_bar"))
        .withColumn("fault_int", when(col("tag") == "fault_code",
                                      col("value").cast("int")))
    )


def build_minute_agg(spread_df):
    """
    1-minute tumbling window per asset.
    Watermark of 2 minutes allows Spark to handle up to 2 minutes of late data
    before finalising a window.
    """
    return (
        spread_df
        .withWatermark("ts", "2 minutes")
        .groupBy(
            col("asset_id"),
            window(col("ts"), "1 minute").alias("w"),
        )
        .agg(
            avg("temp_c").alias("avg_temp_c"),
            avg("vib_mm_s").alias("avg_vib_mm_s"),
            avg("current_a").alias("avg_current_a"),
            avg("flow_m3h").alias("avg_flow_m3h"),
            avg("pres_bar").alias("avg_pressure_bar"),
            count("*").alias("event_count"),
            spark_max("fault_int").alias("fault_code_max"),
        )
        .withColumn("minute_ts", col("w.start"))
        .drop("w")
    )


# ── Anomaly detection + risk scoring (runs in driver, inside foreachBatch) ────

def detect_and_score(rows):
    """
    Pure-Python threshold scan over the aggregated rows.
    Returns (anomaly_rows, risk_rows) — lists of dicts ready for createDataFrame.

    For 5 assets this collect() is trivial.  For 1000+ assets, move the
    threshold logic into a Spark UDF or vectorised operation.
    """
    anomaly_rows, risk_rows = [], []

    for r in rows:
        asset_id  = r["asset_id"]
        minute_ts = r["minute_ts"]
        anoms     = []

        for col_name, cfg in THRESHOLDS.items():
            value = r[col_name]
            if value is None:
                continue

            if cfg["low"]:
                if   value < cfg["crit"]: sev, desc = 2, f"below critical {cfg['crit']}"
                elif value < cfg["warn"]: sev, desc = 1, f"below warning {cfg['warn']}"
                else: continue
            else:
                if   value > cfg["crit"]: sev, desc = 2, f"exceeds critical {cfg['crit']}"
                elif value > cfg["warn"]: sev, desc = 1, f"exceeds warning {cfg['warn']}"
                else: continue

            anoms.append({
                "asset_id":     asset_id,
                "minute_ts":    minute_ts,
                "anomaly_type": cfg["type"],
                "signal":       cfg["tag"],
                "score":        float(value),
                "severity":     sev,
                "reason":       f"{cfg['tag']}={value:.3f} {desc}",
                "evidence_json": None,
            })

        anomaly_rows.extend(anoms)

        # Risk score: 20 pts per critical anomaly, 10 per warning, cap 100.
        score  = min(sum(20 if a["severity"] == 2 else 10 for a in anoms), 100)
        level  = ("CRITICAL" if score >= 90 else
                  "HIGH"     if score >= 70 else
                  "MEDIUM"   if score >= 30 else "LOW")
        mode   = anoms[0]["anomaly_type"] if anoms else None
        drivers = ",".join(dict.fromkeys(a["anomaly_type"] for a in anoms)) or None

        risk_rows.append({
            "asset_id":     asset_id,
            "minute_ts":    minute_ts,
            "risk_score":   float(score),
            "risk_level":   level,
            "failure_mode": mode,
            "top_drivers":  drivers,
            "evidence_json": json.dumps({"anomaly_count": len(anoms)}),
        })

    return anomaly_rows, risk_rows


def _upsert_risk_current(spark: SparkSession, r: dict) -> None:
    """
    MERGE dbo.asset_risk_current for one asset using a raw JDBC statement.
    String values are single-quote escaped before interpolation.
    """
    def q(v):  # SQL-escape a nullable string
        return f"N'{str(v).replace(chr(39), chr(39)*2)}'" if v is not None else "NULL"

    sql = f"""
    MERGE dbo.asset_risk_current AS tgt
    USING (SELECT
        N'{r["asset_id"]}' AS asset_id,
        CAST(N'{r["minute_ts"]}' AS DATETIME2(0)) AS as_of_minute_ts,
        {r["risk_score"]}   AS risk_score,
        N'{r["risk_level"]}' AS risk_level,
        {q(r["failure_mode"])} AS failure_mode,
        {q(r["top_drivers"])}  AS top_drivers,
        {q(r["evidence_json"])} AS evidence_json
    ) AS src ON tgt.asset_id = src.asset_id
    WHEN MATCHED THEN UPDATE SET
        as_of_minute_ts = src.as_of_minute_ts,
        risk_score      = src.risk_score,
        risk_level      = src.risk_level,
        failure_mode    = src.failure_mode,
        top_drivers     = src.top_drivers,
        evidence_json   = src.evidence_json,
        updated_at      = SYSUTCDATETIME()
    WHEN NOT MATCHED THEN INSERT
        (asset_id, as_of_minute_ts, risk_score, risk_level,
         failure_mode, top_drivers, evidence_json)
    VALUES
        (src.asset_id, src.as_of_minute_ts, src.risk_score, src.risk_level,
         src.failure_mode, src.top_drivers, src.evidence_json);
    """
    try:
        jdbc_execute(spark, sql)
    except Exception as exc:
        log.error(f"MERGE risk_current failed for {r['asset_id']}: {exc}")


# ── Analytics foreachBatch handler ───────────────────────────────────────────

def write_analytics_batch(batch_df, batch_id):
    """
    Receives completed 1-minute windows (update output mode).
    One call = one or more finalised windows.
    """
    if batch_df.isEmpty():
        return

    spark = SparkSession.getActiveSession()
    rows  = batch_df.collect()
    log.info(f"[analytics] batch={batch_id}  windows={len(rows)}")

    # ── dbo.asset_minute_fact ─────────────────────────────────────────────────
    try:
        jdbc_write(
            batch_df.select(
                "asset_id", "minute_ts",
                "avg_temp_c", "avg_vib_mm_s", "avg_pressure_bar",
                "avg_current_a", "avg_flow_m3h",
                "event_count", "fault_code_max",
            ),
            "dbo.asset_minute_fact",
        )
    except Exception as e:
        _suppress_dup(e, "asset_minute_fact", batch_id)

    # ── dbo.asset_minute_features ─────────────────────────────────────────────
    # Slope/std columns require historical lookback — left NULL here.
    # The .NET feature pipeline (sp_build_asset_minute_features) can fill them
    # if you want full ML compatibility, or extend this job with a state store.
    try:
        jdbc_write(
            batch_df.select(
                col("asset_id"),
                col("minute_ts"),
                col("avg_temp_c").alias("temp_avg_15m"),
                col("avg_vib_mm_s").alias("vib_avg_15m"),
                col("avg_current_a").alias("current_avg_15m"),
                col("avg_flow_m3h").alias("flow_avg_15m"),
                col("avg_pressure_bar").alias("pressure_avg_15m"),
                lit(None).cast("double").alias("vib_std_60m"),
                lit(None).cast("double").alias("temp_slope_15m"),
                lit(None).cast("double").alias("vib_slope_15m"),
                lit(None).cast("double").alias("current_slope_15m"),
                lit(None).cast("double").alias("flow_drop_pct_15m"),
                col("event_count").cast("int").alias("run_minutes_60m"),
                when(col("event_count") >= 5, True)
                    .otherwise(False).alias("quality_gate_ok"),
            ),
            "dbo.asset_minute_features",
        )
    except Exception as e:
        _suppress_dup(e, "asset_minute_features", batch_id)

    # ── Anomaly detection + risk scoring ──────────────────────────────────────
    anomaly_rows, risk_rows = detect_and_score(rows)

    if anomaly_rows:
        try:
            jdbc_write(
                spark.createDataFrame(anomaly_rows),
                "dbo.asset_anomaly_events",
            )
        except Exception as e:
            _suppress_dup(e, "asset_anomaly_events", batch_id)

    if risk_rows:
        # ── dbo.asset_risk_minute ──────────────────────────────────────────
        try:
            jdbc_write(
                spark.createDataFrame(risk_rows).select(
                    "asset_id", "minute_ts", "risk_score",
                    "risk_level", "failure_mode", "top_drivers", "evidence_json",
                ),
                "dbo.asset_risk_minute",
            )
        except Exception as e:
            _suppress_dup(e, "asset_risk_minute", batch_id)

        # ── dbo.asset_risk_current (MERGE per asset) ───────────────────────
        for r in risk_rows:
            _upsert_risk_current(spark, r)


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    spark = build_spark()
    spark.sparkContext.setLogLevel("WARN")

    log.info("Industrial Telemetry Streaming — starting")
    log.info(f"  Kafka : {KAFKA_SERVERS}  topic={KAFKA_TOPIC}")
    log.info(f"  JDBC  : sqlserver:1433   db=Industrail_AI")
    log.info(f"  Checkpoints: {CHECKPOINT_DIR}")

    raw_df    = read_kafka(spark)
    events_df = parse_events(raw_df)

    # Stream 1 — raw events ──────────────────────────────────────────────────
    raw_query = (
        events_df.writeStream
        .foreachBatch(write_raw_batch)
        .option("checkpointLocation", f"{CHECKPOINT_DIR}/raw")
        .trigger(processingTime="10 seconds")
        .start()
    )

    # Stream 2 — 1-minute windowed analytics ─────────────────────────────────
    minute_agg = build_minute_agg(spread_tags(events_df))

    analytics_query = (
        minute_agg.writeStream
        .foreachBatch(write_analytics_batch)
        .option("checkpointLocation", f"{CHECKPOINT_DIR}/analytics")
        .outputMode("update")
        .trigger(processingTime="60 seconds")
        .start()
    )

    log.info(f"Streams running — raw={raw_query.id}  analytics={analytics_query.id}")
    spark.streams.awaitAnyTermination()


if __name__ == "__main__":
    main()
