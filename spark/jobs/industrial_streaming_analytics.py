"""
Industrial Analytics — Spark Structured Streaming
==================================================
Reads from Kafka topic `industrial-telemetry`, processes in real-time,
and writes results to SQL Server (operational) and Hive (persistent).

Write targets
  ┌──────────────────────────────┬─────────────┬──────┐
  │ Table                        │ SQL Server  │ Hive │
  ├──────────────────────────────┼─────────────┼──────┤
  │ telemetry_raw                │ append 10s  │  ✓   │
  │ asset_minute_fact            │ append 60s  │  ✓   │
  │ asset_minute_features        │ append 60s  │  ✓   │
  │ asset_anomaly_events         │ append 60s  │  ✓   │
  │ asset_risk_minute            │ append 60s  │  ✓   │
  │ asset_risk_current           │ MERGE 60s   │  —   │  ← dashboard
  │ asset_enriched_minute (Hive) │  —          │  ✓   │  ← SQL join
  └──────────────────────────────┴─────────────┴──────┘

Spark SQL join (bonus)
  asset_profiles.csv is loaded at startup as a static broadcast DataFrame
  and registered as a Spark SQL temp view.  Each analytics micro-batch is
  also registered as a temp view so a plain SQL SELECT … JOIN … can enrich
  the streaming aggregates with static asset metadata (location, criticality,
  manufacturer, maintenance schedule).  The criticality_weight column is used
  to adjust the raw risk score, making HIGH-criticality assets score higher
  for the same sensor readings.

Hive uses an embedded Derby metastore and stores data as Parquet files
in the managed warehouse.  SQL Server remains the operational serving
database for the .NET API and Grafana dashboard.

Hive writes are non-fatal: if the metastore is unavailable, the error
is logged and SQL Server writes continue uninterrupted.
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, coalesce, from_json, to_timestamp,
    avg, count, max as spark_max,
    when, lit, window, broadcast,
)
from pyspark.sql.types import (
    StructType, StructField,
    StringType, DoubleType, IntegerType, TimestampType,
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

CHECKPOINT_DIR      = "/opt/spark/checkpoints"
HIVE_WAREHOUSE_DIR  = "/opt/spark/hive-warehouse"
DERBY_METASTORE_DIR = "/opt/spark/metastore/derby-metastore"
HIVE_DB             = "industrial"

ASSET_PROFILES_CSV  = "/opt/spark/jobs/asset_profiles.csv"

JDBC_OPTS = {
    "url":      JDBC_URL,
    "user":     JDBC_USER,
    "password": JDBC_PASSWORD,
    "driver":   JDBC_DRIVER,
}

# Anomaly thresholds — mirrors .NET HybridAnomalyDetector thresholds.
THRESHOLDS = {
    "avg_temp_c":       {"tag": "temp_bearing_c",         "type": "HIGH_TEMP",    "low": False, "warn": 75.0, "crit": 88.0},
    "avg_vib_mm_s":     {"tag": "vib_rms_mm_s",           "type": "HIGH_VIB",     "low": False, "warn":  4.0, "crit":  7.0},
    "avg_current_a":    {"tag": "motor_current_a",        "type": "HIGH_CURRENT", "low": False, "warn": 25.0, "crit": 30.0},
    "avg_flow_m3h":     {"tag": "flow_rate_m3_h",         "type": "LOW_FLOW",     "low": True,  "warn": 85.0, "crit": 70.0},
    "avg_pressure_bar": {"tag": "discharge_pressure_bar", "type": "LOW_PRESSURE", "low": True,  "warn":  6.5, "crit":  5.5},
}


# ── Schemas for Python-built DataFrames (evidence_json is nullable String) ────

ANOMALY_SCHEMA = StructType([
    StructField("asset_id",      StringType(),    True),
    StructField("minute_ts",     TimestampType(), True),
    StructField("anomaly_type",  StringType(),    True),
    StructField("signal",        StringType(),    True),
    StructField("score",         DoubleType(),    True),
    StructField("severity",      IntegerType(),   True),
    StructField("reason",        StringType(),    True),
    StructField("evidence_json", StringType(),    True),
])

RISK_SCHEMA = StructType([
    StructField("asset_id",      StringType(),    True),
    StructField("minute_ts",     TimestampType(), True),
    StructField("risk_score",    DoubleType(),    True),
    StructField("risk_level",    StringType(),    True),
    StructField("failure_mode",  StringType(),    True),
    StructField("top_drivers",   StringType(),    True),
    StructField("evidence_json", StringType(),    True),
])


# ── Telemetry JSON schema  (matches TelemetryEvent produced by .NET) ──────────

TELEMETRY_SCHEMA = StructType([
    StructField("asset_id",   StringType(),  True),
    StructField("asset_type", StringType(),  True),
    StructField("ts",         StringType(),  True),   # ISO-8601 Z string
    StructField("tag",        StringType(),  True),
    StructField("value",      DoubleType(),  True),
    StructField("quality",    IntegerType(), True),
])


# ── SQL Server helpers ────────────────────────────────────────────────────────

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
    The MSSQL JDBC driver is already on Spark's classpath (via --packages).
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
    SQL Server puts the error number in a property, not the message text,
    so we match on the message text instead.
    """
    msg = str(exc)
    if ("2627" in msg or "2601" in msg
            or "PRIMARY KEY" in msg
            or "duplicate key" in msg.lower()):
        log.warning(f"Duplicate key in {table} batch={batch_id} — skipped (checkpoint replay)")
    else:
        raise exc


# ── Hive helpers ──────────────────────────────────────────────────────────────

def init_hive_db(spark: SparkSession) -> None:
    """Create the Hive database on first run (idempotent)."""
    spark.sql(f"CREATE DATABASE IF NOT EXISTS {HIVE_DB}")
    log.info(f"[hive] Database '{HIVE_DB}' ready — warehouse: {HIVE_WAREHOUSE_DIR}")


# ── Static dataset — asset profiles ───────────────────────────────────────────

def load_asset_profiles(spark: SparkSession) -> None:
    """
    Load asset_profiles.csv as a broadcast static DataFrame and register it
    as a Spark SQL temp view so foreachBatch handlers can join against it
    with plain SQL without re-reading the file on every micro-batch.

    Columns: asset_id, location, asset_category, manufacturer, install_year,
             rated_capacity_pct, maintenance_interval_days, criticality,
             criticality_weight
    """
    profiles_df = (
        spark.read
        .option("header", "true")
        .option("inferSchema", "true")
        .csv(ASSET_PROFILES_CSV)
    )
    # broadcast hint keeps the small CSV in executor memory — no shuffle needed
    from pyspark.sql.functions import broadcast
    spark.createDataFrame(
        broadcast(profiles_df).collect(),
        schema=profiles_df.schema,
    ).createOrReplaceTempView("asset_profiles")

    count = profiles_df.count()
    log.info(f"[profiles] Loaded {count} asset profiles from {ASSET_PROFILES_CSV}")


def hive_write(df, table: str, batch_id: int = -1) -> None:
    """
    Append a DataFrame to a Hive managed table (Parquet).

    Non-fatal: logs errors and returns so SQL Server writes are unaffected.
    First call creates the table; subsequent calls append.
    """
    qualified = f"{HIVE_DB}.{table}"
    try:
        df.write.mode("append").saveAsTable(qualified)
        log.debug(f"[hive] batch={batch_id} → {qualified}")
    except Exception as exc:
        log.error(f"[hive] {qualified} batch={batch_id}: {exc}")


# ── SparkSession ──────────────────────────────────────────────────────────────

def build_spark() -> SparkSession:
    return (
        SparkSession.builder
        .appName("IndustrialTelemetryStreaming")
        # ── Hive catalog (embedded Derby metastore) ──────────────────────────
        .enableHiveSupport()
        .config("spark.sql.warehouse.dir", HIVE_WAREHOUSE_DIR)
        .config("spark.hadoop.javax.jdo.option.ConnectionURL",
                f"jdbc:derby:{DERBY_METASTORE_DIR};create=true")
        .config("spark.hadoop.javax.jdo.option.ConnectionDriverName",
                "org.apache.derby.jdbc.EmbeddedDriver")
        .config("spark.hadoop.datanucleus.autoCreateSchema", "true")
        .config("spark.hadoop.hive.metastore.schema.verification", "false")
        # ── Streaming tuning ─────────────────────────────────────────────────
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


# ── Stream 1 — Raw telemetry ──────────────────────────────────────────────────
# Destinations: dbo.telemetry_raw (SQL Server)  +  industrial.telemetry_raw (Hive)

def write_raw_batch(batch_df, batch_id):
    if batch_df.isEmpty():
        return
    cnt = batch_df.count()
    log.info(f"[raw] batch={batch_id}  rows={cnt}")

    raw_df = batch_df.select(
        "asset_id", "asset_type", "ts", "tag",
        "value", "quality", "kafka_offset", "kafka_partition",
    )

    # Hive first — persistent storage layer
    hive_write(raw_df, "telemetry_raw", batch_id)

    # SQL Server — operational log
    jdbc_write(raw_df, "dbo.telemetry_raw")


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


# ── Anomaly detection + risk scoring ──────────────────────────────────────────

def detect_and_score(rows, criticality_weights: dict = None):
    """
    Pure-Python threshold scan over the aggregated rows.

    criticality_weights — dict of {asset_id: float} from the static asset
    profiles join.  A weight > 1.0 (HIGH criticality) scales the raw risk
    score up so that the same sensor reading matters more for critical assets.

    Returns (anomaly_rows, risk_rows) — lists of dicts ready for createDataFrame.
    """
    weights       = criticality_weights or {}
    anomaly_rows  = []
    risk_rows     = []

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
                "asset_id":      asset_id,
                "minute_ts":     minute_ts,
                "anomaly_type":  cfg["type"],
                "signal":        cfg["tag"],
                "score":         float(value),
                "severity":      sev,
                "reason":        f"{cfg['tag']}={value:.3f} {desc}",
                "evidence_json": None,
            })

        anomaly_rows.extend(anoms)

        # Base score + criticality weight from static asset profiles join
        base_score = sum(20 if a["severity"] == 2 else 10 for a in anoms)
        weight     = weights.get(asset_id, 1.0)
        score      = min(round(base_score * weight), 100)

        level   = ("CRITICAL" if score >= 90 else
                   "HIGH"     if score >= 70 else
                   "MEDIUM"   if score >= 30 else "LOW")
        mode    = anoms[0]["anomaly_type"] if anoms else None
        drivers = ",".join(dict.fromkeys(a["anomaly_type"] for a in anoms)) or None

        risk_rows.append({
            "asset_id":      asset_id,
            "minute_ts":     minute_ts,
            "risk_score":    float(score),
            "risk_level":    level,
            "failure_mode":  mode,
            "top_drivers":   drivers,
            "evidence_json": json.dumps({
                "anomaly_count":       len(anoms),
                "criticality_weight":  weight,
            }),
        })

    return anomaly_rows, risk_rows


def _upsert_risk_current(spark: SparkSession, r: dict) -> None:
    """MERGE dbo.asset_risk_current for one asset (SQL Server only)."""
    def q(v):
        return f"N'{str(v).replace(chr(39), chr(39)*2)}'" if v is not None else "NULL"

    sql = f"""
    MERGE dbo.asset_risk_current AS tgt
    USING (SELECT
        N'{r["asset_id"]}' AS asset_id,
        CAST(N'{r["minute_ts"]}' AS DATETIME2(0)) AS as_of_minute_ts,
        {r["risk_score"]}    AS risk_score,
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
# Each table is written to both SQL Server and Hive.
# asset_risk_current is SQL Server only (requires MERGE for the .NET dashboard).

def write_analytics_batch(batch_df, batch_id):
    """
    Receives completed 1-minute windows (update output mode).
    One call = one or more finalised windows.
    """
    if batch_df.isEmpty():
        return

    try:
        _write_analytics_batch_inner(batch_df, batch_id)
    except Exception as exc:
        log.error(f"[analytics] batch={batch_id} UNHANDLED: {exc}", exc_info=True)


def _write_analytics_batch_inner(batch_df, batch_id):
    spark = SparkSession.getActiveSession()
    rows  = batch_df.collect()
    log.info(f"[analytics] batch={batch_id}  windows={len(rows)}")

    # ── asset_minute_fact ─────────────────────────────────────────────────────
    fact_df = batch_df.select(
        "asset_id", "minute_ts",
        "avg_temp_c", "avg_vib_mm_s", "avg_pressure_bar",
        "avg_current_a", "avg_flow_m3h",
        "event_count", "fault_code_max",
    )
    # Hive first — a dup-key abort on SQL Server must not block the Hive path.
    hive_write(fact_df, "asset_minute_fact", batch_id)
    try:
        jdbc_write(fact_df, "dbo.asset_minute_fact")
    except Exception as e:
        _suppress_dup(e, "asset_minute_fact", batch_id)

    # ── Spark SQL join: streaming batch ⋈ static asset profiles ──────────────
    # Use the DataFrame API join so both sides share the same SparkSession.
    # A temp view registered via batch_df.createOrReplaceTempView is invisible
    # to SparkSession.getActiveSession().sql() because they are different session
    # objects inside foreachBatch.
    try:
        profiles_df = spark.table("asset_profiles")
        s = batch_df.alias("s")
        p = broadcast(profiles_df).alias("p")
        enriched_df = s.join(p, col("s.asset_id") == col("p.asset_id"), "left").select(
            col("s.asset_id"),
            col("s.minute_ts"),
            col("s.avg_temp_c"),
            col("s.avg_vib_mm_s"),
            col("s.avg_current_a"),
            col("s.avg_flow_m3h"),
            col("s.avg_pressure_bar"),
            col("s.event_count"),
            col("s.fault_code_max"),
            coalesce(col("p.location"),                   lit("Unknown")).alias("location"),
            coalesce(col("p.asset_category"),             lit("Unknown")).alias("asset_category"),
            coalesce(col("p.manufacturer"),               lit("Unknown")).alias("manufacturer"),
            coalesce(col("p.install_year"),               lit(0)).alias("install_year"),
            coalesce(col("p.rated_capacity_pct"),         lit(100.0)).alias("rated_capacity_pct"),
            coalesce(col("p.maintenance_interval_days"),  lit(90)).alias("maintenance_interval_days"),
            coalesce(col("p.criticality"),                lit("MEDIUM")).alias("criticality"),
            coalesce(col("p.criticality_weight"),         lit(1.0)).alias("criticality_weight"),
        )
        hive_write(enriched_df, "asset_enriched_minute", batch_id)
        log.info(f"[sql-join] batch={batch_id} enriched rows with asset profiles")
    except Exception as exc:
        log.error(f"[sql-join] batch={batch_id}: {exc}")

    # ── asset_minute_features ─────────────────────────────────────────────────
    feat_df = batch_df.select(
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
    )
    hive_write(feat_df, "asset_minute_features", batch_id)
    try:
        jdbc_write(feat_df, "dbo.asset_minute_features")
    except Exception as e:
        _suppress_dup(e, "asset_minute_features", batch_id)

    # ── Anomaly detection + risk scoring ──────────────────────────────────────
    try:
        weight_map = {
            r["asset_id"]: float(r["criticality_weight"])
            for r in spark.sql(
                "SELECT asset_id, criticality_weight FROM asset_profiles"
            ).collect()
        }
    except Exception:
        weight_map = {}

    anomaly_rows, risk_rows = detect_and_score(rows, weight_map)

    if anomaly_rows:
        anom_df = spark.createDataFrame(anomaly_rows, ANOMALY_SCHEMA)
        hive_write(anom_df, "asset_anomaly_events", batch_id)
        try:
            jdbc_write(anom_df, "dbo.asset_anomaly_events")
        except Exception as e:
            _suppress_dup(e, "asset_anomaly_events", batch_id)

    if risk_rows:
        risk_df = spark.createDataFrame(risk_rows, RISK_SCHEMA).select(
            "asset_id", "minute_ts", "risk_score",
            "risk_level", "failure_mode", "top_drivers", "evidence_json",
        )
        hive_write(risk_df, "asset_risk_minute", batch_id)
        try:
            jdbc_write(risk_df, "dbo.asset_risk_minute")
        except Exception as e:
            _suppress_dup(e, "asset_risk_minute", batch_id)

        for r in risk_rows:
            _upsert_risk_current(spark, r)


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    spark = build_spark()
    spark.sparkContext.setLogLevel("WARN")

    log.info("Industrial Telemetry Streaming — starting")
    log.info(f"  Kafka      : {KAFKA_SERVERS}  topic={KAFKA_TOPIC}")
    log.info(f"  SQL Server : sqlserver:1433   db=Industrail_AI")
    log.info(f"  Hive       : {HIVE_WAREHOUSE_DIR}   db={HIVE_DB}")
    log.info(f"  Derby      : {DERBY_METASTORE_DIR}")
    log.info(f"  Checkpoints: {CHECKPOINT_DIR}")

    try:
        init_hive_db(spark)
    except Exception as exc:
        log.error(f"[hive] Metastore init failed — Hive writes will be skipped: {exc}")

    # Load static asset profiles CSV and register as Spark SQL temp view.
    # This is the "join streaming with static dataset" step — the view persists
    # for the lifetime of the SparkSession and is joined in every analytics batch.
    try:
        load_asset_profiles(spark)
    except Exception as exc:
        log.error(f"[profiles] Failed to load asset profiles — enrichment disabled: {exc}")

    raw_df    = read_kafka(spark)
    events_df = parse_events(raw_df)

    # Stream 1 — raw events (10 s micro-batches) ─────────────────────────────
    raw_query = (
        events_df.writeStream
        .foreachBatch(write_raw_batch)
        .option("checkpointLocation", f"{CHECKPOINT_DIR}/raw")
        .trigger(processingTime="10 seconds")
        .start()
    )

    # Stream 2 — 1-minute windowed analytics ──────────────────────────────────
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
