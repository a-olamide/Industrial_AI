"""
Demo script — queries all Hive Parquet tables using DuckDB.
Run with:  python3 demo_hive_query.py
"""
import duckdb

BASE = "spark/hive-warehouse/industrial.db"

TABLES = [
    "telemetry_raw",
    "asset_minute_fact",
    "asset_anomaly_events",
    "asset_risk_minute",
    "asset_enriched_minute",
]


def run(sql):
    con = duckdb.connect()
    rel = con.query(sql)
    cols = [d[0] for d in rel.description]
    rows = rel.fetchall()
    return cols, rows


def _fmt(val):
    return "" if val is None else str(val)


def print_table(cols, rows):
    if not rows:
        print("  (table exists but no rows yet)")
        return
    widths = [max(len(c), max((len(_fmt(r[i])) for r in rows), default=0))
              for i, c in enumerate(cols)]
    sep = "  " + "  ".join("-" * w for w in widths)
    header = "  " + "  ".join(c.ljust(widths[i]) for i, c in enumerate(cols))
    print(header)
    print(sep)
    for row in rows:
        print("  " + "  ".join(_fmt(row[i]).ljust(widths[i]) for i in range(len(cols))))


def query(label, sql):
    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print("=" * 60)
    try:
        cols, rows = run(sql)
        print_table(cols, rows)
    except Exception as e:
        print(f"  (no data yet — {e})")


def table_path(name):
    return f"{BASE}/{name}/*.parquet"


# ── Row counts ────────────────────────────────────────────────────────────────
print(f"\n{'=' * 60}")
print("  Hive table row counts")
print("=" * 60)
for t in TABLES:
    try:
        con = duckdb.connect()
        n = con.query(f"SELECT COUNT(*) FROM read_parquet('{table_path(t)}')").fetchone()[0]
        print(f"  {t:<30} {n:>8} rows")
    except Exception as e:
        print(f"  {t:<30}        0 rows  (no parquet files yet)")

# ── Raw telemetry ─────────────────────────────────────────────────────────────
query("Latest 5 raw telemetry events", f"""
    SELECT asset_id, asset_type, ts, tag, ROUND(value, 2) AS value, quality
    FROM read_parquet('{BASE}/telemetry_raw/*.parquet')
    ORDER BY ts DESC
    LIMIT 5
""")

# ── Minute aggregates ─────────────────────────────────────────────────────────
query("Latest minute aggregates per asset", f"""
    SELECT asset_id, minute_ts,
           ROUND(avg_temp_c, 1)       AS avg_temp_c,
           ROUND(avg_vib_mm_s, 3)     AS avg_vib_mm_s,
           ROUND(avg_current_a, 2)    AS avg_current_a,
           ROUND(avg_flow_m3h, 2)     AS avg_flow_m3h,
           event_count
    FROM read_parquet('{BASE}/asset_minute_fact/*.parquet')
    ORDER BY minute_ts DESC, asset_id
    LIMIT 10
""")

# ── Anomalies ─────────────────────────────────────────────────────────────────
query("Recent anomaly events", f"""
    SELECT asset_id, minute_ts, anomaly_type, signal,
           severity, ROUND(score, 2) AS score, reason
    FROM read_parquet('{BASE}/asset_anomaly_events/*.parquet')
    ORDER BY minute_ts DESC
    LIMIT 10
""")

# ── Risk scores ───────────────────────────────────────────────────────────────
query("Risk scores over time (most recent per asset)", f"""
    SELECT asset_id, minute_ts,
           ROUND(risk_score, 1) AS risk_score,
           risk_level, failure_mode, top_drivers
    FROM read_parquet('{BASE}/asset_risk_minute/*.parquet')
    ORDER BY minute_ts DESC
    LIMIT 10
""")

# ── Spark SQL JOIN result ─────────────────────────────────────────────
query("Spark SQL JOIN: streaming data enriched with static asset profiles", f"""
    SELECT
        asset_id,
        minute_ts,
        ROUND(avg_temp_c, 1)    AS avg_temp_c,
        ROUND(avg_vib_mm_s, 3)  AS avg_vib_mm_s,
        location,
        asset_category,
        manufacturer,
        criticality,
        criticality_weight,
        maintenance_interval_days
    FROM read_parquet('{BASE}/asset_enriched_minute/*.parquet')
    ORDER BY minute_ts DESC
    LIMIT 10
""")
