"""
Demo script — queries all Hive Parquet tables using DuckDB.
Run with:  python3 demo_hive_query.py
"""
import duckdb

BASE = "spark/hive-warehouse/industrial.db"


def query(label, sql):
    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print("=" * 60)
    try:
        df = duckdb.query(sql).df()
        if df.empty:
            print("  (table exists but no rows yet)")
        else:
            print(df.to_string(index=False))
    except Exception as e:
        print(f"  (no data yet — {e})")


# ── Row counts ────────────────────────────────────────────────────────────────
query("Hive table row counts", f"""
    SELECT 'telemetry_raw'          AS hive_table, COUNT(*) AS rows
    FROM read_parquet('{BASE}/telemetry_raw/**/*.parquet')
    UNION ALL
    SELECT 'asset_minute_fact',     COUNT(*)
    FROM read_parquet('{BASE}/asset_minute_fact/**/*.parquet')
    UNION ALL
    SELECT 'asset_anomaly_events',  COUNT(*)
    FROM read_parquet('{BASE}/asset_anomaly_events/**/*.parquet')
    UNION ALL
    SELECT 'asset_risk_minute',     COUNT(*)
    FROM read_parquet('{BASE}/asset_risk_minute/**/*.parquet')
    UNION ALL
    SELECT 'asset_enriched_minute', COUNT(*)
    FROM read_parquet('{BASE}/asset_enriched_minute/**/*.parquet')
""")

# ── Raw telemetry ─────────────────────────────────────────────────────────────
query("Latest 5 raw telemetry events", f"""
    SELECT asset_id, asset_type, ts, tag, ROUND(value, 2) AS value, quality
    FROM read_parquet('{BASE}/telemetry_raw/**/*.parquet')
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
    FROM read_parquet('{BASE}/asset_minute_fact/**/*.parquet')
    ORDER BY minute_ts DESC, asset_id
    LIMIT 10
""")

# ── Anomalies ─────────────────────────────────────────────────────────────────
query("Recent anomaly events", f"""
    SELECT asset_id, minute_ts, anomaly_type, signal,
           severity, ROUND(score, 2) AS score, reason
    FROM read_parquet('{BASE}/asset_anomaly_events/**/*.parquet')
    ORDER BY minute_ts DESC
    LIMIT 10
""")

# ── Risk scores ───────────────────────────────────────────────────────────────
query("Risk scores over time (most recent per asset)", f"""
    SELECT asset_id, minute_ts,
           ROUND(risk_score, 1) AS risk_score,
           risk_level, failure_mode, top_drivers
    FROM read_parquet('{BASE}/asset_risk_minute/**/*.parquet')
    ORDER BY minute_ts DESC
    LIMIT 10
""")

# ── Spark SQL JOIN result (BONUS) ─────────────────────────────────────────────
query("BONUS — Spark SQL JOIN: streaming data enriched with static asset profiles", f"""
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
    FROM read_parquet('{BASE}/asset_enriched_minute/**/*.parquet')
    ORDER BY minute_ts DESC
    LIMIT 10
""")
