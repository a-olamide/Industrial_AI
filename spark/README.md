# Spark Structured Streaming — Industrial Analytics

## Architecture

```
.NET Producer
    │  JSON messages (35/sec, 5 assets × 7 tags)
    ▼
Kafka  topic: industrial-telemetry  (kafka:29092 inside Docker)
    │
    ▼
Spark Structured Streaming  (local[2], bitnami/spark:3.5)
    │
    ├── Stream 1 — every 10 s → dbo.telemetry_raw         (raw event log)
    │
    └── Stream 2 — every 60 s (1-min windows)
            ├── dbo.asset_minute_fact       (aggregated values)
            ├── dbo.asset_minute_features   (ML-ready features)
            ├── dbo.asset_anomaly_events    (threshold breaches)
            ├── dbo.asset_risk_minute       (risk score history)
            └── dbo.asset_risk_current      (latest risk per asset — MERGE)

.NET API + Blazor UI reads from SQL Server as before.
.NET Recommendation Worker continues to own dbo.asset_recommendations.
```

## Folder structure

```
spark/
├── submit.sh                           shell script run by Docker container
├── checkpoints/                        Spark streaming state (git-tracked dir)
│   ├── raw/                            created at runtime by Stream 1
│   └── analytics/                      created at runtime by Stream 2
├── jobs/
│   └── industrial_streaming_analytics.py   main PySpark job
└── README.md                           this file
```

## Prerequisites

| Tool | Version | Notes |
|------|---------|-------|
| Docker Desktop | 4.x+ | 3 GB RAM allocated minimum |
| .NET 9 SDK | 9.x | for the producer |

No local Spark or Java installation required — everything runs in Docker.

## Quick start

### 1. Start all infrastructure

```bash
cd /path/to/IndustrailAI
docker-compose up -d
```

Services start in dependency order:
- `sqlserver` → healthy → `sqlserver-init` (schema) → exits
- `kafka` → healthy → `kafka-init` (topic) → exits
- `spark` → starts streaming job (waits for kafka + sqlserver healthy)
- `grafana` → starts independently

First boot downloads Spark packages from Maven Central (~2 min).
Subsequent boots use the `spark_ivy_cache` volume and start in seconds.

### 2. Verify Kafka topic exists

```bash
docker logs industrial_kafka_init
# Expected: Created topic industrial-telemetry.
```

### 3. Start the .NET producer

```bash
cd src/IndustrialTelemetry.Producer   # or IndustrialTelemetry.Producer/
dotnet run --configuration Release
```

### 4. Watch Spark consume and write

```bash
# Follow Spark logs
docker logs -f industrial_spark

# Open Spark job UI (only available while the job is running)
open http://localhost:4040
```

### 5. Verify data in SQL Server

```bash
docker exec -it industrial_sqlserver \
  /opt/mssql-tools18/bin/sqlcmd \
  -S localhost -U sa -P "IndustrialAI#2026" -C \
  -Q "USE [Industrail_AI];
      SELECT TOP 5 asset_id, tag, value, ts FROM dbo.telemetry_raw ORDER BY id DESC;
      SELECT asset_id, minute_ts, avg_temp_c, risk_level
        FROM dbo.asset_minute_fact f
        JOIN dbo.asset_risk_current r ON f.asset_id = r.asset_id
       ORDER BY minute_ts DESC;"
```

## Responsibility split — Spark vs .NET

| Concern | Owner | Reason |
|---------|-------|--------|
| Raw telemetry ingest | Spark | high-throughput streaming |
| 1-min aggregation | Spark | native windowing |
| Anomaly detection (threshold) | Spark | co-located with aggregation |
| Risk scoring | Spark | derived from anomalies |
| ML feature engineering (slopes, std) | .NET stored procs OR extend Spark | requires historical state; easier in SQL |
| Recommendations | .NET worker | cooldown logic, ack tracking, business rules |
| Dashboard API | .NET API | reads SQL Server, unchanged |

## Troubleshooting

### Kafka connection error in Spark logs

```
org.apache.kafka.common.errors.TimeoutException: Topic ... not present
```

Check: `docker logs industrial_kafka_init` — the topic might not have been created yet.
Fix: `docker-compose up kafka-init`

### Package download fails on first start

Spark needs Maven Central access. If you're behind a proxy, add:
```yaml
environment:
  - JAVA_OPTS=-Dhttps.proxyHost=... -Dhttps.proxyPort=...
```
to the `spark:` service in docker-compose.yml.

### SQL Server JDBC errors

```
com.microsoft.sqlserver.jdbc.SQLServerException: Login failed for user 'sa'
```

The sqlserver container may still be starting. Spark has `depends_on: service_healthy` so this
should not happen in normal operation. If it does: `docker-compose restart spark`.

### Full reset (clean slate)

`submit.sh` automatically wipes three things on every container restart so they stay in sync:

| What | Why |
|------|-----|
| `spark/hive-warehouse/industrial.db/` | Prevents Hive `LOCATION_ALREADY_EXISTS` on `saveAsTable` |
| `spark/metastore/derby-metastore/` | Metastore and warehouse must be wiped together — stale metastore entries pointing to deleted directories cause silent write failures |
| `spark/checkpoints/raw` and `spark/checkpoints/analytics` | Stale checkpoints trigger SQL Server duplicate-key errors on replay, causing the analytics query to crash before Hive tables are written |

This means **all Hive Parquet files are deleted on each Spark restart** — data is re-streamed from Kafka after the container comes back up.

To trigger a manual reset, simply restart the container:
```bash
docker compose restart spark
```

Spark starts fresh from the latest Kafka offset (`startingOffsets: "latest"`).
To reprocess historical messages, temporarily change that to `"earliest"` in the job before restarting.

### Duplicate rows after Spark restart

No longer observed — stale checkpoints (the root cause) are wiped automatically by `submit.sh`.
If you do see SQL Server error 2627/2601 in the logs, the job catches them and logs a warning
rather than crashing.

### Apple Silicon (M1/M2/M3) notes

`bitnami/spark:3.5` publishes a native `linux/arm64` image. Docker Desktop uses it automatically.
If you see `WARNING: The requested image's platform (linux/amd64) does not match`, edit
`docker-compose.yml` and add `platform: linux/arm64` to the `spark:` service. Spark itself
and all the JVM JARs run natively on ARM — no Rosetta emulation needed.

## Extending the job

### Add 15-minute rolling features (slopes, std)

Use Spark's `Session Window` or maintain state with `flatMapGroupsWithState`.
Alternatively, keep running `sp_build_asset_minute_features` as a scheduled SQL Agent job
and point the .NET anomaly worker at the SQL-computed features.

### Scale to a real Spark cluster

Replace `--master local[2]` with `--master spark://spark-master:7077` and add
`spark-master` + `spark-worker` services (bitnami images support this via `SPARK_MODE=master/worker`).
The job code does not change.
