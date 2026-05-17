# Industrial AI — Predictive Maintenance Platform

A real-time industrial asset monitoring system built on a modern big-data stack.
Simulated sensor telemetry flows from a Kafka producer through Apache Spark Structured Streaming, into SQL Server and Hive, and is served via a .NET API to a Blazor frontend and a Grafana dashboard.

---

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────────┐
│                         Data Ingestion                           │
│                                                                  │
│   .NET Producer  ──►  Kafka topic: industrial-telemetry (6 p.)  │
└──────────────────────────────┬───────────────────────────────────┘
                               │
┌──────────────────────────────▼───────────────────────────────────┐
│               Apache Spark Structured Streaming                   │
│                                                                  │
│  Stream 1 (10s)  — raw telemetry events → SQL Server + Hive     │
│  Stream 2 (60s)  — 1-minute tumbling window:                     │
│    ├── aggregate per asset per minute                            │
│    ├── Spark SQL JOIN with static asset_profiles.csv  ← BONUS   │
│    ├── threshold anomaly detection                               │
│    └── criticality-weighted risk scoring                         │
└──────────┬────────────────────────────────────┬──────────────────┘
           │                                    │
           ▼                                    ▼
    SQL Server 2022                    Hive (Parquet / Derby)
  (operational store)              (analytical / persistent store)
           │
    ┌──────┴────────┐
    │               │
    ▼               ▼
 .NET API        Grafana
    │           (port 3000)
    ▼
Blazor UI
(port 5055)
```

---

## Services

| Service | Technology | Port | Purpose |
|---|---|---|---|
| `sqlserver` | SQL Server 2022 | 1433 | Operational database — serves API and Grafana |
| `kafka` | Apache Kafka 3.9 (KRaft) | 9092 | Message broker — receives telemetry events |
| `spark` | Apache Spark 3.5.1 | 4040 (UI) | Streaming analytics engine |
| `grafana` | Grafana OSS | 3000 | Operations dashboard |
| `IndustrialTelemetry.Producer` | .NET 9 | — | Simulates 5-asset sensor feeds to Kafka |
| `IndustrialAnalytics.Api` | .NET 9 Minimal API | 5025 | REST API serving the Blazor UI |
| `IndustrialAnalytics.Worker` | .NET 9 Worker Service | — | Background risk/anomaly/recommendation workers |
| `IndustrialAnalytics.Ui` | Blazor Server | 5055 | Interactive web frontend |

---

## Data Flow

### 1. Telemetry Production
`IndustrialTelemetry.Producer` simulates 5 industrial assets publishing sensor readings every second to Kafka.

| Asset | Type | Key Signals |
|---|---|---|
| COMP_001 | Compressor | temp_bearing_c, motor_current_a, discharge_pressure_bar |
| FAN_001 | Fan | temp_bearing_c, vib_rms_mm_s |
| GBX_001 | Gearbox | temp_bearing_c, vib_rms_mm_s, motor_current_a |
| PUMP_001 | Pump | temp_bearing_c, vib_rms_mm_s, flow_rate_m3_h |
| PUMP_002 | Pump | temp_bearing_c, vib_rms_mm_s, flow_rate_m3_h |

Each event: `asset_id`, `asset_type`, `ts` (ISO-8601), `tag`, `value`, `quality` (OPC UA quality code).
Only events with quality code `192` (Good) pass the Spark filter.

### 2. Kafka
- Topic: `industrial-telemetry`, 6 partitions, KRaft mode (no ZooKeeper)
- Two listeners: `EXTERNAL://localhost:9092` (host producer), `PLAINTEXT://kafka:29092` (Spark inside Docker)

### 3. Spark Structured Streaming

**Stream 1 — Raw (10-second micro-batches)**
- Deserialises JSON from Kafka using a fixed schema
- Writes to `dbo.telemetry_raw` (SQL Server) and `industrial.telemetry_raw` (Hive Parquet)

**Stream 2 — Analytics (60-second micro-batches)**
- 1-minute tumbling window with 2-minute late-data watermark
- Aggregates per asset: avg temperature, vibration, current, flow, pressure; event count; max fault code
- **Spark SQL JOIN with static asset profiles** (see bonus section)
- Threshold-based anomaly detection; severity 1 (warning) or 2 (critical)
- Risk scoring adjusted by asset criticality weight from the static join
- Writes to 5 SQL Server tables + 5 Hive tables

### 4. SQL Server Tables (Operational)

| Table | Write mode | Description |
|---|---|---|
| `dbo.telemetry_raw` | append | Every raw sensor event |
| `dbo.asset_minute_fact` | append | Per-asset per-minute averages |
| `dbo.asset_minute_features` | append | ML-ready feature vectors |
| `dbo.asset_anomaly_events` | append | Detected threshold violations |
| `dbo.asset_risk_minute` | append | Risk score history per asset |
| `dbo.asset_risk_current` | MERGE | Latest risk snapshot per asset (dashboard source) |

### 5. Hive Tables (Analytical / Parquet)

| Table | Description |
|---|---|
| `industrial.telemetry_raw` | Raw event archive |
| `industrial.asset_minute_fact` | Minute aggregates |
| `industrial.asset_minute_features` | Feature vectors |
| `industrial.asset_anomaly_events` | Anomaly history |
| `industrial.asset_risk_minute` | Risk score history |
| `industrial.asset_enriched_minute` | Streaming + static profiles join output |

---

## Spark SQL Join — Streaming with Static Dataset (Bonus)

A core architectural feature: enriching the streaming aggregates with a static reference dataset using Spark SQL at query time.

**Static dataset** — `spark/jobs/asset_profiles.csv`:

| asset_id | location | asset_category | manufacturer | criticality | criticality_weight |
|---|---|---|---|---|---|
| COMP_001 | Building-A | Compressor | Atlas Copco | HIGH | 1.25 |
| FAN_001 | Building-B | Fan | Ebm-papst | MEDIUM | 1.00 |
| GBX_001 | Production-Floor | Gearbox | SEW-EURODRIVE | HIGH | 1.25 |
| PUMP_001 | Utility-Room | Pump | Grundfos | MEDIUM | 1.00 |
| PUMP_002 | Utility-Room | Pump | Grundfos | LOW | 0.80 |

**How it works:**
1. At Spark startup, the CSV is loaded as a broadcast static DataFrame and registered as a Spark SQL temp view `asset_profiles`
2. Each analytics micro-batch is registered as a temp view `streaming_minute_batch`
3. A Spark SQL `LEFT JOIN` enriches every streaming window:

```sql
SELECT
    s.asset_id, s.minute_ts,
    s.avg_temp_c, s.avg_vib_mm_s, s.avg_current_a, s.avg_flow_m3h, s.avg_pressure_bar,
    s.event_count, s.fault_code_max,
    COALESCE(p.location,                  'Unknown') AS location,
    COALESCE(p.asset_category,            'Unknown') AS asset_category,
    COALESCE(p.manufacturer,              'Unknown') AS manufacturer,
    COALESCE(p.maintenance_interval_days, 90)        AS maintenance_interval_days,
    COALESCE(p.criticality,               'MEDIUM')  AS criticality,
    COALESCE(p.criticality_weight,        1.0)       AS criticality_weight
FROM streaming_minute_batch s
LEFT JOIN asset_profiles p ON s.asset_id = p.asset_id
```

4. The `criticality_weight` adjusts risk scores: HIGH-criticality assets (COMP_001, GBX_001) score 25% higher for the same sensor readings — making them prioritised for maintenance
5. The enriched result is persisted to `industrial.asset_enriched_minute` in Hive as Parquet

---

## REST API Endpoints

Base URL: `http://localhost:5025`

| Method | Path | Description |
|---|---|---|
| GET | `/api/v1/assets` | List all assets with current risk |
| GET | `/api/v1/assets/{id}/summary` | Latest risk + recommendation counts |
| GET | `/api/v1/assets/{id}/risk` | Risk score time series (`from`, `to`, `stepMinutes`) |
| GET | `/api/v1/assets/{id}/anomalies` | Anomaly events in time range |
| GET | `/api/v1/assets/{id}/recommendations` | Open/acked/closed recommendations |
| POST | `/api/v1/assets/{id}/insights` | AI-generated insight (Ollama LLM) |
| PUT | `/api/v1/recommendations/{id}/ack` | Acknowledge a recommendation |
| PUT | `/api/v1/recommendations/{id}/close` | Close a recommendation |

---

## Grafana Dashboard

URL: `http://localhost:3000` — credentials: `admin` / `admin`

Navigate: **Dashboards → Industrial AI → Industrial AI — Asset Monitoring**

| Panel | Visualisation | Source |
|---|---|---|
| Current Risk Score by Asset | Bar gauge (gradient, green→red) | `asset_risk_current` |
| Risk Level by Asset | Table with colour-coded risk_score | `asset_risk_current` |
| Avg Temperature per Minute | Time series (°C, thresholds at 75/88°C) | `asset_minute_fact` |
| Avg Vibration per Minute | Time series (mm/s, thresholds at 4/7) | `asset_minute_fact` |
| Risk Score Over Time | Time series (0–100, multi-asset) | `asset_risk_minute` |
| Anomaly Events | Table with severity colour coding | `asset_anomaly_events` |

An **Asset** dropdown at the top (multi-select, includes "All") filters all panels simultaneously using `${asset_id:sqlstring}` in each panel's SQL.

---

## Blazor UI

URL: `http://localhost:5055`

**Fleet view** — lists all assets with their current risk badge (LOW/MEDIUM/HIGH/CRITICAL) and count of open recommendations.

**Asset detail** — for each asset:
- Date/time range picker (defaults to today)
- Risk trend sparkline (SVG) with anomaly event markers as red dashed lines
- **Anomaly trend chart** (SVG) — each anomaly plotted by time (x) and severity 1–5 (y), coloured by signal type; hover any dot for full details
- Anomaly events table (sorted newest first, severity colour-coded)
- Risk points table
- Open recommendations with ACK and CLOSE actions
- AI-generated insight card (headline, likely causes, next steps) via Ollama

---

## AI Features

### Anomaly Detection (Deterministic)
Spark and .NET workers both apply threshold-based detection:

| Signal | Anomaly Type | Warning | Critical |
|---|---|---|---|
| temp_bearing_c | HIGH_TEMP | > 75°C | > 88°C |
| vib_rms_mm_s | HIGH_VIB | > 4 mm/s | > 7 mm/s |
| motor_current_a | HIGH_CURRENT | > 25 A | > 30 A |
| flow_rate_m3_h | LOW_FLOW | < 85 m³/h | < 70 m³/h |
| discharge_pressure_bar | LOW_PRESSURE | < 6.5 bar | < 5.5 bar |

Severity 1 (warning) adds 10 to the base risk score; severity 2 (critical) adds 20.
The base score is then multiplied by the asset's `criticality_weight` (0.80–1.25) from the static profiles join.

### Risk Scoring
Risk score (0–100) → level: LOW < 30, MEDIUM 30–70, HIGH 70–90, CRITICAL ≥ 90
Updated every minute per asset; `asset_risk_current` is MERGE-upserted for instant dashboard reads.

### Generative AI (Ollama)
The `/insights` API endpoint uses a local Ollama LLM constrained to a strict JSON schema:
- Headline + 2-sentence summary
- Likely causes (from detected anomaly drivers only)
- Recommended next steps
- Disclaimer

The LLM never detects anomalies or computes risk — it only explains deterministic evidence already in the database.

---

## Running the Platform

### Prerequisites
- Docker Desktop (4 GB RAM minimum for SQL Server + Kafka + Spark)
- .NET 9 SDK

### Start infrastructure

```bash
docker compose up -d
```

SQL Server and Kafka initialise automatically.
Spark downloads Maven packages on first boot (~2 min); subsequent restarts are fast.

### Start the telemetry producer

```bash
cd src/IndustrialTelemetry.Producer && dotnet run
```

### Start the API

```bash
cd src/IndustrialAnalytics.Api && dotnet run
```

### (Optional) Start background workers

```bash
cd src/IndustrialAnalytics.Worker && dotnet run
```

### Start the Blazor UI

```bash
cd src/IndustrialAnalytics.Ui && dotnet run
```

### Verify data is flowing

```bash
# Spark logs (watch for [raw] and [analytics] batch lines)
docker compose logs -f spark

# Query SQL Server
docker exec -it industrial_sqlserver \
  /opt/mssql-tools18/bin/sqlcmd -S localhost -U sa -P 'IndustrialAI#2026' -C \
  -Q "SELECT asset_id, risk_score, risk_level, failure_mode FROM Industrail_AI.dbo.asset_risk_current"
```

### Query Hive Parquet files with DuckDB (optional)

```bash
python3 -m pip install duckdb
python3 -c "
import duckdb
print(duckdb.query(\"SELECT asset_id, COUNT(*) AS rows FROM read_parquet('spark/hive-warehouse/industrial.db/asset_enriched_minute/**/*.parquet') GROUP BY 1\").df())
"
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Message broker | Apache Kafka 3.9 (KRaft — no ZooKeeper) |
| Stream processing | Apache Spark 3.5.1 Structured Streaming |
| Analytical store | Apache Hive (Parquet) with embedded Derby metastore |
| Operational store | Microsoft SQL Server 2022 |
| REST API | .NET 9 Minimal API + Dapper |
| Background workers | .NET 9 Worker Services |
| Frontend | Blazor Server (.NET 9) |
| Dashboard | Grafana OSS with provisioned MSSQL datasource |
| Containerisation | Docker Compose |
| Stream processing language | Python 3 / PySpark |
| API / UI language | C# 13 / .NET 9 |
| Generative AI | Ollama (local LLM, OpenAI-compatible API) |
