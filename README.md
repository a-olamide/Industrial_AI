# Industrial AI Platform — Predictive Maintenance, Risk Scoring & Insights

An end-to-end **Industrial AI** system that ingests raw telemetry, aggregates to minute-level facts, engineers features, detects anomalies, scores asset risk, produces maintenance recommendations, and generates **executive-friendly AI insights** (LLM) from deterministic evidence.

> **Core idea:** separate **signal detection** (deterministic + auditable) from **language generation** (LLM explanation only).

---

## What I Built

This project combines three major capabilities:

### 1) Predictive AI (Detection)
- Converts high-frequency telemetry to structured facts/features.
- Detects:
  - **spikes** (sudden events) using rolling statistics / Z-score
  - **drift / degradation** using trend slopes over rolling windows and persistence checks

### 2) Prescriptive AI (Decisioning)
- Converts risk/anomaly patterns into **actionable recommendations**
- Adds **cooldown logic** to avoid noisy repeated alerts

### 3) Generative AI (Explanation)
- Uses **Ollama (local LLM)** to generate an executive insight card:
  - headline + summary
  - likely causes / drivers
  - next steps
  - confidence + disclaimer  
- Guardrails: LLM is constrained to **only the facts provided** (no hallucinated sensors).

---

## High-level Architecture
![generated-image](https://github.com/user-attachments/assets/13994273-d34b-4447-aef7-b4ca217d17ba)

### Data Model (Core Tables)
### Table	(Description)

```sql
dbo.telemetry_raw	Raw high-frequency telemetry from industrial assets (timestamp, asset_id, tag, value, quality).

dbo.asset_minute_fact	Minute-level aggregated facts (avg temperature, vibration, current, flow, pressure, run_state, fault count).

dbo.asset_minute_features	Engineered features from minute facts (rolling averages, rolling standard deviation, trend slopes).

dbo.asset_anomaly_events	Detected anomalies with type, signal, severity, score, and human-readable reason.

dbo.asset_risk_minute	Time-series risk score per asset per minute.

dbo.asset_risk_current	Latest risk snapshot per asset (used by UI and APIs).

dbo.asset_recommendations	Prescriptive maintenance recommendations with lifecycle (OPEN / ACKED / CLOSED).

dbo.worker_checkpoint	Checkpoints used by background workers to process data incrementally.
```

### Stored Procedures
```sql
dbo.sp_build_asset_minute_fact(@from, @to)
```
- Aggregates raw telemetry into minute-granularity facts:
 - Averages analog signals (temperature, vibration, current, flow, pressure)
 - Majority-vote run state
 - Fault counts per minute
 - Data quality percentage

```python
dbo.sp_build_asset_minute_features(@from, @to)
```
  - Builds analytical features from minute facts:
  - Rolling averages (e.g. 15-minute)
  - Rolling standard deviation (e.g. 60-minute)
  - Rolling slopes (trend detection)
These features are used by anomaly detection and risk scoring workers.

## Workers (Background Processing)
## AnomalyWorker
- Consumes minute-level features
- Detects:
  - Spikes using rolling statistics and Z-scores
  - Drift / degradation using slope analysis and persistence rules
- Writes anomaly events to asset_anomaly_events

## RiskWorker

- Computes rolling risk scores from anomalies
- Produces:
  - asset_risk_minute (risk time series)
  - asset_risk_current (latest snapshot)
- Identifies dominant risk drivers

## RecommendationWorker
- Converts risk states and anomaly patterns into actionable recommendations
- Applies cooldown rules to avoid alert fatigue
- Writes results to asset_recommendations

## API Endpoints (Examples)
```
GET /api/v1/assets
GET /api/v1/assets/{assetId}/summary
GET /api/v1/assets/{assetId}/risk?from=...&to=...
GET /api/v1/assets/{assetId}/anomalies?from=...&to=...
GET /api/v1/assets/{assetId}/recommendations?status=OPEN
GET /api/v1/assets/{assetId}/insight?from=...&to=...
```

All endpoints return structured JSON suitable for dashboards and downstream systems.

## Generative AI (Ollama Insight)
A dedicated insight endpoint generates natural-language explanations of asset health using a local LLM (Ollama).

## Key Principles

- The LLM does not detect anomalies or compute risk.
- It receives only deterministic facts (risk trends, anomalies, recommendations).
- Output is constrained to a strict JSON schema.
- Safe fallback is returned if the model fails.

## Purpose

Translate complex industrial signals into:
- Executive-friendly summaries
- Likely causes
- Recommended next steps
- Evidence-backed explanations

## Running the Demo (Suggested Flow)
Insert telemetry into dbo.telemetry_raw

Run:
```sql
EXEC dbo.sp_build_asset_minute_fact @from, @to;
EXEC dbo.sp_build_asset_minute_features @from, @to;
```

## Start background workers:

- AnomalyWorker
- RiskWorker
- RecommendationWorker

## Launch API and UI
Explore:

 - Risk trends and anomalies
 - Active recommendations
 - AI-generated asset insights

## Tech Stack
 - Backend: .NET (Minimal APIs, Background Services)
 - Database: SQL Server (T-SQL, MERGE, time-series aggregation)
 - Analytics: Rolling statistics, slope-based trend analysis
 - Generative AI: Ollama (local LLM inference)
 - Frontend: Blazor
 - Logging: Serilog

