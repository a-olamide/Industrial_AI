-- ============================================================
-- Industrial Analytics Platform — Schema Initialisation
-- Compatible with SQL Server 2022 (Docker container)
-- Idempotent: every CREATE is guarded with IF OBJECT_ID checks
-- ============================================================

IF NOT EXISTS (SELECT 1 FROM sys.databases WHERE name = N'Industrail_AI')
BEGIN
    CREATE DATABASE [Industrail_AI];
    PRINT 'Database Industrail_AI created.';
END
GO

USE [Industrail_AI];
GO

-- ── 1. worker_checkpoint ─────────────────────────────────────
-- Tracks the last-processed minute per background worker.
IF OBJECT_ID('dbo.worker_checkpoint', 'U') IS NULL
BEGIN
    CREATE TABLE dbo.worker_checkpoint (
        worker_name    NVARCHAR(128) NOT NULL,
        last_minute_ts DATETIME2(0)  NOT NULL,
        updated_at     DATETIME2(0)  NOT NULL CONSTRAINT df_wc_updated_at DEFAULT SYSUTCDATETIME(),

        CONSTRAINT pk_worker_checkpoint PRIMARY KEY (worker_name)
    );
    PRINT 'Created dbo.worker_checkpoint';
END
GO

-- ── 2. asset_minute_features ─────────────────────────────────
-- Pre-computed rolling statistical features per asset-minute.
-- Written by the ingestion pipeline; read-only within the app.
IF OBJECT_ID('dbo.asset_minute_features', 'U') IS NULL
BEGIN
    CREATE TABLE dbo.asset_minute_features (
        asset_id           NVARCHAR(64) NOT NULL,
        minute_ts          DATETIME2(0) NOT NULL,

        temp_avg_15m       FLOAT        NULL,
        vib_avg_15m        FLOAT        NULL,
        current_avg_15m    FLOAT        NULL,
        flow_avg_15m       FLOAT        NULL,
        pressure_avg_15m   FLOAT        NULL,
        vib_std_60m        FLOAT        NULL,
        temp_slope_15m     FLOAT        NULL,
        vib_slope_15m      FLOAT        NULL,
        current_slope_15m  FLOAT        NULL,
        flow_drop_pct_15m  FLOAT        NULL,
        run_minutes_60m    INT          NULL,
        quality_gate_ok    BIT          NOT NULL CONSTRAINT df_amf_quality DEFAULT 0,

        created_at         DATETIME2(0) NOT NULL CONSTRAINT df_amf_created_at DEFAULT SYSUTCDATETIME(),

        CONSTRAINT pk_asset_minute_features PRIMARY KEY (asset_id, minute_ts)
    );

    CREATE INDEX ix_amf_minute_ts
        ON dbo.asset_minute_features (minute_ts)
        INCLUDE (asset_id, quality_gate_ok);

    PRINT 'Created dbo.asset_minute_features';
END
GO

-- ── 3. asset_anomaly_events ──────────────────────────────────
-- One row per detected anomaly signal per asset-minute.
-- The unique index on (asset_id, minute_ts, anomaly_type, signal)
-- is what causes error 2601/2627 that AnomalyEventRepository suppresses.
IF OBJECT_ID('dbo.asset_anomaly_events', 'U') IS NULL
BEGIN
    CREATE TABLE dbo.asset_anomaly_events (
        anomaly_id    BIGINT        NOT NULL IDENTITY(1,1),
        asset_id      NVARCHAR(64)  NOT NULL,
        minute_ts     DATETIME2(0)  NOT NULL,
        anomaly_type  NVARCHAR(64)  NOT NULL,
        signal        NVARCHAR(64)  NOT NULL,
        score         FLOAT         NULL,
        severity      TINYINT       NOT NULL,
        reason        NVARCHAR(512) NULL,
        evidence_json NVARCHAR(MAX) NULL,
        created_at    DATETIME2(0)  NOT NULL CONSTRAINT df_aae_created_at DEFAULT SYSUTCDATETIME(),

        CONSTRAINT pk_asset_anomaly_events PRIMARY KEY (anomaly_id)
    );

    CREATE UNIQUE INDEX uix_aae_natural
        ON dbo.asset_anomaly_events (asset_id, minute_ts, anomaly_type, signal);

    CREATE INDEX ix_aae_asset_minute
        ON dbo.asset_anomaly_events (asset_id, minute_ts);

    PRINT 'Created dbo.asset_anomaly_events';
END
GO

-- ── 4. asset_risk_minute ─────────────────────────────────────
-- Risk score history — one row per asset per computed minute.
IF OBJECT_ID('dbo.asset_risk_minute', 'U') IS NULL
BEGIN
    CREATE TABLE dbo.asset_risk_minute (
        asset_id      NVARCHAR(64)  NOT NULL,
        minute_ts     DATETIME2(0)  NOT NULL,
        risk_score    FLOAT         NOT NULL,
        risk_level    NVARCHAR(16)  NOT NULL,
        failure_mode  NVARCHAR(128) NULL,
        top_drivers   NVARCHAR(512) NULL,
        evidence_json NVARCHAR(MAX) NULL,
        computed_at   DATETIME2(0)  NOT NULL CONSTRAINT df_arm_computed_at DEFAULT SYSUTCDATETIME(),

        CONSTRAINT pk_asset_risk_minute PRIMARY KEY (asset_id, minute_ts)
    );

    CREATE INDEX ix_arm_minute_ts
        ON dbo.asset_risk_minute (minute_ts)
        INCLUDE (asset_id, risk_score, risk_level);

    PRINT 'Created dbo.asset_risk_minute';
END
GO

-- ── 5. asset_risk_current ────────────────────────────────────
-- Latest risk snapshot per asset — upserted by RiskRepository on every run.
IF OBJECT_ID('dbo.asset_risk_current', 'U') IS NULL
BEGIN
    CREATE TABLE dbo.asset_risk_current (
        asset_id        NVARCHAR(64)  NOT NULL,
        as_of_minute_ts DATETIME2(0)  NOT NULL,
        risk_score      FLOAT         NOT NULL,
        risk_level      NVARCHAR(16)  NOT NULL,
        failure_mode    NVARCHAR(128) NULL,
        top_drivers     NVARCHAR(512) NULL,
        evidence_json   NVARCHAR(MAX) NULL,
        updated_at      DATETIME2(0)  NOT NULL CONSTRAINT df_arc_updated_at DEFAULT SYSUTCDATETIME(),

        CONSTRAINT pk_asset_risk_current PRIMARY KEY (asset_id)
    );

    PRINT 'Created dbo.asset_risk_current';
END
GO

-- ── 6. asset_recommendations ─────────────────────────────────
-- Generated maintenance recommendations with full workflow state
-- (OPEN → ACKED → CLOSED).
IF OBJECT_ID('dbo.asset_recommendations', 'U') IS NULL
BEGIN
    CREATE TABLE dbo.asset_recommendations (
        recommendation_id BIGINT        NOT NULL IDENTITY(1,1),
        asset_id          NVARCHAR(64)  NOT NULL,
        as_of_minute_ts   DATETIME2(0)  NOT NULL,
        rec_type          NVARCHAR(64)  NOT NULL,
        priority          TINYINT       NOT NULL,
        title             NVARCHAR(256) NOT NULL,
        description       NVARCHAR(MAX) NOT NULL,
        status            NVARCHAR(16)  NOT NULL CONSTRAINT df_ar_status DEFAULT 'OPEN',
        confidence        DECIMAL(7,4)  NOT NULL,
        drivers           NVARCHAR(MAX) NULL,
        evidence_json     NVARCHAR(MAX) NULL,
        cooldown_until    DATETIME2(0)  NULL,
        state_fingerprint NVARCHAR(256) NULL,
        acknowledged_at   DATETIME2(0)  NULL,
        acknowledged_by   NVARCHAR(128) NULL,
        ack_until         DATETIME2(0)  NULL,
        closed_at         DATETIME2(0)  NULL,
        closed_by         NVARCHAR(128) NULL,
        created_at        DATETIME2(0)  NOT NULL CONSTRAINT df_ar_created_at DEFAULT SYSUTCDATETIME(),
        updated_at        DATETIME2(0)  NOT NULL CONSTRAINT df_ar_updated_at DEFAULT SYSUTCDATETIME(),

        CONSTRAINT pk_asset_recommendations PRIMARY KEY (recommendation_id)
    );

    CREATE INDEX ix_ar_asset_status
        ON dbo.asset_recommendations (asset_id, status)
        INCLUDE (created_at, rec_type, as_of_minute_ts);

    CREATE INDEX ix_ar_status_created
        ON dbo.asset_recommendations (status, created_at DESC);

    PRINT 'Created dbo.asset_recommendations';
END
GO

-- ── 7. telemetry_raw ─────────────────────────────────────────
-- Raw event log written by Spark Stream 1 (append-only).
-- Serves as the source-of-truth audit trail from Kafka.
IF OBJECT_ID('dbo.telemetry_raw', 'U') IS NULL
BEGIN
    CREATE TABLE dbo.telemetry_raw (
        id               BIGINT        NOT NULL IDENTITY(1,1),
        asset_id         NVARCHAR(64)  NOT NULL,
        asset_type       NVARCHAR(64)  NULL,
        ts               DATETIME2(3)  NOT NULL,
        tag              NVARCHAR(64)  NOT NULL,
        value            FLOAT         NULL,
        quality          INT           NOT NULL,
        kafka_offset     BIGINT        NULL,
        kafka_partition  INT           NULL,
        inserted_at      DATETIME2(0)  NOT NULL CONSTRAINT df_tr_inserted_at DEFAULT SYSUTCDATETIME(),

        CONSTRAINT pk_telemetry_raw PRIMARY KEY (id)
    );

    CREATE INDEX ix_tr_asset_ts ON dbo.telemetry_raw (asset_id, ts);
    CREATE INDEX ix_tr_ts       ON dbo.telemetry_raw (ts);

    PRINT 'Created dbo.telemetry_raw';
END
GO

-- ── 8. asset_minute_fact ─────────────────────────────────────
-- 1-minute windowed aggregations written by Spark Stream 2.
-- Lightweight fact table; asset_minute_features holds the ML-ready columns.
IF OBJECT_ID('dbo.asset_minute_fact', 'U') IS NULL
BEGIN
    CREATE TABLE dbo.asset_minute_fact (
        asset_id          NVARCHAR(64)  NOT NULL,
        minute_ts         DATETIME2(0)  NOT NULL,
        avg_temp_c        FLOAT         NULL,
        avg_vib_mm_s      FLOAT         NULL,
        avg_pressure_bar  FLOAT         NULL,
        avg_current_a     FLOAT         NULL,
        avg_flow_m3h      FLOAT         NULL,
        event_count       INT           NOT NULL,
        fault_code_max    INT           NULL,
        computed_at       DATETIME2(0)  NOT NULL CONSTRAINT df_amf2_computed_at DEFAULT SYSUTCDATETIME(),

        CONSTRAINT pk_asset_minute_fact PRIMARY KEY (asset_id, minute_ts)
    );

    CREATE INDEX ix_amf2_minute_ts
        ON dbo.asset_minute_fact (minute_ts)
        INCLUDE (asset_id, avg_temp_c, avg_vib_mm_s);

    PRINT 'Created dbo.asset_minute_fact';
END
GO

PRINT '-------------------------------------------------------';
PRINT 'Schema initialisation complete — Industrail_AI is ready.';
PRINT '-------------------------------------------------------';
GO
