"""Persist streaming inference results as Digital Twin state in SQL Server.

Integration choice
------------------
Spark writes to SQL Server directly over JDBC. That is not a new idea in
this repository - ``spark/jobs/industrial_streaming_analytics.py``
already writes six tables this way and already executes a MERGE through
py4j for ``asset_risk_current``. Following the same path means:

- no new broker, database, service or HTTP hop;
- the SQL Server JDBC driver is already a known, pinned dependency;
- the "current state + history" shape mirrors the existing
  ``asset_risk_current`` / ``asset_risk_minute`` pair, so the .NET
  repositories read it with their existing conventions.

An ingestion REST endpoint in the .NET API was the alternative. It was
rejected because it adds a network hop and a second deployable to the
critical path for data Spark can already write, and it would have been
the only writer in the system that does not use JDBC.

Volume makes this safe: one row per 2048-sample window, so a 12 kHz
asset produces ~6 rows/second even at full real-time replay.

What is written
---------------
``dbo.asset_twin_inference_history``  append-only, one row per window.
``dbo.asset_twin_current``            one row per asset, upserted.

Both carry the same column groups, and demo ground truth lives in
``demo_*`` columns that are written for scoring and never read back into
inference.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Sequence


JDBC_DRIVER = "com.microsoft.sqlserver.jdbc.SQLServerDriver"

DEFAULT_JDBC_URL = (
    "jdbc:sqlserver://sqlserver:1433;"
    "databaseName=Industrail_AI;"
    "encrypt=true;"
    "trustServerCertificate=true;"
)
DEFAULT_JDBC_USER = "sa"
DEFAULT_JDBC_PASSWORD = "IndustrialAI#2026"

# Column order shared by both statements. Declared once so the INSERT
# and the MERGE can never drift apart.
PAYLOAD_COLUMNS: tuple[str, ...] = (
    "asset_id",
    "window_start_sequence",
    "window_end_sequence",
    "sample_count",
    "is_anomalous",
    "anomaly_score",
    "anomaly_threshold",
    "predicted_class",
    "confidence",
    "class_probabilities_json",
    "vibration_rms",
    "vibration_std",
    "vibration_peak",
    "vibration_peak_to_peak",
    "vibration_kurtosis",
    "vibration_skewness",
    "crest_factor",
    "motor_load_hp",
    "rotational_speed_rpm",
    "demo_recording_id",
    "demo_fault_class",
    "demo_fault_severity_in",
)

# Insert only when this window has not already been recorded. Spark's
# `update` output mode can re-emit a completed group, and Kafka delivery
# is at-least-once, so replay must not duplicate history rows.
HISTORY_INSERT_SQL = f"""
INSERT INTO dbo.asset_twin_inference_history (
    asset_id, inferred_at_utc,
    window_start_sequence, window_end_sequence, sample_count,
    is_anomalous, anomaly_score, anomaly_threshold,
    predicted_class, confidence, class_probabilities_json,
    vibration_rms, vibration_std, vibration_peak, vibration_peak_to_peak,
    vibration_kurtosis, vibration_skewness, crest_factor,
    motor_load_hp, rotational_speed_rpm,
    demo_recording_id, demo_fault_class, demo_fault_severity_in
)
SELECT ?, SYSUTCDATETIME(),
       ?, ?, ?,
       ?, ?, ?,
       ?, ?, ?,
       ?, ?, ?, ?,
       ?, ?, ?,
       ?, ?,
       ?, ?, ?
WHERE NOT EXISTS (
    SELECT 1 FROM dbo.asset_twin_inference_history
    WHERE asset_id = ? AND window_end_sequence = ?
);
""".strip()

# Current state: last write wins. The writer applies rows in
# (asset_id, window_end_sequence) order within a batch, so the newest
# window of a batch is the one that survives. A fresh replay of a
# different recording restarts sequence numbers at 0 and is expected to
# take over the twin - which is exactly what a demo wants.
CURRENT_MERGE_SQL = """
MERGE dbo.asset_twin_current AS target
USING (SELECT ? AS asset_id) AS source
   ON target.asset_id = source.asset_id
WHEN MATCHED THEN UPDATE SET
    last_updated_utc         = SYSUTCDATETIME(),
    window_start_sequence    = ?,
    window_end_sequence      = ?,
    sample_count             = ?,
    is_anomalous             = ?,
    anomaly_score            = ?,
    anomaly_threshold        = ?,
    predicted_class          = ?,
    confidence               = ?,
    class_probabilities_json = ?,
    vibration_rms            = ?,
    vibration_std            = ?,
    vibration_peak           = ?,
    vibration_peak_to_peak   = ?,
    vibration_kurtosis       = ?,
    vibration_skewness       = ?,
    crest_factor             = ?,
    motor_load_hp            = ?,
    rotational_speed_rpm     = ?,
    demo_recording_id        = ?,
    demo_fault_class         = ?,
    demo_fault_severity_in   = ?,
    updated_at               = SYSUTCDATETIME()
WHEN NOT MATCHED THEN INSERT (
    asset_id, last_updated_utc,
    window_start_sequence, window_end_sequence, sample_count,
    is_anomalous, anomaly_score, anomaly_threshold,
    predicted_class, confidence, class_probabilities_json,
    vibration_rms, vibration_std, vibration_peak, vibration_peak_to_peak,
    vibration_kurtosis, vibration_skewness, crest_factor,
    motor_load_hp, rotational_speed_rpm,
    demo_recording_id, demo_fault_class, demo_fault_severity_in
) VALUES (
    ?, SYSUTCDATETIME(),
    ?, ?, ?,
    ?, ?, ?,
    ?, ?, ?,
    ?, ?, ?, ?,
    ?, ?, ?,
    ?, ?,
    ?, ?, ?
);
""".strip()


def result_to_row(result: dict[str, Any]) -> dict[str, Any]:
    """Flatten an inference result into the persistence column shape.

    This is the single mapping from MODEL OUTPUT to Digital Twin state.
    Ground truth is copied into ``demo_*`` columns only - it is never
    merged into the model-output or feature groups.
    """
    features = result["features"]
    operating = result.get("operatingContext") or {}
    anomaly = result["anomaly"]
    classification = result["classification"]
    truth = result.get("groundTruth") or {}

    probabilities = classification.get("probabilities") or {}
    return {
        "asset_id": str(result["assetId"]),
        "window_start_sequence": int(result["windowStartSequence"]),
        "window_end_sequence": int(result["windowEndSequence"]),
        "sample_count": int(result["sampleCount"]),
        # MODEL OUTPUT
        "is_anomalous": bool(anomaly["isAnomalous"]),
        "anomaly_score": float(anomaly["score"]),
        "anomaly_threshold": float(anomaly["threshold"]),
        "predicted_class": str(classification["predictedClass"]),
        "confidence": (
            None
            if classification.get("confidence") is None
            else float(classification["confidence"])
        ),
        "class_probabilities_json": (
            json.dumps(probabilities, separators=(",", ":")) if probabilities else None
        ),
        # ENGINEERED FEATURES
        "vibration_rms": float(features["vibrationRms"]),
        "vibration_std": float(features["vibrationStd"]),
        "vibration_peak": float(features["vibrationPeak"]),
        "vibration_peak_to_peak": float(features["vibrationPeakToPeak"]),
        "vibration_kurtosis": float(features["vibrationKurtosis"]),
        "vibration_skewness": float(features["vibrationSkewness"]),
        "crest_factor": float(features["crestFactor"]),
        # OPERATING CONTEXT
        "motor_load_hp": (
            None if operating.get("motorLoadHp") is None else float(operating["motorLoadHp"])
        ),
        "rotational_speed_rpm": (
            None
            if operating.get("rotationalSpeedRpm") is None
            else float(operating["rotationalSpeedRpm"])
        ),
        # DEMO GROUND TRUTH ONLY
        "demo_recording_id": truth.get("recordingId"),
        "demo_fault_class": truth.get("faultClass"),
        "demo_fault_severity_in": (
            None
            if truth.get("faultSeverityIn") is None
            else float(truth["faultSeverityIn"])
        ),
    }


def _ordered_values(row: dict[str, Any]) -> list[Any]:
    return [row[column] for column in PAYLOAD_COLUMNS]


@dataclass(frozen=True)
class TwinSinkConfig:
    url: str = DEFAULT_JDBC_URL
    user: str = DEFAULT_JDBC_USER
    password: str = DEFAULT_JDBC_PASSWORD


class TwinSink:
    """Writes Digital Twin state through the JVM's JDBC driver via py4j.

    The driver jar is already on Spark's classpath (``--packages
    com.microsoft.sqlserver:mssql-jdbc``), exactly as the existing
    analytics job arranges it, so no Python DB driver is introduced.
    """

    def __init__(self, spark, config: TwinSinkConfig) -> None:
        self._spark = spark
        self._config = config

    def _connect(self):
        """Open a JDBC connection without going through DriverManager.

        ``--packages`` puts mssql-jdbc on Spark's *context* classloader,
        but ``java.sql.DriverManager`` only consults drivers registered
        with the system classloader, so it answers "No suitable driver
        found" even though the jar is demonstrably loaded. Instantiating
        the driver and calling ``connect`` directly sidesteps that
        lookup entirely and needs no extraClassPath tuning.
        """
        jvm = self._spark.sparkContext._jvm
        loader = jvm.java.lang.Thread.currentThread().getContextClassLoader()
        # Ensure the class is initialised on the context classloader
        # before py4j is asked to construct it.
        jvm.java.lang.Class.forName(JDBC_DRIVER, True, loader)
        # py4j instantiates by package path. Class.getDeclaredConstructor()
        # is varargs and py4j cannot resolve it with zero arguments, so
        # this direct form is used instead.
        driver = jvm.com.microsoft.sqlserver.jdbc.SQLServerDriver()

        properties = jvm.java.util.Properties()
        properties.setProperty("user", self._config.user)
        properties.setProperty("password", self._config.password)

        connection = driver.connect(self._config.url, properties)
        if connection is None:
            raise RuntimeError(
                f"{JDBC_DRIVER} refused the URL {self._config.url!r}"
            )
        return connection

    @staticmethod
    def _bind(statement, index: int, value: Any) -> None:
        if value is None:
            # VARCHAR is accepted by the driver for any nullable column.
            statement.setNull(index, 12)
        elif isinstance(value, bool):
            statement.setBoolean(index, value)
        elif isinstance(value, int):
            statement.setLong(index, value)
        elif isinstance(value, float):
            statement.setDouble(index, value)
        else:
            statement.setString(index, str(value))

    def write(self, results: Sequence[dict[str, Any]]) -> tuple[int, int]:
        """Persist results. Returns ``(history_rows, current_upserts)``."""
        if not results:
            return (0, 0)

        rows = [result_to_row(r) for r in results]
        # Apply in window order so the newest window of the batch is the
        # one that ends up as current state.
        rows.sort(key=lambda r: (r["asset_id"], r["window_end_sequence"]))

        connection = self._connect()
        history_written = 0
        current_written = 0
        try:
            connection.setAutoCommit(False)

            history = connection.prepareStatement(HISTORY_INSERT_SQL)
            try:
                for row in rows:
                    values = _ordered_values(row)
                    for offset, value in enumerate(values, start=1):
                        self._bind(history, offset, value)
                    # Trailing NOT EXISTS parameters.
                    self._bind(history, len(values) + 1, row["asset_id"])
                    self._bind(history, len(values) + 2, row["window_end_sequence"])
                    history_written += int(history.executeUpdate())
            finally:
                history.close()

            current = connection.prepareStatement(CURRENT_MERGE_SQL)
            try:
                for row in rows:
                    values = _ordered_values(row)
                    # MERGE binds the payload three times: the USING
                    # key, the UPDATE list, and the INSERT list.
                    self._bind(current, 1, row["asset_id"])
                    for offset, value in enumerate(values[1:], start=2):
                        self._bind(current, offset, value)
                    base = len(values)
                    for offset, value in enumerate(values, start=base + 1):
                        self._bind(current, offset, value)
                    current_written += int(current.executeUpdate())
            finally:
                current.close()

            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

        return (history_written, current_written)


__all__ = [
    "CURRENT_MERGE_SQL",
    "DEFAULT_JDBC_PASSWORD",
    "DEFAULT_JDBC_URL",
    "DEFAULT_JDBC_USER",
    "HISTORY_INSERT_SQL",
    "JDBC_DRIVER",
    "PAYLOAD_COLUMNS",
    "TwinSink",
    "TwinSinkConfig",
    "result_to_row",
]
