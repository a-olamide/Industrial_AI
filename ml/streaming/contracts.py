"""Message contracts for the online vibration-inference pipeline.

Two contracts live here:

1. :class:`VibrationTelemetryEvent` - what the replay simulator publishes
   to Kafka, one message per raw vibration sample.
2. :class:`InferenceResult` - what the Spark job emits per completed
   2048-sample window.

The single most important property of this module is the **physical
separation of ground truth from sensor telemetry**.

A CWRU recording knows its own fault class, defect diameter and
recording id. A real machine does not. Those three fields travel in a
nested ``sourceScenario`` object, are modelled by a separate frozen
dataclass (:class:`SourceScenario`), and are never reachable from the
feature path: :func:`feature_inputs` returns a dict built from the
sensor/operating fields only, and :data:`GROUND_TRUTH_FIELDS` is
asserted against every feature contract in the test-suite.

Field naming follows the camelCase shape specified for this pipeline.
Note that this deliberately differs from the legacy
``industrial-telemetry`` topic, whose ``TelemetryEvent`` uses snake_case
(``asset_id``/``tag``/``value``) and carries one row per *sensor tag*
rather than per vibration sample. The two topics carry different
contracts, so they stay separate rather than being forced together.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

from pyspark.sql.types import (
    BooleanType,
    DoubleType,
    IntegerType,
    LongType,
    MapType,
    StringType,
    StructField,
    StructType,
)


# Kafka topic for the raw vibration sample stream. Dotted namespace so it
# is unambiguous against the legacy per-tag topic `industrial-telemetry`.
VIBRATION_TOPIC = "industrial.telemetry.vibration"
INFERENCE_TOPIC = "industrial.inference.vibration"

# Non-overlapping window length, identical to the offline training
# pipeline (ml/src/feature_extraction.py::DEFAULT_WINDOW_SIZE).
WINDOW_SIZE = 2048

# Ground-truth / demo-only fields. These exist so a demonstration can be
# scored against reality. They must never reach feature engineering or
# model inference.
GROUND_TRUTH_FIELDS: tuple[str, ...] = (
    "faultClass",
    "faultSeverityIn",
    "recordingId",
)

# The operating-context fields that ARE legitimate model inputs: a real
# plant knows its own motor load and shaft speed from the drive.
OPERATING_CONTEXT_FIELDS: tuple[str, ...] = (
    "motorLoadHp",
    "rotationalSpeedRpm",
)

SENSOR_FIELDS: tuple[str, ...] = (
    "assetId",
    "timestampUtc",
    "sequenceNumber",
    "vibration",
)


@dataclass(frozen=True)
class SourceScenario:
    """DEMO / GROUND-TRUTH ONLY. Never an input to feature engineering.

    A deployed asset cannot report its own fault class or defect
    diameter; that is precisely what the models are asked to infer.
    This object is carried alongside the telemetry so the demo can be
    scored, and is kept in its own type so that passing it into the
    feature path requires a deliberate, visible mistake rather than an
    accidental dict spread.
    """

    faultClass: str
    faultSeverityIn: float | None
    recordingId: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class OperatingContext:
    """Drive-reported operating point. A legitimate model input."""

    motorLoadHp: float
    rotationalSpeedRpm: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class VibrationTelemetryEvent:
    """One raw drive-end vibration sample as published to Kafka.

    ``sequenceNumber`` is the sample index within the replayed recording
    and is monotonically increasing per asset. It is what makes window
    assignment deterministic and replay-safe: window index is exactly
    ``sequenceNumber // WINDOW_SIZE``, so a window never depends on
    micro-batch boundaries, arrival order or partition assignment.
    """

    assetId: str
    timestampUtc: str
    sequenceNumber: int
    vibration: float
    motorLoadHp: float
    rotationalSpeedRpm: float
    sourceScenario: SourceScenario | None = None

    @property
    def windowIndex(self) -> int:
        return int(self.sequenceNumber) // WINDOW_SIZE

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "assetId": self.assetId,
            "timestampUtc": self.timestampUtc,
            "sequenceNumber": int(self.sequenceNumber),
            "vibration": float(self.vibration),
            "motorLoadHp": float(self.motorLoadHp),
            "rotationalSpeedRpm": float(self.rotationalSpeedRpm),
        }
        if self.sourceScenario is not None:
            payload["sourceScenario"] = self.sourceScenario.to_dict()
        return payload

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), separators=(",", ":"))

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "VibrationTelemetryEvent":
        scenario = payload.get("sourceScenario")
        return cls(
            assetId=str(payload["assetId"]),
            timestampUtc=str(payload["timestampUtc"]),
            sequenceNumber=int(payload["sequenceNumber"]),
            vibration=float(payload["vibration"]),
            motorLoadHp=float(payload["motorLoadHp"]),
            rotationalSpeedRpm=float(payload["rotationalSpeedRpm"]),
            sourceScenario=(
                SourceScenario(
                    faultClass=str(scenario["faultClass"]),
                    faultSeverityIn=(
                        None
                        if scenario.get("faultSeverityIn") is None
                        else float(scenario["faultSeverityIn"])
                    ),
                    recordingId=str(scenario["recordingId"]),
                )
                if scenario
                else None
            ),
        )

    @classmethod
    def from_json(cls, raw: str) -> "VibrationTelemetryEvent":
        return cls.from_dict(json.loads(raw))


def feature_inputs(event: VibrationTelemetryEvent) -> dict[str, float]:
    """The ONLY sanctioned path from a telemetry event to model inputs.

    Returns the sensor reading plus the drive-reported operating point.
    ``sourceScenario`` is structurally unreachable from here - the
    function never touches it - so ground truth cannot leak into the
    feature vector by accident.
    """
    values = {
        "vibration": float(event.vibration),
        "motor_load_hp": float(event.motorLoadHp),
        "rotational_speed_rpm": float(event.rotationalSpeedRpm),
    }
    for forbidden in GROUND_TRUTH_FIELDS:
        assert forbidden not in values, f"ground truth {forbidden!r} leaked into features"
    return values


# ---------------------------------------------------------------------------
# Spark schemas
# ---------------------------------------------------------------------------

SOURCE_SCENARIO_SCHEMA = StructType(
    [
        StructField("faultClass", StringType(), True),
        StructField("faultSeverityIn", DoubleType(), True),
        StructField("recordingId", StringType(), True),
    ]
)

# Schema used by from_json() in the Spark job. sourceScenario is parsed so
# the demo can join ground truth onto results AFTER inference; the
# inference path selects only the sensor/operating columns.
VIBRATION_TELEMETRY_SCHEMA = StructType(
    [
        StructField("assetId", StringType(), True),
        StructField("timestampUtc", StringType(), True),
        StructField("sequenceNumber", LongType(), True),
        StructField("vibration", DoubleType(), True),
        StructField("motorLoadHp", DoubleType(), True),
        StructField("rotationalSpeedRpm", DoubleType(), True),
        StructField("sourceScenario", SOURCE_SCENARIO_SCHEMA, True),
    ]
)


# ---------------------------------------------------------------------------
# Inference result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AnomalyResult:
    isAnomalous: bool
    score: float
    threshold: float
    scoreDirection: str = (
        "score_samples: HIGHER = more normal, LOWER = more anomalous; "
        "ANOMALOUS when score < threshold"
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ClassificationResult:
    predictedClass: str
    confidence: float
    probabilities: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class InferenceResult:
    """One structured result per completed 2048-sample window."""

    assetId: str
    windowIndex: int
    windowStartSequence: int
    windowEndSequence: int
    timestampUtc: str
    sampleCount: int
    features: dict[str, float]
    operatingContext: dict[str, float]
    anomaly: dict[str, Any]
    classification: dict[str, Any]
    # Demo/evaluation only; attached AFTER inference, never before.
    groundTruth: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "assetId": self.assetId,
            "windowIndex": self.windowIndex,
            "windowStartSequence": self.windowStartSequence,
            "windowEndSequence": self.windowEndSequence,
            "timestampUtc": self.timestampUtc,
            "sampleCount": self.sampleCount,
            "features": self.features,
            "operatingContext": self.operatingContext,
            "anomaly": self.anomaly,
            "classification": self.classification,
        }
        if self.groundTruth is not None:
            payload["groundTruth"] = self.groundTruth
        return payload

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), separators=(",", ":"))


INFERENCE_RESULT_SCHEMA = StructType(
    [
        StructField("assetId", StringType(), True),
        StructField("windowIndex", LongType(), True),
        StructField("windowStartSequence", LongType(), True),
        StructField("windowEndSequence", LongType(), True),
        StructField("timestampUtc", StringType(), True),
        StructField("sampleCount", IntegerType(), True),
        StructField("vibrationRms", DoubleType(), True),
        StructField("vibrationStd", DoubleType(), True),
        StructField("vibrationPeak", DoubleType(), True),
        StructField("vibrationPeakToPeak", DoubleType(), True),
        StructField("vibrationKurtosis", DoubleType(), True),
        StructField("vibrationSkewness", DoubleType(), True),
        StructField("crestFactor", DoubleType(), True),
        StructField("motorLoadHp", DoubleType(), True),
        StructField("rotationalSpeedRpm", DoubleType(), True),
        StructField("anomalyIsAnomalous", BooleanType(), True),
        StructField("anomalyScore", DoubleType(), True),
        StructField("anomalyThreshold", DoubleType(), True),
        StructField("predictedClass", StringType(), True),
        StructField("confidence", DoubleType(), True),
        StructField("probabilities", MapType(StringType(), DoubleType()), True),
        StructField("groundTruthFaultClass", StringType(), True),
        StructField("groundTruthSeverityIn", DoubleType(), True),
        StructField("groundTruthRecordingId", StringType(), True),
    ]
)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


__all__ = [
    "GROUND_TRUTH_FIELDS",
    "INFERENCE_RESULT_SCHEMA",
    "INFERENCE_TOPIC",
    "OPERATING_CONTEXT_FIELDS",
    "SENSOR_FIELDS",
    "SOURCE_SCENARIO_SCHEMA",
    "VIBRATION_TELEMETRY_SCHEMA",
    "VIBRATION_TOPIC",
    "WINDOW_SIZE",
    "AnomalyResult",
    "ClassificationResult",
    "InferenceResult",
    "OperatingContext",
    "SourceScenario",
    "VibrationTelemetryEvent",
    "feature_inputs",
    "utc_now_iso",
]
