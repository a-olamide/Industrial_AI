"""Load the frozen Experiment-2/3 models and score streaming windows.

Nothing here trains, refits or mutates a model. The two artifacts are
loaded read-only:

- ``ml/models/rf_multiseverity_cwru.joblib`` - Experiment 2's Random
  Forest multi-severity fault classifier (9 features).
- ``ml/models/isolation_forest_cwru.joblib`` - Experiment 3's Isolation
  Forest anomaly detector (7 vibration features, inside a
  StandardScaler pipeline).

Feature order is never inferred by position or by luck. For each model
the order is read from the experiment's saved JSON metadata and then
cross-checked against the estimator's own ``feature_names_in_``. If the
two disagree, or if the streaming pipeline cannot supply a required
column, loading fails loudly rather than silently scoring a permuted
vector - which would produce confident, completely wrong predictions.

The Isolation Forest decision uses the threshold recorded in the
Experiment-3 metadata (the label-free 1% train-score quantile), NOT
sklearn's default ``predict()`` boundary, so online decisions match the
offline evaluation exactly.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np
import pandas as pd

from .contracts import (
    GROUND_TRUTH_FIELDS,
    AnomalyResult,
    ClassificationResult,
    InferenceResult,
)
from .stream_features import FEATURE_JSON_NAMES, StreamingWindowFeatures


DEFAULT_MODELS_DIR = Path(__file__).resolve().parents[2] / "ml" / "models"

CLASSIFIER_STEM = "rf_multiseverity_cwru"
ANOMALY_STEM = "isolation_forest_cwru"

# Columns the streaming pipeline can supply, and where each comes from.
#   - the seven vibration features come from the window statistics
#   - the two operating-point columns come from drive telemetry
STREAMING_FEATURE_SOURCES: dict[str, str] = {
    "vibration_rms": "window",
    "vibration_std": "window",
    "vibration_peak": "window",
    "vibration_peak_to_peak": "window",
    "vibration_kurtosis": "window",
    "vibration_skewness": "window",
    "crest_factor": "window",
    "rotational_speed_rpm": "operating_context",
    "motor_load_hp": "operating_context",
}


class ModelContractError(RuntimeError):
    """Raised when a saved model's feature contract cannot be honoured."""


def _load_metadata(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ModelContractError(
            f"missing model metadata {path}. Model artifacts are gitignored; "
            "re-run the experiment that produces it "
            "(python -m ml.src.experiment2_multiseverity / "
            "python -m ml.src.experiment3_anomaly_detection)."
        )
    return json.loads(path.read_text())


def _estimator_feature_names(model: Any) -> list[str] | None:
    names = getattr(model, "feature_names_in_", None)
    if names is None and hasattr(model, "named_steps"):
        for step in model.named_steps.values():
            names = getattr(step, "feature_names_in_", None)
            if names is not None:
                break
    return None if names is None else [str(n) for n in names]


def _validate_contract(
    label: str,
    model: Any,
    metadata_order: Sequence[str],
    allowed_sources: dict[str, str],
) -> tuple[str, ...]:
    """Reconcile metadata order, estimator order, and what streaming can supply."""
    order = [str(c) for c in metadata_order]
    if not order:
        raise ModelContractError(f"{label}: saved metadata declares no feature columns")

    estimator_order = _estimator_feature_names(model)
    if estimator_order is not None and estimator_order != order:
        raise ModelContractError(
            f"{label}: feature order mismatch between saved metadata and the fitted "
            f"estimator.\n  metadata : {order}\n  estimator: {estimator_order}\n"
            "Refusing to score - a permuted feature vector yields confident but "
            "meaningless predictions."
        )

    unsupplied = [c for c in order if c not in allowed_sources]
    if unsupplied:
        raise ModelContractError(
            f"{label}: streaming pipeline cannot supply required feature(s) "
            f"{unsupplied}. Available sources: {sorted(allowed_sources)}"
        )

    leaked = [c for c in order if c in GROUND_TRUTH_FIELDS or c in {
        "fault_class", "fault_severity_in", "recording_id", "source_file",
        "sampling_rate_hz", "window_id",
    }]
    if leaked:
        raise ModelContractError(
            f"{label}: saved contract contains non-observable column(s) {leaked}"
        )
    return tuple(order)


@dataclass(frozen=True)
class LoadedModels:
    """The two frozen estimators plus their validated feature contracts."""

    classifier: Any
    classifier_features: tuple[str, ...]
    classifier_classes: tuple[str, ...]
    classifier_metadata: dict[str, Any]
    classifier_path: Path

    anomaly: Any
    anomaly_features: tuple[str, ...]
    anomaly_threshold: float
    anomaly_threshold_name: str
    anomaly_metadata: dict[str, Any]
    anomaly_path: Path

    def describe(self) -> str:
        return (
            f"classifier : {self.classifier_path.name} "
            f"({type(self.classifier).__name__}, "
            f"{len(self.classifier_features)} features, "
            f"classes={list(self.classifier_classes)})\n"
            f"             feature order = {list(self.classifier_features)}\n"
            f"anomaly    : {self.anomaly_path.name} "
            f"({type(self.anomaly).__name__}, "
            f"{len(self.anomaly_features)} features)\n"
            f"             feature order = {list(self.anomaly_features)}\n"
            f"             threshold     = {self.anomaly_threshold:.6f} "
            f"({self.anomaly_threshold_name}; ANOMALOUS when score < threshold)"
        )


def load_models(models_dir: str | Path = DEFAULT_MODELS_DIR) -> LoadedModels:
    """Load both frozen artifacts and validate their feature contracts."""
    base = Path(models_dir).expanduser().resolve()

    clf_path = base / f"{CLASSIFIER_STEM}.joblib"
    clf_meta_path = base / f"{CLASSIFIER_STEM}.json"
    if_path = base / f"{ANOMALY_STEM}.joblib"
    if_meta_path = base / f"{ANOMALY_STEM}.json"

    for path in (clf_path, if_path):
        if not path.is_file():
            raise ModelContractError(
                f"missing model artifact {path}. Model binaries are gitignored; "
                "re-run the corresponding experiment to regenerate them."
            )

    clf_meta = _load_metadata(clf_meta_path)
    if_meta = _load_metadata(if_meta_path)

    classifier = joblib.load(clf_path)
    anomaly = joblib.load(if_path)

    classifier_features = _validate_contract(
        "RandomForest (Experiment 2)",
        classifier,
        clf_meta.get("feature_columns", []),
        STREAMING_FEATURE_SOURCES,
    )
    # The anomaly detector must see ONLY the seven vibration features.
    anomaly_allowed = {
        k: v for k, v in STREAMING_FEATURE_SOURCES.items() if v == "window"
    }
    anomaly_features = _validate_contract(
        "IsolationForest (Experiment 3)",
        anomaly,
        if_meta.get("feature_columns", []),
        anomaly_allowed,
    )
    if len(anomaly_features) != 7:
        raise ModelContractError(
            f"IsolationForest must consume exactly 7 vibration features, "
            f"got {len(anomaly_features)}: {list(anomaly_features)}"
        )

    threshold_block = if_meta.get("threshold", {})
    if "value" not in threshold_block:
        raise ModelContractError(
            "Experiment-3 metadata has no frozen threshold value; refusing to "
            "fall back to sklearn's default boundary, which would not match the "
            "offline evaluation."
        )

    classes = tuple(str(c) for c in getattr(classifier, "classes_", ()))
    if not classes:
        raise ModelContractError("classifier exposes no classes_")

    return LoadedModels(
        classifier=classifier,
        classifier_features=classifier_features,
        classifier_classes=classes,
        classifier_metadata=clf_meta,
        classifier_path=clf_path,
        anomaly=anomaly,
        anomaly_features=anomaly_features,
        anomaly_threshold=float(threshold_block["value"]),
        anomaly_threshold_name=str(threshold_block.get("selected", "unknown")),
        anomaly_metadata=if_meta,
        anomaly_path=if_path,
    )


def build_feature_row(
    features: StreamingWindowFeatures,
    motor_load_hp: float,
    rotational_speed_rpm: float,
    order: Sequence[str],
) -> pd.DataFrame:
    """Materialise a single-row frame in EXACTLY the model's column order."""
    supply: dict[str, float] = features.as_dict()
    supply["motor_load_hp"] = float(motor_load_hp)
    supply["rotational_speed_rpm"] = float(rotational_speed_rpm)

    missing = [c for c in order if c not in supply]
    if missing:
        raise ModelContractError(f"cannot supply feature(s) {missing}")
    # Column order is taken from `order`, never from dict insertion order.
    return pd.DataFrame([[supply[c] for c in order]], columns=list(order))


def score_window(
    models: LoadedModels,
    features: StreamingWindowFeatures,
    motor_load_hp: float,
    rotational_speed_rpm: float,
) -> tuple[AnomalyResult, ClassificationResult]:
    """Run both frozen models over one completed window."""
    anomaly_row = build_feature_row(
        features, motor_load_hp, rotational_speed_rpm, models.anomaly_features
    )
    score = float(models.anomaly.score_samples(anomaly_row)[0])
    anomaly = AnomalyResult(
        isAnomalous=bool(score < models.anomaly_threshold),
        score=score,
        threshold=float(models.anomaly_threshold),
    )

    clf_row = build_feature_row(
        features, motor_load_hp, rotational_speed_rpm, models.classifier_features
    )
    predicted = str(models.classifier.predict(clf_row)[0])
    probabilities: dict[str, float] = {}
    confidence = float("nan")
    if hasattr(models.classifier, "predict_proba"):
        proba = np.asarray(models.classifier.predict_proba(clf_row))[0]
        probabilities = {
            str(cls): float(p) for cls, p in zip(models.classifier_classes, proba)
        }
        confidence = float(probabilities.get(predicted, max(proba)))
    classification = ClassificationResult(
        predictedClass=predicted,
        confidence=confidence,
        probabilities=probabilities,
    )
    return anomaly, classification


def infer(models: LoadedModels, window) -> InferenceResult:
    """Score an :class:`~ml.streaming.windowing.AssembledWindow`.

    Ground truth is attached only AFTER both models have run, and is
    never part of either feature row.
    """
    features = window.features()
    anomaly, classification = score_window(
        models, features, window.motorLoadHp, window.rotationalSpeedRpm
    )
    ground_truth = (
        window.sourceScenario.to_dict() if window.sourceScenario is not None else None
    )
    return InferenceResult(
        assetId=window.assetId,
        windowIndex=window.windowIndex,
        windowStartSequence=window.windowStartSequence,
        windowEndSequence=window.windowEndSequence,
        timestampUtc=window.timestampUtc,
        sampleCount=window.sampleCount,
        features=features.as_json_dict(),
        operatingContext={
            "motorLoadHp": float(window.motorLoadHp),
            "rotationalSpeedRpm": float(window.rotationalSpeedRpm),
        },
        anomaly=anomaly.to_dict(),
        classification=classification.to_dict(),
        groundTruth=ground_truth,
    )


__all__ = [
    "ANOMALY_STEM",
    "CLASSIFIER_STEM",
    "DEFAULT_MODELS_DIR",
    "STREAMING_FEATURE_SOURCES",
    "LoadedModels",
    "ModelContractError",
    "build_feature_row",
    "infer",
    "load_models",
    "score_window",
]
