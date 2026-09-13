"""Random Forest baseline trainer for CWRU fault classification.

The trainer is intentionally minimal: it fits a single
``RandomForestClassifier`` on the canonical MachineFeatureVector,
reports validation accuracy, and persists the trained pipeline plus a
JSON metadata sidecar. Model comparison and hyperparameter search will
be added on top of this scaffold in a later branch.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


FEATURE_COLUMNS: tuple[str, ...] = (
    "vibration_rms",
    "vibration_std",
    "vibration_peak",
    "vibration_peak_to_peak",
    "vibration_kurtosis",
    "vibration_skewness",
    "crest_factor",
    "rotational_speed_rpm",
    "motor_load_hp",
)

LABEL_COLUMN = "fault_class"


@dataclass(frozen=True)
class TrainingResult:
    model_path: Path
    metadata_path: Path
    validation_accuracy: float
    validation_macro_f1: float


def _split_xy(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    missing = [c for c in FEATURE_COLUMNS if c not in frame.columns]
    if missing:
        raise KeyError(f"missing feature columns: {missing}")
    if LABEL_COLUMN not in frame.columns:
        raise KeyError(f"missing label column {LABEL_COLUMN!r}")
    return frame[list(FEATURE_COLUMNS)].copy(), frame[LABEL_COLUMN].copy()


def build_pipeline(random_state: int = 42, n_estimators: int = 200) -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=n_estimators,
                    random_state=random_state,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def train_and_persist(
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    output_dir: str | Path,
    model_name: str = "rf_baseline",
    random_state: int = 42,
    n_estimators: int = 200,
) -> TrainingResult:
    """Fit the baseline pipeline and persist model + metadata to disk."""
    out = Path(output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    X_train, y_train = _split_xy(train_frame)
    X_val, y_val = _split_xy(validation_frame)

    pipeline = build_pipeline(random_state=random_state, n_estimators=n_estimators)
    pipeline.fit(X_train, y_train)

    y_val_pred = pipeline.predict(X_val)
    accuracy = float(accuracy_score(y_val, y_val_pred))
    macro_f1 = float(f1_score(y_val, y_val_pred, average="macro"))

    model_path = out / f"{model_name}.joblib"
    metadata_path = out / f"{model_name}.json"
    joblib.dump(pipeline, model_path)
    metadata = {
        "model_name": model_name,
        "feature_columns": list(FEATURE_COLUMNS),
        "label_column": LABEL_COLUMN,
        "n_train_rows": int(len(X_train)),
        "n_validation_rows": int(len(X_val)),
        "validation_accuracy": accuracy,
        "validation_macro_f1": macro_f1,
        "sklearn_estimator": type(pipeline.named_steps["clf"]).__name__,
        "n_estimators": n_estimators,
        "random_state": random_state,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2))

    return TrainingResult(
        model_path=model_path,
        metadata_path=metadata_path,
        validation_accuracy=accuracy,
        validation_macro_f1=macro_f1,
    )
