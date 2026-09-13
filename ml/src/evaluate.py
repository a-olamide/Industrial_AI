"""Evaluate a trained baseline model on the held-out test partition."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from .train_baseline import FEATURE_COLUMNS, LABEL_COLUMN


@dataclass(frozen=True)
class EvaluationReport:
    accuracy: float
    macro_f1: float
    report_text: str
    confusion: pd.DataFrame
    labels: tuple[str, ...]


def _prepare_xy(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    missing = [c for c in FEATURE_COLUMNS if c not in frame.columns]
    if missing:
        raise KeyError(f"missing feature columns: {missing}")
    if LABEL_COLUMN not in frame.columns:
        raise KeyError(f"missing label column {LABEL_COLUMN!r}")
    return frame[list(FEATURE_COLUMNS)].copy(), frame[LABEL_COLUMN].copy()


def evaluate(model_path: str | Path, test_frame: pd.DataFrame) -> EvaluationReport:
    """Load a persisted model and produce metrics + confusion matrix."""
    pipeline = joblib.load(str(model_path))
    X_test, y_test = _prepare_xy(test_frame)
    y_pred = pipeline.predict(X_test)

    labels = tuple(sorted(np.unique(np.concatenate([y_test.to_numpy(), y_pred]))))
    cm = confusion_matrix(y_test, y_pred, labels=list(labels))
    confusion_frame = pd.DataFrame(cm, index=list(labels), columns=list(labels))
    confusion_frame.index.name = "actual"
    confusion_frame.columns.name = "predicted"

    return EvaluationReport(
        accuracy=float(accuracy_score(y_test, y_pred)),
        macro_f1=float(f1_score(y_test, y_pred, average="macro")),
        report_text=classification_report(y_test, y_pred, labels=list(labels), zero_division=0),
        confusion=confusion_frame,
        labels=labels,
    )
