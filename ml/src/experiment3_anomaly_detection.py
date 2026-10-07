"""Experiment 3 - unsupervised anomaly detection on CWRU bearing vibration.

Research question
-----------------
Can an unsupervised anomaly-detection model learn NORMAL bearing
behaviour and flag fault-condition vibration windows as anomalous,
*without using fault labels during model training*?

This asks a fundamentally different question from Experiments 1-2:

    supervised classifier  -> "which known fault class is this?"
    anomaly detector       -> "does this differ from learned normal?"

An Isolation Forest cannot name the physical fault type. It only
produces an anomaly signal. Nothing in this module should be read as
fault diagnosis.

Methodology guarantees
----------------------
- ``fit()`` sees NORMAL windows from motor loads 0 and 1 HP ONLY.
  :func:`fit_isolation_forest` refuses a frame containing any
  non-NORMAL row, so the guarantee is enforced, not merely intended.
- ``fault_class`` is never a feature and never reaches ``fit()``.
  Labels are used strictly AFTER inference, to score detection.
- The decision threshold is selected on VALIDATION (2 HP) and then
  frozen. TEST (3 HP) is scored exactly once, at the end.
- Splits are by complete recording / motor load. No random row
  splitting anywhere.

Relationship to Experiments 1 and 2
-----------------------------------
Both earlier experiments are frozen. This module imports the shared
dataset loader read-only and never writes ``rf_baseline_cwru.*`` or
``rf_multiseverity_cwru.*``. It writes only ``experiment3_*`` figures
and ``isolation_forest_cwru.{joblib,json}``.

Invoke as::

    python -m ml.src.experiment3_anomaly_detection
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    roc_auc_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .dataset_builder import FAULT_SEVERITIES_IN
from .experiment2_multiseverity import load_expanded_frame
from .feature_extraction import DEFAULT_WINDOW_SIZE
from .split_dataset import (
    GROUP_COLUMN,
    LABEL_COLUMN,
    LOAD_COLUMN,
    SEVERITY_COLUMN,
    WINDOW_COLUMN,
    multiseverity_load_split,
)
from .train_baseline import FEATURE_COLUMNS as SUPERVISED_FEATURE_COLUMNS


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "ml" / "models"
FIGURES_DIR = PROJECT_ROOT / "ml" / "reports" / "figures"

EXPERIMENT_NAME = "experiment3_unsupervised_anomaly_detection"
RESEARCH_QUESTION = (
    "Can an unsupervised anomaly-detection model learn NORMAL bearing behaviour "
    "and identify fault-condition vibration windows as anomalous, without using "
    "fault labels during model training?"
)
FIGURE_PREFIX = "experiment3_"
MODEL_STEM = "isolation_forest_cwru"

TRAIN_LOADS: tuple[float, ...] = (0.0, 1.0)
VALIDATION_LOADS: tuple[float, ...] = (2.0,)
TEST_LOADS: tuple[float, ...] = (3.0,)

NORMAL_LABEL = "NORMAL"
FAULT_CLASSES: tuple[str, ...] = ("INNER_RACE", "BALL", "OUTER_RACE")

# Binary anomaly vocabulary. "ANOMALOUS" is the POSITIVE class for
# precision / recall / F1 throughout this module.
LABEL_NORMAL = "NORMAL"
LABEL_ANOMALOUS = "ANOMALOUS"
BINARY_ORDER: tuple[str, ...] = (LABEL_NORMAL, LABEL_ANOMALOUS)

CLASS_COLORS = {
    "NORMAL": "#2b8cbe",
    "INNER_RACE": "#e34a33",
    "BALL": "#31a354",
    "OUTER_RACE": "#756bb1",
}
SEVERITY_COLORS = {0.007: "#4c72b0", 0.014: "#dd8452", 0.021: "#55a868"}

# ---------------------------------------------------------------------------
# Feature contract - deliberately NOT the supervised contract
# ---------------------------------------------------------------------------

# Vibration-derived window statistics only. These are the physical
# signal descriptors; every one of them is computable from a raw
# 2048-sample window with no knowledge of the operating point.
ANOMALY_FEATURE_COLUMNS: tuple[str, ...] = (
    "vibration_rms",
    "vibration_std",
    "vibration_peak",
    "vibration_peak_to_peak",
    "vibration_kurtosis",
    "vibration_skewness",
    "crest_factor",
)

# Operating-point columns that ARE in the supervised contract but are
# dropped here. See :func:`operating_point_feature_diagnostic` for the
# evidence behind the decision.
DROPPED_OPERATING_POINT_COLUMNS: tuple[str, ...] = (
    "rotational_speed_rpm",
    "motor_load_hp",
)

# Columns that must never enter X under any circumstances.
EXCLUDED_FROM_X: tuple[str, ...] = (
    "fault_class",
    "fault_severity_in",
    "recording_id",
    "window_id",
    "source",
    "source_file",
    "asset_id",
    "sampling_rate_hz",
)

RANDOM_STATE = 42
N_ESTIMATORS = 200
MAX_SAMPLES = "auto"

# Train-score quantiles evaluated as label-free threshold candidates.
TRAIN_QUANTILE_CANDIDATES: tuple[float, ...] = (0.01, 0.05, 0.10)

# A label-free threshold is preferred over the validation-F1-optimal
# threshold when it is within this much F1 of the best candidate.
# Rationale: a production deployment has no fault labels to tune on, so
# a threshold derived purely from healthy-machine data is worth a small
# measured concession.
LABEL_FREE_PREFERENCE_TOLERANCE = 0.01

LIMITATIONS: tuple[str, ...] = (
    "CWRU NORMAL recordings are published at 48 kHz while the selected fault "
    "recordings are 12 kHz Drive End. No resampling is performed in this "
    "experiment.",
    "A fixed 2048-sample window therefore spans ~42.7 ms for NORMAL and "
    "~170.7 ms for fault recordings - different physical durations for the "
    "same nominal window size.",
    "Experiment 2 already showed NORMAL-vs-FAULT to be the easy axis of this "
    "dataset (NORMAL test RMS 0.060-0.071 with zero overlap against any fault "
    "class). Experiment 3 measures exactly that axis, so strong anomaly "
    "results partly reflect acquisition and dataset characteristics rather "
    "than detector sophistication.",
    "Because NORMAL and FAULT recordings differ in BOTH health state and "
    "sampling rate, the two effects are confounded and cannot be separated "
    "with this data. A same-sampling-rate NORMAL baseline would be required "
    "to attribute the separation to bearing health alone.",
    "Results must not be presented as production industrial performance.",
    "Isolation Forest provides an anomaly signal only. It does not identify "
    "the physical fault type; naming the fault remains the supervised "
    "classifier's job.",
    "Training used 355 NORMAL windows from two recordings (Normal_0, "
    "Normal_1). That is a narrow notion of 'normal' - one healthy bearing at "
    "two load points - so the model has seen little legitimate operational "
    "variety.",
    "Fault windows at 0 and 1 HP exist in the dataset but are deliberately "
    "unused: the detector must never see a fault during training.",
    "All features are time-domain window statistics. No frequency-domain "
    "features are used.",
    "Windows from one recording are strongly correlated, which is why "
    "splitting is by complete recording rather than by random rows.",
)


def _hr(title: str) -> None:
    line = "=" * 72
    print(f"\n{line}\n{title}\n{line}")


def _fig_path(name: str) -> Path:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    return FIGURES_DIR / f"{FIGURE_PREFIX}{name}"


def _save(fig, path: Path) -> Path:
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"wrote {path}")
    return path


# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AnomalySplits:
    """NORMAL-only training frame plus mixed validation / test frames."""

    train_normal: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    unused_train_load_faults: int


def build_anomaly_splits(frame: pd.DataFrame) -> AnomalySplits:
    """Split by complete recording / motor load, then keep NORMAL for training."""
    _hr("STEP 1 - SPLIT BY RECORDING / MOTOR LOAD (no random row splitting)")
    split, audit = multiseverity_load_split(
        frame,
        train_loads=TRAIN_LOADS,
        validation_loads=VALIDATION_LOADS,
        test_loads=TEST_LOADS,
        expected_severities=FAULT_SEVERITIES_IN,
    )

    train_normal = split.train[split.train[LABEL_COLUMN] == NORMAL_LABEL].reset_index(
        drop=True
    )
    unused = int(len(split.train) - len(train_normal))

    # Enforced invariants for the unsupervised design.
    assert set(train_normal[LABEL_COLUMN]) == {NORMAL_LABEL}, (
        "training frame must contain NORMAL windows only"
    )
    assert set(train_normal[LOAD_COLUMN].astype(float)) == {float(v) for v in TRAIN_LOADS}
    assert set(split.validation[LOAD_COLUMN].astype(float)) == {
        float(v) for v in VALIDATION_LOADS
    }
    assert set(split.test[LOAD_COLUMN].astype(float)) == {float(v) for v in TEST_LOADS}
    for name, df in (("validation", split.validation), ("test", split.test)):
        present = set(df[LABEL_COLUMN])
        assert NORMAL_LABEL in present, f"{name} must contain NORMAL windows"
        missing = set(FAULT_CLASSES) - present
        assert not missing, f"{name} is missing fault classes {sorted(missing)}"

    print(
        "Isolation Forest is fit on NORMAL windows only. The fault windows at\n"
        "0 and 1 HP are intentionally discarded - the detector must never see a\n"
        "fault during training."
    )
    rows = [
        {
            "split": "train (fit)",
            "motor_loads_hp": "0+1",
            "recordings": ", ".join(sorted(train_normal[GROUP_COLUMN].unique())),
            "n_windows": len(train_normal),
            "composition": "NORMAL only",
        },
        {
            "split": "validation",
            "motor_loads_hp": "2",
            "recordings": f"{split.validation[GROUP_COLUMN].nunique()} recordings",
            "n_windows": len(split.validation),
            "composition": (
                f"NORMAL {int((split.validation[LABEL_COLUMN] == NORMAL_LABEL).sum())} / "
                f"FAULT {int((split.validation[LABEL_COLUMN] != NORMAL_LABEL).sum())}"
            ),
        },
        {
            "split": "test",
            "motor_loads_hp": "3",
            "recordings": f"{split.test[GROUP_COLUMN].nunique()} recordings",
            "n_windows": len(split.test),
            "composition": (
                f"NORMAL {int((split.test[LABEL_COLUMN] == NORMAL_LABEL).sum())} / "
                f"FAULT {int((split.test[LABEL_COLUMN] != NORMAL_LABEL).sum())}"
            ),
        },
    ]
    print()
    print(pd.DataFrame(rows).to_string(index=False))
    print(
        f"\nfault windows at 0/1 HP deliberately unused for training: {unused:,}"
    )

    print("\nvalidation composition by class / severity:")
    print(
        split.validation.groupby([LABEL_COLUMN, SEVERITY_COLUMN], dropna=False)
        .size()
        .to_string()
    )
    print("\ntest composition by class / severity:")
    print(
        split.test.groupby([LABEL_COLUMN, SEVERITY_COLUMN], dropna=False)
        .size()
        .to_string()
    )
    print(
        "\nrecording-level disjointness asserted by multiseverity_load_split(): "
        "no recording_id and no individual window appears in more than one split."
    )
    return AnomalySplits(
        train_normal=train_normal,
        validation=split.validation,
        test=split.test,
        unused_train_load_faults=unused,
    )


# ---------------------------------------------------------------------------
# Feature decision
# ---------------------------------------------------------------------------


def operating_point_feature_diagnostic(splits: AnomalySplits) -> dict[str, Any]:
    """Decide whether motor_load_hp / RPM may enter the anomaly feature set.

    The supervised contract contains two operating-point columns. This
    function tests - on TRAIN and VALIDATION only - whether they belong
    in an anomaly detector whose splits are defined BY motor load.
    """
    _hr("STEP 2 - FEATURE DECISION: motor_load_hp AND rotational_speed_rpm")
    print(
        "The supervised experiments used 9 columns. Reusing them blindly here would "
        "be a design error. Start with the structural fact:\n"
    )
    ranges = []
    for column in DROPPED_OPERATING_POINT_COLUMNS:
        for name, df in (
            ("train (NORMAL fit data)", splits.train_normal),
            ("validation", splits.validation),
            ("test", splits.test),
        ):
            values = df[column].astype(float)
            ranges.append(
                {
                    "feature": column,
                    "split": name,
                    "min": float(values.min()),
                    "max": float(values.max()),
                    "n_unique": int(values.nunique()),
                }
            )
    range_table = pd.DataFrame(ranges)
    print(range_table.to_string(index=False))
    print(
        "\nBoth columns are constant within a recording and the split is BY motor "
        "load, so their train / validation / test value sets are DISJOINT BY "
        "CONSTRUCTION. Every validation and test window - healthy or faulty - sits "
        "outside the operating-point range the detector saw while learning "
        "'normal'. These columns describe the experimental condition, not the "
        "vibration signature."
    )

    # Three variants, scored on VALIDATION only. The rejected variants
    # are never evaluated on the test split.
    print(
        "\nEmpirical check (VALIDATION only, 5th-percentile train-score threshold).\n"
        "AUC is threshold-free: it ranks FAULT above NORMAL by anomaly score, so\n"
        "0.5 means the feature set carries no separation at all."
    )
    variants = (
        ("vibration_only (7)", ANOMALY_FEATURE_COLUMNS),
        ("supervised_contract (9)", tuple(SUPERVISED_FEATURE_COLUMNS)),
        ("operating_point_only (2)", DROPPED_OPERATING_POINT_COLUMNS),
    )
    results: dict[str, dict[str, float]] = {}
    rows = []
    is_normal = (splits.validation[LABEL_COLUMN] == NORMAL_LABEL).to_numpy()
    for variant, columns in variants:
        model = _build_pipeline()
        model.fit(splits.train_normal[list(columns)])
        train_scores = model.score_samples(splits.train_normal[list(columns)])
        val_scores = model.score_samples(splits.validation[list(columns)])
        threshold = float(np.percentile(train_scores, 5.0))
        flags = val_scores < threshold
        metrics = {
            "validation_auc": float(
                roc_auc_score((~is_normal).astype(int), -val_scores)
            ),
            "validation_false_positive_rate_on_normal": float(flags[is_normal].mean()),
            "validation_fault_detection_rate": float(flags[~is_normal].mean()),
        }
        results[variant] = metrics
        rows.append({"feature_set": variant, "n_features": len(columns), **metrics})
    print()
    print(pd.DataFrame(rows).round(4).to_string(index=False))

    op_auc = results["operating_point_only (2)"]["validation_auc"]
    op_flagged = results["operating_point_only (2)"]["validation_fault_detection_rate"]
    print(
        "\nThis did NOT go the way the structural argument predicts, and the reason "
        "matters.\n"
        f"The operating-point-only detector scores AUC {op_auc:.4f} and flags "
        f"{op_flagged:.1%} of validation windows. An Isolation Forest measures "
        "isolation DEPTH, not distance: the NORMAL training data contains only two "
        "distinct operating points ((1796 RPM, 0 HP) and (1772 RPM, 1 HP)), so those "
        "two axes can be split at most once and every point - in-range or far "
        "outside it - terminates at the same shallow depth. Isolation Forest is "
        "simply blind to the out-of-range-ness here, so including the two columns "
        "neither exploits the split nor measurably hurts validation; it mostly "
        "dilutes per-split feature sampling."
    )
    print(
        "\nDECISION: use the 7 vibration-derived features only. The justification is "
        "validity, NOT measured validation harm - validation does not penalize "
        "including them, and claiming otherwise would overstate the evidence.\n"
        "  1. Those columns encode the experimental condition that DEFINES the "
        "split, so any credit they earn is unattributable between 'vibration is "
        "abnormal' and 'operating point is unfamiliar'. Every validation window "
        "shares the same unseen load, so no aggregate metric computed here can "
        "separate the two explanations.\n"
        "  2. Isolation Forest's blindness to out-of-range values is an accident of "
        "this algorithm and of having only two training operating points. A "
        "distance-based detector (LOF, one-class SVM, Mahalanobis) or a forest "
        "trained over more load levels WOULD key on them.\n"
        "  3. The research question is about vibration anomaly. A detector should "
        "raise an alarm because the machine's vibration changed, not because it is "
        "running at a load the training set happened not to cover."
    )
    print(
        "\nThe rejected variants were scored on VALIDATION only and are never "
        "evaluated on the test split, so no test information entered this decision."
    )
    return {
        "operating_point_ranges": range_table.to_dict(orient="records"),
        "validation_variant_comparison": results,
        "decision": "vibration_only (7 features)",
        "decision_basis": (
            "validity / attributability, not measured validation harm - validation "
            "metrics did not penalize including the operating-point columns"
        ),
    }


def print_feature_contract() -> None:
    _hr("STEP 3 - FINAL ANOMALY FEATURE CONTRACT")
    print(f"X = {len(ANOMALY_FEATURE_COLUMNS)} vibration-derived features:")
    for i, column in enumerate(ANOMALY_FEATURE_COLUMNS, start=1):
        print(f"  {i}. {column}")
    print("\ny = none. The model is fit unsupervised; no target is passed to fit().")
    print(f"\ndropped from the supervised contract: {list(DROPPED_OPERATING_POINT_COLUMNS)}")
    print(f"never permitted in X:                 {list(EXCLUDED_FROM_X)}")
    overlap = set(ANOMALY_FEATURE_COLUMNS) & set(EXCLUDED_FROM_X)
    assert not overlap, f"excluded column(s) leaked into X: {sorted(overlap)}"
    assert LABEL_COLUMN not in ANOMALY_FEATURE_COLUMNS
    assert SEVERITY_COLUMN not in ANOMALY_FEATURE_COLUMNS
    print("\nleakage assertions passed.")


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


def _build_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "detector",
                IsolationForest(
                    n_estimators=N_ESTIMATORS,
                    max_samples=MAX_SAMPLES,
                    contamination="auto",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def fit_isolation_forest(train_normal: pd.DataFrame) -> Pipeline:
    """Fit on NORMAL windows only. Refuses any frame containing a fault."""
    _hr("STEP 4 - FIT ISOLATION FOREST (NORMAL WINDOWS ONLY)")
    if LABEL_COLUMN in train_normal.columns:
        contaminating = sorted(set(train_normal[LABEL_COLUMN]) - {NORMAL_LABEL})
        if contaminating:
            raise ValueError(
                "Isolation Forest must be fit on NORMAL windows only; received "
                f"rows labelled {contaminating}. Refusing to fit."
            )

    X_train = train_normal[list(ANOMALY_FEATURE_COLUMNS)].copy()
    assert list(X_train.columns) == list(ANOMALY_FEATURE_COLUMNS)
    for forbidden in EXCLUDED_FROM_X:
        assert forbidden not in X_train.columns

    model = _build_pipeline()
    # No y argument: the label column never reaches the estimator.
    model.fit(X_train)

    detector = model.named_steps["detector"]
    print(
        f"fit on {len(X_train):,} NORMAL windows x "
        f"{len(ANOMALY_FEATURE_COLUMNS)} vibration features"
    )
    print(f"recordings used: {sorted(train_normal[GROUP_COLUMN].unique())}")
    print(
        f"\nIsolationForest(n_estimators={detector.n_estimators}, "
        f"max_samples={detector.max_samples!r} -> {detector.max_samples_} per tree, "
        f"contamination={detector.contamination!r}, "
        f"max_features={detector.max_features}, "
        f"random_state={detector.random_state})"
    )
    print(
        "\nStandardScaler is fit on the NORMAL training windows and kept inside the "
        "pipeline so the saved artifact is self-contained and reproducible. It is "
        "a strictly monotonic per-feature affine map, so it does not change which "
        "points an axis-aligned isolation tree separates; it is there for artifact "
        "hygiene and so a distance-based detector could be swapped in later "
        "without changing the contract."
    )
    print(
        "\nNo label was passed to fit(). The signature used is model.fit(X) with X "
        "restricted to the 7 vibration columns."
    )
    return model


# ---------------------------------------------------------------------------
# Scoring and thresholds
# ---------------------------------------------------------------------------


def score(model: Pipeline, frame: pd.DataFrame) -> np.ndarray:
    """Continuous Isolation Forest ``score_samples``.

    DIRECTION (stated once, used everywhere): HIGHER = more normal,
    LOWER = more anomalous. Values are negative by construction; a
    window is flagged ANOMALOUS when its score falls BELOW the
    threshold.
    """
    return model.score_samples(frame[list(ANOMALY_FEATURE_COLUMNS)])


def classify(scores: np.ndarray, threshold: float) -> np.ndarray:
    return np.where(scores < threshold, LABEL_ANOMALOUS, LABEL_NORMAL)


def binary_truth(frame: pd.DataFrame) -> np.ndarray:
    """Ground truth in anomaly vocabulary. Used only AFTER inference."""
    return np.where(
        frame[LABEL_COLUMN] == NORMAL_LABEL, LABEL_NORMAL, LABEL_ANOMALOUS
    )


def anomaly_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Metrics with ANOMALOUS as the positive class."""
    cm = confusion_matrix(y_true, y_pred, labels=list(BINARY_ORDER))
    tn, fp, fn, tp = cm.ravel()
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_anomalous": float(
            precision_score(y_true, y_pred, pos_label=LABEL_ANOMALOUS, zero_division=0)
        ),
        "recall_anomalous": float(
            recall_score(y_true, y_pred, pos_label=LABEL_ANOMALOUS, zero_division=0)
        ),
        "f1_anomalous": float(
            f1_score(y_true, y_pred, pos_label=LABEL_ANOMALOUS, zero_division=0)
        ),
        "false_positive_rate_on_normal": float(fp / (fp + tn)) if (fp + tn) else 0.0,
        "false_negative_rate_on_faults": float(fn / (fn + tp)) if (fn + tp) else 0.0,
        "true_negatives_normal_kept_normal": int(tn),
        "false_positives_normal_flagged": int(fp),
        "false_negatives_faults_missed": int(fn),
        "true_positives_faults_detected": int(tp),
    }


def confusion_frame(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    cm = pd.DataFrame(
        confusion_matrix(y_true, y_pred, labels=list(BINARY_ORDER)),
        index=list(BINARY_ORDER),
        columns=list(BINARY_ORDER),
    )
    cm.index.name = "actual"
    cm.columns.name = "predicted"
    return cm


def _print_metrics(title: str, metrics: dict[str, float], cm: pd.DataFrame) -> None:
    print(f"\n{title}")
    print(f"  accuracy                      = {metrics['accuracy']:.4f}")
    print(f"  precision (ANOMALOUS)         = {metrics['precision_anomalous']:.4f}")
    print(
        f"  recall / detection (ANOMALOUS) = {metrics['recall_anomalous']:.4f}"
    )
    print(f"  F1 (ANOMALOUS)                = {metrics['f1_anomalous']:.4f}")
    print(
        f"  false-positive rate on NORMAL = "
        f"{metrics['false_positive_rate_on_normal']:.4f} "
        f"({metrics['false_positives_normal_flagged']} healthy windows flagged)"
    )
    print(
        f"  false-negative rate on faults = "
        f"{metrics['false_negative_rate_on_faults']:.4f} "
        f"({metrics['false_negatives_faults_missed']} fault windows missed)"
    )
    print("\n  confusion matrix (rows=actual, cols=predicted):")
    print("    " + cm.to_string().replace("\n", "\n    "))


@dataclass
class ThresholdCandidate:
    name: str
    threshold: float
    derivation: str
    label_free: bool
    metrics: dict[str, float] = None  # filled on validation


def build_threshold_candidates(
    model: Pipeline, train_normal: pd.DataFrame
) -> list[ThresholdCandidate]:
    """Threshold candidates. None of them looks at the test split."""
    train_scores = score(model, train_normal)
    detector = model.named_steps["detector"]
    candidates = [
        ThresholdCandidate(
            name="sklearn_default",
            threshold=float(detector.offset_),
            derivation=(
                "sklearn's contamination='auto' decision boundary, i.e. "
                f"offset_={float(detector.offset_):.4f} on the score_samples scale "
                "(equivalently decision_function < 0). An absolute path-length "
                "criterion; it consults no data beyond the fitted trees."
            ),
            label_free=True,
        )
    ]
    for q in TRAIN_QUANTILE_CANDIDATES:
        candidates.append(
            ThresholdCandidate(
                name=f"train_quantile_{q:.2f}",
                threshold=float(np.quantile(train_scores, q)),
                derivation=(
                    f"the {q:.0%} quantile of score_samples over the NORMAL "
                    "TRAINING windows. Derived from healthy data only - no fault "
                    "window and no label is consulted. Interpreted as 'accept a "
                    f"{q:.0%} false-alarm rate on known-healthy data'."
                ),
                label_free=True,
            )
        )
    return candidates


def select_threshold(
    model: Pipeline,
    candidates: list[ThresholdCandidate],
    validation: pd.DataFrame,
) -> tuple[ThresholdCandidate, pd.DataFrame, dict[str, Any]]:
    """Evaluate candidates on VALIDATION, add an F1-optimal sweep, and choose."""
    _hr("STEP 5 - THRESHOLD STRATEGY AND VALIDATION SELECTION")
    print(
        "Contamination is NOT set from the fault proportion in validation or test.\n"
        "Doing so would smuggle the answer into the model: the fraction of broken\n"
        "machines is precisely what a deployed detector does not know. The forest\n"
        "is fit once with contamination='auto' (which affects only the decision\n"
        "offset, never the trees), and the decision threshold is then chosen\n"
        "explicitly from the candidates below.\n"
    )
    val_scores = score(model, validation)
    y_true = binary_truth(validation)

    # F1-optimal sweep over validation scores. Permitted (validation
    # labels may be used); recorded as NOT label-free.
    grid = np.unique(np.quantile(val_scores, np.linspace(0.0, 1.0, 501)))
    best_f1, best_threshold = -1.0, float(grid[0])
    for candidate_threshold in grid:
        f1 = f1_score(
            y_true,
            classify(val_scores, float(candidate_threshold)),
            pos_label=LABEL_ANOMALOUS,
            zero_division=0,
        )
        if f1 > best_f1:
            best_f1, best_threshold = float(f1), float(candidate_threshold)
    candidates = list(candidates) + [
        ThresholdCandidate(
            name="validation_f1_optimal",
            threshold=best_threshold,
            derivation=(
                "swept over 501 quantiles of the VALIDATION score distribution and "
                "kept the threshold maximising F1 for the ANOMALOUS class. Uses "
                "validation labels (permitted); uses no test data."
            ),
            label_free=False,
        )
    ]

    for candidate in candidates:
        candidate.metrics = anomaly_metrics(
            y_true, classify(val_scores, candidate.threshold)
        )

    table = pd.DataFrame(
        [
            {
                "candidate": c.name,
                "threshold": round(c.threshold, 5),
                "label_free": c.label_free,
                "val_accuracy": round(c.metrics["accuracy"], 4),
                "val_precision_ANOM": round(c.metrics["precision_anomalous"], 4),
                "val_recall_ANOM": round(c.metrics["recall_anomalous"], 4),
                "val_f1_ANOM": round(c.metrics["f1_anomalous"], 4),
                "val_fpr_on_NORMAL": round(c.metrics["false_positive_rate_on_normal"], 4),
            }
            for c in candidates
        ]
    )
    print("validation comparison of threshold candidates:")
    print(table.to_string(index=False))

    best = max(candidates, key=lambda c: c.metrics["f1_anomalous"])
    label_free_within_tolerance = [
        c
        for c in candidates
        if c.label_free
        and best.metrics["f1_anomalous"] - c.metrics["f1_anomalous"]
        <= LABEL_FREE_PREFERENCE_TOLERANCE
    ]
    if label_free_within_tolerance:
        selected = max(
            label_free_within_tolerance,
            key=lambda c: (c.metrics["f1_anomalous"], -c.metrics["false_positive_rate_on_normal"]),
        )
        rationale = (
            f"best validation F1 is {best.metrics['f1_anomalous']:.4f} "
            f"({best.name}). {selected.name} is within the documented "
            f"{LABEL_FREE_PREFERENCE_TOLERANCE} F1 tolerance at "
            f"{selected.metrics['f1_anomalous']:.4f} and is LABEL-FREE: it is "
            "derived from healthy data alone, so the same procedure works on a "
            "real machine where no fault labels exist. That robustness is worth "
            "the measured concession, so it is selected."
        )
    else:
        selected = best
        rationale = (
            f"no label-free candidate came within "
            f"{LABEL_FREE_PREFERENCE_TOLERANCE} F1 of the best candidate, so the "
            f"validation-F1-optimal threshold {selected.name} is selected "
            f"(F1={selected.metrics['f1_anomalous']:.4f})."
        )

    print(f"\nSELECTED THRESHOLD: {selected.name} = {selected.threshold:.5f}")
    print(f"derivation: {selected.derivation}")
    print(f"WHY: {rationale}")
    print(
        "\nThe threshold is now FROZEN. The test split has not been scored at this "
        "point in the run."
    )
    _print_metrics(
        "VALIDATION (2 HP) with the selected threshold:",
        selected.metrics,
        confusion_frame(y_true, classify(val_scores, selected.threshold)),
    )
    return selected, table, {"rationale": rationale, "best_candidate": best.name}


# ---------------------------------------------------------------------------
# Breakdowns
# ---------------------------------------------------------------------------


def detection_by_fault_class(
    frame: pd.DataFrame, scores: np.ndarray, threshold: float
) -> pd.DataFrame:
    faults = frame[LABEL_COLUMN] != NORMAL_LABEL
    flags = classify(scores, threshold) == LABEL_ANOMALOUS
    rows = []
    for fault_class in FAULT_CLASSES:
        mask = (frame[LABEL_COLUMN] == fault_class).to_numpy()
        if not mask.any():
            continue
        rows.append(
            {
                "fault_class": fault_class,
                "n_windows": int(mask.sum()),
                "n_detected": int(flags[mask].sum()),
                "detection_rate": float(flags[mask].mean()),
                "mean_score": float(scores[mask].mean()),
            }
        )
    normal_mask = (frame[LABEL_COLUMN] == NORMAL_LABEL).to_numpy()
    rows.append(
        {
            "fault_class": "NORMAL (reference)",
            "n_windows": int(normal_mask.sum()),
            "n_detected": int(flags[normal_mask].sum()),
            "detection_rate": float(flags[normal_mask].mean()),
            "mean_score": float(scores[normal_mask].mean()),
        }
    )
    _ = faults
    return pd.DataFrame(rows)


def detection_by_severity(
    frame: pd.DataFrame, scores: np.ndarray, threshold: float
) -> pd.DataFrame:
    flags = classify(scores, threshold) == LABEL_ANOMALOUS
    rows = []
    for severity in FAULT_SEVERITIES_IN:
        mask = (
            (frame[LABEL_COLUMN] != NORMAL_LABEL)
            & (frame[SEVERITY_COLUMN].astype(float).round(3) == round(severity, 3))
        ).to_numpy()
        if not mask.any():
            continue
        row = {
            "fault_severity_in": severity,
            "n_windows": int(mask.sum()),
            "n_detected": int(flags[mask].sum()),
            "detection_rate": float(flags[mask].mean()),
            "mean_score": float(scores[mask].mean()),
        }
        for fault_class in FAULT_CLASSES:
            sub = mask & (frame[LABEL_COLUMN] == fault_class).to_numpy()
            if sub.any():
                row[f"detect_{fault_class}"] = float(flags[sub].mean())
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Comparison with Experiment 2
# ---------------------------------------------------------------------------


def compare_with_experiment2(
    test: pd.DataFrame, scores: np.ndarray, threshold: float
) -> dict[str, Any]:
    """Cross-tabulate anomaly flags against the frozen Experiment-2 classifier.

    Reads ``rf_multiseverity_cwru.joblib`` strictly read-only; nothing
    about Experiment 2 is refit or rewritten. Skips cleanly when the
    (gitignored) artifact is absent.
    """
    _hr("STEP 8 - RELATIONSHIP TO THE EXPERIMENT-2 SUPERVISED CLASSIFIER")
    artifact = MODELS_DIR / "rf_multiseverity_cwru.joblib"
    if not artifact.is_file():
        print(
            f"{artifact} not present (gitignored) - run "
            "`python -m ml.src.experiment2_multiseverity` to regenerate it. "
            "Skipping the cross-model comparison."
        )
        return {"available": False}

    rf = joblib.load(artifact)
    rf_pred = rf.predict(test[list(SUPERVISED_FEATURE_COLUMNS)])
    truth = test[LABEL_COLUMN].to_numpy()
    rf_correct = rf_pred == truth
    anomalous = classify(scores, threshold) == LABEL_ANOMALOUS
    is_fault = truth != NORMAL_LABEL

    print(
        "These two models answer different questions, so the interesting cell is "
        "the one where the classifier names the wrong fault type but the detector "
        "still says 'this machine is not behaving normally'.\n"
    )
    print(
        f"Experiment-2 RF test accuracy (recomputed read-only): "
        f"{float(rf_correct.mean()):.4f}"
    )
    print(
        f"Experiment-3 IF test detection rate on faults: "
        f"{float(anomalous[is_fault].mean()):.4f}"
    )

    misclassified = is_fault & ~rf_correct
    n_mis = int(misclassified.sum())
    n_mis_flagged = int(anomalous[misclassified].sum())
    print(
        f"\nfault windows the supervised classifier got WRONG: {n_mis}\n"
        f"  of those, flagged ANOMALOUS by Isolation Forest: {n_mis_flagged} "
        f"({(n_mis_flagged / n_mis * 100 if n_mis else 0.0):.1f}%)"
    )

    rows = []
    for severity in FAULT_SEVERITIES_IN:
        mask = is_fault & (
            test[SEVERITY_COLUMN].astype(float).round(3) == round(severity, 3)
        ).to_numpy()
        if not mask.any():
            continue
        sev_mis = mask & ~rf_correct
        rows.append(
            {
                "fault_severity_in": severity,
                "n_fault_windows": int(mask.sum()),
                "exp2_classification_accuracy": float(rf_correct[mask].mean()),
                "exp3_anomaly_detection_rate": float(anomalous[mask].mean()),
                "exp2_misclassified": int(sev_mis.sum()),
                "exp3_detected_among_exp2_errors": int(anomalous[sev_mis].sum()),
            }
        )
    table = pd.DataFrame(rows)
    print("\nper-severity comparison on the 3 HP test split:")
    print(table.round(4).to_string(index=False))
    print(
        "\nIsolation Forest does NOT identify the physical fault type. A window "
        "counted as 'detected' here is only a window whose vibration statistics "
        "fall outside learned normal behaviour."
    )
    return {
        "available": True,
        "experiment2_test_accuracy": float(rf_correct.mean()),
        "experiment3_test_detection_rate": float(anomalous[is_fault].mean()),
        "experiment2_misclassified_fault_windows": n_mis,
        "of_those_flagged_anomalous": n_mis_flagged,
        "per_severity": table.to_dict(orient="records"),
    }


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


SCORE_AXIS_LABEL = (
    "IsolationForest score_samples\n(HIGHER = more normal  |  LOWER = more anomalous)"
)


def figure_score_normal_vs_fault(
    validation: pd.DataFrame,
    val_scores: np.ndarray,
    test: pd.DataFrame,
    test_scores: np.ndarray,
    threshold: float,
) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), sharex=True)
    panels = (
        ("VALIDATION (2 HP)", validation, val_scores),
        ("TEST (3 HP, held out)", test, test_scores),
    )
    for ax, (title, frame, scores) in zip(axes, panels):
        normal = scores[(frame[LABEL_COLUMN] == NORMAL_LABEL).to_numpy()]
        fault = scores[(frame[LABEL_COLUMN] != NORMAL_LABEL).to_numpy()]
        bins = np.linspace(min(scores.min(), threshold) - 0.02, scores.max() + 0.02, 55)
        ax.hist(normal, bins=bins, color=CLASS_COLORS["NORMAL"], alpha=0.75, label="NORMAL (healthy)")
        ax.hist(fault, bins=bins, color="#d62728", alpha=0.6, label="FAULT")
        ax.axvline(
            threshold,
            color="black",
            linestyle="--",
            linewidth=1.6,
            label=f"frozen threshold = {threshold:.4f}",
        )
        ax.set_title(title, fontsize=10)
        ax.set_xlabel(SCORE_AXIS_LABEL, fontsize=8)
        ax.set_ylabel("windows")
        ax.legend(fontsize=7)
        ax.grid(True, axis="y", alpha=0.3)
    fig.suptitle(
        "Experiment 3 - anomaly-score distribution: NORMAL vs FAULT\n"
        "windows LEFT of the dashed line are flagged ANOMALOUS",
        fontsize=11,
    )
    return _save(fig, _fig_path("score_distribution_normal_vs_fault.png"))


def figure_score_by_fault_class(
    test: pd.DataFrame, test_scores: np.ndarray, threshold: float
) -> Path:
    order = ["NORMAL", *FAULT_CLASSES]
    data = [test_scores[(test[LABEL_COLUMN] == c).to_numpy()] for c in order]
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    box = ax.boxplot(data, tick_labels=order, patch_artist=True, widths=0.55, vert=True)
    for patch, cls in zip(box["boxes"], order):
        patch.set_facecolor(CLASS_COLORS[cls])
        patch.set_alpha(0.65)
    ax.axhline(
        threshold,
        color="black",
        linestyle="--",
        linewidth=1.6,
        label=f"frozen threshold = {threshold:.4f} (below = ANOMALOUS)",
    )
    ax.set_ylabel(SCORE_AXIS_LABEL, fontsize=8)
    ax.set_xlabel("true fault class (labels used only AFTER inference)")
    ax.set_title(
        "Experiment 3 - anomaly score by fault class, TEST split (3 HP)\n"
        "Isolation Forest gives an anomaly signal; it does not name the fault type",
        fontsize=10,
    )
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    return _save(fig, _fig_path("score_distribution_by_fault_class.png"))


def figure_score_by_severity(
    test: pd.DataFrame, test_scores: np.ndarray, threshold: float
) -> Path:
    fig, ax = plt.subplots(figsize=(10, 4.8))
    groups: list[tuple[str, np.ndarray, str]] = [
        (
            "NORMAL\n(no severity)",
            test_scores[(test[LABEL_COLUMN] == NORMAL_LABEL).to_numpy()],
            CLASS_COLORS["NORMAL"],
        )
    ]
    for severity in FAULT_SEVERITIES_IN:
        mask = (
            (test[LABEL_COLUMN] != NORMAL_LABEL)
            & (test[SEVERITY_COLUMN].astype(float).round(3) == round(severity, 3))
        ).to_numpy()
        groups.append((f'{severity:.3f}"', test_scores[mask], SEVERITY_COLORS[severity]))

    box = ax.boxplot(
        [g[1] for g in groups],
        tick_labels=[g[0] for g in groups],
        patch_artist=True,
        widths=0.55,
    )
    for patch, group in zip(box["boxes"], groups):
        patch.set_facecolor(group[2])
        patch.set_alpha(0.7)
    ax.axhline(
        threshold,
        color="black",
        linestyle="--",
        linewidth=1.6,
        label=f"frozen threshold = {threshold:.4f} (below = ANOMALOUS)",
    )
    for i, group in enumerate(groups, start=1):
        rate = float((group[1] < threshold).mean())
        ax.text(
            i,
            ax.get_ylim()[1],
            f"flagged\n{rate:.1%}",
            ha="center",
            va="top",
            fontsize=7.5,
        )
    ax.set_ylabel(SCORE_AXIS_LABEL, fontsize=8)
    ax.set_xlabel("fault severity (defect diameter) - NORMAL has no physical severity")
    ax.set_title(
        "Experiment 3 - anomaly score by fault severity, TEST split (3 HP)",
        fontsize=10,
    )
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(True, axis="y", alpha=0.3)
    return _save(fig, _fig_path("score_distribution_by_severity.png"))


def figure_test_confusion(y_true: np.ndarray, y_pred: np.ndarray) -> Path:
    fig, ax = plt.subplots(figsize=(6.0, 5.2))
    ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        labels=list(BINARY_ORDER),
        ax=ax,
        cmap="Blues",
        colorbar=True,
    )
    ax.set_title(
        "Experiment 3 - TEST confusion matrix (3 HP held out)\n"
        "Isolation Forest, fit on NORMAL windows only",
        fontsize=10,
    )
    return _save(fig, _fig_path("test_confusion_matrix.png"))


def figure_threshold_selection(table: pd.DataFrame, selected_name: str) -> Path:
    fig, ax = plt.subplots(figsize=(10, 4.4))
    x = np.arange(len(table))
    series = [
        ("val_f1_ANOM", "#4c72b0"),
        ("val_precision_ANOM", "#dd8452"),
        ("val_recall_ANOM", "#55a868"),
        ("val_fpr_on_NORMAL", "#c44e52"),
    ]
    width = 0.8 / len(series)
    for i, (column, color) in enumerate(series):
        offset = (i - (len(series) - 1) / 2) * width
        ax.bar(x + offset, table[column], width=width, color=color, label=column)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [
            f"{row.candidate}\n{'label-free' if row.label_free else 'uses val labels'}"
            f"\nthr={row.threshold:.4f}"
            for row in table.itertuples()
        ],
        fontsize=7,
    )
    for i, row in enumerate(table.itertuples()):
        if row.candidate == selected_name:
            ax.axvspan(i - 0.45, i + 0.45, color="#ffd92f", alpha=0.25, zorder=0)
    ax.set_ylim(0, 1.18)
    ax.set_ylabel("validation score")
    ax.set_title(
        "Experiment 3 - threshold selection on VALIDATION (2 HP)\n"
        f"highlighted = selected ({selected_name}); test data not consulted",
        fontsize=10,
    )
    ax.legend(fontsize=7, ncol=4, loc="upper center")
    ax.grid(True, axis="y", alpha=0.3)
    return _save(fig, _fig_path("threshold_selection.png"))


def figure_vs_experiment2(comparison: dict[str, Any]) -> Path | None:
    if not comparison.get("available"):
        return None
    table = pd.DataFrame(comparison["per_severity"])
    x = np.arange(len(table))
    width = 0.38
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    ax.bar(
        x - width / 2,
        table["exp2_classification_accuracy"],
        width=width,
        color="#4c72b0",
        label="Exp 2 - supervised: correct fault TYPE",
    )
    ax.bar(
        x + width / 2,
        table["exp3_anomaly_detection_rate"],
        width=width,
        color="#c44e52",
        label="Exp 3 - unsupervised: flagged ANOMALOUS",
    )
    for xi, v in zip(x - width / 2, table["exp2_classification_accuracy"]):
        ax.text(xi, v + 0.015, f"{v:.3f}", ha="center", fontsize=8)
    for xi, v in zip(x + width / 2, table["exp3_anomaly_detection_rate"]):
        ax.text(xi, v + 0.015, f"{v:.3f}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [
            f'{row.fault_severity_in:.3f}"\n(n={int(row.n_fault_windows)})'
            for row in table.itertuples()
        ]
    )
    ax.set_ylim(0, 1.32)
    ax.set_ylabel("rate on TEST fault windows (3 HP)")
    ax.set_xlabel("fault severity (defect diameter)")
    ax.set_title(
        "Experiment 3 vs Experiment 2 on the same held-out fault windows\n"
        "different questions: 'which fault?' vs 'is anything wrong?'",
        fontsize=10,
    )
    ax.legend(fontsize=8, loc="upper center", ncol=2, framealpha=0.95)
    ax.grid(True, axis="y", alpha=0.3)
    return _save(fig, _fig_path("vs_experiment2_by_severity.png"))


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def persist(
    model: Pipeline,
    selected: ThresholdCandidate,
    selection_notes: dict[str, Any],
    candidate_table: pd.DataFrame,
    splits: AnomalySplits,
    val_metrics: dict[str, float],
    val_confusion: pd.DataFrame,
    test_metrics: dict[str, float],
    test_confusion: pd.DataFrame,
    class_table: pd.DataFrame,
    severity_table: pd.DataFrame,
    feature_decision: dict[str, Any],
    comparison: dict[str, Any],
    figures: list[Path],
    dataset_source: str,
) -> tuple[Path, Path]:
    _hr("STEP 9 - PERSIST EXPERIMENT-3 ARTIFACTS")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / f"{MODEL_STEM}.joblib"
    metadata_path = MODELS_DIR / f"{MODEL_STEM}.json"

    for reserved in (
        "rf_baseline_cwru.joblib",
        "rf_baseline_cwru.json",
        "rf_multiseverity_cwru.joblib",
        "rf_multiseverity_cwru.json",
    ):
        assert model_path.name != reserved and metadata_path.name != reserved, (
            f"refusing to overwrite the frozen artifact {reserved}"
        )

    joblib.dump(model, model_path)
    detector = model.named_steps["detector"]
    metadata = {
        "experiment_name": EXPERIMENT_NAME,
        "research_question": RESEARCH_QUESTION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_source": dataset_source,
        "learning_paradigm": "unsupervised anomaly detection (no labels at fit time)",
        "algorithm": "sklearn.ensemble.IsolationForest inside a StandardScaler Pipeline",
        "model_parameters": {
            "n_estimators": detector.n_estimators,
            "max_samples": str(detector.max_samples),
            "max_samples_resolved": int(detector.max_samples_),
            "contamination": str(detector.contamination),
            "max_features": detector.max_features,
            "bootstrap": detector.bootstrap,
            "random_state": detector.random_state,
            "offset_": float(detector.offset_),
        },
        "feature_columns": list(ANOMALY_FEATURE_COLUMNS),
        "dropped_operating_point_columns": list(DROPPED_OPERATING_POINT_COLUMNS),
        "excluded_from_x": list(EXCLUDED_FROM_X),
        "feature_decision": feature_decision,
        "feature_decision_summary": (
            "Only the 7 vibration-derived window statistics are used. "
            "rotational_speed_rpm and motor_load_hp were dropped because they are "
            "constant within a recording and the split is BY motor load, so their "
            "train/validation/test value sets are disjoint by construction; "
            "including them would make 'anomalous' mean 'recorded at an unseen "
            "load' rather than 'vibrating abnormally'."
        ),
        "training_data": {
            "composition": "NORMAL windows only",
            "motor_loads_hp": list(TRAIN_LOADS),
            "recordings": sorted(splits.train_normal[GROUP_COLUMN].unique().tolist()),
            "n_windows_fit": int(len(splits.train_normal)),
            "fault_windows_at_train_loads_deliberately_unused": splits.unused_train_load_faults,
            "labels_used_at_fit_time": False,
        },
        "split_design": {
            "strategy": "by complete recording / motor load; no random row splitting",
            "train_loads_hp": list(TRAIN_LOADS),
            "validation_load_hp": list(VALIDATION_LOADS),
            "test_load_hp": list(TEST_LOADS),
            "validation_recordings": sorted(
                splits.validation[GROUP_COLUMN].unique().tolist()
            ),
            "test_recordings": sorted(splits.test[GROUP_COLUMN].unique().tolist()),
            "n_validation_windows": int(len(splits.validation)),
            "n_test_windows": int(len(splits.test)),
            "window_size": DEFAULT_WINDOW_SIZE,
            "window_hop": DEFAULT_WINDOW_SIZE,
        },
        "score_direction": (
            "score_samples: HIGHER = more normal, LOWER = more anomalous. A window "
            "is ANOMALOUS when score_samples < threshold."
        ),
        "threshold": {
            "selected": selected.name,
            "value": float(selected.threshold),
            "derivation": selected.derivation,
            "label_free": selected.label_free,
            "selection_rationale": selection_notes["rationale"],
            "best_validation_candidate": selection_notes["best_candidate"],
            "label_free_preference_tolerance": LABEL_FREE_PREFERENCE_TOLERANCE,
            "candidates": candidate_table.to_dict(orient="records"),
            "contamination_policy": (
                "contamination is NOT set from the observed fault proportion in "
                "validation or test; the fraction of broken machines is exactly "
                "what a deployed detector does not know. The forest is fit with "
                "contamination='auto' (which affects only the decision offset, "
                "never the trees) and the operating threshold is chosen explicitly."
            ),
        },
        "positive_class": LABEL_ANOMALOUS,
        "class_labels": list(BINARY_ORDER),
        "validation_metrics": val_metrics,
        "validation_confusion_matrix": {
            "labels": list(BINARY_ORDER),
            "rows_actual_cols_predicted": val_confusion.to_numpy().tolist(),
        },
        "test_metrics": test_metrics,
        "test_confusion_matrix": {
            "labels": list(BINARY_ORDER),
            "rows_actual_cols_predicted": test_confusion.to_numpy().tolist(),
        },
        "test_detection_by_fault_class": class_table.to_dict(orient="records"),
        "test_detection_by_severity": severity_table.to_dict(orient="records"),
        "comparison_with_experiment2": comparison,
        "sampling_rate_notes": {
            "NORMAL": 48000,
            "INNER_RACE": 12000,
            "BALL": 12000,
            "OUTER_RACE": 12000,
            "resampling_performed": False,
        },
        "known_limitations": list(LIMITATIONS),
        "figures": [str(p.relative_to(PROJECT_ROOT)) for p in figures],
        "frozen_artifacts_untouched": [
            "ml/models/rf_baseline_cwru.joblib",
            "ml/models/rf_baseline_cwru.json",
            "ml/models/rf_multiseverity_cwru.joblib",
            "ml/models/rf_multiseverity_cwru.json",
        ],
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, default=str))
    print(f"wrote {model_path} ({model_path.stat().st_size / 1024:.1f} KB)")
    print(f"wrote {metadata_path} ({metadata_path.stat().st_size / 1024:.1f} KB)")
    print(
        "\nBoth are gitignored by existing repository policy (ml/models/*.joblib, "
        "ml/models/*.json). No Experiment-1 or Experiment-2 artifact was written."
    )
    return model_path, metadata_path


def print_limitations() -> None:
    _hr("STEP 10 - KNOWN LIMITATIONS")
    for i, text in enumerate(LIMITATIONS, start=1):
        print(f"{i:2d}. {text}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> int:
    frame, dataset_source = load_expanded_frame()
    splits = build_anomaly_splits(frame)
    feature_decision = operating_point_feature_diagnostic(splits)
    print_feature_contract()

    model = fit_isolation_forest(splits.train_normal)

    candidates = build_threshold_candidates(model, splits.train_normal)
    selected, candidate_table, selection_notes = select_threshold(
        model, candidates, splits.validation
    )
    threshold = selected.threshold

    val_scores = score(model, splits.validation)
    val_truth = binary_truth(splits.validation)
    val_metrics = anomaly_metrics(val_truth, classify(val_scores, threshold))
    val_confusion = confusion_frame(val_truth, classify(val_scores, threshold))

    _hr("STEP 6 - FINAL TEST EVALUATION (3 HP, scored once, threshold frozen)")
    test_scores = score(model, splits.test)
    test_truth = binary_truth(splits.test)
    test_pred = classify(test_scores, threshold)
    test_metrics = anomaly_metrics(test_truth, test_pred)
    test_confusion = confusion_frame(test_truth, test_pred)
    _print_metrics("TEST (3 HP held out):", test_metrics, test_confusion)

    _hr("STEP 7 - DETECTION BREAKDOWN BY FAULT CLASS AND SEVERITY")
    class_table = detection_by_fault_class(splits.test, test_scores, threshold)
    print("detection rate by physical fault class (TEST):")
    print(class_table.round(4).to_string(index=False))
    severity_table = detection_by_severity(splits.test, test_scores, threshold)
    print("\ndetection rate by fault severity (TEST):")
    print(severity_table.round(4).to_string(index=False))
    hard = severity_table[severity_table["fault_severity_in"] == 0.014]
    if not hard.empty:
        rate = float(hard["detection_rate"].iloc[0])
        print(
            f"\n0.014\" focus - the severity Experiment 2 found hardest to CLASSIFY: "
            f"{int(hard['n_detected'].iloc[0])}/{int(hard['n_windows'].iloc[0])} "
            f"windows flagged ANOMALOUS ({rate:.4f}). Being detectable as abnormal "
            "is a weaker claim than being correctly typed, and the two can differ."
        )

    comparison = compare_with_experiment2(splits.test, test_scores, threshold)

    _hr("STEP 8b - EXPERIMENT-3 FIGURES")
    figures = [
        figure_score_normal_vs_fault(
            splits.validation, val_scores, splits.test, test_scores, threshold
        ),
        figure_score_by_fault_class(splits.test, test_scores, threshold),
        figure_score_by_severity(splits.test, test_scores, threshold),
        figure_test_confusion(test_truth, test_pred),
        figure_threshold_selection(candidate_table, selected.name),
    ]
    vs_fig = figure_vs_experiment2(comparison)
    if vs_fig is not None:
        figures.append(vs_fig)

    model_path, metadata_path = persist(
        model=model,
        selected=selected,
        selection_notes=selection_notes,
        candidate_table=candidate_table,
        splits=splits,
        val_metrics=val_metrics,
        val_confusion=val_confusion,
        test_metrics=test_metrics,
        test_confusion=test_confusion,
        class_table=class_table,
        severity_table=severity_table,
        feature_decision=feature_decision,
        comparison=comparison,
        figures=figures,
        dataset_source=dataset_source,
    )

    print_limitations()

    _hr("EXPERIMENT 3 COMPLETE")
    print(f"fit on            : {len(splits.train_normal):,} NORMAL windows, no labels")
    print(f"threshold         : {selected.name} = {threshold:.5f}")
    print(
        f"validation (2 HP) : accuracy={val_metrics['accuracy']:.4f}  "
        f"recall(ANOM)={val_metrics['recall_anomalous']:.4f}  "
        f"FPR(NORMAL)={val_metrics['false_positive_rate_on_normal']:.4f}"
    )
    print(
        f"test (3 HP)       : accuracy={test_metrics['accuracy']:.4f}  "
        f"recall(ANOM)={test_metrics['recall_anomalous']:.4f}  "
        f"FPR(NORMAL)={test_metrics['false_positive_rate_on_normal']:.4f}"
    )
    print(f"model artifact    : {model_path}")
    print(f"metadata          : {metadata_path}")
    print("figures:")
    for p in figures:
        print(f"  {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
