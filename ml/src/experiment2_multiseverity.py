"""Experiment 2 - multi-severity CWRU bearing-fault classification.

Research question
-----------------
Can a supervised classifier trained across *multiple* bearing-fault
severities (0.007", 0.014", 0.021") generalize to an unseen motor load?

Design
------
- Input: the expanded 40-recording feature dataset (2048-sample
  non-overlapping windows), all three fault severities.
- Split strictly by complete recording / motor load:
  train = 0 + 1 HP, validation = 2 HP, test = 3 HP.
- Target: ``fault_class`` in {NORMAL, INNER_RACE, BALL, OUTER_RACE}.
- ``fault_severity_in`` is EXPERIMENTAL METADATA and is deliberately
  excluded from ``X``: the physical defect diameter of an unknown
  machine is not observable at inference time, so feeding it to the
  classifier would leak information that production never has.
- Three candidate models (Logistic Regression, Random Forest, Gradient
  Boosting) are compared on VALIDATION only. The winner is then
  evaluated once on the held-out 3 HP TEST recordings.

Relationship to Experiment 1
----------------------------
Experiment 1 (the frozen 0.007"-only Random Forest baseline, driven by
``ml/src/run_experiment.py``) is NOT touched. This module writes its
own figures under the ``experiment2_`` prefix and its own model
artifact under ``<slug>_multiseverity_cwru.joblib``; it never writes
``rf_baseline_cwru.*`` or any baseline figure.

Invoke as::

    python -m ml.src.experiment2_multiseverity
"""

from __future__ import annotations

import json
import sys
import warnings
from contextlib import contextmanager
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
from sklearn.base import BaseEstimator
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from .dataset_builder import (
    CWRU_RECORDINGS,
    FAULT_SEVERITIES_IN,
    build_feature_frame,
)
from .feature_extraction import DEFAULT_WINDOW_SIZE
from .split_dataset import (
    GROUP_COLUMN,
    LOAD_COLUMN,
    SEVERITY_COLUMN,
    WINDOW_COLUMN,
    SplitAudit,
    SplitFrames,
    multiseverity_load_split,
)
from .train_baseline import FEATURE_COLUMNS, LABEL_COLUMN


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "ml" / "data" / "raw" / "cwru"
PROCESSED_DIR = PROJECT_ROOT / "ml" / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "ml" / "models"
FIGURES_DIR = PROJECT_ROOT / "ml" / "reports" / "figures"

EXPANDED_CSV = PROCESSED_DIR / "cwru_features_expanded.csv"

EXPERIMENT_NAME = "experiment2_multiseverity_fault_classification"
RESEARCH_QUESTION = (
    "Can a supervised classifier trained across multiple bearing-fault "
    "severities (0.007\", 0.014\", 0.021\") generalize to an unseen motor load?"
)
FIGURE_PREFIX = "experiment2_"

TRAIN_LOADS: tuple[float, ...] = (0.0, 1.0)
VALIDATION_LOADS: tuple[float, ...] = (2.0,)
TEST_LOADS: tuple[float, ...] = (3.0,)

CLASS_ORDER: tuple[str, ...] = ("NORMAL", "INNER_RACE", "BALL", "OUTER_RACE")
CLASS_COLORS = {
    "NORMAL": "#2b8cbe",
    "INNER_RACE": "#e34a33",
    "BALL": "#31a354",
    "OUTER_RACE": "#756bb1",
}

# Columns that exist in the feature frame but must NEVER enter X.
# ``fault_severity_in`` heads the list: it is the experimental knob,
# not an observable sensor reading.
EXCLUDED_FROM_X: tuple[str, ...] = (
    "fault_severity_in",
    "sampling_rate_hz",
    "recording_id",
    "window_id",
    "source",
    "source_file",
    "asset_id",
    "fault_class",
)

# Candidate model registry. ``rank`` encodes "simpler / more
# interpretable first" and is only consulted to break a validation tie.
MODEL_SLUGS = {
    "logistic_regression": "logreg",
    "random_forest": "rf",
    "gradient_boosting": "gbm",
}
MODEL_RANK = {
    "logistic_regression": 0,
    "random_forest": 1,
    "gradient_boosting": 2,
}
# Absolute macro-F1 gap below which two models are treated as tied and
# the simpler one wins. Documented so selection is reproducible.
TIE_TOLERANCE = 0.005

RANDOM_STATE = 42

SEPARABILITY_PROBE_FEATURES: tuple[str, ...] = (
    "vibration_rms",
    "vibration_std",
    "vibration_peak",
)

LIMITATIONS: tuple[str, ...] = (
    "CWRU is a controlled laboratory bearing dataset collected on a test rig, "
    "not an in-service industrial fleet.",
    "Faults are seeded (machined/EDM defects of known diameter) rather than "
    "naturally occurring progressive degradation.",
    "NORMAL recordings are published at 48 kHz while the selected fault "
    "recordings are 12 kHz Drive End. No resampling is performed in this "
    "experiment.",
    "Because of that sampling-rate mismatch, a fixed 2048-sample window "
    "represents ~42.7 ms for NORMAL and ~170.7 ms for fault recordings - "
    "different physical durations for the same nominal window size.",
    "All features are time-domain window statistics. No frequency-domain "
    "features (FFT bands, envelope spectrum, bearing characteristic "
    "frequencies) are used.",
    "Results must not be read as production-level industrial performance.",
    "The experiment tests generalization to an unseen MOTOR LOAD only. All "
    "three fault severities are present during training, so this is NOT a "
    "test of generalization to an unseen defect size.",
    "Windows sliced from one recording are strongly correlated, which is why "
    "splitting is done by complete recording rather than by random rows.",
    "rotational_speed_rpm is nearly a deterministic function of motor load in "
    "CWRU (1797/1772/1750/1730 RPM for 0/1/2/3 HP), and motor_load_hp is "
    "constant within a recording. Both are in the inherited feature contract "
    "and are kept for contract parity, but at test time both take values the "
    "model never saw in training, so neither can contribute usable signal to "
    "load generalization.",
    "fault_severity_in is excluded from X by design; it is experimental "
    "metadata that would not be observable when diagnosing an unknown machine.",
)


def _hr(title: str) -> None:
    line = "=" * 72
    print(f"\n{line}\n{title}\n{line}")


def _severity_label(value: Any) -> str:
    if value is None:
        return "NORMAL (no severity)"
    try:
        if pd.isna(value):
            return "NORMAL (no severity)"
    except (TypeError, ValueError):
        return str(value)
    return f'{float(value):.3f}"'


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


def load_expanded_frame() -> tuple[pd.DataFrame, str]:
    """Return the expanded 40-recording feature frame.

    Prefers the CSV emitted by ``audit_expanded_dataset`` so Experiment 2
    scores exactly the audited dataset. Falls back to rebuilding from the
    raw ``.mat`` files when the (gitignored) CSV is absent or stale.
    """
    _hr("STEP 0 - LOAD EXPANDED FEATURE DATASET")
    required = set(FEATURE_COLUMNS) | {
        LABEL_COLUMN,
        GROUP_COLUMN,
        WINDOW_COLUMN,
        SEVERITY_COLUMN,
    }
    expected_recordings = len(CWRU_RECORDINGS)

    if EXPANDED_CSV.is_file():
        frame = pd.read_csv(EXPANDED_CSV)
        missing = sorted(required - set(frame.columns))
        n_rec = int(frame[GROUP_COLUMN].nunique()) if GROUP_COLUMN in frame else 0
        if not missing and n_rec == expected_recordings:
            print(f"loaded {EXPANDED_CSV} ({len(frame):,} windows, {n_rec} recordings)")
            return frame, f"csv:{EXPANDED_CSV.relative_to(PROJECT_ROOT)}"
        print(
            f"{EXPANDED_CSV.name} is unusable (missing columns={missing}, "
            f"recordings={n_rec}/{expected_recordings}); rebuilding from raw .mat files"
        )

    print(f"building expanded feature frame from {RAW_DIR} ...")
    frame = build_feature_frame(RAW_DIR, window_size=DEFAULT_WINDOW_SIZE)
    n_rec = int(frame[GROUP_COLUMN].nunique()) if len(frame) else 0
    if n_rec != expected_recordings:
        raise SystemExit(
            f"expected {expected_recordings} recordings, resolved {n_rec}. "
            f"Drop the missing CWRU .mat files into {RAW_DIR} and re-run."
        )
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_csv(EXPANDED_CSV, index=False)
    print(f"wrote {EXPANDED_CSV} ({len(frame):,} windows, {n_rec} recordings)")
    return frame, "rebuilt-from-raw"


def _recording_table(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(
            [LABEL_COLUMN, GROUP_COLUMN, LOAD_COLUMN, SEVERITY_COLUMN], dropna=False
        )
        .size()
        .reset_index(name="n_windows")
        .sort_values([LABEL_COLUMN, SEVERITY_COLUMN, GROUP_COLUMN])
    )


def build_experiment2_split(frame: pd.DataFrame) -> tuple[SplitFrames, SplitAudit]:
    """TASK 1 - recording/load-aware split with programmatic assertions."""
    _hr("TASK 1 - EXPERIMENT-2 SPLIT (by complete recording / motor load)")
    split, audit = multiseverity_load_split(
        frame,
        train_loads=TRAIN_LOADS,
        validation_loads=VALIDATION_LOADS,
        test_loads=TEST_LOADS,
        expected_severities=FAULT_SEVERITIES_IN,
    )
    print(
        "assertions passed: (1) no recording_id in >1 split, (2) all four target "
        "classes in every split, (3) no feature window crosses splits, "
        "(4) all three fault severities present in the fault recordings of every split."
    )

    named = (
        ("train", split.train, TRAIN_LOADS),
        ("validation", split.validation, VALIDATION_LOADS),
        ("test", split.test, TEST_LOADS),
    )
    summary_rows = []
    for name, df, loads in named:
        summary_rows.append(
            {
                "split": name,
                "motor_loads_hp": "+".join(f"{int(v)}" for v in loads),
                "n_recordings": len(audit.recordings[name]),
                "n_windows": audit.window_counts[name],
            }
        )
    print("\nsplit sizes:")
    print(pd.DataFrame(summary_rows).to_string(index=False))

    print("\nrecordings per split:")
    for name, _df, _loads in named:
        ids = audit.recordings[name]
        print(f"  {name:11s} ({len(ids):2d}): {', '.join(ids)}")

    print("\nclass counts per split (windows):")
    class_tbl = pd.DataFrame(
        {name: audit.class_counts[name] for name, _d, _l in named}
    ).reindex(list(CLASS_ORDER)).fillna(0).astype(int)
    print(class_tbl.to_string())

    print("\nseverity counts per split (windows; NONE = NORMAL, no physical severity):")
    sev_tbl = pd.DataFrame(
        {name: audit.severity_counts[name] for name, _d, _l in named}
    ).fillna(0).astype(int)
    print(sev_tbl.sort_index().to_string())

    print("\nload counts per split (windows):")
    load_tbl = pd.DataFrame(
        {name: audit.load_counts[name] for name, _d, _l in named}
    ).fillna(0).astype(int)
    print(load_tbl.sort_index().to_string())

    print("\nrecording-level detail:")
    for name, df, _loads in named:
        print(f"\n  --- {name.upper()} ---")
        tbl = _recording_table(df)
        for _, row in tbl.iterrows():
            print(
                f"    {row[LABEL_COLUMN]:11s} {row[GROUP_COLUMN]:12s} "
                f"load={int(row[LOAD_COLUMN])}HP  "
                f"severity={_severity_label(row[SEVERITY_COLUMN]):>20s}  "
                f"windows={int(row['n_windows']):,}"
            )

    print(
        "\nNo random row-level splitting is performed anywhere in this experiment: "
        "the split keys are motor_load_hp values, and every recording belongs to "
        "exactly one load."
    )
    return split, audit


# ---------------------------------------------------------------------------
# X / y contract
# ---------------------------------------------------------------------------


def define_xy(
    split: SplitFrames,
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.Series], tuple[str, ...]]:
    """TASK 2 - explicit X / y definition with a leakage check."""
    _hr("TASK 2 - X / y DEFINITION (leakage check)")
    feature_columns = tuple(FEATURE_COLUMNS)

    print(f"final X feature list ({len(feature_columns)} columns, inherited from")
    print("ml/src/train_baseline.py::FEATURE_COLUMNS - same contract as Experiment 1):")
    for i, col in enumerate(feature_columns, start=1):
        print(f"  {i}. {col}")
    print(f"\ntarget y: {LABEL_COLUMN}")
    print(f"class labels: {list(CLASS_ORDER)}")

    leaked = [c for c in feature_columns if c in EXCLUDED_FROM_X]
    assert not leaked, f"metadata column(s) leaked into X: {leaked}"
    assert SEVERITY_COLUMN not in feature_columns, (
        f"{SEVERITY_COLUMN!r} must never be a model input - it is experimental "
        "metadata, not an observable signal"
    )
    print(
        f"\nexcluded from X (asserted): {list(EXCLUDED_FROM_X)}\n"
        f"  - {SEVERITY_COLUMN}: physical defect size, unknowable for an unseen machine\n"
        f"  - {GROUP_COLUMN} / {WINDOW_COLUMN} / source / asset_id: provenance identifiers\n"
        f"  - sampling_rate_hz: acquisition metadata; constant per class here, so it "
        "would act as a direct NORMAL-vs-FAULT label"
    )

    X: dict[str, pd.DataFrame] = {}
    y: dict[str, pd.Series] = {}
    for name, df in (
        ("train", split.train),
        ("validation", split.validation),
        ("test", split.test),
    ):
        missing = [c for c in feature_columns if c not in df.columns]
        if missing:
            raise KeyError(f"split {name!r} missing feature columns: {missing}")
        X[name] = df[list(feature_columns)].copy()
        y[name] = df[LABEL_COLUMN].copy()
        print(f"\nX[{name}].shape = {X[name].shape}   y[{name}].shape = {y[name].shape}")
        assert list(X[name].columns) == list(feature_columns)

    return X, y, feature_columns


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


def build_candidate_models() -> dict[str, BaseEstimator]:
    """TASK 3 - three candidate classifiers with deterministic seeds."""
    return {
        # Scaling is mandatory for a linear model on these raw-amplitude
        # features; it lives inside the Pipeline so it is fit on train only.
        "logistic_regression": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=5000,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                        n_jobs=None,
                    ),
                ),
            ]
        ),
        # Same configuration family as the Experiment-1 baseline
        # (200 trees, seed 42, class_weight="balanced"). Trees are
        # scale-invariant so no scaler is attached.
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            random_state=RANDOM_STATE,
            class_weight="balanced",
            n_jobs=-1,
        ),
        # sklearn's GradientBoostingClassifier has NO class_weight
        # parameter. See the printed note in train_candidates().
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=3,
            random_state=RANDOM_STATE,
        ),
    }


def train_candidates(
    models: dict[str, BaseEstimator],
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> dict[str, BaseEstimator]:
    _hr("TASK 3 - TRAIN THREE CANDIDATE CLASSIFIERS")
    print(
        "Class-imbalance handling, stated explicitly:\n"
        "  logistic_regression : class_weight='balanced' (supported)\n"
        "  random_forest       : class_weight='balanced' (supported)\n"
        "  gradient_boosting   : NOT class-weighted. sklearn's\n"
        "      GradientBoostingClassifier exposes no class_weight parameter; the\n"
        "      only equivalent is passing sample_weight at fit() time (e.g.\n"
        "      sklearn.utils.class_weight.compute_sample_weight('balanced', y)).\n"
        "      This experiment deliberately does NOT do that, so GB is trained\n"
        "      under the raw class priors (NORMAL is over-represented). The\n"
        "      three-way comparison is therefore not weighted identically, and\n"
        "      that asymmetry is a real caveat - not something to paper over.\n"
        "No hyperparameter tuning is performed. The test set is not touched here."
    )
    fitted: dict[str, BaseEstimator] = {}
    for name, model in models.items():
        with _suppressed_blas_fp_warnings():
            model.fit(X_train, y_train)
        fitted[name] = model
        print(f"\nfitted {name}: {_describe_estimator(model)}")
        _report_convergence(name, model)
    return fitted


@contextmanager
def _suppressed_blas_fp_warnings():
    """Silence spurious ``matmul`` floating-point RuntimeWarnings.

    numpy 2.0 on Apple's Accelerate BLAS raises "divide by zero /
    overflow / invalid value encountered in matmul" for well-conditioned
    products, so lbfgs floods the log while fitting. The filter is
    deliberately narrow (RuntimeWarning, message must mention matmul)
    and :func:`_report_convergence` independently proves the fit was
    healthy, so nothing real is hidden.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=".*encountered in matmul.*", category=RuntimeWarning
        )
        yield


def _report_convergence(name: str, model: BaseEstimator) -> None:
    """Prove an iterative solver actually converged and stayed finite."""
    inner = _inner_estimator(model)
    n_iter = getattr(inner, "n_iter_", None)
    if n_iter is None:
        return
    iterations = int(np.max(np.atleast_1d(n_iter)))
    max_iter = int(getattr(inner, "max_iter", 0))
    coef = getattr(inner, "coef_", None)
    finite = bool(np.isfinite(coef).all()) if coef is not None else True
    print(
        f"  solver converged in {iterations} iteration(s) of max_iter={max_iter}; "
        f"coefficients finite={finite}"
    )
    assert finite, f"{name}: non-finite coefficients after fit"
    assert iterations < max_iter, (
        f"{name}: solver hit max_iter={max_iter} without converging; "
        "metrics below would be untrustworthy"
    )


def _describe_estimator(model: BaseEstimator) -> str:
    if isinstance(model, Pipeline):
        steps = " -> ".join(type(step).__name__ for _, step in model.steps)
        return f"Pipeline({steps})"
    return type(model).__name__


def _inner_estimator(model: BaseEstimator) -> BaseEstimator:
    return model.named_steps["clf"] if isinstance(model, Pipeline) else model


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def _macro_metrics(y_true, y_pred, labels: list[str]) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(
            precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
        ),
        "macro_recall": float(
            recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
        ),
        "macro_f1": float(
            f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
        ),
    }


def _per_class_frame(y_true, y_pred, labels: list[str]) -> pd.DataFrame:
    precision, recall, fbeta, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    return pd.DataFrame(
        {
            "precision": precision,
            "recall": recall,
            "f1": fbeta,
            "support": support.astype(int),
        },
        index=labels,
    )


def _confusion_frame(y_true, y_pred, labels: list[str]) -> pd.DataFrame:
    cm = pd.DataFrame(
        confusion_matrix(y_true, y_pred, labels=labels), index=labels, columns=labels
    )
    cm.index.name = "actual"
    cm.columns.name = "predicted"
    return cm


@dataclass
class ModelEvaluation:
    name: str
    predictions: np.ndarray
    macro: dict[str, float]
    per_class: pd.DataFrame
    confusion: pd.DataFrame
    report_text: str


def evaluate_model(
    name: str,
    model: BaseEstimator,
    X: pd.DataFrame,
    y: pd.Series,
    labels: list[str],
) -> ModelEvaluation:
    with _suppressed_blas_fp_warnings():
        y_pred = model.predict(X)
    return ModelEvaluation(
        name=name,
        predictions=y_pred,
        macro=_macro_metrics(y, y_pred, labels),
        per_class=_per_class_frame(y, y_pred, labels),
        confusion=_confusion_frame(y, y_pred, labels),
        report_text=classification_report(
            y, y_pred, labels=labels, digits=4, zero_division=0
        ),
    )


def _print_evaluation(ev: ModelEvaluation) -> None:
    m = ev.macro
    print(
        f"\n{ev.name}\n"
        f"  accuracy        = {m['accuracy']:.4f}\n"
        f"  macro precision = {m['macro_precision']:.4f}\n"
        f"  macro recall    = {m['macro_recall']:.4f}\n"
        f"  macro F1        = {m['macro_f1']:.4f}"
    )
    print("\n  per-class precision / recall / F1:")
    print("    " + ev.per_class.round(4).to_string().replace("\n", "\n    "))
    print("\n  confusion matrix (rows=actual, cols=predicted):")
    print("    " + ev.confusion.to_string().replace("\n", "\n    "))


# ---------------------------------------------------------------------------
# Validation comparison + selection
# ---------------------------------------------------------------------------


def validation_comparison(
    fitted: dict[str, BaseEstimator],
    X_val: pd.DataFrame,
    y_val: pd.Series,
    labels: list[str],
) -> tuple[dict[str, ModelEvaluation], pd.DataFrame]:
    _hr("TASK 4 - VALIDATION COMPARISON (motor load = 2 HP)")
    evaluations: dict[str, ModelEvaluation] = {}
    for name, model in fitted.items():
        ev = evaluate_model(name, model, X_val, y_val, labels)
        evaluations[name] = ev
        _print_evaluation(ev)
        print("\n  sklearn classification_report:")
        print("    " + ev.report_text.replace("\n", "\n    "))

    table = pd.DataFrame(
        [
            {
                "model": name,
                "validation_accuracy": round(ev.macro["accuracy"], 4),
                "validation_macro_f1": round(ev.macro["macro_f1"], 4),
            }
            for name, ev in evaluations.items()
        ]
    ).sort_values("validation_macro_f1", ascending=False, kind="stable")
    print("\ncomparison table:")
    print(table.to_string(index=False))
    return evaluations, table


def select_model(evaluations: dict[str, ModelEvaluation]) -> tuple[str, str]:
    """Pick the candidate on VALIDATION macro-F1; ties go to the simpler model."""
    _hr("TASK 4b - MODEL SELECTION (validation only)")
    best_f1 = max(ev.macro["macro_f1"] for ev in evaluations.values())
    contenders = [
        name
        for name, ev in evaluations.items()
        if best_f1 - ev.macro["macro_f1"] <= TIE_TOLERANCE
    ]
    contenders.sort(key=lambda n: (MODEL_RANK[n], -evaluations[n].macro["macro_f1"]))
    selected = contenders[0]

    if len(contenders) == 1:
        rationale = (
            f"{selected} has the highest validation macro-F1 "
            f"({evaluations[selected].macro['macro_f1']:.4f}) and no other candidate "
            f"is within the {TIE_TOLERANCE} tie tolerance."
        )
    else:
        tied = ", ".join(
            f"{n}={evaluations[n].macro['macro_f1']:.4f}" for n in contenders
        )
        rationale = (
            f"{tied} are tied within the documented {TIE_TOLERANCE} macro-F1 "
            f"tolerance (best={best_f1:.4f}), so the simplest / most interpretable "
            f"candidate wins: {selected}."
        )
    print(f"best validation macro-F1 = {best_f1:.4f}")
    print(f"tie tolerance            = {TIE_TOLERANCE}")
    print(f"contenders within tolerance: {contenders}")
    print(f"\nSELECTED MODEL: {selected}")
    print(f"WHY: {rationale}")
    print(
        "\nTest performance played no part in this decision; the test split has "
        "not been predicted on at this point in the run."
    )
    return selected, rationale


# ---------------------------------------------------------------------------
# Test evaluation
# ---------------------------------------------------------------------------


def final_test_evaluation(
    selected: str,
    model: BaseEstimator,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    labels: list[str],
) -> ModelEvaluation:
    _hr(f"TASK 5 - FINAL TEST EVALUATION ({selected}, motor load = 3 HP)")
    ev = evaluate_model(selected, model, X_test, y_test, labels)
    _print_evaluation(ev)
    print("\n  sklearn classification_report:")
    print("    " + ev.report_text.replace("\n", "\n    "))
    return ev


def test_performance_by_severity(
    test_df: pd.DataFrame,
    y_pred: np.ndarray,
    labels: list[str],
) -> pd.DataFrame:
    """TASK 5b - break the test result down by physical fault severity."""
    _hr("TASK 5b - TEST PERFORMANCE BY FAULT SEVERITY")
    frame = test_df.copy()
    frame["_pred"] = y_pred

    print(
        "NORMAL carries no physical fault severity and is reported separately as a\n"
        "reference row, never as a severity bucket."
    )
    rows: list[dict[str, Any]] = []

    for severity in FAULT_SEVERITIES_IN:
        subset = frame[
            (frame[LABEL_COLUMN] != "NORMAL")
            & (frame[SEVERITY_COLUMN].astype(float).round(3) == round(severity, 3))
        ]
        if subset.empty:
            print(f"\n--- severity {severity:.3f}\" --- no test windows")
            continue
        y_true = subset[LABEL_COLUMN]
        y_hat = subset["_pred"].to_numpy()
        present = [c for c in labels if c in set(y_true)]
        accuracy = float(accuracy_score(y_true, y_hat))
        # Macro-F1 is restricted to the classes actually present as TRUE
        # labels in this severity bucket (NORMAL never is). Precision for
        # those classes still accounts for windows wrongly predicted into
        # them, and windows predicted as NORMAL show up as recall loss.
        macro_f1 = float(
            f1_score(y_true, y_hat, labels=present, average="macro", zero_division=0)
        )
        per_class = _per_class_frame(y_true, y_hat, present)
        cm = _confusion_frame(y_true, y_hat, labels).loc[present]
        n_pred_normal = int((y_hat == "NORMAL").sum())

        print(f"\n--- severity {severity:.3f}\" ---")
        print(f"  windows   = {len(subset):,}")
        print(f"  accuracy  = {accuracy:.4f}")
        print(
            f"  macro F1  = {macro_f1:.4f}  "
            f"(restricted to true classes present: {present})"
        )
        print(f"  windows predicted NORMAL (false 'healthy' calls) = {n_pred_normal}")
        print("  per-class precision / recall / F1:")
        print("    " + per_class.round(4).to_string().replace("\n", "\n    "))
        print("  confusion (rows=actual true classes, cols=all predicted classes):")
        print("    " + cm.to_string().replace("\n", "\n    "))

        row: dict[str, Any] = {
            "fault_severity_in": severity,
            "n_windows": int(len(subset)),
            "accuracy": accuracy,
            "macro_f1_present_classes": macro_f1,
            "n_predicted_normal": n_pred_normal,
        }
        for cls in present:
            row[f"recall_{cls}"] = float(per_class.loc[cls, "recall"])
        rows.append(row)

    normal_subset = frame[frame[LABEL_COLUMN] == "NORMAL"]
    if not normal_subset.empty:
        normal_acc = float(
            accuracy_score(normal_subset[LABEL_COLUMN], normal_subset["_pred"])
        )
        print(
            f"\n--- NORMAL reference (no physical severity) ---\n"
            f"  windows  = {len(normal_subset):,}\n"
            f"  recall   = {normal_acc:.4f}"
        )

    severity_table = pd.DataFrame(rows)
    if not severity_table.empty:
        print("\nseverity summary table:")
        print(severity_table.round(4).to_string(index=False))
        spread = severity_table["accuracy"].max() - severity_table["accuracy"].min()
        print(
            f"\naccuracy spread across severities = {spread:.4f} "
            f"({'no measurable severity effect' if spread == 0 else 'performance varies with severity'})"
        )
    return severity_table


# ---------------------------------------------------------------------------
# Error analysis + separability probe
# ---------------------------------------------------------------------------


def error_analysis(test_df: pd.DataFrame, y_pred: np.ndarray) -> pd.DataFrame:
    """TASK 6 - itemize every misclassified window."""
    _hr("TASK 6 - ERROR ANALYSIS")
    frame = test_df.copy()
    frame["_pred"] = y_pred
    wrong = frame[frame["_pred"] != frame[LABEL_COLUMN]]
    print(f"misclassified windows: {len(wrong):,} / {len(frame):,}")

    if wrong.empty:
        print(
            "\nZero misclassifications on the held-out 3 HP test recordings.\n"
            "That is NOT evidence of production readiness. See the separability\n"
            "probe below for the likely reason."
        )
        return pd.DataFrame(
            columns=[
                "true_class",
                "predicted_class",
                "fault_severity_in",
                "recording_id",
                "n_windows",
            ]
        )

    detail = (
        wrong.groupby(
            [LABEL_COLUMN, "_pred", SEVERITY_COLUMN, GROUP_COLUMN], dropna=False
        )
        .size()
        .reset_index(name="n_windows")
        .rename(
            columns={
                LABEL_COLUMN: "true_class",
                "_pred": "predicted_class",
                GROUP_COLUMN: "recording_id",
            }
        )
        .sort_values("n_windows", ascending=False)
    )
    print("\nper (true -> predicted, severity, recording) breakdown:")
    print(detail.to_string(index=False))

    patterns = (
        wrong.groupby([LABEL_COLUMN, "_pred"])
        .size()
        .reset_index(name="n_windows")
        .sort_values("n_windows", ascending=False)
    )
    print("\nmost common confusion patterns:")
    for _, row in patterns.iterrows():
        share = 100.0 * row["n_windows"] / len(wrong)
        print(
            f"  {row[LABEL_COLUMN]} -> {row['_pred']}: "
            f"{int(row['n_windows'])} windows ({share:.1f}% of all errors)"
        )
    return detail


def separability_probe(
    split: SplitFrames,
    X: dict[str, pd.DataFrame],
    y: dict[str, pd.Series],
    labels: list[str],
) -> dict[str, Any]:
    """TASK 6b - is the task trivially separable by a single amplitude feature?"""
    _hr("TASK 6b - SUSPICIOUSLY-EASY-SEPARATION PROBE")
    print(
        "If the full model scores perfectly, the honest question is whether ANY\n"
        "single amplitude statistic already separates the classes. Each probe below\n"
        "is a depth-3 decision tree fit on ONE feature from the train split only."
    )
    probes: dict[str, dict[str, float]] = {}
    for feature in SEPARABILITY_PROBE_FEATURES:
        stump = DecisionTreeClassifier(max_depth=3, random_state=RANDOM_STATE)
        with _suppressed_blas_fp_warnings():
            stump.fit(X["train"][[feature]], y["train"])
            val_pred = stump.predict(X["validation"][[feature]])
            test_pred = stump.predict(X["test"][[feature]])
        probes[feature] = {
            "validation_accuracy": float(accuracy_score(y["validation"], val_pred)),
            "validation_macro_f1": float(
                f1_score(y["validation"], val_pred, labels=labels, average="macro", zero_division=0)
            ),
            "test_accuracy": float(accuracy_score(y["test"], test_pred)),
            "test_macro_f1": float(
                f1_score(y["test"], test_pred, labels=labels, average="macro", zero_division=0)
            ),
        }
    probe_table = pd.DataFrame(probes).T
    print("\nsingle-feature depth-3 tree performance:")
    print(probe_table.round(4).to_string())

    print(
        "\nper-class vibration_rms range by split "
        "(min - max; overlapping ranges would make the task hard):"
    )
    range_rows: list[dict[str, Any]] = []
    for name, df in (
        ("train", split.train),
        ("validation", split.validation),
        ("test", split.test),
    ):
        for cls in labels:
            sub = df[df[LABEL_COLUMN] == cls]["vibration_rms"]
            if sub.empty:
                continue
            range_rows.append(
                {
                    "split": name,
                    "fault_class": cls,
                    "n": int(len(sub)),
                    "rms_min": float(sub.min()),
                    "rms_max": float(sub.max()),
                    "rms_mean": float(sub.mean()),
                }
            )
    range_table = pd.DataFrame(range_rows)
    print(range_table.round(4).to_string(index=False))

    print("\nper (class, severity) vibration_rms range on the TEST split:")
    sev_range = (
        split.test.groupby([LABEL_COLUMN, SEVERITY_COLUMN], dropna=False)["vibration_rms"]
        .agg(["count", "min", "max", "mean"])
        .round(4)
    )
    print(sev_range.to_string())

    # Do the test-split RMS intervals of any two classes overlap at all?
    test_ranges = {
        cls: (
            float(split.test[split.test[LABEL_COLUMN] == cls]["vibration_rms"].min()),
            float(split.test[split.test[LABEL_COLUMN] == cls]["vibration_rms"].max()),
        )
        for cls in labels
        if not split.test[split.test[LABEL_COLUMN] == cls].empty
    }
    overlaps: list[str] = []
    cls_list = list(test_ranges)
    for i, a in enumerate(cls_list):
        for b in cls_list[i + 1 :]:
            lo_a, hi_a = test_ranges[a]
            lo_b, hi_b = test_ranges[b]
            if lo_a <= hi_b and lo_b <= hi_a:
                overlaps.append(f"{a} <-> {b}")
    if overlaps:
        print(
            f"\nTEST vibration_rms intervals OVERLAP for: {overlaps}. A single RMS "
            "threshold cannot separate those pairs, so the model is doing real work "
            "for them."
        )
    else:
        print(
            "\nTEST vibration_rms intervals are DISJOINT for every class pair. The "
            "four classes are linearly orderable by raw amplitude alone at these "
            "severities, which is why high scores here say more about the dataset "
            "than about the classifier."
        )
    return {
        "single_feature_probes": probes,
        "test_rms_overlapping_class_pairs": overlaps,
    }


# ---------------------------------------------------------------------------
# Interpretability
# ---------------------------------------------------------------------------


def interpretability(
    fitted: dict[str, BaseEstimator],
    selected: str,
    feature_columns: tuple[str, ...],
    labels: list[str],
) -> dict[str, Any]:
    """TASK 7 - model-specific importance / coefficient reporting."""
    _hr("TASK 7 - FEATURE IMPORTANCE / INTERPRETABILITY")
    print(
        "Caveats that apply to everything below:\n"
        "  - importance is MODEL-SPECIFIC; two models can rank features differently\n"
        "    while scoring identically.\n"
        "  - these amplitude features are strongly correlated (rms / std / peak /\n"
        "    peak_to_peak), so they SHARE importance; a low score does not mean a\n"
        "    feature is uninformative, only that a correlated sibling absorbed it.\n"
        "  - importance is not physical causation."
    )
    out: dict[str, Any] = {}

    for name, model in fitted.items():
        inner = _inner_estimator(model)
        if hasattr(inner, "feature_importances_"):
            series = pd.Series(
                inner.feature_importances_, index=list(feature_columns)
            ).sort_values(ascending=False)
            print(f"\n{name} - impurity-based feature_importances_:")
            print("  " + series.round(6).to_string().replace("\n", "\n  "))
            out[name] = {
                "kind": "feature_importances_",
                "values": {k: float(v) for k, v in series.items()},
            }
        elif hasattr(inner, "coef_"):
            coef = pd.DataFrame(
                inner.coef_, index=list(inner.classes_), columns=list(feature_columns)
            )
            abs_rank = coef.abs().mean(axis=0).sort_values(ascending=False)
            print(
                f"\n{name} - standardized coefficients "
                "(StandardScaler inside the pipeline means these are directly "
                "comparable across features; one row per class, "
                "one-vs-rest style multinomial weights):"
            )
            print("  " + coef.round(4).to_string().replace("\n", "\n  "))
            print("\n  mean |coefficient| across classes (overall feature ranking):")
            print("  " + abs_rank.round(4).to_string().replace("\n", "\n  "))
            out[name] = {
                "kind": "standardized_coefficients",
                "per_class": {
                    str(cls): {c: float(v) for c, v in coef.loc[cls].items()}
                    for cls in coef.index
                },
                "mean_abs_coefficient": {k: float(v) for k, v in abs_rank.items()},
            }
        else:
            print(f"\n{name} - no built-in importance or coefficient attribute")
            out[name] = {"kind": "unavailable"}

    print(f"\nselected model for the interpretability figure: {selected}")
    return out


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def _fig_path(name: str) -> Path:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    return FIGURES_DIR / f"{FIGURE_PREFIX}{name}"


def _save(fig: plt.Figure, path: Path) -> Path:
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"wrote {path}")
    return path


def figure_model_comparison(table: pd.DataFrame) -> Path:
    ordered = table.sort_values("validation_macro_f1", ascending=True, kind="stable")
    y_pos = np.arange(len(ordered))
    height = 0.38
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.barh(
        y_pos + height / 2,
        ordered["validation_macro_f1"],
        height=height,
        color="#4c72b0",
        label="macro F1",
    )
    ax.barh(
        y_pos - height / 2,
        ordered["validation_accuracy"],
        height=height,
        color="#dd8452",
        label="accuracy",
    )
    for i, (f1v, accv) in enumerate(
        zip(ordered["validation_macro_f1"], ordered["validation_accuracy"])
    ):
        ax.text(f1v + 0.006, i + height / 2, f"{f1v:.4f}", va="center", fontsize=8)
        ax.text(accv + 0.006, i - height / 2, f"{accv:.4f}", va="center", fontsize=8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(ordered["model"])
    ax.set_xlim(0.0, 1.12)
    ax.set_xlabel("validation score (motor load = 2 HP)")
    ax.set_title("Experiment 2 - validation model comparison (multi-severity CWRU)")
    ax.legend(loc="lower right")
    ax.grid(True, axis="x", alpha=0.3)
    return _save(fig, _fig_path("validation_model_comparison.png"))


def figure_confusion(
    y_true, y_pred, labels: list[str], title: str, filename: str
) -> Path:
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        labels=labels,
        ax=ax,
        cmap="Blues",
        colorbar=True,
        xticks_rotation=30,
    )
    ax.set_title(title, fontsize=10)
    return _save(fig, _fig_path(filename))


def figure_interpretability(
    selected: str, model: BaseEstimator, feature_columns: tuple[str, ...]
) -> Path:
    inner = _inner_estimator(model)
    if hasattr(inner, "feature_importances_"):
        series = pd.Series(
            inner.feature_importances_, index=list(feature_columns)
        ).sort_values(ascending=True)
        fig, ax = plt.subplots(figsize=(9, 4.6))
        ax.barh(series.index, series.values, color="#4c72b0")
        ax.set_xlabel("mean decrease in impurity (Gini)")
        ax.set_title(
            f"Experiment 2 - {selected} feature importance (multi-severity)\n"
            "correlated amplitude features share importance; not causal",
            fontsize=10,
        )
        ax.grid(True, axis="x", alpha=0.3)
        return _save(fig, _fig_path("selected_model_interpretability.png"))

    coef = pd.DataFrame(
        inner.coef_, index=list(inner.classes_), columns=list(feature_columns)
    )
    fig, ax = plt.subplots(figsize=(10, 3.6))
    limit = float(np.abs(coef.to_numpy()).max()) or 1.0
    im = ax.imshow(coef.to_numpy(), cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
    ax.set_xticks(np.arange(len(feature_columns)))
    ax.set_xticklabels(feature_columns, rotation=40, ha="right", fontsize=8)
    ax.set_yticks(np.arange(len(coef.index)))
    ax.set_yticklabels(coef.index, fontsize=8)
    for i in range(coef.shape[0]):
        for j in range(coef.shape[1]):
            ax.text(
                j,
                i,
                f"{coef.iat[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=7,
                color="black",
            )
    fig.colorbar(im, ax=ax, label="standardized coefficient")
    ax.set_title(
        f"Experiment 2 - {selected} standardized coefficients (multi-severity)\n"
        "features are z-scored inside the pipeline, so magnitudes are comparable",
        fontsize=10,
    )
    return _save(fig, _fig_path("selected_model_interpretability.png"))


def figure_severity_performance(severity_table: pd.DataFrame, selected: str) -> Path | None:
    if severity_table.empty:
        return None
    recall_cols = [c for c in severity_table.columns if c.startswith("recall_")]
    x = np.arange(len(severity_table))
    series = ["accuracy", "macro_f1_present_classes"] + recall_cols
    width = 0.8 / max(len(series), 1)
    palette = ["#4c72b0", "#dd8452"] + [
        CLASS_COLORS.get(c.replace("recall_", ""), "#937860") for c in recall_cols
    ]

    fig, ax = plt.subplots(figsize=(10, 4.6))
    for i, (col, color) in enumerate(zip(series, palette)):
        offset = (i - (len(series) - 1) / 2) * width
        values = severity_table[col].to_numpy(dtype=float)
        ax.bar(x + offset, values, width=width, color=color, label=col)
        for xi, v in zip(x + offset, values):
            ax.text(xi, v + 0.012, f"{v:.3f}", ha="center", fontsize=6.5, rotation=90)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [
            f'{s:.3f}"\n(n={int(n):,} windows)'
            for s, n in zip(severity_table["fault_severity_in"], severity_table["n_windows"])
        ]
    )
    ax.set_ylim(0.0, 1.38)
    ax.set_ylabel("score")
    ax.set_xlabel("fault severity (defect diameter, inches) - FAULT windows only")
    ax.set_title(
        f"Experiment 2 - {selected} test performance by fault severity (3 HP)\n"
        "NORMAL has no physical severity and is excluded from these buckets",
        fontsize=10,
    )
    ax.legend(fontsize=7, ncol=5, loc="upper center", framealpha=0.95)
    ax.grid(True, axis="y", alpha=0.3)
    return _save(fig, _fig_path("test_performance_by_severity.png"))


def figure_rms_separability(split: SplitFrames, labels: list[str]) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    data = [split.test[split.test[LABEL_COLUMN] == c]["vibration_rms"].values for c in labels]
    box = axes[0].boxplot(
        data, tick_labels=labels, patch_artist=True, showfliers=True, widths=0.55
    )
    for patch, cls in zip(box["boxes"], labels):
        patch.set_facecolor(CLASS_COLORS[cls])
        patch.set_alpha(0.65)
    axes[0].set_title("vibration_rms by class - TEST split (3 HP)", fontsize=10)
    axes[0].set_ylabel("vibration_rms")
    axes[0].tick_params(axis="x", rotation=25)
    axes[0].grid(True, axis="y", alpha=0.3)

    severities = sorted(
        split.test.loc[split.test[LABEL_COLUMN] != "NORMAL", SEVERITY_COLUMN].dropna().unique()
    )
    offsets = np.linspace(-0.26, 0.26, max(len(severities), 1))
    for sev, off in zip(severities, offsets):
        xs, ys = [], []
        for i, cls in enumerate(labels):
            sub = split.test[
                (split.test[LABEL_COLUMN] == cls)
                & (split.test[SEVERITY_COLUMN].astype(float).round(3) == round(float(sev), 3))
            ]["vibration_rms"]
            if sub.empty:
                continue
            xs.append(i + off)
            ys.append(float(sub.mean()))
        axes[1].scatter(xs, ys, s=46, label=f'{float(sev):.3f}"')
    normal = split.test[split.test[LABEL_COLUMN] == "NORMAL"]["vibration_rms"]
    if not normal.empty:
        axes[1].scatter(
            [labels.index("NORMAL")],
            [float(normal.mean())],
            s=46,
            marker="s",
            color="#444444",
            label="NORMAL (no severity)",
        )
    axes[1].set_xticks(np.arange(len(labels)))
    axes[1].set_xticklabels(labels, rotation=25)
    axes[1].set_title("mean vibration_rms by class x severity - TEST split", fontsize=10)
    axes[1].legend(fontsize=7, title="severity")
    axes[1].grid(True, axis="y", alpha=0.3)

    fig.suptitle(
        "Experiment 2 - amplitude separability check: how much does raw RMS alone already explain?",
        fontsize=11,
    )
    return _save(fig, _fig_path("rms_separability_by_class.png"))


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def persist_artifacts(
    selected: str,
    model: BaseEstimator,
    rationale: str,
    split: SplitFrames,
    audit: SplitAudit,
    feature_columns: tuple[str, ...],
    labels: list[str],
    val_evaluations: dict[str, ModelEvaluation],
    comparison_table: pd.DataFrame,
    test_eval: ModelEvaluation,
    severity_table: pd.DataFrame,
    errors: pd.DataFrame,
    interpretation: dict[str, Any],
    probe: dict[str, Any],
    figures: list[Path],
    dataset_source: str,
) -> tuple[Path, Path]:
    _hr("TASK 9 - PERSIST EXPERIMENT-2 ARTIFACTS")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    slug = MODEL_SLUGS[selected]
    model_path = MODELS_DIR / f"{slug}_multiseverity_cwru.joblib"
    metadata_path = MODELS_DIR / f"{slug}_multiseverity_cwru.json"

    for reserved in ("rf_baseline_cwru.joblib", "rf_baseline_cwru.json"):
        assert model_path.name != reserved and metadata_path.name != reserved, (
            f"refusing to overwrite the frozen Experiment-1 artifact {reserved}"
        )

    joblib.dump(model, model_path)

    metadata: dict[str, Any] = {
        "experiment_name": EXPERIMENT_NAME,
        "research_question": RESEARCH_QUESTION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_source": dataset_source,
        "split_design": {
            "strategy": "explicit split by complete recording / motor load; no random row-level splitting",
            "group_column": GROUP_COLUMN,
            "load_column": LOAD_COLUMN,
            "training_loads_hp": list(TRAIN_LOADS),
            "validation_load_hp": list(VALIDATION_LOADS),
            "test_load_hp": list(TEST_LOADS),
            "included_severities_in": list(FAULT_SEVERITIES_IN),
            "window_size": DEFAULT_WINDOW_SIZE,
            "window_hop": DEFAULT_WINDOW_SIZE,
            "train_recordings": list(audit.recordings["train"]),
            "validation_recordings": list(audit.recordings["validation"]),
            "test_recordings": list(audit.recordings["test"]),
            "n_train_recordings": len(audit.recordings["train"]),
            "n_validation_recordings": len(audit.recordings["validation"]),
            "n_test_recordings": len(audit.recordings["test"]),
            "n_train_windows": audit.window_counts["train"],
            "n_validation_windows": audit.window_counts["validation"],
            "n_test_windows": audit.window_counts["test"],
            "class_counts": audit.class_counts,
            "severity_counts": audit.severity_counts,
        },
        "feature_columns": list(feature_columns),
        "label_column": LABEL_COLUMN,
        "class_labels": list(labels),
        "excluded_from_x": list(EXCLUDED_FROM_X),
        "excluded_from_x_reason": (
            "fault_severity_in is experimental metadata: the physical defect size of "
            "an unknown machine is not observable at inference time, so including it "
            "would leak information production never has. The remaining columns are "
            "provenance / acquisition metadata."
        ),
        "selected_model": selected,
        "selection_criterion": "highest validation macro-F1; ties within "
        f"{TIE_TOLERANCE} go to the simpler / more interpretable model",
        "selection_rationale": rationale,
        "model_type": _describe_estimator(model),
        "model_parameters": {
            k: (v if isinstance(v, (int, float, str, bool, type(None))) else str(v))
            for k, v in _inner_estimator(model).get_params().items()
        },
        "random_state": RANDOM_STATE,
        "candidate_validation_metrics": {
            name: ev.macro for name, ev in val_evaluations.items()
        },
        "candidate_validation_table": comparison_table.to_dict(orient="records"),
        "validation_metrics": val_evaluations[selected].macro,
        "validation_per_class": val_evaluations[selected].per_class.to_dict(orient="index"),
        "validation_confusion_matrix": {
            "labels": list(labels),
            "rows_actual_cols_predicted": val_evaluations[selected]
            .confusion.to_numpy()
            .tolist(),
        },
        "test_metrics": test_eval.macro,
        "test_per_class": test_eval.per_class.to_dict(orient="index"),
        "test_confusion_matrix": {
            "labels": list(labels),
            "rows_actual_cols_predicted": test_eval.confusion.to_numpy().tolist(),
        },
        "test_performance_by_severity": severity_table.to_dict(orient="records"),
        "test_misclassifications": errors.to_dict(orient="records"),
        "interpretability": interpretation,
        "separability_probe": probe,
        "class_imbalance_handling": {
            "logistic_regression": "class_weight='balanced'",
            "random_forest": "class_weight='balanced'",
            "gradient_boosting": (
                "none - sklearn GradientBoostingClassifier has no class_weight "
                "parameter; sample_weight at fit() time is the only equivalent and "
                "was deliberately not used, so GB trained under raw class priors"
            ),
        },
        "sampling_rate_notes": {
            "NORMAL": 48000,
            "INNER_RACE": 12000,
            "BALL": 12000,
            "OUTER_RACE": 12000,
            "resampling_performed": False,
            "assumption_source": (
                "CWRU Bearing Data Center vendor documentation - sampling rate is NOT "
                "stored inside the .mat file."
            ),
        },
        "known_limitations": list(LIMITATIONS),
        "figures": [str(p.relative_to(PROJECT_ROOT)) for p in figures],
        "experiment1_baseline_artifacts_untouched": [
            "ml/models/rf_baseline_cwru.joblib",
            "ml/models/rf_baseline_cwru.json",
        ],
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, default=str))

    print(f"wrote {model_path} ({model_path.stat().st_size / 1024:.1f} KB)")
    print(f"wrote {metadata_path} ({metadata_path.stat().st_size / 1024:.1f} KB)")
    print(
        "\nBoth artifacts are gitignored by the existing repository policy "
        "(ml/models/*.joblib, ml/models/*.json). The frozen Experiment-1 artifacts "
        "rf_baseline_cwru.joblib / .json were not written."
    )
    return model_path, metadata_path


def print_limitations() -> None:
    _hr("TASK 10 - KNOWN LIMITATIONS")
    for i, text in enumerate(LIMITATIONS, start=1):
        print(f"{i:2d}. {text}")


def print_experiment1_untouched() -> None:
    _hr("EXPERIMENT-1 INTACTNESS MARKER")
    print(
        "This module wrote nothing under the Experiment-1 namespace:\n"
        "  - ml/models/rf_baseline_cwru.joblib        NOT written\n"
        "  - ml/models/rf_baseline_cwru.json          NOT written\n"
        "  - ml/data/processed/cwru_features.csv      NOT written\n"
        "  - ml/reports/figures/validation_confusion_matrix.png   NOT written\n"
        "  - ml/reports/figures/test_confusion_matrix.png         NOT written\n"
        "  - ml/reports/figures/random_forest_feature_importance.png  NOT written\n"
        "Every Experiment-2 figure carries the 'experiment2_' prefix and the model "
        "artifact carries the '_multiseverity_cwru' suffix.\n"
        "ml/src/run_experiment.py still trains on BASELINE_SPECS (the 16-recording "
        "0.007\" subset) and is not imported by this module."
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> int:
    frame, dataset_source = load_expanded_frame()
    split, audit = build_experiment2_split(frame)
    X, y, feature_columns = define_xy(split)

    labels = [c for c in CLASS_ORDER if c in set(frame[LABEL_COLUMN].unique())]

    fitted = train_candidates(build_candidate_models(), X["train"], y["train"])
    val_evaluations, comparison_table = validation_comparison(
        fitted, X["validation"], y["validation"], labels
    )
    selected, rationale = select_model(val_evaluations)

    test_eval = final_test_evaluation(
        selected, fitted[selected], X["test"], y["test"], labels
    )
    severity_table = test_performance_by_severity(split.test, test_eval.predictions, labels)
    errors = error_analysis(split.test, test_eval.predictions)
    probe = separability_probe(split, X, y, labels)
    interpretation = interpretability(fitted, selected, feature_columns, labels)

    _hr("TASK 8 - EXPERIMENT-2 FIGURES")
    figures: list[Path] = [figure_model_comparison(comparison_table)]
    for name, ev in val_evaluations.items():
        figures.append(
            figure_confusion(
                y["validation"],
                ev.predictions,
                labels,
                f"Experiment 2 - validation confusion matrix\n{name} (2 HP, multi-severity)",
                f"validation_confusion_{name}.png",
            )
        )
    figures.append(
        figure_confusion(
            y["test"],
            test_eval.predictions,
            labels,
            f"Experiment 2 - TEST confusion matrix\n{selected} (3 HP held-out, multi-severity)",
            "test_confusion_matrix.png",
        )
    )
    figures.append(figure_interpretability(selected, fitted[selected], feature_columns))
    severity_fig = figure_severity_performance(severity_table, selected)
    if severity_fig is not None:
        figures.append(severity_fig)
    figures.append(figure_rms_separability(split, labels))

    model_path, metadata_path = persist_artifacts(
        selected=selected,
        model=fitted[selected],
        rationale=rationale,
        split=split,
        audit=audit,
        feature_columns=feature_columns,
        labels=labels,
        val_evaluations=val_evaluations,
        comparison_table=comparison_table,
        test_eval=test_eval,
        severity_table=severity_table,
        errors=errors,
        interpretation=interpretation,
        probe=probe,
        figures=figures,
        dataset_source=dataset_source,
    )

    print_limitations()
    print_experiment1_untouched()

    _hr("EXPERIMENT 2 COMPLETE")
    print(f"selected model : {selected}")
    print(
        f"validation     : accuracy={val_evaluations[selected].macro['accuracy']:.4f}  "
        f"macro-F1={val_evaluations[selected].macro['macro_f1']:.4f}"
    )
    print(
        f"test (3 HP)    : accuracy={test_eval.macro['accuracy']:.4f}  "
        f"macro-F1={test_eval.macro['macro_f1']:.4f}"
    )
    print(f"model artifact : {model_path}")
    print(f"metadata       : {metadata_path}")
    print("figures:")
    for p in figures:
        print(f"  {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
