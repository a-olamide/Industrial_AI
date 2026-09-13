"""End-to-end orchestration for the CWRU bearing-fault baseline experiment.

Runs steps 1-9 from the project brief:
1. Inspect and validate every CWRU .mat file in ml/data/raw/cwru/.
2. Render waveform exploration figures.
3. Extract 2048-sample non-overlapping window features into one DataFrame.
4. Run feature-quality checks and render distribution figures.
5. Split by motor load: train={0,1} HP, validation={2} HP, test={3} HP.
6. Fit RandomForestClassifier, evaluate on validation.
7. Evaluate once on test, dump feature importance figure/table.
8. Render validation and test confusion matrices.
9. Persist the pipeline + metadata JSON to ml/models/.

Invoked as ``python -m ml.src.run_experiment`` from the project root.
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from .cwru_loader import inspect_mat, load_recording
from .dataset_builder import (
    DEFAULT_ASSET_ID,
    SOURCE_TAG,
    RecordingSpec,
    resolve_recording_paths,
)
from .feature_extraction import DEFAULT_WINDOW_SIZE, extract_features
from .split_dataset import load_aware_split, summarize_split
from .train_baseline import FEATURE_COLUMNS, LABEL_COLUMN


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "ml" / "data" / "raw" / "cwru"
PROCESSED_DIR = PROJECT_ROOT / "ml" / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "ml" / "models"
FIGURES_DIR = PROJECT_ROOT / "ml" / "reports" / "figures"

FEATURES_CSV = PROCESSED_DIR / "cwru_features.csv"
MODEL_NAME = "rf_baseline_cwru"

CLASS_ORDER = ("NORMAL", "INNER_RACE", "BALL", "OUTER_RACE")
CLASS_COLORS = {
    "NORMAL": "#2b8cbe",
    "INNER_RACE": "#e34a33",
    "BALL": "#31a354",
    "OUTER_RACE": "#756bb1",
}


def _hr(title: str) -> None:
    line = "=" * 72
    print(f"\n{line}\n{title}\n{line}")


@dataclass
class FileInspection:
    filename: str
    spec: RecordingSpec
    de_key: str
    rpm_key: str | None
    rpm_measured: float | None
    n_samples: int
    minimum: float
    maximum: float
    mean: float
    std: float
    has_nan: bool
    has_inf: bool
    variables: tuple[str, ...]


def inspect_all_files() -> list[FileInspection]:
    _hr("STEP 1 - FILE INSPECTION")
    resolutions, missing = resolve_recording_paths(RAW_DIR)
    if missing:
        raise SystemExit(f"missing recordings: {[m.recording_id for m in missing]}")

    rows: list[dict[str, Any]] = []
    inspections: list[FileInspection] = []
    for res in resolutions:
        info = inspect_mat(res.path)
        rec = load_recording(res.path, expected_experiment_number=res.spec.experiment_number)
        sig = rec.drive_end_signal
        insp = FileInspection(
            filename=res.path.name,
            spec=res.spec,
            de_key=rec.drive_end_key,
            rpm_key=rec.rpm_key,
            rpm_measured=rec.rpm,
            n_samples=int(sig.size),
            minimum=float(sig.min()),
            maximum=float(sig.max()),
            mean=float(sig.mean()),
            std=float(sig.std()),
            has_nan=bool(np.isnan(sig).any()),
            has_inf=bool(np.isinf(sig).any()),
            variables=tuple(v["name"] for v in info["variables"]),
        )
        inspections.append(insp)
        rows.append(
            {
                "filename": insp.filename,
                "class": insp.spec.fault_class,
                "hp": insp.spec.motor_load_hp,
                "meta_rpm": insp.spec.approx_rpm,
                "mat_rpm_key": insp.rpm_key or "-",
                "mat_rpm": f"{insp.rpm_measured:.1f}" if insp.rpm_measured is not None else "-",
                "de_key": insp.de_key,
                "samples": insp.n_samples,
                "min": round(insp.minimum, 4),
                "max": round(insp.maximum, 4),
                "mean": round(insp.mean, 6),
                "std": round(insp.std, 4),
                "nan": insp.has_nan,
                "inf": insp.has_inf,
            }
        )
    inspection_df = pd.DataFrame(rows)
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(inspection_df.to_string(index=False))

    print("\nMATLAB variables seen per file:")
    for insp in inspections:
        print(f"  {insp.filename:16s} -> {list(insp.variables)}")

    _flag_sample_rate_assumptions(inspections)
    _flag_sample_count_outliers(inspection_df)
    return inspections


def _flag_sample_rate_assumptions(inspections: list[FileInspection]) -> None:
    print(
        "\nSampling-rate note: CWRU .mat files do NOT embed a sampling-rate field. "
        "For 0.007\" faults (IR/B/OR@6) the vendor sampling rate is 12,000 Hz; the "
        "Normal_* recordings on the CWRU website are published at 48,000 Hz. "
        "Nothing downstream (RF baseline on time-domain window statistics) reads the "
        "sampling rate directly, but this assumption is documented explicitly and "
        "must be respected before any frequency-domain feature is added."
    )


def _flag_sample_count_outliers(inspection_df: pd.DataFrame) -> None:
    counts = inspection_df.groupby("class")["samples"].agg(["min", "max", "mean"])
    print("\nSample count per class:")
    print(counts.to_string())
    unique = inspection_df["samples"].unique()
    if len(unique) > 1:
        print(
            f"\nNote: sample counts differ across the 16 recordings "
            f"(min={int(inspection_df['samples'].min()):,}, "
            f"max={int(inspection_df['samples'].max()):,}). This is expected: "
            f"CWRU Normal recordings are longer than the fault recordings, and the "
            f"Normal set is also sampled at 48 kHz vs. 12 kHz for the faults, so raw "
            f"sample counts are NOT interchangeable with duration."
        )


def render_waveform_figures(inspections: list[FileInspection]) -> list[Path]:
    _hr("STEP 2 - WAVEFORM FIGURES")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    by_id = {insp.spec.recording_id: insp for insp in inspections}

    normal_insp = by_id["Normal_0"]
    normal = load_recording(
        RAW_DIR / normal_insp.filename,
        expected_experiment_number=normal_insp.spec.experiment_number,
    )
    fig, ax = plt.subplots(figsize=(10, 3.2))
    n_preview = min(5000, normal.drive_end_signal.size)
    ax.plot(normal.drive_end_signal[:n_preview], linewidth=0.5, color=CLASS_COLORS["NORMAL"])
    ax.set_xlabel("sample index")
    ax.set_ylabel("acceleration (a.u.)")
    ax.set_title(f"Normal_0 drive-end waveform - first {n_preview} samples ({normal.drive_end_key})")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    normal_fig = FIGURES_DIR / "normal_waveform.png"
    fig.savefig(normal_fig, dpi=150)
    plt.close(fig)
    saved.append(normal_fig)
    print(f"wrote {normal_fig}")

    representatives = {
        "NORMAL": "Normal_0",
        "INNER_RACE": "IR007_0",
        "BALL": "B007_0",
        "OUTER_RACE": "OR007@6_0",
    }
    fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
    for ax, cls in zip(axes, CLASS_ORDER):
        rec_id = representatives[cls]
        rep_insp = by_id[rec_id]
        rec = load_recording(
            RAW_DIR / rep_insp.filename,
            expected_experiment_number=rep_insp.spec.experiment_number,
        )
        preview_n = min(5000, rec.drive_end_signal.size)
        ax.plot(
            rec.drive_end_signal[:preview_n],
            linewidth=0.5,
            color=CLASS_COLORS[cls],
        )
        ax.set_ylabel("accel (a.u.)")
        ax.set_title(f"{cls} - {rec_id} ({rec.drive_end_key})", loc="left")
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("sample index (0 - 5000)")
    fig.suptitle("CWRU drive-end waveforms by fault class (first 5000 samples)")
    fig.tight_layout()
    fault_fig = FIGURES_DIR / "fault_waveform_comparison.png"
    fig.savefig(fault_fig, dpi=150)
    plt.close(fig)
    saved.append(fault_fig)
    print(f"wrote {fault_fig}")

    return saved


def build_feature_frame_local(inspections: list[FileInspection]) -> pd.DataFrame:
    _hr("STEP 3 - FEATURE EXTRACTION")
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for insp in inspections:
        rec = load_recording(
            RAW_DIR / insp.filename,
            expected_experiment_number=insp.spec.experiment_number,
        )
        rpm = rec.rpm if rec.rpm is not None else float(insp.spec.approx_rpm)
        window_features = extract_features(rec.drive_end_signal, window_size=DEFAULT_WINDOW_SIZE)
        for wf in window_features:
            rows.append(
                {
                    "source": SOURCE_TAG,
                    "asset_id": DEFAULT_ASSET_ID,
                    "recording_id": insp.spec.recording_id,
                    "window_id": wf.window_id,
                    "vibration_rms": wf.vibration_rms,
                    "vibration_std": wf.vibration_std,
                    "vibration_peak": wf.vibration_peak,
                    "vibration_peak_to_peak": wf.vibration_peak_to_peak,
                    "vibration_kurtosis": wf.vibration_kurtosis,
                    "vibration_skewness": wf.vibration_skewness,
                    "crest_factor": wf.crest_factor,
                    "rotational_speed_rpm": rpm,
                    "motor_load_hp": float(insp.spec.motor_load_hp),
                    "fault_class": insp.spec.fault_class,
                }
            )
    frame = pd.DataFrame(rows)
    print(f"total feature rows: {len(frame):,}")
    print("\nrows per recording:")
    print(frame.groupby("recording_id").size().sort_values(ascending=False).to_string())
    print("\nrows per fault class:")
    print(frame["fault_class"].value_counts().to_string())
    print("\ndescriptive stats of vibration features by fault class:")
    stats = (
        frame.groupby("fault_class")[
            [
                "vibration_rms",
                "vibration_std",
                "vibration_peak",
                "vibration_peak_to_peak",
                "vibration_kurtosis",
                "vibration_skewness",
                "crest_factor",
            ]
        ]
        .describe()
        .T
    )
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(stats.to_string())

    frame.to_csv(FEATURES_CSV, index=False)
    print(f"\nwrote {FEATURES_CSV} ({FEATURES_CSV.stat().st_size / 1024:.1f} KB)")
    return frame


def feature_quality_checks(frame: pd.DataFrame) -> list[Path]:
    _hr("STEP 4 - FEATURE QUALITY CHECKS")
    numeric_cols = list(FEATURE_COLUMNS)
    numeric_frame = frame[numeric_cols]

    missing = numeric_frame.isna().sum()
    inf_counts = np.isinf(numeric_frame.to_numpy()).sum(axis=0)
    variances = numeric_frame.var(ddof=0)
    scale = numeric_frame.agg(["min", "max", "mean", "std"]).T

    print("missing values per feature:")
    print(missing.to_string())
    print("\ninfinite value counts per feature:")
    for col, cnt in zip(numeric_cols, inf_counts):
        print(f"  {col:26s} {int(cnt)}")
    print("\nvariance per feature (zero-variance features are ignored by the RF but flagged here):")
    print(variances.to_string())
    print("\nfeature scale summary:")
    print(scale.to_string())
    print("\nclass counts:")
    print(frame["fault_class"].value_counts().to_string())

    zero_var = variances[variances == 0].index.tolist()
    if zero_var:
        print(f"\nWARNING: zero-variance features detected: {zero_var}")

    q1 = numeric_frame.quantile(0.25)
    q3 = numeric_frame.quantile(0.75)
    iqr = q3 - q1
    outlier_mask = (
        (numeric_frame < (q1 - 3.0 * iqr)) | (numeric_frame > (q3 + 3.0 * iqr))
    ).any(axis=1)
    print(
        f"\nrows flagged as multi-feature outliers (>3.0 * IQR beyond quartiles on any feature): "
        f"{int(outlier_mask.sum()):,} / {len(numeric_frame):,}"
    )

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    saved.append(_class_distribution_figure(frame, "vibration_rms", "feature_distribution_rms.png"))
    saved.append(
        _class_distribution_figure(frame, "vibration_kurtosis", "feature_distribution_kurtosis.png")
    )
    return saved


def _class_distribution_figure(frame: pd.DataFrame, column: str, filename: str) -> Path:
    fig, ax = plt.subplots(figsize=(9, 4.5))
    data = [frame.loc[frame["fault_class"] == cls, column].values for cls in CLASS_ORDER]
    box = ax.boxplot(
        data,
        tick_labels=list(CLASS_ORDER),
        patch_artist=True,
        showfliers=True,
        widths=0.55,
    )
    for patch, cls in zip(box["boxes"], CLASS_ORDER):
        patch.set_facecolor(CLASS_COLORS[cls])
        patch.set_alpha(0.6)
    ax.set_xlabel("fault class")
    ax.set_ylabel(column)
    ax.set_title(f"{column} distribution by fault class (2048-sample windows)")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    path = FIGURES_DIR / filename
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"wrote {path}")
    return path


def load_split_and_report(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    _hr("STEP 5 - LOAD-AWARE SPLIT (50/25/25 recording-level)")
    split = load_aware_split(
        frame,
        train_loads=(0.0, 1.0),
        validation_loads=(2.0,),
        test_loads=(3.0,),
    )
    print("Split summary (rows / groups / class counts):")
    print(summarize_split(split).to_string(index=False))
    print("\nTRAIN recordings (0 HP + 1 HP):")
    _print_recordings(split.train)
    print("\nVALIDATION recordings (2 HP):")
    _print_recordings(split.validation)
    print("\nTEST recordings (3 HP):")
    _print_recordings(split.test)
    print(
        "\nWhy 50/25/25 and not 70/15/15: only four independent motor-load "
        "recordings exist per fault class in this milestone dataset (0, 1, 2, 3 HP). "
        "Two loads for train (0 & 1 HP) and one load each for validation (2 HP) and "
        "test (3 HP) is the finest recording-level partition that keeps every fault "
        "class in every split without leaking windows between splits. A future "
        "experiment with additional fault diameters and outer-race positions will "
        "support broader group-aware cross-validation."
    )
    return split.train, split.validation, split.test


def _print_recordings(df: pd.DataFrame) -> None:
    if df.empty:
        print("  (empty)")
        return
    grouped = (
        df.groupby(["fault_class", "recording_id", "motor_load_hp"])
        .size()
        .reset_index(name="n_windows")
        .sort_values(["fault_class", "recording_id"])
    )
    for _, row in grouped.iterrows():
        print(
            f"  {row['fault_class']:11s} {row['recording_id']:12s} "
            f"load={int(row['motor_load_hp'])}HP  windows={int(row['n_windows']):,}"
        )


def _metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def _print_metrics(label: str, metrics: dict[str, float]) -> None:
    print(
        f"{label:11s} accuracy={metrics['accuracy']:.4f}  "
        f"macro-P={metrics['macro_precision']:.4f}  "
        f"macro-R={metrics['macro_recall']:.4f}  "
        f"macro-F1={metrics['macro_f1']:.4f}"
    )


def train_and_evaluate(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[RandomForestClassifier, dict[str, Any]]:
    _hr("STEP 6 - RANDOM FOREST BASELINE")
    X_train = train_df[list(FEATURE_COLUMNS)]
    y_train = train_df[LABEL_COLUMN]
    X_val = val_df[list(FEATURE_COLUMNS)]
    y_val = val_df[LABEL_COLUMN]
    X_test = test_df[list(FEATURE_COLUMNS)]
    y_test = test_df[LABEL_COLUMN]

    clf = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    y_val_pred = clf.predict(X_val)
    y_test_pred = clf.predict(X_test)

    val_metrics = _metrics(y_val, y_val_pred)
    test_metrics = _metrics(y_test, y_test_pred)

    print("VALIDATION metrics (recordings at 2 HP):")
    _print_metrics("validation", val_metrics)
    print("\nper-class classification report (VALIDATION):")
    print(
        classification_report(
            y_val,
            y_val_pred,
            labels=list(CLASS_ORDER),
            digits=4,
            zero_division=0,
        )
    )
    print("VALIDATION confusion matrix (rows=actual, cols=predicted):")
    print(
        pd.DataFrame(
            confusion_matrix(y_val, y_val_pred, labels=list(CLASS_ORDER)),
            index=list(CLASS_ORDER),
            columns=list(CLASS_ORDER),
        ).to_string()
    )

    _hr("STEP 8a - VALIDATION CONFUSION MATRIX FIGURE")
    val_cm_path = _confusion_figure(
        y_val,
        y_val_pred,
        title="Validation confusion matrix (2 HP recordings)",
        filename="validation_confusion_matrix.png",
    )

    _hr("STEP 6b - FINAL TEST EVALUATION (single pass)")
    print("TEST metrics (recordings at 3 HP):")
    _print_metrics("test", test_metrics)
    print("\nper-class classification report (TEST):")
    print(
        classification_report(
            y_test,
            y_test_pred,
            labels=list(CLASS_ORDER),
            digits=4,
            zero_division=0,
        )
    )
    print("TEST confusion matrix (rows=actual, cols=predicted):")
    print(
        pd.DataFrame(
            confusion_matrix(y_test, y_test_pred, labels=list(CLASS_ORDER)),
            index=list(CLASS_ORDER),
            columns=list(CLASS_ORDER),
        ).to_string()
    )

    _hr("STEP 8b - TEST CONFUSION MATRIX FIGURE")
    test_cm_path = _confusion_figure(
        y_test,
        y_test_pred,
        title="Test confusion matrix (3 HP recordings)",
        filename="test_confusion_matrix.png",
    )

    _hr("STEP 7 - FEATURE IMPORTANCE")
    importance_series = pd.Series(clf.feature_importances_, index=list(FEATURE_COLUMNS))
    importance_series = importance_series.sort_values(ascending=False)
    print("random-forest feature importance:")
    print(importance_series.to_string())
    importance_path = _importance_figure(importance_series)

    return clf, {
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "val_predictions": y_val_pred,
        "test_predictions": y_test_pred,
        "figures": {
            "validation_confusion_matrix": str(val_cm_path.relative_to(PROJECT_ROOT)),
            "test_confusion_matrix": str(test_cm_path.relative_to(PROJECT_ROOT)),
            "feature_importance": str(importance_path.relative_to(PROJECT_ROOT)),
        },
        "feature_importance": {k: float(v) for k, v in importance_series.items()},
    }


def _confusion_figure(y_true: pd.Series, y_pred: np.ndarray, title: str, filename: str) -> Path:
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        labels=list(CLASS_ORDER),
        ax=ax,
        cmap="Blues",
        colorbar=True,
        xticks_rotation=30,
    )
    ax.set_title(title)
    fig.tight_layout()
    path = FIGURES_DIR / filename
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"wrote {path}")
    return path


def _importance_figure(importance: pd.Series) -> Path:
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ordered = importance.sort_values(ascending=True)
    ax.barh(ordered.index, ordered.values, color="#4c72b0")
    ax.set_xlabel("mean decrease in impurity (Gini)")
    ax.set_title("Random Forest feature importance (baseline, 200 trees)")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    path = FIGURES_DIR / "random_forest_feature_importance.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"wrote {path}")
    return path


def persist_model_and_metadata(
    clf: RandomForestClassifier,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    results: dict[str, Any],
    inspections: list[FileInspection],
) -> tuple[Path, Path]:
    _hr("STEP 9 - PERSIST MODEL + METADATA")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / f"{MODEL_NAME}.joblib"
    metadata_path = MODELS_DIR / f"{MODEL_NAME}.json"
    joblib.dump(clf, model_path)
    metadata = {
        "model_type": type(clf).__name__,
        "model_parameters": {
            "n_estimators": clf.n_estimators,
            "random_state": clf.random_state,
            "class_weight": "balanced",
            "n_jobs": -1,
        },
        "feature_names": list(FEATURE_COLUMNS),
        "label_column": LABEL_COLUMN,
        "class_order": list(CLASS_ORDER),
        "window_size": DEFAULT_WINDOW_SIZE,
        "window_hop": DEFAULT_WINDOW_SIZE,
        "sampling_rate_notes": {
            "NORMAL": 48000,
            "INNER_RACE": 12000,
            "BALL": 12000,
            "OUTER_RACE": 12000,
            "assumption_source": "CWRU Bearing Data Center vendor documentation - sampling rate is NOT stored inside the .mat file.",
        },
        "training_source_recordings": sorted(train_df["recording_id"].unique().tolist()),
        "validation_source_recordings": sorted(val_df["recording_id"].unique().tolist()),
        "test_source_recordings": sorted(test_df["recording_id"].unique().tolist()),
        "n_train_rows": int(len(train_df)),
        "n_validation_rows": int(len(val_df)),
        "n_test_rows": int(len(test_df)),
        "validation_metrics": results["val_metrics"],
        "test_metrics": results["test_metrics"],
        "feature_importance": results["feature_importance"],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2))
    print(f"wrote {model_path} ({model_path.stat().st_size / 1024:.1f} KB)")
    print(f"wrote {metadata_path}")
    return model_path, metadata_path


def main() -> None:
    inspections = inspect_all_files()
    waveform_figs = render_waveform_figures(inspections)
    feature_frame = build_feature_frame_local(inspections)
    quality_figs = feature_quality_checks(feature_frame)
    train_df, val_df, test_df = load_split_and_report(feature_frame)
    clf, results = train_and_evaluate(train_df, val_df, test_df)
    model_path, metadata_path = persist_model_and_metadata(
        clf, train_df, val_df, test_df, results, inspections
    )

    _hr("EXPERIMENT COMPLETE")
    all_figs = waveform_figs + quality_figs + [
        FIGURES_DIR / "validation_confusion_matrix.png",
        FIGURES_DIR / "test_confusion_matrix.png",
        FIGURES_DIR / "random_forest_feature_importance.png",
    ]
    print("figures written:")
    for f in all_figs:
        print(f"  {f}")
    print(f"model: {model_path}")
    print(f"metadata: {metadata_path}")
    print(f"features CSV: {FEATURES_CSV}")


if __name__ == "__main__":
    main()
