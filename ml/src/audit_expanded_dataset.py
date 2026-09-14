"""Dataset audit for the expanded CWRU set (0.007", 0.014", 0.021").

Purpose: inspect and validate the 40-recording expanded CWRU dataset
BEFORE any Experiment-2 model is trained. This script:

- Loads every recording listed in :data:`CWRU_RECORDINGS` via the
  existing safe loader (`ml/src/cwru_loader.py`).
- Reports per-file metadata + anomalies.
- Emits the expanded feature CSV at
  ``ml/data/processed/cwru_features_expanded.csv`` (gitignored) using
  the same 2048-sample non-overlapping windows as the baseline.
- Prints a compact audit table grouped by (class, severity, load).
- Writes exploratory figures under ``ml/reports/figures/`` with an
  ``expanded_`` prefix so the frozen baseline figures are not
  overwritten.

Deliberately DOES NOT:
- Split into train/validation/test.
- Fit any model.
- Touch ``rf_baseline_cwru.joblib`` or any baseline artifact.

Invoke as::

    python -m ml.src.audit_expanded_dataset
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .cwru_loader import inspect_mat, load_recording
from .dataset_builder import (
    BASELINE_SPECS,
    CWRU_RECORDINGS,
    DEFAULT_ASSET_ID,
    FAULT_SEVERITIES_IN,
    SOURCE_TAG,
    RecordingSpec,
    resolve_recording_paths,
)
from .feature_extraction import DEFAULT_WINDOW_SIZE, extract_features


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "ml" / "data" / "raw" / "cwru"
PROCESSED_DIR = PROJECT_ROOT / "ml" / "data" / "processed"
FIGURES_DIR = PROJECT_ROOT / "ml" / "reports" / "figures"

EXPANDED_CSV = PROCESSED_DIR / "cwru_features_expanded.csv"

CLASS_ORDER = ("NORMAL", "INNER_RACE", "BALL", "OUTER_RACE")
SEVERITY_ORDER = ("N/A",) + tuple(f"{s:.3f}" for s in FAULT_SEVERITIES_IN)

# Distinct color per (severity) so severity comparisons read easily.
SEVERITY_COLORS = {
    "N/A": "#2b8cbe",
    "0.007": "#fee08b",
    "0.014": "#fdae61",
    "0.021": "#d53e4f",
}


def _hr(title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


def _severity_label(value) -> str:
    if value is None:
        return "N/A"
    try:
        if pd.isna(value):
            return "N/A"
    except (TypeError, ValueError):
        pass
    return f"{float(value):.3f}"


@dataclass
class FileInspection:
    spec: RecordingSpec
    filename: str
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

    @property
    def severity_label(self) -> str:
        return _severity_label(self.spec.fault_severity_in)


def inspect_all_recordings() -> tuple[list[FileInspection], list[RecordingSpec], list[str]]:
    """Return (found_inspections, missing_specs, unexpected_files)."""
    _hr("STEP A - RAW RECORDING INSPECTION")
    resolutions, missing = resolve_recording_paths(RAW_DIR, CWRU_RECORDINGS)

    expected_names = {n for spec in CWRU_RECORDINGS for n in spec.candidate_filenames}
    on_disk = {p.name for p in RAW_DIR.glob("*.mat")}
    unexpected = sorted(on_disk - expected_names)

    inspections: list[FileInspection] = []
    for res in resolutions:
        info = inspect_mat(res.path)
        rec = load_recording(res.path, expected_experiment_number=res.spec.experiment_number)
        signal = rec.drive_end_signal
        insp = FileInspection(
            spec=res.spec,
            filename=res.path.name,
            de_key=rec.drive_end_key,
            rpm_key=rec.rpm_key,
            rpm_measured=rec.rpm,
            n_samples=int(signal.size),
            minimum=float(signal.min()),
            maximum=float(signal.max()),
            mean=float(signal.mean()),
            std=float(signal.std()),
            has_nan=bool(np.isnan(signal).any()),
            has_inf=bool(np.isinf(signal).any()),
            variables=tuple(v["name"] for v in info["variables"]),
        )
        inspections.append(insp)

    print(f"expected recordings   : {len(CWRU_RECORDINGS)}")
    print(f"resolved on disk      : {len(inspections)}")
    print(f"missing (not on disk) : {len(missing)}")
    print(f"unexpected .mat files : {len(unexpected)}")

    if missing:
        print("\nMissing recordings:")
        for spec in missing:
            print(
                f"  - {spec.recording_id:12s} class={spec.fault_class:11s} "
                f"severity={_severity_label(spec.fault_severity_in):5s} "
                f"load={spec.motor_load_hp}HP  candidates={spec.candidate_filenames}"
            )
    if unexpected:
        print("\nUnexpected .mat files (present on disk but not in metadata):")
        for name in unexpected:
            print(f"  - {name}")
    return inspections, missing, unexpected


def print_per_file_table(inspections: list[FileInspection]) -> None:
    rows = []
    for insp in inspections:
        rows.append(
            {
                "filename": insp.filename,
                "class": insp.spec.fault_class,
                "severity": insp.severity_label,
                "hp": insp.spec.motor_load_hp,
                "meta_rpm": insp.spec.approx_rpm,
                "mat_rpm": f"{insp.rpm_measured:.1f}" if insp.rpm_measured is not None else "-",
                "sr_hz": insp.spec.sampling_rate_hz,
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
    df = pd.DataFrame(rows).sort_values(["class", "severity", "hp"]).reset_index(drop=True)
    with pd.option_context("display.max_columns", None, "display.width", 220):
        print(df.to_string(index=False))
    return df


def flag_anomalies(inspections: list[FileInspection]) -> None:
    _hr("STEP B - ANOMALY CHECKS")
    print("Signal integrity:")
    problem = [i for i in inspections if i.has_nan or i.has_inf]
    if not problem:
        print("  no NaN or Inf in any drive-end signal.")
    else:
        for insp in problem:
            print(f"  {insp.filename}: has_nan={insp.has_nan} has_inf={insp.has_inf}")

    print("\nAmbiguous MATLAB variable checks:")
    ambiguous = []
    for insp in inspections:
        de_keys = [v for v in insp.variables if v.endswith("_DE_time")]
        if len(de_keys) > 1:
            ambiguous.append((insp.filename, de_keys, insp.de_key))
    if not ambiguous:
        print("  every file has exactly one _DE_time key or a resolved experiment-number match.")
    else:
        for name, de_keys, chosen in ambiguous:
            print(
                f"  {name}: multiple _DE_time keys {de_keys} -> chose {chosen!r} "
                f"via experiment_number hint."
            )

    print("\nSample-count sanity (per class):")
    df = pd.DataFrame(
        [
            {
                "class": insp.spec.fault_class,
                "severity": insp.severity_label,
                "samples": insp.n_samples,
                "sr_hz": insp.spec.sampling_rate_hz,
            }
            for insp in inspections
        ]
    )
    stats = (
        df.groupby(["class", "severity"])["samples"]
        .agg(["min", "max", "mean", "count"])
        .round(0)
        .astype({"min": int, "max": int, "mean": int, "count": int})
    )
    print(stats.to_string())

    print("\nSampling-rate distribution (per class):")
    sr_dist = df.groupby("class")["sr_hz"].agg(lambda s: sorted(set(int(x) for x in s)))
    print(sr_dist.to_string())

    print("\nRPM sanity (compare metadata vs. embedded RPM where present):")
    rpm_rows = []
    for insp in inspections:
        if insp.rpm_measured is not None:
            drift = float(insp.rpm_measured) - float(insp.spec.approx_rpm)
            rpm_rows.append(
                {
                    "recording": insp.spec.recording_id,
                    "meta_rpm": insp.spec.approx_rpm,
                    "mat_rpm": round(float(insp.rpm_measured), 1),
                    "delta": round(drift, 1),
                }
            )
    rpm_df = pd.DataFrame(rpm_rows).sort_values("recording").reset_index(drop=True)
    with pd.option_context("display.max_columns", None, "display.width", 120):
        print(rpm_df.to_string(index=False))
    max_drift = float(rpm_df["delta"].abs().max()) if not rpm_df.empty else 0.0
    print(f"\nmax |mat_rpm - meta_rpm| across recordings with embedded RPM: {max_drift:.1f}")


def build_expanded_frame(inspections: list[FileInspection]) -> pd.DataFrame:
    _hr("STEP C - EXPANDED FEATURE EXTRACTION (2048-sample non-overlap)")
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for insp in inspections:
        rec = load_recording(
            RAW_DIR / insp.filename,
            expected_experiment_number=insp.spec.experiment_number,
        )
        rpm = rec.rpm if rec.rpm is not None else float(insp.spec.approx_rpm)
        for wf in extract_features(rec.drive_end_signal, window_size=DEFAULT_WINDOW_SIZE):
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
                    "rotational_speed_rpm": float(rpm),
                    "motor_load_hp": float(insp.spec.motor_load_hp),
                    "fault_class": insp.spec.fault_class,
                    # Metadata columns - NOT ML input features.
                    "fault_severity_in": insp.spec.fault_severity_in,
                    "sampling_rate_hz": int(insp.spec.sampling_rate_hz),
                }
            )
    frame = pd.DataFrame(rows)
    frame.to_csv(EXPANDED_CSV, index=False)
    print(f"wrote {EXPANDED_CSV} ({EXPANDED_CSV.stat().st_size / 1024:.1f} KB)")
    print(f"total feature rows: {len(frame):,}")
    return frame


def print_audit_table(inspections: list[FileInspection], frame: pd.DataFrame) -> pd.DataFrame:
    _hr("STEP D - AUDIT TABLE")
    total_raw = sum(insp.n_samples for insp in inspections)
    print(f"A. Total recordings resolved: {len(inspections)}")
    print(f"   Total raw drive-end samples: {total_raw:,}")
    print(f"   Total 2048-sample feature windows: {len(frame):,}")

    print("\nB. Recordings grouped by fault_class:")
    rec_df = pd.DataFrame(
        [
            {
                "class": insp.spec.fault_class,
                "severity": insp.severity_label,
                "hp": insp.spec.motor_load_hp,
                "sr_hz": insp.spec.sampling_rate_hz,
                "samples": insp.n_samples,
                "recording_id": insp.spec.recording_id,
            }
            for insp in inspections
        ]
    )
    print(rec_df.groupby("class").size().to_string())
    print("\n   Recordings grouped by fault_severity_in:")
    print(rec_df.groupby("severity").size().to_string())
    print("\n   Recordings grouped by motor_load_hp:")
    print(rec_df.groupby("hp").size().to_string())

    print("\nC. Feature windows by fault_class:")
    print(frame.groupby("fault_class").size().to_string())
    print("\n   Feature windows by fault_severity_in:")
    print(frame["fault_severity_in"].fillna("N/A").astype(str).replace({"nan": "N/A"}).value_counts().sort_index().to_string())
    print("\n   Feature windows by motor_load_hp:")
    print(frame.groupby("motor_load_hp").size().to_string())

    print("\nD. Raw sample counts per recording:")
    print(rec_df[["recording_id", "samples"]].sort_values("recording_id").to_string(index=False))

    print("\nE. Feature-window counts per recording:")
    win_per_rec = frame.groupby("recording_id").size().reset_index(name="n_windows")
    print(win_per_rec.sort_values("recording_id").to_string(index=False))

    print("\nF. Sampling-rate distribution:")
    print(rec_df["sr_hz"].value_counts().sort_index().to_string())

    # G: missing expected combinations
    print("\nG. Missing expected (class, severity, load) combinations:")
    expected_combos = set()
    for spec in CWRU_RECORDINGS:
        expected_combos.add((spec.fault_class, _severity_label(spec.fault_severity_in), spec.motor_load_hp))
    seen_combos = set(
        (row["class"], row["severity"], int(row["hp"])) for _, row in rec_df.iterrows()
    )
    missing_combos = sorted(expected_combos - seen_combos)
    if not missing_combos:
        print("  none - every (class, severity, load) expected by the metadata table is present.")
    else:
        for cls, sev, hp in missing_combos:
            print(f"  - {cls} severity={sev} load={hp}HP")

    # H: pivot table
    print("\nH. Recording and window counts by (class, severity, load):")
    ns = (
        rec_df.groupby(["class", "severity", "hp"])
        .size()
        .reset_index(name="n_recordings")
    )
    nw = (
        frame.assign(severity=lambda d: d["fault_severity_in"].map(_severity_label))
        .groupby(["fault_class", "severity", "motor_load_hp"])
        .size()
        .reset_index(name="n_windows")
        .rename(columns={"fault_class": "class", "motor_load_hp": "hp"})
    )
    nw["hp"] = nw["hp"].astype(int)
    merged = ns.merge(nw, on=["class", "severity", "hp"], how="left").fillna({"n_windows": 0})
    merged["n_windows"] = merged["n_windows"].astype(int)
    merged = merged.sort_values(["class", "severity", "hp"]).reset_index(drop=True)
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(merged.to_string(index=False))
    return merged


def render_severity_figures(frame: pd.DataFrame, inspections: list[FileInspection]) -> list[Path]:
    _hr("STEP E - EXPLORATORY FIGURES")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    frame_labelled = frame.assign(
        severity=frame["fault_severity_in"].map(_severity_label)
    )

    saved.append(_class_severity_boxplot(frame_labelled, "vibration_rms",
                                         "expanded_rms_by_class_severity.png",
                                         "Vibration RMS by fault class and severity"))
    saved.append(_class_severity_boxplot(frame_labelled, "vibration_kurtosis",
                                         "expanded_kurtosis_by_class_severity.png",
                                         "Vibration kurtosis by fault class and severity"))
    saved.append(_severity_waveform_figure(inspections, chosen_class="INNER_RACE"))
    return saved


def _class_severity_boxplot(
    frame: pd.DataFrame,
    column: str,
    filename: str,
    title: str,
) -> Path:
    fig, ax = plt.subplots(figsize=(11, 5))
    positions = []
    tick_positions = []
    tick_labels = []
    box_data = []
    box_colors = []

    slot = 0.0
    class_gap = 1.0
    for cls in CLASS_ORDER:
        class_start = slot
        n_boxes = 0
        for sev in SEVERITY_ORDER:
            sub = frame[(frame["fault_class"] == cls) & (frame["severity"] == sev)][column]
            if sub.empty:
                continue
            positions.append(slot)
            box_data.append(sub.values)
            box_colors.append(SEVERITY_COLORS[sev])
            slot += 1.0
            n_boxes += 1
        if n_boxes:
            tick_positions.append((class_start + slot - 1.0) / 2.0)
            tick_labels.append(cls)
            slot += class_gap

    if not box_data:
        raise RuntimeError(f"no data to plot for column {column!r}")

    box = ax.boxplot(
        box_data,
        positions=positions,
        widths=0.75,
        patch_artist=True,
        showfliers=True,
    )
    for patch, color in zip(box["boxes"], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels)
    ax.set_ylabel(column)
    ax.set_title(title + " (2048-sample windows)")
    ax.grid(True, axis="y", alpha=0.3)

    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=SEVERITY_COLORS[sev], alpha=0.75, edgecolor="black")
        for sev in SEVERITY_ORDER
    ]
    ax.legend(handles, [f"severity {sev}" for sev in SEVERITY_ORDER], title="fault_severity_in", loc="upper left")

    fig.tight_layout()
    path = FIGURES_DIR / filename
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"wrote {path}")
    return path


def _severity_waveform_figure(inspections: list[FileInspection], chosen_class: str) -> Path:
    # Pick load 0 HP for all severities so RPM/load is held constant and
    # the only thing varying is severity - the actual research question.
    picks = {}
    for insp in inspections:
        if insp.spec.fault_class == chosen_class and insp.spec.motor_load_hp == 0:
            picks[insp.severity_label] = insp

    ordered_labels = [s for s in ("0.007", "0.014", "0.021") if s in picks]
    fig, axes = plt.subplots(len(ordered_labels), 1, figsize=(11, 6), sharex=True, sharey=True)
    if len(ordered_labels) == 1:
        axes = [axes]

    for ax, sev in zip(axes, ordered_labels):
        insp = picks[sev]
        rec = load_recording(
            RAW_DIR / insp.filename,
            expected_experiment_number=insp.spec.experiment_number,
        )
        signal = rec.drive_end_signal
        n_preview = min(5000, signal.size)
        ax.plot(signal[:n_preview], linewidth=0.5, color=SEVERITY_COLORS[sev])
        ax.set_ylabel("accel (a.u.)")
        ax.set_title(
            f"{insp.spec.recording_id} - severity={sev}\", "
            f"load={insp.spec.motor_load_hp}HP, "
            f"{rec.drive_end_key}",
            loc="left",
        )
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("sample index (0 - 5000)")
    fig.suptitle(
        f"CWRU {chosen_class} drive-end waveform vs. severity (load fixed at 0 HP)"
    )
    fig.tight_layout()
    path = FIGURES_DIR / "expanded_severity_waveforms.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"wrote {path}")
    return path


def print_baseline_intactness_marker() -> None:
    _hr("STEP F - BASELINE INTACTNESS")
    baseline_ids = {s.recording_id for s in BASELINE_SPECS}
    print(f"BASELINE_SPECS still exposes {len(baseline_ids)} recording ids.")
    expected = {
        "Normal_0", "Normal_1", "Normal_2", "Normal_3",
        "IR007_0", "IR007_1", "IR007_2", "IR007_3",
        "B007_0", "B007_1", "B007_2", "B007_3",
        "OR007@6_0", "OR007@6_1", "OR007@6_2", "OR007@6_3",
    }
    missing_from_baseline = expected - baseline_ids
    extra_in_baseline = baseline_ids - expected
    if not missing_from_baseline and not extra_in_baseline:
        print("BASELINE_SPECS matches the frozen 16-recording 0.007\" set exactly.")
    else:
        print(f"BASELINE drift detected. missing={sorted(missing_from_baseline)} extra={sorted(extra_in_baseline)}")


def main() -> int:
    inspections, missing, unexpected = inspect_all_recordings()
    if missing:
        print(
            "\nProceeding with the recordings that ARE present. Add the missing "
            "files to ml/data/raw/cwru/ to include them in the next audit."
        )
    print_per_file_table(inspections)
    flag_anomalies(inspections)
    frame = build_expanded_frame(inspections)
    print_audit_table(inspections, frame)
    figures = render_severity_figures(frame, inspections)
    print_baseline_intactness_marker()

    _hr("AUDIT COMPLETE - NO MODEL TRAINED")
    print("figures written:")
    for f in figures:
        print(f"  {f}")
    print(f"feature CSV: {EXPANDED_CSV}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
