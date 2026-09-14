"""Assemble the canonical MachineFeatureVector dataset from CWRU recordings.

The CWRU download exposes recordings as MATLAB files whose filenames on
the vendor website may be either the friendly experiment tag (e.g.
``IR007_0.mat``) or the raw experiment number (e.g. ``105.mat``). The
mapping between the two is not derivable from the file content, so we
maintain it explicitly in :data:`CWRU_RECORDINGS`.

The initial fault classes are:

- ``NORMAL``
- ``INNER_RACE``
- ``BALL``
- ``OUTER_RACE``

Each ``RecordingSpec`` below documents its class, motor load in HP, the
approximate shaft speed (RPM) reported by the vendor, the fault
severity in inches (0.007", 0.014", or 0.021" - equivalent to the
CWRU-documented "fault diameter"), and - for outer-race faults - the
fault position relative to the load zone.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from .cwru_loader import CwruRecording, load_recording
from .feature_extraction import (
    DEFAULT_WINDOW_SIZE,
    WindowFeatures,
    extract_features,
)


FAULT_CLASSES: tuple[str, ...] = ("NORMAL", "INNER_RACE", "BALL", "OUTER_RACE")

SOURCE_TAG = "CWRU_12kHz_DE"
DEFAULT_ASSET_ID = "cwru_bearing_12kDE"


@dataclass(frozen=True)
class RecordingSpec:
    """Vendor-documented metadata for one CWRU experiment.

    ``experiment_number`` is the numeric CWRU experiment id
    (e.g. Normal at 2 HP is experiment 99). Because some downloads
    bundle multiple experiments in one ``.mat`` file, the loader uses
    this hint to select the correct ``X{nnn}_*`` variables.

    ``candidate_filenames`` lists the plausible on-disk names for the
    same experiment. The user drops the downloaded ``.mat`` files into
    ``ml/data/raw/cwru/`` and the loader searches this list in order.

    ``sampling_rate_hz`` documents the vendor's published sampling rate
    for the recording. It is NOT stored inside the ``.mat`` file and
    the current time-domain baseline does not consume it, but it must
    be honored before any frequency-domain feature is added.
    """

    recording_id: str
    fault_class: str
    motor_load_hp: int
    approx_rpm: int
    fault_severity_in: float | None
    outer_race_position: str | None
    experiment_number: int
    sampling_rate_hz: int
    candidate_filenames: tuple[str, ...]


# Vendor metadata for the CWRU 12 kHz Drive End recordings covering
# fault severities 0.007", 0.014", and 0.021" across NORMAL and
# INNER_RACE / BALL / OUTER_RACE@6 faults at motor loads 0, 1, 2, 3 HP.
# See: https://engineering.case.edu/bearingdatacenter.
# Numeric filenames match CWRU's original download URLs.
# Normal recordings are published at 48 kHz; 12 kHz Drive End fault
# recordings are at 12 kHz. Sampling rate is NOT stored inside the
# .mat file; this table is the single source of truth.
CWRU_RECORDINGS: tuple[RecordingSpec, ...] = (
    # NORMAL (48 kHz)
    RecordingSpec("Normal_0",  "NORMAL",     0, 1797, None,  None,       97, 48000, ("Normal_0.mat", "97.mat", "097.mat")),
    RecordingSpec("Normal_1",  "NORMAL",     1, 1772, None,  None,       98, 48000, ("Normal_1.mat", "98.mat", "098.mat")),
    RecordingSpec("Normal_2",  "NORMAL",     2, 1750, None,  None,       99, 48000, ("Normal_2.mat", "99.mat", "099.mat")),
    RecordingSpec("Normal_3",  "NORMAL",     3, 1730, None,  None,      100, 48000, ("Normal_3.mat", "100.mat")),
    # INNER_RACE, 0.007"
    RecordingSpec("IR007_0",   "INNER_RACE", 0, 1797, 0.007, None,      105, 12000, ("IR007_0.mat", "105.mat")),
    RecordingSpec("IR007_1",   "INNER_RACE", 1, 1772, 0.007, None,      106, 12000, ("IR007_1.mat", "106.mat")),
    RecordingSpec("IR007_2",   "INNER_RACE", 2, 1750, 0.007, None,      107, 12000, ("IR007_2.mat", "107.mat")),
    RecordingSpec("IR007_3",   "INNER_RACE", 3, 1730, 0.007, None,      108, 12000, ("IR007_3.mat", "108.mat")),
    # INNER_RACE, 0.014"
    RecordingSpec("IR014_0",   "INNER_RACE", 0, 1797, 0.014, None,      169, 12000, ("IR014_0.mat", "169.mat")),
    RecordingSpec("IR014_1",   "INNER_RACE", 1, 1772, 0.014, None,      170, 12000, ("IR014_1.mat", "170.mat")),
    RecordingSpec("IR014_2",   "INNER_RACE", 2, 1750, 0.014, None,      171, 12000, ("IR014_2.mat", "171.mat")),
    RecordingSpec("IR014_3",   "INNER_RACE", 3, 1730, 0.014, None,      172, 12000, ("IR014_3.mat", "172.mat")),
    # INNER_RACE, 0.021"
    RecordingSpec("IR021_0",   "INNER_RACE", 0, 1797, 0.021, None,      209, 12000, ("IR021_0.mat", "209.mat")),
    RecordingSpec("IR021_1",   "INNER_RACE", 1, 1772, 0.021, None,      210, 12000, ("IR021_1.mat", "210.mat")),
    RecordingSpec("IR021_2",   "INNER_RACE", 2, 1750, 0.021, None,      211, 12000, ("IR021_2.mat", "211.mat")),
    RecordingSpec("IR021_3",   "INNER_RACE", 3, 1730, 0.021, None,      212, 12000, ("IR021_3.mat", "212.mat")),
    # BALL, 0.007"
    RecordingSpec("B007_0",    "BALL",       0, 1797, 0.007, None,      118, 12000, ("B007_0.mat", "118.mat")),
    RecordingSpec("B007_1",    "BALL",       1, 1772, 0.007, None,      119, 12000, ("B007_1.mat", "119.mat")),
    RecordingSpec("B007_2",    "BALL",       2, 1750, 0.007, None,      120, 12000, ("B007_2.mat", "120.mat")),
    RecordingSpec("B007_3",    "BALL",       3, 1730, 0.007, None,      121, 12000, ("B007_3.mat", "121.mat")),
    # BALL, 0.014"
    RecordingSpec("B014_0",    "BALL",       0, 1797, 0.014, None,      185, 12000, ("B014_0.mat", "185.mat")),
    RecordingSpec("B014_1",    "BALL",       1, 1772, 0.014, None,      186, 12000, ("B014_1.mat", "186.mat")),
    RecordingSpec("B014_2",    "BALL",       2, 1750, 0.014, None,      187, 12000, ("B014_2.mat", "187.mat")),
    RecordingSpec("B014_3",    "BALL",       3, 1730, 0.014, None,      188, 12000, ("B014_3.mat", "188.mat")),
    # BALL, 0.021"
    RecordingSpec("B021_0",    "BALL",       0, 1797, 0.021, None,      222, 12000, ("B021_0.mat", "222.mat")),
    RecordingSpec("B021_1",    "BALL",       1, 1772, 0.021, None,      223, 12000, ("B021_1.mat", "223.mat")),
    RecordingSpec("B021_2",    "BALL",       2, 1750, 0.021, None,      224, 12000, ("B021_2.mat", "224.mat")),
    RecordingSpec("B021_3",    "BALL",       3, 1730, 0.021, None,      225, 12000, ("B021_3.mat", "225.mat")),
    # OUTER_RACE @6, 0.007"
    RecordingSpec("OR007@6_0", "OUTER_RACE", 0, 1797, 0.007, "6_oclock", 130, 12000, ("OR007@6_0.mat", "130.mat")),
    RecordingSpec("OR007@6_1", "OUTER_RACE", 1, 1772, 0.007, "6_oclock", 131, 12000, ("OR007@6_1.mat", "131.mat")),
    RecordingSpec("OR007@6_2", "OUTER_RACE", 2, 1750, 0.007, "6_oclock", 132, 12000, ("OR007@6_2.mat", "132.mat")),
    RecordingSpec("OR007@6_3", "OUTER_RACE", 3, 1730, 0.007, "6_oclock", 133, 12000, ("OR007@6_3.mat", "133.mat")),
    # OUTER_RACE @6, 0.014"
    RecordingSpec("OR014@6_0", "OUTER_RACE", 0, 1797, 0.014, "6_oclock", 197, 12000, ("OR014@6_0.mat", "197.mat")),
    RecordingSpec("OR014@6_1", "OUTER_RACE", 1, 1772, 0.014, "6_oclock", 198, 12000, ("OR014@6_1.mat", "198.mat")),
    RecordingSpec("OR014@6_2", "OUTER_RACE", 2, 1750, 0.014, "6_oclock", 199, 12000, ("OR014@6_2.mat", "199.mat")),
    RecordingSpec("OR014@6_3", "OUTER_RACE", 3, 1730, 0.014, "6_oclock", 200, 12000, ("OR014@6_3.mat", "200.mat")),
    # OUTER_RACE @6, 0.021"
    RecordingSpec("OR021@6_0", "OUTER_RACE", 0, 1797, 0.021, "6_oclock", 234, 12000, ("OR021@6_0.mat", "234.mat")),
    RecordingSpec("OR021@6_1", "OUTER_RACE", 1, 1772, 0.021, "6_oclock", 235, 12000, ("OR021@6_1.mat", "235.mat")),
    RecordingSpec("OR021@6_2", "OUTER_RACE", 2, 1750, 0.021, "6_oclock", 236, 12000, ("OR021@6_2.mat", "236.mat")),
    RecordingSpec("OR021@6_3", "OUTER_RACE", 3, 1730, 0.021, "6_oclock", 237, 12000, ("OR021@6_3.mat", "237.mat")),
)


RECORDINGS_BY_ID: dict[str, RecordingSpec] = {r.recording_id: r for r in CWRU_RECORDINGS}


# The 16-recording subset used by the frozen 0.007" baseline experiment
# (Experiment 1). Kept as a distinct constant so extending
# ``CWRU_RECORDINGS`` cannot silently change the baseline's behavior.
BASELINE_SPECS: tuple[RecordingSpec, ...] = tuple(
    r for r in CWRU_RECORDINGS
    if r.fault_class == "NORMAL" or r.fault_severity_in == 0.007
)

# Convenience severity tuple. NORMAL is the None-severity anchor and
# is intentionally not listed here.
FAULT_SEVERITIES_IN: tuple[float, ...] = (0.007, 0.014, 0.021)


@dataclass(frozen=True)
class RecordingResolution:
    spec: RecordingSpec
    path: Path


def resolve_recording_paths(
    raw_dir: str | Path,
    specs: Iterable[RecordingSpec] = CWRU_RECORDINGS,
) -> tuple[list[RecordingResolution], list[RecordingSpec]]:
    """Locate the on-disk ``.mat`` file for each spec.

    Returns ``(found, missing)``. Missing recordings are returned rather
    than raised so callers can surface a clear "drop these files here"
    message.
    """
    raw = Path(raw_dir).expanduser().resolve()
    found: list[RecordingResolution] = []
    missing: list[RecordingSpec] = []
    for spec in specs:
        match: Path | None = None
        for name in spec.candidate_filenames:
            candidate = raw / name
            if candidate.is_file():
                match = candidate
                break
        if match is None:
            missing.append(spec)
        else:
            found.append(RecordingResolution(spec=spec, path=match))
    return found, missing


def _row_from_features(
    features: WindowFeatures,
    spec: RecordingSpec,
    recording: CwruRecording,
    asset_id: str,
) -> dict[str, object]:
    return {
        "source": SOURCE_TAG,
        "asset_id": asset_id,
        "recording_id": spec.recording_id,
        "window_id": features.window_id,
        "vibration_rms": features.vibration_rms,
        "vibration_std": features.vibration_std,
        "vibration_peak": features.vibration_peak,
        "vibration_peak_to_peak": features.vibration_peak_to_peak,
        "vibration_kurtosis": features.vibration_kurtosis,
        "vibration_skewness": features.vibration_skewness,
        "crest_factor": features.crest_factor,
        "rotational_speed_rpm": float(recording.rpm) if recording.rpm is not None else float(spec.approx_rpm),
        "motor_load_hp": float(spec.motor_load_hp),
        "fault_class": spec.fault_class,
        # Metadata columns - not used as ML input features.
        "fault_severity_in": spec.fault_severity_in,
        "sampling_rate_hz": int(spec.sampling_rate_hz),
    }


def build_feature_frame(
    raw_dir: str | Path,
    window_size: int = DEFAULT_WINDOW_SIZE,
    hop: int | None = None,
    asset_id: str = DEFAULT_ASSET_ID,
    specs: Iterable[RecordingSpec] | None = None,
) -> pd.DataFrame:
    """Load available CWRU recordings and emit the canonical feature frame.

    Missing recordings are skipped silently so the caller can incrementally
    populate ``ml/data/raw/cwru/`` and re-run. To surface which recordings
    are still missing, call :func:`resolve_recording_paths` directly.

    Pass ``specs=BASELINE_SPECS`` to restrict to the 16-recording 0.007"
    baseline. When ``specs`` is ``None`` the full :data:`CWRU_RECORDINGS`
    table is used.
    """
    selected_specs = tuple(specs) if specs is not None else CWRU_RECORDINGS
    resolutions, _missing = resolve_recording_paths(raw_dir, selected_specs)
    rows: list[dict[str, object]] = []
    for resolution in resolutions:
        recording = load_recording(
            resolution.path,
            expected_experiment_number=resolution.spec.experiment_number,
        )
        window_features = extract_features(
            recording.drive_end_signal,
            window_size=window_size,
            hop=hop,
        )
        for wf in window_features:
            rows.append(_row_from_features(wf, resolution.spec, recording, asset_id))
    columns = [
        "source",
        "asset_id",
        "recording_id",
        "window_id",
        "vibration_rms",
        "vibration_std",
        "vibration_peak",
        "vibration_peak_to_peak",
        "vibration_kurtosis",
        "vibration_skewness",
        "crest_factor",
        "rotational_speed_rpm",
        "motor_load_hp",
        "fault_class",
        "fault_severity_in",
        "sampling_rate_hz",
    ]
    return pd.DataFrame(rows, columns=columns)
