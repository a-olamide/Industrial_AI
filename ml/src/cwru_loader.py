"""Conservative loader for CWRU Bearing Data Center MATLAB files.

The Case Western Reserve University (CWRU) dataset is distributed as
MATLAB v5 ``.mat`` files. Each file contains one bearing experiment with
per-channel time-domain vibration signals and, for many recordings, an
RPM measurement. Variable names inside the file are prefixed with the
experiment number (e.g. ``X097_DE_time``), so we cannot hard-code a
single key.

This module is intentionally read-only and side-effect free; higher
level orchestration (windowing, feature extraction, dataset assembly)
lives in :mod:`ml.src.feature_extraction` and
:mod:`ml.src.dataset_builder`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import loadmat


DRIVE_END_SUFFIX = "_DE_time"
FAN_END_SUFFIX = "_FE_time"
BASE_ACC_SUFFIX = "_BA_time"
RPM_SUFFIX = "RPM"


class CwruLoaderError(RuntimeError):
    """Raised when a CWRU ``.mat`` file cannot be interpreted."""


@dataclass(frozen=True)
class CwruRecording:
    """A single loaded CWRU experiment.

    Attributes
    ----------
    path:
        Absolute path to the source ``.mat`` file.
    variables:
        All non-metadata MATLAB variable names present in the file.
    drive_end_key:
        Name of the drive-end time-domain vibration variable that was
        selected (e.g. ``"X097_DE_time"``).
    drive_end_signal:
        Flattened 1-D ``float`` vibration signal from the drive-end
        accelerometer.
    fan_end_key / fan_end_signal:
        Fan-end vibration when present in the file, otherwise ``None``.
    rpm_key / rpm:
        Measured shaft speed if the file exposes an ``*RPM`` variable,
        otherwise ``None``. RPM is never fabricated.
    """

    path: Path
    variables: tuple[str, ...]
    drive_end_key: str
    drive_end_signal: np.ndarray
    fan_end_key: str | None
    fan_end_signal: np.ndarray | None
    rpm_key: str | None
    rpm: float | None


def _user_keys(mat: dict[str, Any]) -> list[str]:
    return [k for k in mat.keys() if not k.startswith("__")]


def _find_suffixed(
    keys: list[str],
    suffix: str,
    preferred_prefix: str | None = None,
) -> str | None:
    matches = [k for k in keys if k.endswith(suffix)]
    if not matches:
        return None
    if preferred_prefix is not None:
        for k in matches:
            if k.startswith(preferred_prefix):
                return k
    if len(matches) > 1:
        matches.sort()
    return matches[0]


def _flatten_signal(raw: Any, key: str) -> np.ndarray:
    arr = np.asarray(raw)
    if arr.size == 0:
        raise CwruLoaderError(f"Variable {key!r} is empty")
    flat = np.squeeze(arr)
    if flat.ndim != 1:
        raise CwruLoaderError(
            f"Variable {key!r} has shape {arr.shape}; expected a 1-D vibration signal"
        )
    return np.ascontiguousarray(flat, dtype=np.float64)


def _scalar_rpm(raw: Any, key: str) -> float:
    arr = np.asarray(raw).squeeze()
    if arr.ndim != 0:
        raise CwruLoaderError(
            f"Variable {key!r} has shape {np.asarray(raw).shape}; expected a scalar RPM"
        )
    value = float(arr)
    if not np.isfinite(value) or value <= 0:
        raise CwruLoaderError(f"Variable {key!r} = {value!r} is not a valid RPM")
    return value


def load_recording(
    mat_path: str | Path,
    expected_experiment_number: int | None = None,
) -> CwruRecording:
    """Load a single CWRU ``.mat`` file.

    Parameters
    ----------
    mat_path:
        Path to the ``.mat`` file on disk.
    expected_experiment_number:
        Optional hint for the numeric CWRU experiment (e.g. ``99``).
        Some CWRU downloads bundle two experiments in one file (e.g.
        ``99.mat`` also contains ``X098_*`` variables). When provided,
        the loader prefers variables whose key starts with
        ``f"X{n:03d}_"`` or ``f"X{n:03d}"``. Falls back to alphabetical
        selection when no hint is given or no matching key exists.

    Raises
    ------
    FileNotFoundError:
        If the file does not exist.
    CwruLoaderError:
        If no drive-end time-domain signal can be located, or if the
        signal is malformed.
    """
    path = Path(mat_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"CWRU .mat file not found: {path}")

    mat = loadmat(str(path), squeeze_me=False, struct_as_record=True)
    keys = _user_keys(mat)
    if not keys:
        raise CwruLoaderError(f"{path.name} contains no user variables")

    if expected_experiment_number is not None:
        prefix_padded = f"X{int(expected_experiment_number):03d}"
    else:
        prefix_padded = None

    de_key = _find_suffixed(keys, DRIVE_END_SUFFIX, preferred_prefix=prefix_padded)
    if de_key is None:
        raise CwruLoaderError(
            f"{path.name} has no drive-end variable (expected a key ending in {DRIVE_END_SUFFIX!r}). "
            f"Available variables: {keys}"
        )
    de_signal = _flatten_signal(mat[de_key], de_key)

    fe_key = _find_suffixed(keys, FAN_END_SUFFIX, preferred_prefix=prefix_padded)
    fe_signal = _flatten_signal(mat[fe_key], fe_key) if fe_key else None

    # RPM variables follow the pattern XnnnRPM (no underscore before "RPM"),
    # so the same experiment prefix filter applies.
    rpm_key = _find_suffixed(keys, RPM_SUFFIX, preferred_prefix=prefix_padded)
    rpm_value = _scalar_rpm(mat[rpm_key], rpm_key) if rpm_key else None

    return CwruRecording(
        path=path,
        variables=tuple(keys),
        drive_end_key=de_key,
        drive_end_signal=de_signal,
        fan_end_key=fe_key,
        fan_end_signal=fe_signal,
        rpm_key=rpm_key,
        rpm=rpm_value,
    )


def inspect_mat(mat_path: str | Path) -> dict[str, Any]:
    """Return a lightweight description of a ``.mat`` file for exploration."""
    path = Path(mat_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"CWRU .mat file not found: {path}")

    mat = loadmat(str(path), squeeze_me=False, struct_as_record=True)
    keys = _user_keys(mat)
    variables = []
    for key in keys:
        arr = np.asarray(mat[key])
        variables.append({"name": key, "shape": tuple(arr.shape), "dtype": str(arr.dtype)})
    return {
        "path": str(path),
        "variables": variables,
        "drive_end_candidates": [k for k in keys if k.endswith(DRIVE_END_SUFFIX)],
        "fan_end_candidates": [k for k in keys if k.endswith(FAN_END_SUFFIX)],
        "base_acc_candidates": [k for k in keys if k.endswith(BASE_ACC_SUFFIX)],
        "rpm_candidates": [k for k in keys if k.endswith(RPM_SUFFIX)],
    }
