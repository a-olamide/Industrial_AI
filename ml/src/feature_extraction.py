"""Window-based time-domain feature extraction for vibration signals.

This module deliberately covers only the initial time-domain feature
set described in the ML feature contract. Frequency-domain features
(FFT bands, envelope spectrum, bearing characteristic frequencies) are
intentionally out of scope for the CWRU foundation milestone.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Iterator

import numpy as np


DEFAULT_WINDOW_SIZE = 2048


@dataclass(frozen=True)
class WindowFeatures:
    """Time-domain descriptors for a single vibration window."""

    window_id: int
    start_index: int
    end_index: int
    vibration_rms: float
    vibration_std: float
    vibration_peak: float
    vibration_peak_to_peak: float
    vibration_kurtosis: float
    vibration_skewness: float
    crest_factor: float

    def as_dict(self) -> dict[str, float | int]:
        return asdict(self)


def iter_windows(
    signal: np.ndarray,
    window_size: int = DEFAULT_WINDOW_SIZE,
    hop: int | None = None,
) -> Iterator[tuple[int, int, np.ndarray]]:
    """Yield ``(window_id, start_index, samples)`` for each full window.

    A partial trailing window (fewer than ``window_size`` samples) is
    discarded so every emitted window has the same length.
    """
    if window_size <= 0:
        raise ValueError("window_size must be positive")
    step = window_size if hop is None else hop
    if step <= 0:
        raise ValueError("hop must be positive")

    flat = np.ascontiguousarray(np.asarray(signal).ravel(), dtype=np.float64)
    n = flat.size
    window_id = 0
    for start in range(0, n - window_size + 1, step):
        end = start + window_size
        yield window_id, start, flat[start:end]
        window_id += 1


def _kurtosis(x: np.ndarray, mean: float, std: float) -> float:
    if std == 0.0:
        return 0.0
    return float(np.mean(((x - mean) / std) ** 4) - 3.0)


def _skewness(x: np.ndarray, mean: float, std: float) -> float:
    if std == 0.0:
        return 0.0
    return float(np.mean(((x - mean) / std) ** 3))


def compute_window_features(samples: np.ndarray, window_id: int, start_index: int) -> WindowFeatures:
    """Compute time-domain features for one window of vibration samples."""
    x = np.ascontiguousarray(np.asarray(samples).ravel(), dtype=np.float64)
    if x.size == 0:
        raise ValueError("window is empty")

    mean = float(np.mean(x))
    std = float(np.std(x, ddof=0))
    rms = float(np.sqrt(np.mean(x * x)))
    abs_peak = float(np.max(np.abs(x)))
    peak_to_peak = float(np.max(x) - np.min(x))
    crest = float(abs_peak / rms) if rms > 0.0 else 0.0

    return WindowFeatures(
        window_id=window_id,
        start_index=start_index,
        end_index=start_index + x.size,
        vibration_rms=rms,
        vibration_std=std,
        vibration_peak=abs_peak,
        vibration_peak_to_peak=peak_to_peak,
        vibration_kurtosis=_kurtosis(x, mean, std),
        vibration_skewness=_skewness(x, mean, std),
        crest_factor=crest,
    )


def extract_features(
    signal: np.ndarray,
    window_size: int = DEFAULT_WINDOW_SIZE,
    hop: int | None = None,
) -> list[WindowFeatures]:
    """Slice ``signal`` into windows and return features for each."""
    return [
        compute_window_features(samples, wid, start)
        for wid, start, samples in iter_windows(signal, window_size=window_size, hop=hop)
    ]
