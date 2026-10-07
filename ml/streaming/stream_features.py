"""Streaming-side implementation of the seven vibration features.

This is a DELIBERATELY INDEPENDENT implementation of the feature
contract defined in ``ml/src/feature_extraction.py``. It accumulates
one sample at a time (as a streaming system must) using online central
moments, rather than operating on a materialised numpy array.

Keeping it independent is the point: ``ml/tests/test_streaming_pipeline.py``
asserts that this implementation and the offline training code agree on
real CWRU windows to a documented tolerance. If the two were the same
function the parity test would prove nothing.

Exact definitions reproduced from the training code
---------------------------------------------------
Read off ``ml/src/feature_extraction.py::compute_window_features``:

- ``vibration_rms``          = sqrt(mean(x^2))
- ``vibration_std``          = np.std(x, ddof=0)  -> POPULATION std, not sample
- ``vibration_peak``         = max(|x|)           -> absolute peak, not max(x)
- ``vibration_peak_to_peak`` = max(x) - min(x)
- ``vibration_kurtosis``     = mean(((x-mu)/sigma)^4) - 3  -> EXCESS kurtosis,
                               population (biased) moments, sigma is ddof=0
- ``vibration_skewness``     = mean(((x-mu)/sigma)^3)      -> population (biased)
- ``crest_factor``           = max(|x|) / rms, and exactly 0.0 when rms == 0

Two edge cases are inherited verbatim and must not be "improved":

- when ``sigma == 0`` the training code returns kurtosis 0.0 and
  skewness 0.0 (not NaN, not -3.0);
- when ``rms == 0`` the crest factor is 0.0 (not NaN, not infinity).

A note on scipy: the training pipeline does NOT call
``scipy.stats.kurtosis`` / ``scipy.stats.skew``; it hand-rolls both in
numpy. The hand-rolled definitions happen to coincide with scipy's
defaults (``bias=True``, and ``fisher=True`` for kurtosis), but the
numpy implementation is the normative one and is what is reproduced
here.

Spark equivalence
-----------------
Apache Spark's built-in ``stddev_pop``, ``kurtosis`` and ``skewness``
aggregates use the same population/biased definitions and the same
excess-kurtosis convention, which is why
``ml/streaming/spark_inference_job.py`` can compute the window features
with native Spark SQL aggregates. That equivalence is measured, not
assumed - see the parity test.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence


FEATURE_NAMES: tuple[str, ...] = (
    "vibration_rms",
    "vibration_std",
    "vibration_peak",
    "vibration_peak_to_peak",
    "vibration_kurtosis",
    "vibration_skewness",
    "crest_factor",
)

# camelCase projection used in the JSON inference contract.
FEATURE_JSON_NAMES: dict[str, str] = {
    "vibration_rms": "vibrationRms",
    "vibration_std": "vibrationStd",
    "vibration_peak": "vibrationPeak",
    "vibration_peak_to_peak": "vibrationPeakToPeak",
    "vibration_kurtosis": "vibrationKurtosis",
    "vibration_skewness": "vibrationSkewness",
    "crest_factor": "crestFactor",
}


class OnlineMoments:
    """Numerically stable online central moments up to the 4th order.

    Uses the standard Welford / Pebay incremental updates. This is the
    same family of algorithm Spark's ``kurtosis``/``skewness``
    aggregates use internally, and it avoids the catastrophic
    cancellation that a naive sum-of-powers accumulator suffers when the
    mean is far from zero.
    """

    __slots__ = ("n", "mean", "m2", "m3", "m4", "sum_sq", "minimum", "maximum", "max_abs")

    def __init__(self) -> None:
        self.n = 0
        self.mean = 0.0
        self.m2 = 0.0
        self.m3 = 0.0
        self.m4 = 0.0
        self.sum_sq = 0.0
        self.minimum = math.inf
        self.maximum = -math.inf
        self.max_abs = 0.0

    def add(self, x: float) -> None:
        value = float(x)
        n1 = self.n
        self.n += 1
        n = self.n

        delta = value - self.mean
        delta_n = delta / n
        delta_n2 = delta_n * delta_n
        term = delta * delta_n * n1

        self.mean += delta_n
        self.m4 += (
            term * delta_n2 * (n * n - 3 * n + 3)
            + 6 * delta_n2 * self.m2
            - 4 * delta_n * self.m3
        )
        self.m3 += term * delta_n * (n - 2) - 3 * delta_n * self.m2
        self.m2 += term

        self.sum_sq += value * value
        if value < self.minimum:
            self.minimum = value
        if value > self.maximum:
            self.maximum = value
        abs_value = abs(value)
        if abs_value > self.max_abs:
            self.max_abs = abs_value

    def extend(self, values: Iterable[float]) -> None:
        for value in values:
            self.add(value)


@dataclass(frozen=True)
class StreamingWindowFeatures:
    vibration_rms: float
    vibration_std: float
    vibration_peak: float
    vibration_peak_to_peak: float
    vibration_kurtosis: float
    vibration_skewness: float
    crest_factor: float

    def as_dict(self) -> dict[str, float]:
        return {name: float(getattr(self, name)) for name in FEATURE_NAMES}

    def as_json_dict(self) -> dict[str, float]:
        return {
            FEATURE_JSON_NAMES[name]: float(getattr(self, name))
            for name in FEATURE_NAMES
        }

    def as_ordered_values(self, order: Sequence[str]) -> list[float]:
        """Project to an explicit feature order (e.g. a model's contract)."""
        missing = [name for name in order if not hasattr(self, name)]
        if missing:
            raise KeyError(f"streaming features cannot supply {missing}")
        return [float(getattr(self, name)) for name in order]


def features_from_moments(acc: OnlineMoments) -> StreamingWindowFeatures:
    """Derive the seven features from accumulated moments."""
    if acc.n == 0:
        raise ValueError("cannot compute features from an empty window")

    n = acc.n
    rms = math.sqrt(acc.sum_sq / n)
    # Population variance: m2 / n  (ddof=0), matching np.std(x, ddof=0).
    variance = acc.m2 / n
    std = math.sqrt(variance) if variance > 0.0 else 0.0
    peak = acc.max_abs
    peak_to_peak = acc.maximum - acc.minimum

    if std == 0.0:
        # Training code returns plain 0.0 for both when sigma == 0.
        kurtosis = 0.0
        skewness = 0.0
    else:
        kurtosis = (n * acc.m4) / (acc.m2 * acc.m2) - 3.0
        skewness = math.sqrt(float(n)) * acc.m3 / (acc.m2 ** 1.5)

    crest = (peak / rms) if rms > 0.0 else 0.0

    return StreamingWindowFeatures(
        vibration_rms=rms,
        vibration_std=std,
        vibration_peak=peak,
        vibration_peak_to_peak=peak_to_peak,
        vibration_kurtosis=kurtosis,
        vibration_skewness=skewness,
        crest_factor=crest,
    )


def compute_streaming_features(samples: Iterable[float]) -> StreamingWindowFeatures:
    """Accumulate ``samples`` one at a time and emit the seven features."""
    acc = OnlineMoments()
    acc.extend(samples)
    return features_from_moments(acc)


__all__ = [
    "FEATURE_JSON_NAMES",
    "FEATURE_NAMES",
    "OnlineMoments",
    "StreamingWindowFeatures",
    "compute_streaming_features",
    "features_from_moments",
]
