"""Per-asset, non-overlapping 2048-sample window assembly.

Window assignment is a pure function of the sample's own sequence
number::

    windowIndex = sequenceNumber // WINDOW_SIZE

That choice is deliberate and does most of the correctness work for
free:

- **Non-overlapping by construction.** Window *k* owns sequence numbers
  ``[k*2048, k*2048+2047]`` and nothing else. No sliding, no shared
  samples between consecutive windows - matching the offline
  ``iter_windows(..., hop=None)`` behaviour used during training.
- **Asset isolation.** State is keyed by ``(assetId, windowIndex)``, so
  a sample from ``MOTOR_002`` can never land in a ``MOTOR_001`` window
  even when both arrive in the same Kafka partition or micro-batch.
- **Micro-batch independent.** Spark may split or merge batches, retry,
  or deliver out of order. Because the window a sample belongs to is
  carried by the sample itself, the resulting windows are identical
  regardless. This is what makes the streaming features reproducible
  against the offline pipeline.
- **Partial windows are dropped.** A window is emitted only once it has
  exactly ``WINDOW_SIZE`` distinct samples, mirroring the offline
  extractor, which discards a short trailing window.

Duplicate deliveries (Kafka at-least-once) are absorbed because samples
are stored per sequence number rather than appended to a list.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator

from .contracts import WINDOW_SIZE, SourceScenario, VibrationTelemetryEvent
from .stream_features import (
    OnlineMoments,
    StreamingWindowFeatures,
    features_from_moments,
)


@dataclass(frozen=True)
class AssembledWindow:
    """A complete 2048-sample window, ready for feature use and inference."""

    assetId: str
    windowIndex: int
    windowStartSequence: int
    windowEndSequence: int
    timestampUtc: str
    sampleCount: int
    samples: tuple[float, ...]
    motorLoadHp: float
    rotationalSpeedRpm: float
    # Demo-only. Carried beside the window, never fed to the models.
    sourceScenario: SourceScenario | None = None

    def features(self) -> StreamingWindowFeatures:
        acc = OnlineMoments()
        acc.extend(self.samples)
        return features_from_moments(acc)


class WindowAssembler:
    """Accumulates telemetry events into complete per-asset windows.

    Used directly by the Spark job's ``foreachBatch`` fallback path and
    by the offline demo/tests. The Spark SQL path in
    ``spark_inference_job.py`` expresses the same rule declaratively as
    ``groupBy(assetId, sequenceNumber div WINDOW_SIZE)``.
    """

    def __init__(self, window_size: int = WINDOW_SIZE) -> None:
        if window_size <= 0:
            raise ValueError("window_size must be positive")
        self.window_size = window_size
        # (assetId, windowIndex) -> {sequenceNumber: event}
        self._pending: dict[tuple[str, int], dict[int, VibrationTelemetryEvent]] = {}
        self._emitted: set[tuple[str, int]] = set()

    def add(self, event: VibrationTelemetryEvent) -> AssembledWindow | None:
        """Add one sample; return a window when this sample completes one."""
        window_index = int(event.sequenceNumber) // self.window_size
        key = (event.assetId, window_index)
        if key in self._emitted:
            # Late/duplicate delivery for a window already emitted.
            return None

        bucket = self._pending.setdefault(key, {})
        bucket[int(event.sequenceNumber)] = event

        if len(bucket) < self.window_size:
            return None

        window = self._materialise(event.assetId, window_index, bucket)
        del self._pending[key]
        self._emitted.add(key)
        return window

    def add_all(
        self, events: Iterable[VibrationTelemetryEvent]
    ) -> Iterator[AssembledWindow]:
        for event in events:
            window = self.add(event)
            if window is not None:
                yield window

    def _materialise(
        self,
        asset_id: str,
        window_index: int,
        bucket: dict[int, VibrationTelemetryEvent],
    ) -> AssembledWindow:
        # Sequence ordering is restored here, so an out-of-order Kafka
        # delivery produces the same window as an in-order one.
        ordered = [bucket[seq] for seq in sorted(bucket)]
        expected_start = window_index * self.window_size
        expected = list(range(expected_start, expected_start + self.window_size))
        actual = [int(e.sequenceNumber) for e in ordered]
        if actual != expected:
            raise ValueError(
                f"window ({asset_id}, {window_index}) has non-contiguous sequence "
                f"numbers: expected {expected[0]}..{expected[-1]}, got "
                f"{actual[0]}..{actual[-1]} ({len(actual)} samples)"
            )

        last = ordered[-1]
        return AssembledWindow(
            assetId=asset_id,
            windowIndex=window_index,
            windowStartSequence=actual[0],
            windowEndSequence=actual[-1],
            timestampUtc=last.timestampUtc,
            sampleCount=len(ordered),
            samples=tuple(float(e.vibration) for e in ordered),
            # Operating point is constant within a CWRU recording; the
            # window's own last sample is the authoritative reading.
            motorLoadHp=float(last.motorLoadHp),
            rotationalSpeedRpm=float(last.rotationalSpeedRpm),
            sourceScenario=last.sourceScenario,
        )

    @property
    def pending_window_count(self) -> int:
        return len(self._pending)

    @property
    def completed_window_count(self) -> int:
        return len(self._emitted)


__all__ = ["AssembledWindow", "WindowAssembler"]
