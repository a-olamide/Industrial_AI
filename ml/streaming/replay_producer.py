"""Telemetry replay simulator: a CWRU recording as live industrial telemetry.

Replays the drive-end vibration signal of a CWRU recording sample by
sample onto Kafka (or a file / stdout sink), as if a real accelerometer
on a real asset were streaming it.

The recording's ``.mat`` file is read through the existing, audited
loader (``ml/src/cwru_loader.py``), which is what resolves the
``X{nnn}_DE_time`` variable correctly - including the ``Normal_2`` file
that bundles two experiments. Vendor metadata (motor load, shaft speed,
fault class, defect diameter) comes from
``ml/src/dataset_builder.py::CWRU_RECORDINGS``; none of it is invented
here.

Ground truth handling
---------------------
Fault class, defect diameter and recording id are published inside a
nested ``sourceScenario`` object so a demonstration can be scored. They
are demo metadata, not telemetry. ``--no-ground-truth`` omits them
entirely, which is the honest way to show that the models never needed
them: the pipeline produces identical predictions with or without the
block.

Replay speed
------------
Real CWRU acquisition is 12 kHz (48 kHz for NORMAL recordings), so
real-time replay of a 10-second recording means 120,000 messages.
``--speed`` multiplies the nominal sample rate; ``--speed 0`` (the
default) means "as fast as possible", which is what demonstrations
want. ``--speed 1`` replays at true acquisition rate.

Usage::

    # Write a file the Spark file-source demo can consume
    python -m ml.streaming.replay_producer --recording OR014@6_3 \\
        --asset MOTOR_001 --windows 3 --sink file \\
        --out /tmp/vibration.jsonl

    # Publish to Kafka
    python -m ml.streaming.replay_producer --recording Normal_3 \\
        --asset MOTOR_001 --windows 3 --sink kafka \\
        --bootstrap localhost:9092
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterator

from ..src.cwru_loader import load_recording
from ..src.dataset_builder import RECORDINGS_BY_ID, resolve_recording_paths
from .contracts import (
    VIBRATION_TOPIC,
    WINDOW_SIZE,
    SourceScenario,
    VibrationTelemetryEvent,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "ml" / "data" / "raw" / "cwru"

DEFAULT_ASSET_ID = "MOTOR_001"


@dataclass(frozen=True)
class ReplayPlan:
    recording_id: str
    asset_id: str
    n_samples: int
    sampling_rate_hz: int
    motor_load_hp: float
    rotational_speed_rpm: float
    fault_class: str
    fault_severity_in: float | None


def build_events(
    recording_id: str,
    asset_id: str = DEFAULT_ASSET_ID,
    windows: int | None = None,
    max_samples: int | None = None,
    include_ground_truth: bool = True,
    raw_dir: Path = RAW_DIR,
    start_time: datetime | None = None,
) -> tuple[list[VibrationTelemetryEvent], ReplayPlan]:
    """Load a recording and materialise its telemetry events."""
    spec = RECORDINGS_BY_ID.get(recording_id)
    if spec is None:
        raise SystemExit(
            f"unknown recording {recording_id!r}. Known ids: "
            f"{sorted(RECORDINGS_BY_ID)[:8]} ... ({len(RECORDINGS_BY_ID)} total)"
        )
    found, missing = resolve_recording_paths(raw_dir, [spec])
    if missing:
        raise SystemExit(
            f"{recording_id}: no .mat file found in {raw_dir}. Expected one of "
            f"{list(spec.candidate_filenames)}."
        )

    recording = load_recording(
        found[0].path, expected_experiment_number=spec.experiment_number
    )
    signal = recording.drive_end_signal

    limit = len(signal)
    if windows is not None:
        limit = min(limit, int(windows) * WINDOW_SIZE)
    if max_samples is not None:
        limit = min(limit, int(max_samples))
    signal = signal[:limit]

    rpm = float(recording.rpm) if recording.rpm is not None else float(spec.approx_rpm)
    scenario = (
        SourceScenario(
            faultClass=spec.fault_class,
            faultSeverityIn=spec.fault_severity_in,
            recordingId=spec.recording_id,
        )
        if include_ground_truth
        else None
    )

    base = start_time or datetime.now(timezone.utc)
    period = timedelta(seconds=1.0 / float(spec.sampling_rate_hz))
    events = [
        VibrationTelemetryEvent(
            assetId=asset_id,
            timestampUtc=(base + period * i).isoformat(),
            sequenceNumber=i,
            vibration=float(value),
            motorLoadHp=float(spec.motor_load_hp),
            rotationalSpeedRpm=rpm,
            sourceScenario=scenario,
        )
        for i, value in enumerate(signal)
    ]
    plan = ReplayPlan(
        recording_id=spec.recording_id,
        asset_id=asset_id,
        n_samples=len(events),
        sampling_rate_hz=int(spec.sampling_rate_hz),
        motor_load_hp=float(spec.motor_load_hp),
        rotational_speed_rpm=rpm,
        fault_class=spec.fault_class,
        fault_severity_in=spec.fault_severity_in,
    )
    return events, plan


def _paced(
    events: list[VibrationTelemetryEvent], speed: float, sampling_rate_hz: int
) -> Iterator[VibrationTelemetryEvent]:
    """Yield events, optionally throttled to ``speed`` x real acquisition rate."""
    if speed <= 0.0:
        yield from events
        return
    interval = 1.0 / (float(sampling_rate_hz) * float(speed))
    next_at = time.perf_counter()
    for event in events:
        now = time.perf_counter()
        if next_at > now:
            time.sleep(next_at - now)
        yield event
        next_at += interval


def emit_to_file(
    events: list[VibrationTelemetryEvent], out_path: Path, speed: float, rate: int
) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with out_path.open("w", encoding="utf-8") as handle:
        for event in _paced(events, speed, rate):
            handle.write(event.to_json())
            handle.write("\n")
            count += 1
    return count


def emit_to_stdout(
    events: list[VibrationTelemetryEvent], speed: float, rate: int, preview: int
) -> int:
    count = 0
    for event in _paced(events, speed, rate):
        if preview <= 0 or count < preview:
            print(event.to_json())
        count += 1
    if preview > 0 and count > preview:
        print(f"... ({count - preview} further events suppressed by --preview)")
    return count


def emit_to_kafka(
    events: list[VibrationTelemetryEvent],
    bootstrap: str,
    topic: str,
    speed: float,
    rate: int,
) -> int:
    try:
        from kafka import KafkaProducer  # type: ignore
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise SystemExit(
            "the kafka sink needs a Kafka client: pip install kafka-python-ng"
        ) from exc

    producer = KafkaProducer(
        bootstrap_servers=bootstrap.split(","),
        # Keyed by assetId so every sample of one asset lands on one
        # partition and per-asset ordering is preserved end to end.
        key_serializer=lambda k: k.encode("utf-8"),
        value_serializer=lambda v: v.encode("utf-8"),
        linger_ms=20,
        acks="all",
    )
    count = 0
    try:
        for event in _paced(events, speed, rate):
            producer.send(topic, key=event.assetId, value=event.to_json())
            count += 1
        producer.flush()
    finally:
        producer.close()
    return count


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Replay a CWRU recording as streaming vibration telemetry."
    )
    parser.add_argument("--recording", required=True, help="e.g. Normal_3, OR014@6_3")
    parser.add_argument("--asset", default=DEFAULT_ASSET_ID)
    parser.add_argument(
        "--windows",
        type=int,
        default=None,
        help=f"emit only the first N complete {WINDOW_SIZE}-sample windows",
    )
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--speed",
        type=float,
        default=0.0,
        help="replay rate multiplier; 0 = as fast as possible (default), "
        "1 = true acquisition rate",
    )
    parser.add_argument(
        "--sink", choices=("kafka", "file", "stdout"), default="stdout"
    )
    parser.add_argument("--out", type=Path, default=None, help="path for --sink file")
    parser.add_argument("--bootstrap", default="localhost:9092")
    parser.add_argument("--topic", default=VIBRATION_TOPIC)
    parser.add_argument("--preview", type=int, default=3, help="stdout sink only")
    parser.add_argument(
        "--no-ground-truth",
        action="store_true",
        help="omit the sourceScenario block entirely",
    )
    args = parser.parse_args(argv)

    events, plan = build_events(
        recording_id=args.recording,
        asset_id=args.asset,
        windows=args.windows,
        max_samples=args.max_samples,
        include_ground_truth=not args.no_ground_truth,
    )

    print(
        f"[replay] recording={plan.recording_id} asset={plan.asset_id} "
        f"samples={plan.n_samples:,} rate={plan.sampling_rate_hz} Hz "
        f"load={plan.motor_load_hp:g} HP rpm={plan.rotational_speed_rpm:g}",
        file=sys.stderr,
    )
    print(
        f"[replay] complete windows available: {plan.n_samples // WINDOW_SIZE} "
        f"(ground truth {'omitted' if args.no_ground_truth else 'attached as sourceScenario'})",
        file=sys.stderr,
    )

    started = time.perf_counter()
    if args.sink == "file":
        if args.out is None:
            raise SystemExit("--sink file requires --out")
        sent = emit_to_file(events, args.out, args.speed, plan.sampling_rate_hz)
        target = str(args.out)
    elif args.sink == "kafka":
        sent = emit_to_kafka(
            events, args.bootstrap, args.topic, args.speed, plan.sampling_rate_hz
        )
        target = f"kafka://{args.bootstrap}/{args.topic}"
    else:
        sent = emit_to_stdout(events, args.speed, plan.sampling_rate_hz, args.preview)
        target = "stdout"

    elapsed = time.perf_counter() - started
    print(
        f"[replay] emitted {sent:,} events to {target} in {elapsed:.2f}s "
        f"({sent / elapsed:,.0f} msg/s)",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
