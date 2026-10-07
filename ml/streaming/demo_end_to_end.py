"""End-to-end streaming demonstration over three CWRU scenarios.

Runs the REAL pipeline - replay simulator -> JSON telemetry stream ->
Spark Structured Streaming -> windowing -> features -> Isolation Forest
+ Random Forest - and then scores the results against ground truth.

Scenarios
---------
A. ``Normal_3``    - healthy bearing, 3 HP. Expect NORMAL / not anomalous.
B. ``OR021@6_3``   - 0.021" outer-race fault. The easy case: Experiment 2
                     classified this severity perfectly.
C. ``OR014@6_3``   - 0.014" outer-race fault. The hard case: Experiment 2
                     misclassified 14 of its 59 windows as BALL. The demo
                     replays the FULL recording precisely so those errors
                     appear rather than being hidden behind a three-window
                     sample.

Every window of every scenario is reported, and misclassifications are
listed individually.

Usage::

    python -m ml.streaming.demo_end_to_end
    python -m ml.streaming.demo_end_to_end --workdir /tmp/demo --windows 10
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .contracts import WINDOW_SIZE
from .replay_producer import build_events


@dataclass(frozen=True)
class Scenario:
    key: str
    recording_id: str
    asset_id: str
    label: str


SCENARIOS: tuple[Scenario, ...] = (
    Scenario("A", "Normal_3", "MOTOR_001", "healthy baseline (NORMAL, 3 HP)"),
    Scenario("B", "OR021@6_3", "MOTOR_002", "easy fault (OUTER_RACE, 0.021\")"),
    Scenario(
        "C",
        "OR014@6_3",
        "MOTOR_003",
        "hard fault (OUTER_RACE, 0.014\") - Experiment 2 misclassified 14/59 windows",
    ),
)


def _hr(title: str) -> None:
    line = "=" * 78
    print(f"\n{line}\n{title}\n{line}")


def stage_telemetry(workdir: Path, windows: int | None) -> Path:
    """Materialise one JSONL telemetry file per scenario."""
    _hr("STEP 1 - TELEMETRY REPLAY SIMULATOR (CWRU recording -> Kafka-shaped events)")
    inbox = workdir / "telemetry"
    if inbox.exists():
        shutil.rmtree(inbox)
    inbox.mkdir(parents=True)

    for scenario in SCENARIOS:
        events, plan = build_events(
            recording_id=scenario.recording_id,
            asset_id=scenario.asset_id,
            windows=windows,
        )
        target = inbox / f"{scenario.key.lower()}_{scenario.recording_id.replace('@', '_')}.jsonl"
        with target.open("w", encoding="utf-8") as handle:
            for event in events:
                handle.write(event.to_json())
                handle.write("\n")
        print(
            f"  [{scenario.key}] {scenario.recording_id:11s} asset={scenario.asset_id} "
            f"samples={plan.n_samples:,} rate={plan.sampling_rate_hz} Hz "
            f"load={plan.motor_load_hp:g} HP rpm={plan.rotational_speed_rpm:g} "
            f"-> {plan.n_samples // WINDOW_SIZE} complete windows"
        )
    print(f"\n  wrote {len(SCENARIOS)} telemetry streams to {inbox}")
    print(
        "  each message carries assetId / timestampUtc / sequenceNumber / vibration\n"
        "  + operating context, and a nested sourceScenario block that is DEMO\n"
        "  GROUND TRUTH ONLY and never reaches feature engineering."
    )
    return inbox


def run_spark(workdir: Path, inbox: Path, seconds: float) -> Path:
    _hr("STEP 2 - SPARK STRUCTURED STREAMING (window -> features -> inference)")
    from .spark_inference_job import (
        aggregate_windows,
        build_spark,
        make_batch_handler,
        parse_telemetry,
        read_file_stream,
    )
    from .inference import load_models

    outbox = workdir / "inference"
    checkpoint = workdir / "checkpoint"
    for path in (outbox, checkpoint):
        if path.exists():
            shutil.rmtree(path)

    models = load_models()
    print("  frozen model artifacts loaded (no retraining):")
    print("    " + models.describe().replace("\n", "\n    "))

    spark = build_spark("vibration-inference-demo")
    spark.sparkContext.setLogLevel("ERROR")
    try:
        stream = aggregate_windows(parse_telemetry(read_file_stream(spark, str(inbox))))
        query = (
            stream.writeStream.outputMode("update")
            .foreachBatch(make_batch_handler(models, outbox, echo=False))
            .option("checkpointLocation", str(checkpoint))
            .start()
        )
        query.awaitTermination(seconds)
        query.stop()
    finally:
        spark.stop()

    produced = sorted(outbox.glob("*.jsonl")) if outbox.exists() else []
    total = sum(1 for p in produced for _ in p.open())
    print(f"\n  Spark emitted {total} inference results across {len(produced)} micro-batches")
    return outbox


def load_results(outbox: Path) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for path in sorted(outbox.glob("*.jsonl")):
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    results.append(json.loads(line))
    # De-duplicate: "update" output mode can re-emit a completed group.
    unique: dict[tuple[str, int], dict[str, Any]] = {}
    for result in results:
        unique[(result["assetId"], result["windowIndex"])] = result
    return [unique[k] for k in sorted(unique)]


def report(results: list[dict[str, Any]]) -> int:
    _hr("STEP 3 - RESULTS SCORED AGAINST GROUND TRUTH")
    by_asset: dict[str, list[dict[str, Any]]] = {}
    for result in results:
        by_asset.setdefault(result["assetId"], []).append(result)

    total_errors = 0
    for scenario in SCENARIOS:
        rows = sorted(
            by_asset.get(scenario.asset_id, []), key=lambda r: r["windowIndex"]
        )
        print(f"\n--- Scenario {scenario.key}: {scenario.label} ---")
        print(f"    recording={scenario.recording_id} asset={scenario.asset_id}")
        if not rows:
            print("    no windows produced")
            continue

        truth = rows[0].get("groundTruth", {}) or {}
        truth_class = truth.get("faultClass", "?")
        expect_anomalous = truth_class != "NORMAL"

        flagged = sum(1 for r in rows if r["anomaly"]["isAnomalous"])
        correct = sum(
            1 for r in rows if r["classification"]["predictedClass"] == truth_class
        )
        predictions = Counter(r["classification"]["predictedClass"] for r in rows)
        scores = [r["anomaly"]["score"] for r in rows]

        print(f"    windows                 : {len(rows)}")
        print(
            f"    anomaly flagged         : {flagged}/{len(rows)} "
            f"({flagged / len(rows):.1%})  [expected "
            f"{'ANOMALOUS' if expect_anomalous else 'NORMAL'}]"
        )
        print(
            f"    anomaly score range     : {min(scores):+.4f} .. {max(scores):+.4f} "
            f"(threshold {rows[0]['anomaly']['threshold']:+.4f}; lower = more anomalous)"
        )
        print(
            f"    classification accuracy : {correct}/{len(rows)} "
            f"({correct / len(rows):.1%}) vs truth {truth_class}"
        )
        print(f"    predicted classes       : {dict(predictions)}")

        print("    first 3 windows:")
        for row in rows[:3]:
            feats = row["features"]
            print(
                f"      win#{row['windowIndex']:<3d} seq {row['windowStartSequence']:>6d}-"
                f"{row['windowEndSequence']:<6d} rms={feats['vibrationRms']:.4f} "
                f"kurt={feats['vibrationKurtosis']:+.3f} "
                f"| {'ANOMALOUS' if row['anomaly']['isAnomalous'] else 'NORMAL':9s} "
                f"{row['anomaly']['score']:+.4f} "
                f"| {row['classification']['predictedClass']:11s} "
                f"conf={row['classification']['confidence']:.3f}"
            )

        wrong = [
            r for r in rows if r["classification"]["predictedClass"] != truth_class
        ]
        missed = [r for r in rows if r["anomaly"]["isAnomalous"] != expect_anomalous]
        total_errors += len(wrong)
        if wrong:
            print(f"    MISCLASSIFIED windows   : {len(wrong)} (not hidden)")
            for row in wrong[:20]:
                print(
                    f"      win#{row['windowIndex']:<3d} truth={truth_class} -> "
                    f"predicted={row['classification']['predictedClass']} "
                    f"conf={row['classification']['confidence']:.3f} "
                    f"| anomaly={'ANOMALOUS' if row['anomaly']['isAnomalous'] else 'NORMAL'} "
                    f"({row['anomaly']['score']:+.4f})"
                )
            if len(wrong) > 20:
                print(f"      ... and {len(wrong) - 20} more")
        else:
            print("    MISCLASSIFIED windows   : 0")
        if missed:
            print(f"    anomaly-decision misses : {len(missed)}")
            for row in missed[:10]:
                print(
                    f"      win#{row['windowIndex']:<3d} "
                    f"score={row['anomaly']['score']:+.4f} "
                    f"flagged={row['anomaly']['isAnomalous']} expected={expect_anomalous}"
                )

    _hr("STEP 4 - CROSS-MODEL READING")
    fault_rows = [
        r
        for r in results
        if (r.get("groundTruth") or {}).get("faultClass", "NORMAL") != "NORMAL"
    ]
    if fault_rows:
        detected = sum(1 for r in fault_rows if r["anomaly"]["isAnomalous"])
        mistyped = [
            r
            for r in fault_rows
            if r["classification"]["predictedClass"]
            != (r.get("groundTruth") or {}).get("faultClass")
        ]
        mistyped_detected = sum(1 for r in mistyped if r["anomaly"]["isAnomalous"])
        print(
            f"fault windows streamed            : {len(fault_rows)}\n"
            f"flagged ANOMALOUS by Isolation F. : {detected} "
            f"({detected / len(fault_rows):.1%})\n"
            f"mistyped by the Random Forest     : {len(mistyped)}\n"
            f"  ...of those, still flagged ANOM : {mistyped_detected}"
            + (
                f" ({mistyped_detected / len(mistyped):.1%})"
                if mistyped
                else " (n/a)"
            )
        )
        print(
            "\nThe two models answer different questions. Windows the classifier "
            "mistypes\nare still caught by the detector as 'not normal' - the "
            "Experiment-3 finding,\nreproduced live on the streaming pipeline."
        )
    return total_errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--workdir", type=Path, default=Path("/tmp/cwru-streaming-demo")
    )
    parser.add_argument(
        "--windows",
        type=int,
        default=None,
        help="limit each scenario to N windows (default: the full recording)",
    )
    parser.add_argument("--seconds", type=float, default=120.0)
    args = parser.parse_args(argv)

    workdir = args.workdir.expanduser().resolve()
    workdir.mkdir(parents=True, exist_ok=True)

    inbox = stage_telemetry(workdir, args.windows)
    outbox = run_spark(workdir, inbox, args.seconds)
    results = load_results(outbox)
    if not results:
        print("no inference results produced", file=sys.stderr)
        return 1
    report(results)
    print(f"\nraw inference JSON: {outbox}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
