"""Spark Structured Streaming: Kafka vibration telemetry -> ML inference.

Pipeline
--------
::

    Kafka industrial.telemetry.vibration   (one message per raw sample)
        -> from_json(VIBRATION_TELEMETRY_SCHEMA)
        -> windowIndex = sequenceNumber div 2048
        -> groupBy(assetId, windowIndex)          [non-overlapping, per asset]
        -> native Spark SQL aggregates            [the 7 vibration features]
        -> HAVING count = 2048                    [drop partial windows]
        -> foreachBatch: Isolation Forest + Random Forest inference
        -> structured inference result

Why windowIndex instead of a time window
----------------------------------------
``sequenceNumber div 2048`` makes the window a property of the data,
not of the clock or of Spark's batching. Windows are therefore exactly
non-overlapping, exactly 2048 samples, isolated per asset, and
identical no matter how micro-batches land - which is what lets the
online features match the offline training pipeline bit for bit.
Event-time tumbling windows would not give that guarantee at 12 kHz
with variable network delay.

Feature parity
--------------
The seven features are computed with *native Spark SQL aggregates*:
``stddev_pop``, ``kurtosis``, ``skewness``, ``max(abs(..))``. Spark's
``kurtosis``/``skewness`` use the same population (biased) moments and
the same excess-kurtosis convention as
``ml/src/feature_extraction.py``, and ``stddev_pop`` matches
``np.std(ddof=0)``. That equivalence is measured in
``ml/tests/test_streaming_pipeline.py`` rather than assumed.

Running
-------
Inside Docker (Kafka source)::

    spark-submit --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.1 \\
        /opt/spark/jobs/vibration_inference_job.py --source kafka

Locally against a JSONL file produced by the replay simulator, which
needs no broker::

    python -m ml.streaming.spark_inference_job --source file \\
        --input /tmp/vibration_stream --output /tmp/inference_out
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from .contracts import (
    VIBRATION_TELEMETRY_SCHEMA,
    VIBRATION_TOPIC,
    WINDOW_SIZE,
)
from .inference import LoadedModels, load_models
from .stream_features import StreamingWindowFeatures


DEFAULT_KAFKA_BOOTSTRAP = "kafka:29092"

# Column order the aggregate stage produces. Kept explicit so the
# inference stage never depends on positional luck.
FEATURE_AGG_COLUMNS: tuple[str, ...] = (
    "vibration_rms",
    "vibration_std",
    "vibration_peak",
    "vibration_peak_to_peak",
    "vibration_kurtosis",
    "vibration_skewness",
    "crest_factor",
)


def build_spark(app_name: str = "vibration-inference") -> SparkSession:
    return (
        SparkSession.builder.appName(app_name)
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.streaming.stopGracefullyOnShutdown", "true")
        .getOrCreate()
    )


def read_kafka_stream(
    spark: SparkSession, bootstrap: str, topic: str, starting_offsets: str
) -> DataFrame:
    return (
        spark.readStream.format("kafka")
        .option("kafka.bootstrap.servers", bootstrap)
        .option("subscribe", topic)
        .option("startingOffsets", starting_offsets)
        .option("failOnDataLoss", "false")
        .load()
        .select(F.col("value").cast("string").alias("raw"))
    )


def read_file_stream(spark: SparkSession, input_dir: str) -> DataFrame:
    """JSONL file source - the broker-free path used by the local demo."""
    return (
        spark.readStream.format("text")
        .option("maxFilesPerTrigger", 1)
        .load(input_dir)
        .select(F.col("value").alias("raw"))
    )


def parse_telemetry(raw: DataFrame) -> DataFrame:
    """Parse the JSON envelope and assign each sample to its window."""
    parsed = raw.select(
        F.from_json(F.col("raw"), VIBRATION_TELEMETRY_SCHEMA).alias("e")
    ).select("e.*")

    return parsed.filter(
        F.col("assetId").isNotNull()
        & F.col("sequenceNumber").isNotNull()
        & F.col("vibration").isNotNull()
    ).withColumn(
        # The whole windowing contract, in one deterministic expression.
        "windowIndex",
        (F.col("sequenceNumber") / F.lit(WINDOW_SIZE)).cast("long"),
    )


def aggregate_windows(events: DataFrame) -> DataFrame:
    """Group to complete 2048-sample windows and compute the 7 features.

    Only ``assetId`` and ``windowIndex`` form the grouping key, so
    samples from different assets can never share a window.
    ``sourceScenario`` is carried through with ``max()`` purely so the
    demo can score results; it takes no part in any feature expression.
    """
    vibration = F.col("vibration")
    aggregated = events.groupBy("assetId", "windowIndex").agg(
        F.count(F.lit(1)).alias("sampleCount"),
        F.sum("sequenceNumber").alias("sequenceChecksum"),
        F.min("sequenceNumber").alias("windowStartSequence"),
        F.max("sequenceNumber").alias("windowEndSequence"),
        F.max("timestampUtc").alias("timestampUtc"),
        # --- the seven vibration features, native Spark aggregates ---
        F.sqrt(F.avg(vibration * vibration)).alias("vibration_rms"),
        F.stddev_pop(vibration).alias("vibration_std"),
        F.max(F.abs(vibration)).alias("vibration_peak"),
        (F.max(vibration) - F.min(vibration)).alias("vibration_peak_to_peak"),
        F.kurtosis(vibration).alias("vibration_kurtosis"),
        F.skewness(vibration).alias("vibration_skewness"),
        # --- operating context (drive-reported; a legitimate input) ---
        F.max("motorLoadHp").alias("motor_load_hp"),
        F.max("rotationalSpeedRpm").alias("rotational_speed_rpm"),
        # --- ground truth, demo scoring only ---
        F.max("sourceScenario.faultClass").alias("groundTruthFaultClass"),
        F.max("sourceScenario.faultSeverityIn").alias("groundTruthSeverityIn"),
        F.max("sourceScenario.recordingId").alias("groundTruthRecordingId"),
    )

    # Partial windows are dropped, exactly as the offline extractor
    # discards a short trailing window.
    #
    # Completeness is checked WITHOUT count(distinct), which Spark
    # rejects inside a streaming aggregation. Instead a window must
    # satisfy all four of:
    #   count == 2048, min == k*2048, max == k*2048+2047, and
    #   sum(sequenceNumber) == the arithmetic series for that range.
    # Together those admit only the exact set {k*2048 .. k*2048+2047}
    # with no duplicate and no gap, so an at-least-once redelivery
    # cannot masquerade as a complete window.
    expected_start = F.col("windowIndex") * F.lit(WINDOW_SIZE)
    expected_checksum = (
        F.lit(WINDOW_SIZE) * expected_start
        + F.lit(WINDOW_SIZE * (WINDOW_SIZE - 1) // 2)
    )
    complete = aggregated.filter(
        (F.col("sampleCount") == F.lit(WINDOW_SIZE))
        & (F.col("windowStartSequence") == expected_start)
        & (
            F.col("windowEndSequence")
            == expected_start + F.lit(WINDOW_SIZE - 1)
        )
        & (F.col("sequenceChecksum") == expected_checksum)
    )

    # crest_factor = peak / rms, defined as exactly 0.0 when rms == 0,
    # matching the training implementation rather than yielding NaN/inf.
    return complete.withColumn(
        "crest_factor",
        F.when(F.col("vibration_rms") > F.lit(0.0),
               F.col("vibration_peak") / F.col("vibration_rms"))
        .otherwise(F.lit(0.0)),
    ).withColumn(
        # Same guard the training code applies when sigma == 0.
        "vibration_kurtosis",
        F.when(F.col("vibration_std") > F.lit(0.0), F.col("vibration_kurtosis"))
        .otherwise(F.lit(0.0)),
    ).withColumn(
        "vibration_skewness",
        F.when(F.col("vibration_std") > F.lit(0.0), F.col("vibration_skewness"))
        .otherwise(F.lit(0.0)),
    )


def score_rows(models: LoadedModels, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Run both frozen models over aggregated window rows.

    Takes plain dicts so it is unit-testable without a Spark session.
    """
    from .inference import score_window

    results: list[dict[str, Any]] = []
    for row in rows:
        features = StreamingWindowFeatures(
            vibration_rms=float(row["vibration_rms"]),
            vibration_std=float(row["vibration_std"]),
            vibration_peak=float(row["vibration_peak"]),
            vibration_peak_to_peak=float(row["vibration_peak_to_peak"]),
            vibration_kurtosis=float(row["vibration_kurtosis"]),
            vibration_skewness=float(row["vibration_skewness"]),
            crest_factor=float(row["crest_factor"]),
        )
        anomaly, classification = score_window(
            models,
            features,
            motor_load_hp=float(row["motor_load_hp"]),
            rotational_speed_rpm=float(row["rotational_speed_rpm"]),
        )
        payload: dict[str, Any] = {
            "assetId": row["assetId"],
            "windowIndex": int(row["windowIndex"]),
            "windowStartSequence": int(row["windowStartSequence"]),
            "windowEndSequence": int(row["windowEndSequence"]),
            "timestampUtc": row.get("timestampUtc"),
            "sampleCount": int(row["sampleCount"]),
            "features": features.as_json_dict(),
            "operatingContext": {
                "motorLoadHp": float(row["motor_load_hp"]),
                "rotationalSpeedRpm": float(row["rotational_speed_rpm"]),
            },
            "anomaly": anomaly.to_dict(),
            "classification": classification.to_dict(),
        }
        # Ground truth is attached only AFTER inference has completed.
        if row.get("groundTruthFaultClass"):
            payload["groundTruth"] = {
                "faultClass": row["groundTruthFaultClass"],
                "faultSeverityIn": row.get("groundTruthSeverityIn"),
                "recordingId": row.get("groundTruthRecordingId"),
            }
        results.append(payload)
    return results


def make_batch_handler(models: LoadedModels, output_dir: Path | None, echo: bool):
    """Build the foreachBatch callback that performs model inference."""

    def handle(batch_df: DataFrame, batch_id: int) -> None:
        rows = [r.asDict(recursive=True) for r in batch_df.collect()]
        if not rows:
            return
        rows.sort(key=lambda r: (r["assetId"], r["windowIndex"]))
        results = score_rows(models, rows)

        if output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)
            target = output_dir / f"batch-{batch_id:06d}.jsonl"
            with target.open("w", encoding="utf-8") as handle_out:
                for result in results:
                    handle_out.write(json.dumps(result, separators=(",", ":")))
                    handle_out.write("\n")

        if echo:
            for result in results:
                gt = result.get("groundTruth") or {}
                print(
                    f"[batch {batch_id}] {result['assetId']} "
                    f"win#{result['windowIndex']} "
                    f"seq {result['windowStartSequence']}-{result['windowEndSequence']} "
                    f"| anomaly={'ANOMALOUS' if result['anomaly']['isAnomalous'] else 'NORMAL':9s} "
                    f"score={result['anomaly']['score']:+.4f} "
                    f"| class={result['classification']['predictedClass']:11s} "
                    f"conf={result['classification']['confidence']:.3f} "
                    f"| truth={gt.get('faultClass', '-')}",
                    flush=True,
                )

    return handle


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", choices=("kafka", "file"), default="kafka")
    parser.add_argument("--bootstrap", default=DEFAULT_KAFKA_BOOTSTRAP)
    parser.add_argument("--topic", default=VIBRATION_TOPIC)
    parser.add_argument("--starting-offsets", default="earliest")
    parser.add_argument("--input", default=None, help="directory for --source file")
    parser.add_argument("--checkpoint", default="/opt/spark/checkpoints/vibration")
    parser.add_argument("--output", default=None, help="directory for result JSONL")
    parser.add_argument("--models-dir", default=None)
    parser.add_argument(
        "--await-seconds",
        type=float,
        default=0.0,
        help="stop after N seconds (0 = run until terminated)",
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    models = load_models(args.models_dir) if args.models_dir else load_models()
    print("[inference] loaded frozen model artifacts:", flush=True)
    print(models.describe(), flush=True)

    spark = build_spark()
    spark.sparkContext.setLogLevel("WARN")

    if args.source == "kafka":
        raw = read_kafka_stream(spark, args.bootstrap, args.topic, args.starting_offsets)
        print(
            f"[inference] reading kafka://{args.bootstrap}/{args.topic}", flush=True
        )
    else:
        if not args.input:
            raise SystemExit("--source file requires --input")
        raw = read_file_stream(spark, args.input)
        print(f"[inference] reading file stream {args.input}", flush=True)

    windows = aggregate_windows(parse_telemetry(raw))

    query = (
        windows.writeStream
        # Complete windows are emitted once; "update" keeps re-emitting a
        # group as more samples arrive, so the terminal filter on
        # distinctSequences == 2048 is what gates a window to exactly one
        # emission per micro-batch in which it completes.
        .outputMode("update")
        .foreachBatch(
            make_batch_handler(
                models,
                Path(args.output) if args.output else None,
                echo=not args.quiet,
            )
        )
        .option("checkpointLocation", args.checkpoint)
        .start()
    )

    if args.await_seconds > 0:
        query.awaitTermination(args.await_seconds)
        query.stop()
    else:
        query.awaitTermination()
    spark.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
