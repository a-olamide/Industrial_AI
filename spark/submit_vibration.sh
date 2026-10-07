#!/bin/bash
# Launch the Kafka -> Spark -> ML inference stream.
#
# Separate from submit.sh so the legacy industrial_streaming_analytics
# job keeps its own lifecycle, checkpoints and reset behaviour untouched.
set -e

CHECKPOINT="${VIBRATION_CHECKPOINT:-/opt/spark/checkpoints/vibration}"
BOOTSTRAP="${KAFKA_BOOTSTRAP:-kafka:29092}"
TOPIC="${VIBRATION_TOPIC:-industrial.telemetry.vibration}"
OUTPUT="${INFERENCE_OUTPUT:-/opt/spark/inference-output}"

echo "[submit_vibration] checkpoint=$CHECKPOINT topic=$TOPIC bootstrap=$BOOTSTRAP"

# Stale checkpoints replay old offsets against a changed schema; this job
# is a demonstrator, so it starts clean each boot.
rm -rf "$CHECKPOINT"
mkdir -p "$OUTPUT"

export PYTHONPATH=/opt:${PYTHONPATH}

exec /opt/spark/bin/spark-submit \
  --master "local[2]" \
  --packages "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.1" \
  --conf "spark.jars.ivy=/home/spark/.ivy2" \
  --conf "spark.sql.shuffle.partitions=4" \
  --conf "spark.streaming.stopGracefullyOnShutdown=true" \
  --conf "spark.ui.port=4041" \
  --conf "spark.ui.host=0.0.0.0" \
  /opt/spark/jobs/vibration_inference_job.py \
    --source kafka \
    --bootstrap "$BOOTSTRAP" \
    --topic "$TOPIC" \
    --checkpoint "$CHECKPOINT" \
    --output "$OUTPUT"
