#!/bin/bash
set -e

echo "[submit.sh] Preparing Spark Ivy cache..."

mkdir -p /home/spark/.ivy2/cache
mkdir -p /home/spark/.ivy2/jars

echo "[submit.sh] Starting Spark streaming job..."

exec /opt/spark/bin/spark-submit \
  --master local[2] \
  --packages "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.1,com.microsoft.sqlserver:mssql-jdbc:12.8.1.jre11" \
  --conf "spark.jars.ivy=/home/spark/.ivy2" \
  --conf "spark.sql.shuffle.partitions=4" \
  --conf "spark.streaming.stopGracefullyOnShutdown=true" \
  --conf "spark.ui.port=4040" \
  --conf "spark.ui.host=0.0.0.0" \
  /opt/spark/jobs/industrial_streaming_analytics.py