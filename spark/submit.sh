#!/bin/bash
# Spark job submission script — sourced inside bitnami/spark:3.5 container.
# Packages are pulled from Maven Central on first run, then cached in /root/.ivy2.
set -e

# Load bitnami's env (sets JAVA_HOME, SPARK_HOME, PATH)
. /opt/bitnami/scripts/spark/setenv.sh

echo "[submit.sh] Spark home : $SPARK_HOME"
echo "[submit.sh] Java home  : $JAVA_HOME"
echo "[submit.sh] Starting streaming job..."

exec spark-submit \
  --master local[2] \
  --packages \
    "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0,com.microsoft.sqlserver:mssql-jdbc:12.8.1.jre11" \
  --conf "spark.sql.shuffle.partitions=4" \
  --conf "spark.streaming.stopGracefullyOnShutdown=true" \
  --conf "spark.ui.port=4040" \
  --conf "spark.ui.host=0.0.0.0" \
  /opt/spark/jobs/industrial_streaming_analytics.py
