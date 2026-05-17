#!/bin/bash
set -e

echo "[submit.sh] Preparing Spark Ivy cache..."

mkdir -p /home/spark/.ivy2/cache
mkdir -p /home/spark/.ivy2/jars

echo "[submit.sh] Clearing stale Derby metastore locks and syncing Hive warehouse..."
rm -f /opt/spark/metastore/derby-metastore/db.lck
rm -f /opt/spark/metastore/derby-metastore/dbex.lck
# Hive warehouse and metastore must stay in sync — wipe the managed db dir
# so saveAsTable creates fresh tables rather than hitting LOCATION_ALREADY_EXISTS
rm -rf /opt/spark/hive-warehouse/industrial.db

echo "[submit.sh] Starting Spark streaming job..."

exec /opt/spark/bin/spark-submit \
  --master local[2] \
  --packages "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.1,com.microsoft.sqlserver:mssql-jdbc:12.8.1.jre11" \
  --conf "spark.jars.ivy=/home/spark/.ivy2" \
  --conf "spark.sql.shuffle.partitions=4" \
  --conf "spark.streaming.stopGracefullyOnShutdown=true" \
  --conf "spark.ui.port=4040" \
  --conf "spark.ui.host=0.0.0.0" \
  --conf "spark.sql.warehouse.dir=/opt/spark/hive-warehouse" \
  --conf "spark.hadoop.javax.jdo.option.ConnectionURL=jdbc:derby:/opt/spark/metastore/derby-metastore;create=true" \
  --conf "spark.hadoop.javax.jdo.option.ConnectionDriverName=org.apache.derby.jdbc.EmbeddedDriver" \
  --conf "spark.hadoop.datanucleus.autoCreateSchema=true" \
  --conf "spark.hadoop.hive.metastore.schema.verification=false" \
  /opt/spark/jobs/industrial_streaming_analytics.py
