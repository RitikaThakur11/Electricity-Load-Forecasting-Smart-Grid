
from pathlib import Path

import pyspark
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType,
    StructField,
    StringType,
    DoubleType,
)

# --------------------------------------------------
# 1. CONFIGURATION
# --------------------------------------------------

KAFKA_SERVER = "localhost:9092"
TOPIC_NAME = "smart-meter-readings"

OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Use separate folders for this new streaming query.
SUMMARY_DIR = (OUTPUT_DIR / "kafka_meter_summary").resolve()
CHECKPOINT_DIR = str(
    (OUTPUT_DIR / "kafka_summary_checkpoint").resolve()
)

SPARK_VERSION = pyspark.__version__
KAFKA_CONNECTOR = (
    f"org.apache.spark:spark-sql-kafka-0-10_2.12:{SPARK_VERSION}"
)

# --------------------------------------------------
# 2. DEFINE THE KAFKA MESSAGE SCHEMA
# --------------------------------------------------

schema = StructType([
    StructField("timestamp", StringType(), True),
    StructField("meter", StringType(), True),
    StructField("consumption_kWh", DoubleType(), True),
    StructField("voltage_V", DoubleType(), True),
    StructField("current_A", DoubleType(), True),
    StructField("frequency_Hz", DoubleType(), True),
])

# --------------------------------------------------
# 3. CREATE SPARK SESSION
# --------------------------------------------------

spark = None
query = None

try:
    spark = (
        SparkSession.builder
        .appName("SmartGridConsumptionSummary")
        .master("local[*]")
        .config("spark.jars.packages", KAFKA_CONNECTOR)
        .config("spark.sql.session.timeZone", "UTC")
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("WARN")

    print("\n========================================")
    print(" SMART GRID REAL-TIME CONSUMPTION ANALYSIS")
    print("========================================")
    print(f"PySpark version: {pyspark.__version__}")
    print(f"Running Spark version: {spark.version}")
    print(f"Kafka topic: {TOPIC_NAME}")
    print(f"CSV output directory: {SUMMARY_DIR}")
    print("Press Ctrl+C to stop.\n")

    # --------------------------------------------------
    # 4. READ KAFKA STREAM
    # --------------------------------------------------

    kafka_stream = (
        spark.readStream
        .format("kafka")
        .option("kafka.bootstrap.servers", KAFKA_SERVER)
        .option("subscribe", TOPIC_NAME)
        .option("startingOffsets", "earliest")
        .option("maxOffsetsPerTrigger", 1000)
        .load()
    )

    # --------------------------------------------------
    # 5. CONVERT JSON INTO COLUMNS
    # --------------------------------------------------

    readings = (
        kafka_stream
        .select(
            F.from_json(
                F.col("value").cast("string"),
                schema
            ).alias("reading")
        )
        .select("reading.*")
        .filter(
            F.col("meter").isNotNull()
            & F.col("consumption_kWh").isNotNull()
        )
    )

    # --------------------------------------------------
    # 6. PROCESS EACH MICRO-BATCH
    # --------------------------------------------------

    def process_batch(batch_df, batch_id):

        if batch_df.isEmpty():
            print(f"Batch {batch_id}: no readings.")
            return

        summary = (
            batch_df
            .groupBy("meter")
            .agg(
                F.count("*").alias("reading_count"),
                F.round(
                    F.avg("consumption_kWh"), 6
                ).alias("average_consumption_kWh"),
                F.round(
                    F.sum("consumption_kWh"), 6
                ).alias("total_consumption_kWh"),
                F.round(
                    F.max("consumption_kWh"), 6
                ).alias("peak_consumption_kWh"),
                F.round(
                    F.avg("voltage_V"), 2
                ).alias("average_voltage_V"),
                F.round(
                    F.avg("current_A"), 3
                ).alias("average_current_A"),
                F.round(
                    F.avg("frequency_Hz"), 3
                ).alias("average_frequency_Hz"),
            )
            .withColumn("batch_id", F.lit(batch_id))
            .select(
                "batch_id",
                "meter",
                "reading_count",
                "average_consumption_kWh",
                "total_consumption_kWh",
                "peak_consumption_kWh",
                "average_voltage_V",
                "average_current_A",
                "average_frequency_Hz",
            )
        )

        print(f"\n========== BATCH {batch_id} SUMMARY ==========")

        summary.orderBy(
            F.desc("total_consumption_kWh")
        ).show(20, truncate=False)

        # Save the summary as CSV files.
        # Spark writes part files inside this directory.
        (
            summary.write
            .mode("append")
            .option("header", "true")
            .csv(str(SUMMARY_DIR))
        )

        print(f"Batch {batch_id} summary saved.")

    # --------------------------------------------------
    # 7. START STRUCTURED STREAMING
    # --------------------------------------------------

    query = (
        readings.writeStream
        .foreachBatch(process_batch)
        .option("checkpointLocation", CHECKPOINT_DIR)
        .trigger(processingTime="5 seconds")
        .start()
    )

    print("Streaming started successfully!")
    print("Calculating meter summaries and saving CSV output...\n")

    query.awaitTermination()

except KeyboardInterrupt:
    print("\nStopping streaming application...")

except Exception as error:
    print(f"\nStreaming error: {error}")
    raise

finally:
    if query is not None:
        try:
            query.stop()
        except Exception:
            pass

    if spark is not None:
        try:
            spark.stop()
        except Exception:
            pass

    print("Spark application stopped.")
