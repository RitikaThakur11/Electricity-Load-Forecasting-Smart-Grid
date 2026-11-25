
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    avg,
    col,
    count,
    dayofweek,
    hour,
    max as spark_max,
    min as spark_min,
    round as spark_round,
)

# Project paths
DATA_PATH = "data/smart_grid_sample.csv"
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Start Spark
spark = (
    SparkSession.builder
    .appName("SmartGridSQLAnalysis")
    .master("local[*]")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")

print("\n========== SMART GRID SPARK SQL ANALYSIS ==========")

try:
    # Load CSV into a Spark DataFrame.
    df = (
        spark.read
        .option("header", True)
        .option("inferSchema", True)
        .csv(DATA_PATH)
    )

    # Convert timestamp text to Spark timestamp type.
    df = df.withColumn(
        "x_Timestamp",
        col("x_Timestamp").cast("timestamp"),
    )

    print("\n========== DATASET SCHEMA ==========")
    df.printSchema()

    print("\nTotal readings:", df.count())
    print("Unique meters:", df.select("meter").distinct().count())

    # Register a temporary SQL table.
    df.createOrReplaceTempView("smart_meter_readings")

    print("\n========== 1. CONSUMPTION SUMMARY ==========")
    spark.sql("""
        SELECT
            COUNT(*) AS total_readings,
            ROUND(AVG(t_kWh), 6) AS average_consumption_kWh,
            ROUND(MAX(t_kWh), 6) AS maximum_consumption_kWh,
            ROUND(MIN(t_kWh), 6) AS minimum_consumption_kWh
        FROM smart_meter_readings
    """).show(truncate=False)

    print("\n========== 2. CONSUMPTION BY METER ==========")
    meter_summary = spark.sql("""
        SELECT
            meter,
            COUNT(*) AS readings,
            ROUND(AVG(t_kWh), 6) AS average_kWh,
            ROUND(MAX(t_kWh), 6) AS peak_kWh
        FROM smart_meter_readings
        GROUP BY meter
        ORDER BY average_kWh DESC
    """)

    meter_summary.show(10, truncate=False)

    meter_summary.toPandas().to_csv(
        OUTPUT_DIR / "spark_meter_summary.csv",
        index=False,
    )

    print("\n========== 3. CONSUMPTION BY HOUR ==========")
    hourly_summary = (
        df.groupBy(hour("x_Timestamp").alias("hour"))
        .agg(
            count("*").alias("readings"),
            spark_round(avg("t_kWh"), 6).alias("average_kWh"),
            spark_round(spark_max("t_kWh"), 6).alias("peak_kWh"),
        )
        .orderBy("hour")
    )

    hourly_summary.show(24, truncate=False)

    hourly_summary.toPandas().to_csv(
        OUTPUT_DIR / "spark_hourly_summary.csv",
        index=False,
    )

    print("\n========== 4. PEAK CONSUMPTION READINGS ==========")
    spark.sql("""
        SELECT
            x_Timestamp,
            meter,
            t_kWh,
            `z_Avg Voltage (Volt)` AS voltage
        FROM smart_meter_readings
        ORDER BY t_kWh DESC
        LIMIT 10
    """).show(truncate=False)

    print("\nSaved reports:")
    print("output/spark_meter_summary.csv")
    print("output/spark_hourly_summary.csv")

    print("\nSpark SQL analysis completed successfully.")

finally:
    spark.stop()
