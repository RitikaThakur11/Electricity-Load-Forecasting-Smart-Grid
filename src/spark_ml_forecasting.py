
from pathlib import Path

from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator


# --------------------------------------------------
# 1. Project paths
# --------------------------------------------------

DATA_PATH = "data/smart_grid_features.csv"
OUTPUT_DIR = Path("output")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. Start Spark
# --------------------------------------------------

spark = (
    SparkSession.builder
    .appName("SmartGridPySparkML")
    .master("local[*]")
    .config("spark.sql.session.timeZone", "UTC")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")


try:
    print("\n========== PYSPARK ML FORECASTING ==========")

    # --------------------------------------------------
    # 3. Load the feature-engineered dataset
    # --------------------------------------------------

    df = (
        spark.read
        .option("header", True)
        .option("inferSchema", True)
        .csv(DATA_PATH)
    )

    df = df.withColumn(
        "x_Timestamp",
        F.to_timestamp("x_Timestamp")
    )

    numeric_features = [
        "z_Avg Voltage (Volt)",
        "z_Avg Current (Amp)",
        "y_Freq (Hz)",
        "hour",
        "day",
        "day_of_week",
        "month",
        "is_weekend",
        "previous_load",
        "load_2_steps_ago",
        "load_3_steps_ago",
    ]

    # Remove records with missing required values
    df = df.dropna(
        subset=[
            "x_Timestamp",
            "meter",
            "t_kWh",
        ] + numeric_features
    )

    # --------------------------------------------------
    # 4. Split chronologically within each meter
    # --------------------------------------------------

    # Each meter's earlier readings go into training.
    # Its later readings go into testing.

    window = (
        Window
        .partitionBy("meter")
        .orderBy("x_Timestamp")
    )

    meter_window = Window.partitionBy("meter")

    df = (
        df
        .withColumn("_row", F.row_number().over(window))
        .withColumn("_total", F.count("*").over(meter_window))
        .withColumn("_split", F.floor(F.col("_total") * 0.8))
    )

    train_df = (
        df
        .filter(F.col("_row") <= F.col("_split"))
        .drop("_row", "_total", "_split")
    )

    test_df = (
        df
        .filter(F.col("_row") > F.col("_split"))
        .drop("_row", "_total", "_split")
    )

    print("Training rows:", train_df.count())
    print("Testing rows:", test_df.count())
    print("Number of meters:", df.select("meter").distinct().count())

    # --------------------------------------------------
    # 5. Prepare categorical and numerical features
    # --------------------------------------------------

    indexer = StringIndexer(
        inputCol="meter",
        outputCol="meter_index",
        handleInvalid="keep"
    )

    encoder = OneHotEncoder(
        inputCols=["meter_index"],
        outputCols=["meter_vector"],
        handleInvalid="keep"
    )

    assembler = VectorAssembler(
        inputCols=numeric_features + ["meter_vector"],
        outputCol="features",
        handleInvalid="keep"
    )

    # --------------------------------------------------
    # 6. Build the Random Forest model
    # --------------------------------------------------

    forest = RandomForestRegressor(
        featuresCol="features",
        labelCol="t_kWh",
        predictionCol="prediction",
        numTrees=100,
        maxDepth=8,
        seed=42
    )

    pipeline = Pipeline(
        stages=[
            indexer,
            encoder,
            assembler,
            forest
        ]
    )

    # --------------------------------------------------
    # 7. Train the model
    # --------------------------------------------------

    print("\nTraining Spark ML Random Forest model...")

    model = pipeline.fit(train_df)

    print("Model training completed.")

    # --------------------------------------------------
    # 8. Generate predictions
    # --------------------------------------------------

    predictions = model.transform(test_df).cache()

    # --------------------------------------------------
    # 9. Evaluate the model
    # --------------------------------------------------

    print("\n========== MODEL EVALUATION ==========")

    for metric in ["mae", "rmse", "r2"]:
        evaluator = RegressionEvaluator(
            labelCol="t_kWh",
            predictionCol="prediction",
            metricName=metric
        )

        score = evaluator.evaluate(predictions)

        print(f"{metric.upper()}: {score:.6f}")

    # --------------------------------------------------
    # 10. Save predictions
    # --------------------------------------------------

    # Convert the timestamp to a string in Spark before
    # converting the results to pandas. This avoids the
    # pandas timestamp timezone-conversion error.

    results = (
        predictions
        .select(
            F.date_format(
                "x_Timestamp",
                "yyyy-MM-dd HH:mm:ss"
            ).alias("x_Timestamp"),
            "meter",
            "t_kWh",
            "prediction"
        )
        .orderBy("meter", "x_Timestamp")
        .toPandas()
    )

    results = results.rename(
        columns={
            "t_kWh": "actual_kWh",
            "prediction": "predicted_kWh"
        }
    )

    predictions_path = OUTPUT_DIR / "spark_ml_predictions.csv"

    results.to_csv(
        predictions_path,
        index=False
    )

    # --------------------------------------------------
    # 11. Save the trained Spark ML pipeline
    # --------------------------------------------------

    model_path = OUTPUT_DIR / "spark_ml_model"

    model.write().overwrite().save(str(model_path))

    # --------------------------------------------------
    # 12. Display completion details
    # --------------------------------------------------

    print("\n========== SAMPLE PREDICTIONS ==========")

    print(results.head(10).to_string(index=False))

    print("\nSaved predictions:", predictions_path)
    print("Saved trained model:", model_path)
    print("\nPySpark ML forecasting completed successfully!")

    predictions.unpersist()

finally:
    spark.stop()
