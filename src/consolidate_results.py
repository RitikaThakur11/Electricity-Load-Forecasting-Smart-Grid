
from pathlib import Path
import glob

import pandas as pd
import matplotlib.pyplot as plt

# Project paths
PROJECT_DIR = Path(__file__).resolve().parent.parent
INPUT_DIR = PROJECT_DIR / "output" / "kafka_meter_summary"
OUTPUT_DIR = PROJECT_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# Find all Spark CSV part files
files = glob.glob(str(INPUT_DIR / "*.csv"))

if not files:
    raise FileNotFoundError(
        f"No CSV files found in {INPUT_DIR}"
    )

# Read and combine the CSV files
frames = []

for file in files:
    df = pd.read_csv(file)

    if not df.empty:
        frames.append(df)

if not frames:
    raise ValueError("All CSV files are empty.")

results = pd.concat(frames, ignore_index=True)

# Convert numeric columns
numeric_columns = [
    "batch_id",
    "reading_count",
    "average_consumption_kWh",
    "total_consumption_kWh",
    "peak_consumption_kWh",
    "average_voltage_V",
    "average_current_A",
    "average_frequency_Hz",
]

for column in numeric_columns:
    results[column] = pd.to_numeric(
        results[column], errors="coerce"
    )

results = results.dropna(
    subset=["meter", "batch_id"]
)

results["batch_id"] = results["batch_id"].astype(int)

# Save one consolidated CSV
consolidated_file = OUTPUT_DIR / "kafka_meter_summary_all.csv"
results.to_csv(consolidated_file, index=False)

print(f"Part files processed: {len(files)}")
print(f"Summary rows: {len(results)}")
print(f"Unique meters: {results['meter'].nunique()}")
print(f"Unique batches: {results['batch_id'].nunique()}")
print(f"Consolidated CSV saved to: {consolidated_file}")

# Aggregate across batches for each meter.
# This is an aggregate of the saved batch summaries.
meter_summary = (
    results.groupby("meter", as_index=False)
    .agg(
        total_recorded_consumption_kWh=(
            "total_consumption_kWh", "sum"
        ),
        average_batch_consumption_kWh=(
            "average_consumption_kWh", "mean"
        ),
        peak_recorded_reading_kWh=(
            "peak_consumption_kWh", "max"
        ),
    )
    .sort_values(
        "total_recorded_consumption_kWh",
        ascending=False,
    )
)

meter_summary.to_csv(
    OUTPUT_DIR / "kafka_meter_totals.csv",
    index=False,
)

# Chart 1: Total recorded consumption by meter
plt.figure(figsize=(12, 6))
plt.bar(
    meter_summary["meter"],
    meter_summary["total_recorded_consumption_kWh"],
)
plt.title("Recorded Electricity Consumption by Meter")
plt.xlabel("Smart Meter")
plt.ylabel("Summed Batch Consumption (kWh)")
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "kafka_consumption_by_meter.png",
    dpi=150,
)
plt.close()

# Chart 2: Average consumption by batch
batch_summary = (
    results.groupby("batch_id", as_index=False)
    .agg(
        average_consumption_kWh=(
            "average_consumption_kWh", "mean"
        )
    )
    .sort_values("batch_id")
)

plt.figure(figsize=(10, 5))
plt.plot(
    batch_summary["batch_id"],
    batch_summary["average_consumption_kWh"],
    marker="o",
)
plt.title("Average Meter Consumption Across Kafka Batches")
plt.xlabel("Batch ID")
plt.ylabel("Mean of Meter Average Consumption (kWh)")
plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "kafka_batch_consumption.png",
    dpi=150,
)
plt.close()

print("\nCharts saved:")
print("output/kafka_consumption_by_meter.png")
print("output/kafka_batch_consumption.png")
print("\nTop 10 meters by summed batch consumption:")
print(meter_summary.head(10).to_string(index=False))
