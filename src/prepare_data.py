
from pathlib import Path
import pandas as pd

DATA_PATH = Path("data/CEEW - Smart meter data Bareilly 2020.csv")
OUTPUT_PATH = Path("data/smart_grid_sample.csv")

TARGET_ROWS = 10000
CHUNK_SIZE = 100000

print("Scanning the original dataset in chunks...")

# Collect a limited number of readings from each meter.
meter_samples = {}
rows_per_meter = 250

for chunk in pd.read_csv(DATA_PATH, chunksize=CHUNK_SIZE):
    for meter_id, group in chunk.groupby("meter", sort=False):
        if meter_id not in meter_samples:
            meter_samples[meter_id] = []

        remaining = rows_per_meter - sum(
            len(part) for part in meter_samples[meter_id]
        )

        if remaining > 0:
            meter_samples[meter_id].append(group.head(remaining))

    # Stop once enough meters have been found.
    if len(meter_samples) >= 40:
        break

if not meter_samples:
    raise ValueError("No meter readings were found.")

# Combine the collected readings.
frames = [
    pd.concat(parts, ignore_index=True)
    for parts in meter_samples.values()
    if parts
]

df = pd.concat(frames, ignore_index=True)

# Keep at most 10,000 readings.
df = df.head(TARGET_ROWS)

# Clean and organize the data.
df["x_Timestamp"] = pd.to_datetime(
    df["x_Timestamp"], errors="coerce"
)

df = df.dropna(subset=["x_Timestamp", "t_kWh", "meter"])
df = df.sort_values(["meter", "x_Timestamp"]).reset_index(drop=True)

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUTPUT_PATH, index=False)

print("\n========== PREPARED DATASET ==========")   
print("Rows saved:", len(df))
print("Unique meters:", df["meter"].nunique())
print("Meters:", sorted(df["meter"].unique()))
print("Date range:", df["x_Timestamp"].min(), "to", df["x_Timestamp"].max())
print("Missing values:")
print(df.isnull().sum().to_string())
print("\nSaved to:", OUTPUT_PATH)
