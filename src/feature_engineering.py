
from pathlib import Path
import pandas as pd

INPUT_PATH = Path("data/smart_grid_sample.csv")
OUTPUT_PATH = Path("data/smart_grid_features.csv")

print("Loading prepared smart meter data...")

df = pd.read_csv(INPUT_PATH)
df["x_Timestamp"] = pd.to_datetime(df["x_Timestamp"])

# Sort readings chronologically within each meter.
df = df.sort_values(["meter", "x_Timestamp"]).reset_index(drop=True)

# Time-based features.
df["hour"] = df["x_Timestamp"].dt.hour
df["day"] = df["x_Timestamp"].dt.day
df["day_of_week"] = df["x_Timestamp"].dt.dayofweek
df["month"] = df["x_Timestamp"].dt.month
df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

# Previous consumption readings for the same meter.
df["previous_load"] = df.groupby("meter")["t_kWh"].shift(1)
df["load_2_steps_ago"] = df.groupby("meter")["t_kWh"].shift(2)
df["load_3_steps_ago"] = df.groupby("meter")["t_kWh"].shift(3)

# Remove rows without enough historical readings.
df = df.dropna(
    subset=["previous_load", "load_2_steps_ago", "load_3_steps_ago"]
).copy()

# Save the engineered dataset.
df.to_csv(OUTPUT_PATH, index=False)

print("\n========== FEATURE ENGINEERING RESULTS ==========")
print("Rows:", len(df))
print("Meters:", df["meter"].nunique())
print("Columns:", len(df.columns))
print("\nFeature columns:")
print(df.columns.tolist())
print("\nSample rows:")
print(df.head().to_string(index=False))
print("\nSaved to:", OUTPUT_PATH)
