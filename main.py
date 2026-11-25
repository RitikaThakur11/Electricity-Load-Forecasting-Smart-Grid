
from pathlib import Path
import pandas as pd

DATA_PATH = Path("data/CEEW - Smart meter data Bareilly 2020.csv")

print("Loading the first 10,000 readings...")

df = pd.read_csv(DATA_PATH, nrows=10000)

print("\n========== DATASET OVERVIEW ==========")
print("Rows loaded:", len(df))
print("Columns:", len(df.columns))

print("\n========== COLUMN NAMES ==========")
for column in df.columns:
    print(repr(column))

print("\n========== FIRST FIVE ROWS ==========")
print(df.head().to_string())

print("\n========== DATA TYPES ==========")
print(df.dtypes.to_string())

print("\n========== MISSING VALUES ==========")
print(df.isnull().sum().to_string())

print("\n========== DATASET INSPECTION COMPLETE ==========")
