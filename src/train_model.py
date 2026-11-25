
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DATA_PATH = Path("data/smart_grid_features.csv")
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("Loading feature-engineered data...")
df = pd.read_csv(DATA_PATH)
df["x_Timestamp"] = pd.to_datetime(df["x_Timestamp"])

# Input features and target.
feature_columns = [
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

target_column = "t_kWh"

# Use a chronological split to avoid training on future readings.
df = df.sort_values(["meter", "x_Timestamp"]).reset_index(drop=True)
split_index = int(len(df) * 0.8)

train_df = df.iloc[:split_index]
test_df = df.iloc[split_index:]

X_train = train_df[feature_columns]
y_train = train_df[target_column]
X_test = test_df[feature_columns]
y_test = test_df[target_column]

print("Training rows:", len(train_df))
print("Testing rows:", len(test_df))

model = RandomForestRegressor(
    n_estimators=100,
    max_depth=12,
    random_state=42,
    n_jobs=-1,
)

print("\nTraining Random Forest...")
model.fit(X_train, y_train)

predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
rmse = mean_squared_error(y_test, predictions) ** 0.5
r2 = r2_score(y_test, predictions)

print("\n========== MODEL EVALUATION ==========")
print(f"MAE:  {mae:.6f} kWh")
print(f"RMSE: {rmse:.6f} kWh")
print(f"R²:   {r2:.4f}")

# Save actual and predicted values for analysis.
results = test_df[["x_Timestamp", "meter", target_column]].copy()
results["predicted_kWh"] = predictions
results.to_csv(OUTPUT_DIR / "prediction_results.csv", index=False)

# Plot actual versus predicted values.
plot_data = results.head(200)

plt.figure(figsize=(12, 5))
plt.plot(plot_data[target_column].to_numpy(), label="Actual consumption")
plt.plot(plot_data["predicted_kWh"].to_numpy(), label="Predicted consumption")
plt.title("Actual vs Predicted Electricity Consumption")
plt.xlabel("Test reading")
plt.ylabel("Consumption (kWh)")
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "actual_vs_predicted.png", dpi=150)
plt.close()

print("\nSaved prediction results to output/prediction_results.csv")
print("Saved chart to output/actual_vs_predicted.png")
