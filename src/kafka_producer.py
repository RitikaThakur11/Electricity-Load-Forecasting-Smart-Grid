
import csv
import json
import time
from pathlib import Path

from kafka import KafkaProducer
from kafka.errors import KafkaError


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "smart_grid_sample.csv"

KAFKA_SERVER = "localhost:9092"
TOPIC_NAME = "smart-meter-readings"

# Use a short delay for the first test.
DELAY_SECONDS = 0.01


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    producer = KafkaProducer(
        bootstrap_servers=[KAFKA_SERVER],
        value_serializer=lambda value: json.dumps(value).encode("utf-8"),
        acks="all",
        retries=3,
    )

    sent_count = 0

    try:
        print("\n========== SMART GRID KAFKA PRODUCER ==========")
        print(f"Reading dataset: {DATA_PATH}")
        print(f"Kafka topic: {TOPIC_NAME}\n")

        with DATA_PATH.open(
            "r",
            newline="",
            encoding="utf-8-sig"
        ) as csv_file:

            reader = csv.DictReader(csv_file)

            required_columns = {
                "x_Timestamp",
                "t_kWh",
                "z_Avg Voltage (Volt)",
                "z_Avg Current (Amp)",
                "y_Freq (Hz)",
                "meter",
            }

            actual_columns = set(reader.fieldnames or [])
            missing_columns = required_columns - actual_columns

            if missing_columns:
                raise ValueError(
                    f"Missing CSV columns: {sorted(missing_columns)}"
                )

            for row in reader:
                message = {
                    "timestamp": row["x_Timestamp"],
                    "meter": row["meter"],
                    "consumption_kWh": float(row["t_kWh"]),
                    "voltage_V": float(row["z_Avg Voltage (Volt)"]),
                    "current_A": float(row["z_Avg Current (Amp)"]),
                    "frequency_Hz": float(row["y_Freq (Hz)"]),
                }

                producer.send(
                    TOPIC_NAME,
                    value=message
                ).get(timeout=30)

                sent_count += 1

                if sent_count <= 5 or sent_count % 100 == 0:
                    print(
                        f"Sent #{sent_count}: "
                        f"meter={message['meter']}, "
                        f"consumption={message['consumption_kWh']} kWh"
                    )

                time.sleep(DELAY_SECONDS)

        producer.flush()
        print(f"\nSUCCESS: Sent {sent_count} readings to Kafka.")

    except (KafkaError, ValueError, OSError) as error:
        print(f"\nProducer error: {error}")
        raise

    finally:
        producer.close()


if __name__ == "__main__":
    main()
