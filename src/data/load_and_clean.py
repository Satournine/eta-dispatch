import pandas as pd
from pathlib import Path

RAW_DATA_PATH = Path("data/raw/yellow_tripdata_2025-06.parquet")
PROCESSED_DATA_PATH = Path("data/processed/yellow_tripdata_2025-06-cleaned.parquet")


def load_raw_data(path: Path) -> pd.DataFrame:
    print("▶️ Loading raw data from:", path)
    df = pd.read_parquet(path)
    print("Loaded Shape: ", df.shape)
    print("Columns: ", df.columns.tolist())
    return df

def clean_eta_data(df: pd.DataFrame) -> pd.DataFrame:
    df["tpep_pickup_datetime"] = pd.to_datetime(df["tpep_pickup_datetime"])
    df["tpep_dropoff_datetime"] = pd.to_datetime(df["tpep_dropoff_datetime"])
    df["eta_sec"] = (df["tpep_dropoff_datetime"] - df["tpep_pickup_datetime"]).dt.total_seconds()

    df = df[
        (df["eta_sec"] >= 120) &
        (df["eta_sec"] <= 7200) &
        (df["trip_distance"] >= 0.2)
    ]
    km = df["trip_distance"] * 1.60934
    h = df["eta_sec"] / 3600.0
    speed_kmh = km / h
    df = df[(speed_kmh >= 3) & (speed_kmh <= 120)]
    keep = [
        "tpep_pickup_datetime","tpep_dropoff_datetime",
        "PULocationID","DOLocationID","trip_distance","eta_sec"
    ]
    df = df[keep]

    return df

def save_clean_data(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    print(f"Saved cleaned data to {path}")

if __name__ == "__main__":
    df_raw = load_raw_data(RAW_DATA_PATH)
    df_clean = clean_eta_data(df_raw)
    save_clean_data(df_clean, PROCESSED_DATA_PATH)