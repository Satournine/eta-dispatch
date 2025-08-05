import pandas as pd
from pathlib import Path
from src.features.encode_speed import target_encode_kfold

def load_clean_data(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    print("Loaded clean data", df.shape)
    return df

def load_distance_matrix(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    print("Loaded distance matrix: ", df.shape)
    return df

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df["pickup_hour"] = df["tpep_pickup_datetime"].dt.hour
    df["pickup_weekday"] = df["tpep_pickup_datetime"].dt.weekday
    df["is_weekend"] = df["pickup_weekday"] >= 5
    df["pickup_month"] = df["tpep_pickup_datetime"].dt.month
    print("✅ Time-based features added:", df.shape)
    return df

def join_features(trips: pd.DataFrame, distance: pd.DataFrame) -> pd.DataFrame:
    enriched = trips.merge(
        distance,
        on=["PULocationID", "DOLocationID"],
        how="left"
    )
    print("Merged features:", enriched.shape)
    return enriched

def compute_actual_speed(df: pd.DataFrame) -> pd.DataFrame:
    df["actual_speed_kmh"] = df["great_circle_km"] / (df["eta_sec"] / 3600)
    df = df.replace([float("inf"), -float("inf")], pd.NA)
    print("✅ Computed actual_speed_kmh")
    return df

def save_features(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    print(f"Saved features to {path}")


if __name__ == "__main__":
    raw_path = Path("data/processed/yellow_tripdata_2025-06-cleaned.parquet")
    distance_path = Path("data/geo/zone_distance_matrix.parquet")
    output_path = Path("data/processed/features_yellow_tripdata_2025-06.parquet")

    df_trips = load_clean_data(raw_path)
    df_distances = load_distance_matrix(distance_path)
    df_features = join_features(df_trips, df_distances)
    df_features = compute_actual_speed(df_features)
    df_features = add_time_features(df_features)
    group_cols = ["PULocationID", "DOLocationID", "pickup_hour"]
    df_features["historical_speed_kmh"] = target_encode_kfold(
        df=df_features,
        group_cols=group_cols,
        target_col="actual_speed_kmh",
        k=5,
        agg_func="median"
    )
    print("✅ historical_speed_kmh added via leakage-free encoding")

    save_features(df_features, output_path)