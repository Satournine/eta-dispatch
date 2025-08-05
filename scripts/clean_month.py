print("▶️ Script started...")

from src.data.load_and_clean import (
    load_raw_data,
    clean_eta_data,
    save_clean_data,
    RAW_DATA_PATH,
    PROCESSED_DATA_PATH,
)
print("▶️ Imports successful ✅")

if __name__ == "__name__":
    df_raw = load_raw_data(RAW_DATA_PATH)
    df_clean = clean_eta_data(df_raw)
    save_clean_data(df_clean, PROCESSED_DATA_PATH)