import pandas as pd
from sklearn.model_selection import KFold
from typing import List

def target_encode_kfold(
        df: pd.DataFrame,
        group_cols: List[str],
        target_col: str,
        k: int = 5,
        agg_func = "median",
        seed: int = 42,
        default_value: float = None) -> pd.Series:
    df = df.copy()
    encoded = pd.Series(index = df.index, dtype="float64")
    kf = KFold(n_splits=k, shuffle=True, random_state=seed)

    for train_idx, val_idx in kf.split(df):
        train_df = df.iloc[train_idx]
        val_df = df.iloc[val_idx]

        agg = train_df.groupby(group_cols)[target_col].agg(agg_func)
        val_encoded = val_df[group_cols].merge(
            agg.reset_index(),
            on=group_cols,
            how="left"
        )[[target_col]].set_index(val_df.index)
    
    if default_value is None:
        default_value = df[target_col].agg(agg_func)
    
    encoded = encoded.fillna(default_value)
    return encoded