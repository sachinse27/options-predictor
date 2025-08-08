import pandas as pd

def make_label(df: pd.DataFrame, horizon: int = 3) -> pd.DataFrame:
    df = df.sort_values("date").copy()
    fwd = df["close"].shift(-horizon) / df["close"] - 1
    df["y_ret"] = fwd
    df["y_cls"] = (fwd > 0).astype(int)
    return df
