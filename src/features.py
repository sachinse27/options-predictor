import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange

FEATURES = [
    "r1","r3","r5",
    "dist_ema9","dist_ema21","dist_ema50",
    "rsi14","atr_p","rv10","rv20",
    "dow","month"
]

def realized_vol(ret: pd.Series, win: int) -> pd.Series:
    return ret.rolling(win).std() * np.sqrt(252)

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("date").copy()
    ret = df["close"].pct_change()
    df["r1"]  = ret
    df["r3"]  = df["close"].pct_change(3)
    df["r5"]  = df["close"].pct_change(5)
    for w in (9,21,50):
        ema = df["close"].ewm(span=w, adjust=False).mean()
        df[f"dist_ema{w}"] = df["close"]/ema - 1
    df["rsi14"] = RSIIndicator(df["close"],14).rsi()
    atr = AverageTrueRange(df["high"],df["low"],df["close"],14).average_true_range()
    df["atr_p"] = atr/df["close"]
    df["rv10"] = realized_vol(ret,10)
    df["rv20"] = realized_vol(ret,20)
    df["dow"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    return df
