import pandas as pd
import numpy as np
import yfinance as yf
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange

FEATURES = [
    "r1","r3","r5",
    "dist_ema9","dist_ema21","dist_ema50",
    "rsi14",
    "atr_p","rv10","rv20",
    "gap_ret","intra_ret",
    "mkt_r1","mkt_r5","vix","dvix1","regime",
    "dow","month",
    "earn_pre5","earn_post3",  # optional
]

# <- NEW: only these are required for training/prediction
FEATURES_REQUIRED = [
    "r1","r3","r5",
    "dist_ema9","dist_ema21","dist_ema50",
    "rsi14",
    "atr_p","rv10","rv20",
    "gap_ret","intra_ret",
    "mkt_r1","mkt_r5","vix","dvix1","regime",
    "dow","month",
]

def realized_vol(ret: pd.Series, win: int) -> pd.Series:
    return ret.rolling(win).std() * np.sqrt(252)

def _download_series(ticker: str, start: str, col: str = "Adj Close") -> pd.Series:
    df = yf.download(ticker, start=start, progress=False, auto_adjust=True, group_by="column", threads=False)
    if df is None or df.empty:
        return pd.Series(dtype="float64", name="value")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.index.name = "date"
    df = df.rename(columns=str.lower).reset_index()

    # pick a reasonable numeric column
    for c in [col.lower(), "adj close", "close"]:
        if c in df.columns:
            chosen = c
            break
    else:
        numeric = [c for c in df.columns if df[c].dtype.kind in "fc"]
        if not numeric:
            return pd.Series(dtype="float64", name="value")
        chosen = numeric[0]

    s = df.rename(columns={chosen: "value"}).set_index("date")["value"]
    s = s.replace([np.inf, -np.inf], np.nan).dropna().sort_index()
    s.name = "value"
    return s

def _merge_context(df: pd.DataFrame, start: str) -> pd.DataFrame:
    spy = _download_series("SPY", start).pct_change().rename("mkt_r1")
    mkt = spy.to_frame()
    mkt["mkt_r5"] = mkt["mkt_r1"].rolling(5).sum()

    vix_lvl = _download_series("^VIX", start).rename("vix")
    vix = vix_lvl.to_frame()
    vix["dvix1"] = vix["vix"].pct_change()

    out = df.merge(mkt, left_on="date", right_index=True, how="left")
    out = out.merge(vix, left_on="date", right_index=True, how="left")

    # --- fill context safely ---
    for c in ["mkt_r1", "mkt_r5"]:
        if c not in out.columns:
            out[c] = 0.0
        out[c] = out[c].fillna(0.0)

    if "vix" not in out.columns:
        out["vix"] = 0.0
    if not out["vix"].notna().any():
        out["vix"] = 0.0
    out["vix"] = out["vix"].ffill().bfill().fillna(0.0)

    if "dvix1" not in out.columns:
        out["dvix1"] = 0.0
    out["dvix1"] = out["dvix1"].fillna(0.0)

    # regime from available vix
    v = out["vix"]
    if len(v.dropna()) > 50:
        q1, q2 = v.quantile([0.33, 0.66])
        out["regime"] = pd.cut(out["vix"], bins=[-np.inf, q1, q2, np.inf], labels=[0,1,2]).astype("int32")
    else:
        out["regime"] = 1

    return out


def add_features(df: pd.DataFrame, start_for_context: str | None = None) -> pd.DataFrame:
    df = df.sort_values("date").copy()

    ret = df["close"].pct_change()
    df["r1"]  = ret
    df["r3"]  = df["close"].pct_change(3)
    df["r5"]  = df["close"].pct_change(5)

    for w in (9, 21, 50):
        ema = df["close"].ewm(span=w, adjust=False).mean()
        df[f"dist_ema{w}"] = df["close"]/ema - 1

    df["rsi14"] = RSIIndicator(df["close"], 14).rsi()
    atr = AverageTrueRange(df["high"], df["low"], df["close"], 14).average_true_range()
    df["atr_p"] = atr/df["close"]

    df["rv10"] = realized_vol(ret, 10)
    df["rv20"] = realized_vol(ret, 20)

    prev_close = df["close"].shift(1)
    df["gap_ret"]   = (df["open"] - prev_close) / prev_close
    df["intra_ret"] = (df["close"] - df["open"]) / df["open"]

    df["dow"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month

    start_ctx = start_for_context or (df["date"].min().strftime("%Y-%m-%d"))
    df = _merge_context(df, start_ctx)

    # Optional earnings flags from data/earnings.csv
    try:
        earn = pd.read_csv("data/earnings.csv", parse_dates=["date"])
        df["earn_pre5"] = 0
        df["earn_post3"] = 0
        if "ticker" in df.columns:
            sym = df["ticker"].iloc[0]
            dates = earn[earn["symbol"].str.upper() == str(sym).upper()]["date"].dt.normalize().unique()
        else:
            dates = earn["date"].dt.normalize().unique()
        if len(dates):
            on = df["date"].dt.normalize()
            df["earn_pre5"] = on.isin(pd.DatetimeIndex([d - pd.Timedelta(days=i) for d in dates for i in range(1,6)])).astype(int)
            df["earn_post3"] = on.isin(pd.DatetimeIndex([d + pd.Timedelta(days=i) for d in dates for i in range(1,4)])).astype(int)
    except Exception:
        df["earn_pre5"] = np.nan
        df["earn_post3"] = np.nan
    # ---- make features robust: replace inf → NaN, then fill ----
    for c in [
        "r1","r3","r5",
        "dist_ema9","dist_ema21","dist_ema50",
        "rsi14","atr_p","rv10","rv20",
        "gap_ret","intra_ret",
        "mkt_r1","mkt_r5","vix","dvix1","regime",
        "dow","month",
        "earn_pre5","earn_post3",
    ]:
        if c not in df.columns:
            df[c] = np.nan
        df[c] = df[c].replace([np.inf, -np.inf], np.nan)

    # Fill context & technicals so the *latest row* is complete
    df[[
        "mkt_r1","mkt_r5","vix","dvix1","regime",
        "dist_ema9","dist_ema21","dist_ema50",
        "rsi14","atr_p","rv10","rv20",
        "gap_ret","intra_ret",
    ]] = df[[
        "mkt_r1","mkt_r5","vix","dvix1","regime",
        "dist_ema9","dist_ema21","dist_ema50",
        "rsi14","atr_p","rv10","rv20",
        "gap_ret","intra_ret",
    ]].ffill().bfill()

    # Calendar is always present; earnings flags are optional → fill 0
    df["earn_pre5"] = df["earn_pre5"].fillna(0)
    df["earn_post3"] = df["earn_post3"].fillna(0)

    # If regime still NaN (e.g., no VIX), set neutral
    df["regime"] = df["regime"].fillna(1)

    return df
