import os
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from .fetch_data import get_ohlcv
from .features import add_features, FEATURES

LOG_PATH_PARQUET = Path("data/preds.parquet")
LOG_PATH_CSV = Path("data/preds.csv")

def _load_model():
    model = joblib.load("data/model.joblib")
    calib = joblib.load("data/calibrator.joblib")
    return model, calib

def _ensure_log():
    if LOG_PATH_PARQUET.exists():
        df = pd.read_parquet(LOG_PATH_PARQUET)
    elif LOG_PATH_CSV.exists():
        df = pd.read_csv(LOG_PATH_CSV, parse_dates=["asof_date"])
        df.to_parquet(LOG_PATH_PARQUET, index=False)
        df = pd.read_parquet(LOG_PATH_PARQUET)
    else:
        df = pd.DataFrame(columns=[
            "asof_date","symbol","p_up","pred_exp_ret","horizon_days",
            "spot","resolved_on","actual_ret","correct"
        ])
    return df

def _save_log(df: pd.DataFrame):
    df.to_parquet(LOG_PATH_PARQUET, index=False)
    df.to_csv(LOG_PATH_CSV, index=False)

def _expected_move_estimate(p_up: float, hist: pd.DataFrame, H: int) -> float:
    """Optional: expected H-day return ≈ p*avg_up - (1-p)*avg_down (magnitudes from history)."""
    tmp = hist.copy()
    tmp = tmp.sort_values("date")
    fwd = tmp["close"].shift(-H)/tmp["close"] - 1
    up = fwd[fwd > 0].dropna()
    dn = (-fwd[fwd <= 0]).dropna()
    if len(up) < 10 or len(dn) < 10:
        # fallback: use rolling realized vol as proxy
        vol = tmp["close"].pct_change().rolling(20).std().iloc[-1] or 0.01
        avg_up = vol * np.sqrt(H) * 0.8
        avg_dn = vol * np.sqrt(H) * 0.8
    else:
        avg_up = up.mean()
        avg_dn = dn.mean()
    return float(p_up*avg_up - (1.0 - p_up)*avg_dn)

def make_prediction(symbol: str, horizon_days: int, train_start: str = "2015-01-01"):
    model, calib = _load_model()
    hist = get_ohlcv(symbol, start=train_start)
    feat = add_features(hist).dropna()
    if len(feat) < 200:
        raise ValueError(f"Not enough history for {symbol}")
    x = feat.iloc[-1:]
    p_raw = model.predict_proba(x[FEATURES])[:,1][0]
    p_up = float(calib.transform([p_raw])[0])
    spot = float(hist["close"].iloc[-1])
    pred_mu = _expected_move_estimate(p_up, hist, horizon_days)
    return {
        "asof_date": pd.to_datetime(hist["date"].iloc[-1]).normalize(),
        "symbol": symbol.upper(),
        "p_up": p_up,
        "pred_exp_ret": pred_mu,  # can be positive or negative
        "horizon_days": int(horizon_days),
        "spot": spot,
        "resolved_on": pd.NaT,
        "actual_ret": np.nan,
        "correct": pd.NA
    }

def log_predictions(symbols, horizon_days: int, train_start: str = "2015-01-01"):
    log = _ensure_log()
    new_rows = []
    for sym in symbols:
        row = make_prediction(sym, horizon_days, train_start=train_start)
        # Avoid duplicate same-day entries per symbol
        dup = (log["symbol"]==row["symbol"]) & (log["asof_date"]==row["asof_date"])
        if dup.any():
            continue
        new_rows.append(row)
    if new_rows:
        log = pd.concat([log, pd.DataFrame(new_rows)], ignore_index=True)
        _save_log(log)
    return log

def score_predictions(train_start: str = "2015-01-01"):
    """Fill in actual_ret and correct once enough days have passed."""
    log = _ensure_log()
    if log.empty:
        return log
    # group by symbol, pull future price
    symbols = log["symbol"].unique().tolist()
    latest_by_sym = {}

    # pull full history once per symbol
    for sym in symbols:
        latest_by_sym[sym] = get_ohlcv(sym, start=train_start).sort_values("date")

    updated = []
    for idx, r in log.iterrows():
        if pd.notna(r["correct"]):  # already scored
            continue
        sym = r["symbol"]
        H = int(r["horizon_days"])
        hist = latest_by_sym[sym]
        # find asof row
        mask = hist["date"] == r["asof_date"]
        if not mask.any():
            # trading holiday or mismatch: align to next available trading date
            after = hist[hist["date"] >= r["asof_date"]]
            if after.empty:
                continue
            start_idx = after.index[0]
        else:
            start_idx = hist[mask].index[0]
        end_idx = start_idx + H
        if end_idx >= len(hist):
            continue  # not enough future yet
        s0 = float(hist.loc[start_idx, "close"])
        sH = float(hist.loc[end_idx, "close"])
        actual_ret = sH/s0 - 1.0
        correct = int((actual_ret > 0) == (r["p_up"] >= 0.5))
        r["resolved_on"] = pd.to_datetime(hist.loc[end_idx, "date"]).normalize()
        r["actual_ret"] = actual_ret
        r["correct"] = correct
        updated.append((idx, r))

    if updated:
        for idx, r in updated:
            log.loc[idx] = r
        _save_log(log)
    return log
