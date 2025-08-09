import os
from pathlib import Path
import pandas as pd
import numpy as np
from .fetch_data import get_ohlcv
from .features import add_features, FEATURES_REQUIRED
from .auto_train import ensure_trained
from .options_utils import option_ev_over_h

LOG_PATH_PARQUET = Path("data/preds.parquet")
LOG_PATH_CSV = Path("data/preds.csv")

def _ensure_log():
    if LOG_PATH_PARQUET.exists():
        df = pd.read_parquet(LOG_PATH_PARQUET)
    elif LOG_PATH_CSV.exists():
        df = pd.read_csv(LOG_PATH_CSV, parse_dates=["asof_date","resolved_on"])
        df.to_parquet(LOG_PATH_PARQUET, index=False)
        df = pd.read_parquet(LOG_PATH_PARQUET)
    else:
        df = pd.DataFrame(columns=[
            "asof_date","symbol","p_up","pred_exp_ret","horizon_days",
            "spot","resolved_on","actual_ret","correct",
            "opt_symbol","opt_side","opt_ev"
        ])
    return df

def _save_log(df: pd.DataFrame):
    df.to_parquet(LOG_PATH_PARQUET, index=False)
    df.to_csv(LOG_PATH_CSV, index=False)

def make_prediction(symbol: str, horizon_days: int, train_start: str = "2015-01-01"):
    cls, calib, reg = ensure_trained(symbol, horizon_days=horizon_days, start=train_start)
    hist = get_ohlcv(symbol, start=train_start)
    feat = add_features(hist).dropna()
    if len(feat) < 200:
        raise ValueError(f"Not enough history for {symbol}")
    x = feat.iloc[-1:]
    p_raw = cls.predict_proba(x[FEATURES_REQUIRED])[:,1][0]
    p_up = float(calib.transform([p_raw])[0])
    muhat = float(reg.predict(x[FEATURES_REQUIRED])[0])
    spot = float(hist["close"].iloc[-1])

    side = "call" if p_up >= 0.5 else "put"
    opt = option_ev_over_h(symbol, side, mu_hat=muhat, H_days=horizon_days)
    contract = opt.get("contractSymbol") if isinstance(opt, dict) else None
    ev_est = opt.get("EV_estimate") if isinstance(opt, dict) else None

    return {
        "asof_date": pd.to_datetime(hist["date"].iloc[-1]).normalize(),
        "symbol": symbol.upper(),
        "p_up": p_up,
        "pred_exp_ret": muhat,
        "horizon_days": int(horizon_days),
        "spot": spot,
        "resolved_on": pd.NaT,
        "actual_ret": np.nan,
        "correct": pd.NA,
        "opt_symbol": contract,
        "opt_side": side,
        "opt_ev": ev_est,
    }

def log_predictions(symbols, horizon_days: int, train_start: str = "2015-01-01"):
    log = _ensure_log()
    new_rows = []
    for sym in symbols:
        row = make_prediction(sym, horizon_days, train_start=train_start)
        dup = (log["symbol"]==row["symbol"]) & (log["asof_date"]==row["asof_date"])
        if dup.any():
            continue
        new_rows.append(row)
    if new_rows:
        log = pd.concat([log, pd.DataFrame(new_rows)], ignore_index=True)
        _save_log(log)
    return log

def score_predictions(train_start: str = "2015-01-01"):
    log = _ensure_log()
    if log.empty:
        return log
    symbols = log["symbol"].unique().tolist()
    latest_by_sym = {}
    for sym in symbols:
        latest_by_sym[sym] = get_ohlcv(sym, start=train_start).sort_values("date")

    updated = []
    for idx, r in log.iterrows():
        if pd.notna(r["correct"]):
            continue
        sym = r["symbol"]
        H = int(r["horizon_days"])
        hist = latest_by_sym[sym]
        mask = hist["date"] == r["asof_date"]
        if not mask.any():
            after = hist[hist["date"] >= r["asof_date"]]
            if after.empty:
                continue
            start_idx = after.index[0]
        else:
            start_idx = hist[mask].index[0]
        end_idx = start_idx + H
        if end_idx >= len(hist):
            continue
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
