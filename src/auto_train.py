# src/auto_train.py
import os
from pathlib import Path
import pandas as pd
import joblib
from sklearn.isotonic import IsotonicRegression
from sklearn.ensemble import HistGradientBoostingClassifier

from .fetch_data import get_ohlcv
from .features import add_features, FEATURES

MODELS_DIR = Path("data/models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

def model_paths(symbol: str):
    s = symbol.upper()
    return MODELS_DIR / f"model_{s}.joblib", MODELS_DIR / f"calib_{s}.joblib"

def trained_exists(symbol: str) -> bool:
    m, c = model_paths(symbol)
    return m.exists() and c.exists()

def train_symbol(symbol: str, horizon_days: int = 3, start="2015-01-01"):
    # 1) fetch & feature
    hist = get_ohlcv(symbol, start=start)
    df = add_features(hist)
    df = df.sort_values("date").copy()
    # labels
    fwd = df["close"].shift(-horizon_days) / df["close"] - 1
    df["y_cls"] = (fwd > 0).astype(int)
    df = df.dropna(subset=FEATURES + ["y_cls"])

    if len(df) < 600:
        raise ValueError(f"Not enough history to train a model for {symbol} (need ~600+ rows).")

    # 2) split: last 1y for calibration/validation
    cutoff = df["date"].max() - pd.Timedelta(days=365)
    train = df[df["date"] < cutoff]
    val   = df[df["date"] >= cutoff]
    if len(val) < 100:
        # fall back: 80/20 time split
        cut = int(len(df) * 0.8)
        train, val = df.iloc[:cut], df.iloc[cut:]

    # 3) model
    model = HistGradientBoostingClassifier(
        learning_rate=0.03, max_iter=600, random_state=42
    )
    model.fit(train[FEATURES], train["y_cls"])

    # 4) calibrator
    iso = IsotonicRegression(out_of_bounds="clip")
    p_val = model.predict_proba(val[FEATURES])[:, 1]
    iso.fit(p_val, val["y_cls"])

    # 5) save
    mpath, cpath = model_paths(symbol)
    joblib.dump(model, mpath)
    joblib.dump(iso, cpath)
    return str(mpath), str(cpath)

def ensure_trained(symbol: str, horizon_days: int = 3, start="2015-01-01"):
    """Train if missing; return loaded model + calibrator."""
    mpath, cpath = model_paths(symbol)
    if not trained_exists(symbol):
        train_symbol(symbol, horizon_days=horizon_days, start=start)
    model = joblib.load(mpath)
    calib = joblib.load(cpath)
    return model, calib
