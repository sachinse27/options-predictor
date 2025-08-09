from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from sklearn.isotonic import IsotonicRegression
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

from .fetch_data import get_ohlcv
from .features import add_features, FEATURES_REQUIRED

MODELS_DIR = Path("data/models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Simple identity calibrator fallback (when isotonic is not reliable)
class IdentityCalibrator:
    def transform(self, x):
        # accept list/array; return numpy array
        return np.asarray(x, dtype=float)

def _paths(symbol: str, H: int):
    s = symbol.upper()
    h = int(H)
    return {
        "cls":  MODELS_DIR / f"model_cls_{s}_H{h}.joblib",
        "cal":  MODELS_DIR / f"calib_cls_{s}_H{h}.joblib",
        "reg":  MODELS_DIR / f"model_reg_{s}_H{h}.joblib",
    }

def trained_exists(symbol: str, H: int) -> bool:
    p = _paths(symbol, H)
    return p["cls"].exists() and p["cal"].exists() and p["reg"].exists()

def _label(df: pd.DataFrame, H: int):
    fwd = df["close"].shift(-H) / df["close"] - 1.0
    y_cls = (fwd > 0).astype(int)
    y_reg = fwd
    return y_cls, y_reg

def train_symbol(symbol: str, horizon_days: int = 3, start: str = "2015-01-01"):
    hist = get_ohlcv(symbol, start=start)
    df = add_features(hist).sort_values("date")

    # labels
    y_cls, y_reg = _label(df, horizon_days)
    df = df.assign(y_cls=y_cls, y_reg=y_reg)
    df = df.dropna(subset=FEATURES_REQUIRED + ["y_cls", "y_reg"]).reset_index(drop=True)

    # Make it train on newer/shorter histories
    if len(df) < 200:
        raise ValueError(f"Not enough history to train {symbol} (need ~300 rows). "
                         "Try again later or reduce feature window sizes.")

    # time-based split with fallback
    cutoff = df["date"].max() - pd.Timedelta(days=365)
    train = df[df["date"] < cutoff]
    val   = df[df["date"] >= cutoff]
    if len(val) < 100:
        cut = int(len(df) * 0.8)
        train, val = df.iloc[:cut], df.iloc[cut:]

    # --- Classifier ---
    cls = HistGradientBoostingClassifier(learning_rate=0.03, max_iter=600, random_state=42)
    cls.fit(train[FEATURES_REQUIRED], train["y_cls"])

    # robust calibration
    p_val = cls.predict_proba(val[FEATURES_REQUIRED])[:, 1] if len(val) else np.array([])
    if len(val) < 50 or val["y_cls"].nunique() < 2 or np.allclose(p_val.std(), 0.0):
        calib = IdentityCalibrator()
    else:
        calib = IsotonicRegression(out_of_bounds="clip").fit(p_val, val["y_cls"])

    # --- Regressor (expected H-day return) ---
    reg = HistGradientBoostingRegressor(learning_rate=0.05, max_iter=800, random_state=42)
    reg.fit(train[FEATURES_REQUIRED], train["y_reg"])

    # save
    paths = _paths(symbol, horizon_days)
    joblib.dump(cls, paths["cls"])
    joblib.dump(calib, paths["cal"])
    joblib.dump(reg, paths["reg"])
    return str(paths["cls"]), str(paths["cal"]), str(paths["reg"])

def ensure_trained(symbol: str, horizon_days: int = 3, start: str = "2015-01-01"):
    paths = _paths(symbol, horizon_days)
    if not trained_exists(symbol, horizon_days):
        train_symbol(symbol, horizon_days=horizon_days, start=start)
    return (
        joblib.load(paths["cls"]),
        joblib.load(paths["cal"]),
        joblib.load(paths["reg"]),
    )
