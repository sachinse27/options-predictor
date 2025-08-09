import pandas as pd
import yaml
from pathlib import Path
import joblib
from sklearn.isotonic import IsotonicRegression
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

from .fetch_data import get_ohlcv
from .features import add_features, FEATURES

def load_cfg(path: str | Path = "config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def build_dataset(cfg) -> pd.DataFrame:
    frames = []
    for sym in cfg["symbols"]:
        raw = get_ohlcv(sym, start=cfg["train_start"])
        feat = add_features(raw)
        feat = feat.sort_values("date")
        fwd = feat["close"].shift(-cfg["horizon_days"]) / feat["close"] - 1.0
        feat["y_cls"] = (fwd > 0).astype(int)
        feat["y_reg"] = fwd
        feat["symbol"] = sym
        frames.append(feat)
    data = pd.concat(frames, ignore_index=True)
    return data

def train_and_save(cfg_path: str = "config.yaml"):
    cfg = load_cfg(cfg_path)
    data = build_dataset(cfg).dropna(subset=FEATURES + ["y_cls","y_reg"])
    cutoff = pd.to_datetime(cfg["train_start"]) + pd.Timedelta(days=365*7)
    train = data[data["date"] < cutoff]
    val   = data[data["date"] >= cutoff]
    if len(val) < 100:
        cut = int(len(data)*0.8)
        train, val = data.iloc[:cut], data.iloc[cut:]
    cls = HistGradientBoostingClassifier(learning_rate=0.03, max_iter=600, random_state=42)
    cls.fit(train[FEATURES], train["y_cls"])
    iso = IsotonicRegression(out_of_bounds="clip")
    p_val = cls.predict_proba(val[FEATURES])[:,1]
    iso.fit(p_val, val["y_cls"])
    reg = HistGradientBoostingRegressor(learning_rate=0.05, max_iter=800, random_state=42)
    reg.fit(train[FEATURES], train["y_reg"])
    Path("data").mkdir(exist_ok=True, parents=True)
    joblib.dump(cls, "data/model_cls_GLOBAL.joblib")
    joblib.dump(iso, "data/calib_cls_GLOBAL.joblib")
    joblib.dump(reg, "data/model_reg_GLOBAL.joblib")
    print("Saved global models in data/.")

if __name__ == "__main__":
    train_and_save()
