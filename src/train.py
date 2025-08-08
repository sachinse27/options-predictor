import pandas as pd
import yaml
from pathlib import Path
#from lightgbm import LGBMClassifier

#from xgboost import XGBClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
import joblib

from .fetch_data import get_ohlcv
from .features import add_features, FEATURES
from .labeling import make_label

def load_cfg(path: str | Path = "config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def build_dataset(cfg) -> pd.DataFrame:
    frames = []
    for sym in cfg["symbols"]:
        raw = get_ohlcv(sym, start=cfg["train_start"])
        feat = add_features(raw)
        lab  = make_label(feat, cfg["horizon_days"])
        lab["symbol"] = sym
        frames.append(lab)
    data = pd.concat(frames, ignore_index=True)
    return data

def train_and_save(cfg_path: str = "config.yaml"):
    cfg = load_cfg(cfg_path)
    data = build_dataset(cfg)

    cutoff = pd.to_datetime(cfg["test_start"])
    train = data[data["date"] < cutoff].dropna(subset=FEATURES + ["y_cls"]).copy()
    val   = data[(data["date"] >= cutoff) & (data["y_cls"].notna())].dropna(subset=FEATURES + ["y_cls"]).copy()

    model = HistGradientBoostingClassifier(
        max_depth=None,
        learning_rate=0.03,
        max_iter=600,
        l2_regularization=0.0,
        validation_fraction=None,
        random_state=42
    )
    model.fit(train[FEATURES], train["y_cls"])

    iso = IsotonicRegression(out_of_bounds="clip")
    val_probs = model.predict_proba(val[FEATURES])[:,1]
    iso.fit(val_probs, val["y_cls"])

    Path("data").mkdir(exist_ok=True, parents=True)
    joblib.dump(model, "data/model.joblib")
    joblib.dump(iso, "data/calibrator.joblib")
    print("Saved model and calibrator.")

if __name__ == "__main__":
    train_and_save()
