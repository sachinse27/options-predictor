import pandas as pd
from .fetch_data import get_ohlcv
from .features import add_features, FEATURES
from .auto_train import ensure_trained

def predict_symbol(symbol: str, start: str = "2015-01-01", horizon_days: int = 3) -> float:
    model, calib = ensure_trained(symbol, horizon_days=horizon_days, start=start)
    raw = get_ohlcv(symbol, start=start)
    feat = add_features(raw).dropna()
    if len(feat) < 200:
        raise ValueError("Not enough history for this symbol.")
    x = feat.iloc[-1:]
    p_raw = model.predict_proba(x[FEATURES])[:,1][0]
    p = float(calib.transform([p_raw])[0])
    return p
