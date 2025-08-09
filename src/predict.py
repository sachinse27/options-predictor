import pandas as pd
from .fetch_data import get_ohlcv
from .features import add_features, FEATURES_REQUIRED
from .auto_train import ensure_trained

def predict_symbol(symbol: str, start: str = "2015-01-01", horizon_days: int = 3):
    cls, calib, reg = ensure_trained(symbol, horizon_days=horizon_days, start=start)
    raw = get_ohlcv(symbol, start=start)
    feat = add_features(raw)

    if feat.empty:
        raise ValueError(f"No feature rows available for {symbol}")

    x = feat.iloc[[-1]]  # last row
    if x[FEATURES_REQUIRED].isna().any(axis=None):
        # as a last resort, forward/back fill the full frame then re-take last row
        feat = feat.copy()
        feat[FEATURES_REQUIRED] = feat[FEATURES_REQUIRED].ffill().bfill()
        x = feat.iloc[[-1]]
        if x[FEATURES_REQUIRED].isna().any(axis=None):
            missing = [c for c in FEATURES_REQUIRED if pd.isna(x[c].iloc[0])]
            raise ValueError(f"Latest row has missing features for {symbol}: {missing}")

    p_raw = cls.predict_proba(x[FEATURES_REQUIRED])[:, 1][0]
    p_up  = float(calib.transform([p_raw])[0])
    muhat = float(reg.predict(x[FEATURES_REQUIRED])[0])
    return p_up, muhat
