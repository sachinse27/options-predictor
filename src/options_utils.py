# src/options_utils.py
import math
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timezone

# ---- Black-Scholes Greeks (european) ----
def _norm_cdf(x):  # standard normal CDF
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def _norm_pdf(x):
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)

def _d1(S, K, r, sigma, T):
    return (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))

def _d2(d1, sigma, T):
    return d1 - sigma * math.sqrt(T)

def greeks_call(S, K, r, sigma, T):
    d1 = _d1(S, K, r, sigma, T)
    d2 = _d2(d1, sigma, T)
    delta = _norm_cdf(d1)
    gamma = _norm_pdf(d1) / (S * sigma * math.sqrt(T))
    vega  = S * _norm_pdf(d1) * math.sqrt(T) / 100.0  # per 1 vol point (i.e., 1.00 = 100%)
    # theta per day:
    theta = (-(S * _norm_pdf(d1) * sigma) / (2 * math.sqrt(T)) - r * K * math.exp(-r*T) * _norm_cdf(d2)) / 365.0
    return delta, gamma, vega, theta

def greeks_put(S, K, r, sigma, T):
    d1 = _d1(S, K, r, sigma, T)
    d2 = _d2(d1, sigma, T)
    delta = _norm_cdf(d1) - 1.0
    gamma = _norm_pdf(d1) / (S * sigma * math.sqrt(T))
    vega  = S * _norm_pdf(d1) * math.sqrt(T) / 100.0
    theta = (-(S * _norm_pdf(d1) * sigma) / (2 * math.sqrt(T)) + r * K * math.exp(-r*T) * _norm_cdf(-d2)) / 365.0
    return delta, gamma, vega, theta

# ---- Chain fetch & filter ----
def _nearest_expiries(symbol):
    tkr = yf.Ticker(symbol)
    exps = tkr.options or []
    out = []
    for e in exps:
        try:
            dt = datetime.strptime(e, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            out.append(dt)
        except Exception:
            pass
    return sorted(out)

def _dte(exp):
    # Coerce to UTC-aware
    if isinstance(exp, pd.Timestamp):
        exp = exp.tz_localize("UTC") if exp.tzinfo is None else exp.tz_convert("UTC")
    elif exp.tzinfo is None:
        exp = exp.replace(tzinfo=timezone.utc)

    now = datetime.now(timezone.utc)
    return max(1, (exp - now).days)

def select_contract(symbol: str, side: str,
                    target_delta=0.30, dte_min=20, dte_max=45,
                    min_oi=500, max_spread_pct=0.02):
    tkr = yf.Ticker(symbol)
    expiries = _nearest_expiries(symbol)
    expiries = [e for e in expiries if dte_min <= _dte(e) <= dte_max]
    if not expiries:
        # fallback widen DTE window
        expiries = _nearest_expiries(symbol)
        expiries = [e for e in expiries if 7 <= _dte(e) <= 60]
        if not expiries:
            return None

    S = float(tkr.history(period="1d")["Close"].iloc[-1])

    def _try(exp_list, oi, spread):
        best = None
        best_score = -1e9
        best_meta = None
        for exp_dt in exp_list:
            exp_str = exp_dt.strftime("%Y-%m-%d")
            try:
                chain = tkr.option_chain(exp_str)
            except Exception:
                continue
            tbl = chain.calls if side == "call" else chain.puts
            if tbl is None or tbl.empty:
                continue

            df = tbl.copy()
            for col in ["bid","ask","impliedVolatility","openInterest","strike"]:
                if col not in df.columns:
                    continue

            df = df.dropna(subset=["bid","ask","openInterest","strike"])
            if df.empty:
                continue

            # sanitize IV; fill missing with median
            if "impliedVolatility" in df.columns:
                if df["impliedVolatility"].isna().any():
                    med_iv = df["impliedVolatility"].median()
                    df["impliedVolatility"] = df["impliedVolatility"].fillna(med_iv if pd.notna(med_iv) else 0.5)
            else:
                df["impliedVolatility"] = 0.5  # rough fallback

            df["mid"] = (df["bid"] + df["ask"]) / 2.0
            df = df[(df["openInterest"] >= oi) & (df["bid"] > 0)]
            if df.empty:
                continue

            df["spread_pct"] = (df["ask"] - df["bid"]) / df["mid"]
            df = df[df["spread_pct"] <= spread]
            if df.empty:
                continue

            # approximate delta via BS
            r = 0.00
            T = _dte(exp_dt) / 365.0
            deltas = []
            for _, row in df.iterrows():
                sigma = max(1e-6, float(row["impliedVolatility"]))
                K = float(row["strike"])
                if side == "call":
                    dlt, _, _, _ = greeks_call(S, K, r, sigma, T)
                    dlt_abs = abs(dlt)
                else:
                    dlt, _, _, _ = greeks_put(S, K, r, sigma, T)
                    dlt_abs = abs(dlt)
                deltas.append(dlt_abs)
            df["abs_delta"] = deltas
            df["delta_diff"] = (df["abs_delta"] - target_delta).abs()

            # rank: closest to target delta, then tightest spread, then highest OI
            df = df.sort_values(["delta_diff","spread_pct","openInterest"], ascending=[True, True, False])
            cand = df.iloc[0]
            score = -cand["delta_diff"] - 5.0 * cand["spread_pct"] + 0.0001 * cand["mid"] + 0.00001 * cand["openInterest"]
            if score > best_score:
                best = cand
                best_score = score
                best_meta = {"exp": exp_dt, "oi": oi, "spread": spread}
        if best is None:
            return None, None
        best = best.copy()
        best["S"] = S
        best["dte"] = _dte(best_meta["exp"])
        best["_meta"] = best_meta
        return best, best_meta

    # Attempt ladder: strict → relaxed
    attempts = [
        (expiries, min_oi, max_spread_pct),
        (expiries, 300, 0.05),
        (expiries, 100, 0.10),
        # widen expiries if still nothing
        ([e for e in _nearest_expiries(symbol) if 7 <= _dte(e) <= 90], 100, 0.15),
    ]
    for exps, oi, spr in attempts:
        best, meta = _try(exps, oi, spr)
        if best is not None:
            return best

    return None

# ---- EV over H days using μ̂ (expected return) ----
def option_ev_over_h(symbol: str, side: str, mu_hat: float, H_days: int,
                     target_delta=0.30, dte_min=20, dte_max=45,
                     min_oi=500, max_spread_pct=0.02):
    row = select_contract(symbol, side,
                          target_delta=target_delta, dte_min=dte_min, dte_max=dte_max,
                          min_oi=min_oi, max_spread_pct=max_spread_pct)
    if row is None:
        return {"error": "No suitable contract found (try widening DTE, lowering OI, or allowing larger spread)."}
    """
    Returns dict with contract info + EV estimate over H days.
    """
    side = side.lower()
    if side not in ("call","put"):
        raise ValueError("side must be 'call' or 'put'")

    row = select_contract(symbol, side)
    if row is None:
        return {"error": "No suitable contract found"}

    S = float(row["S"])
    K = float(row["strike"])
    r = 0.00
    sigma = float(row["impliedVolatility"])  # already in decimal from yfinance (e.g., 0.35)

    #T = max(1e-6, (_dte(_nearest_expiries(symbol)[0]) / 365.0))  # rough T: first eligible expiry
    expiries = _nearest_expiries(symbol)
    if expiries:
        T = _dte(expiries[0]) / 365.0
    else:
        T = 30 / 365.0
    mid = float(row["mid"])
    bid = float(row["bid"])
    ask = float(row["ask"])

    # Greeks today
    if side == "call":
        delta, gamma, vega, theta = greeks_call(S, K, r, sigma, T)
    else:
        delta, gamma, vega, theta = greeks_put(S, K, r, sigma, T)

    # Expectation over H days:
    dS = mu_hat * S               # expected price change over H days
    time_theta = theta * H_days   # per-day theta * H
    # second-order price move approximation (ignore vol change expectation)
    ev_move = delta * dS + 0.5 * gamma * (dS ** 2)
    ev_option = ev_move - time_theta

    # friction: assume you pay the mid + half-spread and exit at mid
    half_spread = 0.5 * (ask - bid)
    ev_net = ev_option - half_spread

    return {
        "contractSymbol": row.get("contractSymbol", ""),
        "side": side,
        "strike": K,
        "dte": row.get("dte", None),
        "iv": sigma,
        "bid": bid,
        "ask": ask,
        "mid": mid,
        "delta_abs": float(row.get("abs_delta", np.nan)),
        "spread_pct": float(row.get("spread_pct", np.nan)),
        "EV_estimate": float(ev_net),
        "EV_components": {
            "delta_dS": float(delta * dS),
            "gamma_term": float(0.5 * gamma * (dS ** 2)),
            "theta_H": float(time_theta),
            "half_spread": float(half_spread),
        }
    }
