# app.py
import os
from pathlib import Path

import pandas as pd
import yaml
import streamlit as st
import plotly.graph_objects as go

from src.predict import predict_symbol
from src.fetch_data import get_ohlcv
from src.auto_train import ensure_trained          # trains per symbol on demand
from src.track import log_predictions, score_predictions, LOG_PATH_PARQUET

st.set_page_config(page_title="Stock Direction Predictor", layout="wide")

# --- config ---
cfg = yaml.safe_load(open("config.yaml"))
HORIZON = int(cfg.get("horizon_days", 3))
TRAIN_START = cfg.get("train_start", "2015-01-01")

# --- helpers ---
def parse_symbols(text: str) -> list[str]:
    return [s.strip().upper() for s in text.split(",") if s.strip()]

def ensure_list_has_symbols(symbols: list[str]) -> list[str]:
    if symbols:
        return symbols
    # fallback to config defaults
    return [s.upper() for s in cfg.get("symbols", ["SPY", "QQQ", "AAPL"])]

# -------------------- UI --------------------
tab1, tab2 = st.tabs(["🔮 Predict (one-off)", "📈 Tracker (log & chart)"])

# ========== TAB 1: PREDICT ==========
with tab1:
    st.title("Stock Direction Predictor")
    st.caption(f"Predicts probability the price will be UP after {HORIZON} trading days. Educational demo, not financial advice.")

    symbol = st.text_input("Symbol (any ticker)", value="SPY").upper()

    if st.button("Predict"):
        try:
            with st.spinner(f"Training/Loading model for {symbol}…"):
                ensure_trained(symbol, horizon_days=HORIZON, start=TRAIN_START)
                p = predict_symbol(symbol, start=TRAIN_START, horizon_days=HORIZON)
            st.metric(f"Probability of Up Move ({HORIZON} days)", f"{p:.1%}")
        except Exception as e:
            st.error(str(e))

# ========== TAB 2: TRACKER ==========
with tab2:
    st.title("Prediction Tracker")
    st.caption("Log today's predictions for a set of symbols; come back later to score and visualize them on the price chart.")

    default_syms = ",".join([s.upper() for s in cfg.get("symbols", ["SPY","QQQ","AAPL"])])
    symbols_text = st.text_input("Symbols to track (comma-separated)", value=default_syms)
    symbols = ensure_list_has_symbols(parse_symbols(symbols_text))

    colA, colB, colC = st.columns([1,1,2])
    with colA:
        if st.button("Log today's predictions"):
            try:
                with st.spinner("Training/loading models & logging…"):
                    for s in symbols:
                        ensure_trained(s, horizon_days=HORIZON, start=TRAIN_START)
                    log_predictions(symbols, HORIZON, train_start=TRAIN_START)
                st.success(f"Logged: {', '.join(symbols)}")
            except Exception as e:
                st.error(str(e))

    with colB:
        if st.button("Update scores (resolve past predictions)"):
            try:
                with st.spinner("Scoring…"):
                    score_predictions(train_start=TRAIN_START)
                st.success("Scores updated.")
            except Exception as e:
                st.error(str(e))

    st.markdown("---")
    st.subheader("Logged Predictions")
    if Path(LOG_PATH_PARQUET).exists():
        log = pd.read_parquet(LOG_PATH_PARQUET)
        st.dataframe(
            log.sort_values(["symbol","asof_date"], ascending=[True, False]),
            use_container_width=True
        )
    else:
        st.info("No predictions logged yet. Enter symbols above and click “Log today's predictions”.")

    st.subheader("Chart with Prediction Markers")
    sym_opts = symbols
    sym_chart = st.selectbox("Symbol to visualize", options=sym_opts, index=0)

    try:
        # price series
        hist = get_ohlcv(sym_chart, start=TRAIN_START).sort_values("date")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=hist["date"], y=hist["close"], mode="lines", name=f"{sym_chart} Close"
        ))

        # markers from log
        if Path(LOG_PATH_PARQUET).exists():
            L = pd.read_parquet(LOG_PATH_PARQUET)
            L = L[L["symbol"] == sym_chart].sort_values("asof_date")

            for _, r in L.iterrows():
                color = "gray"
                label = f"{pd.to_datetime(r['asof_date']).date()}  p={r['p_up']:.0%}"
                if pd.notna(r.get("correct")):
                    is_correct = int(r["correct"]) == 1
                    color = "green" if is_correct else "red"
                    act = r.get("actual_ret")
                    if pd.notna(act):
                        label += f"  {'✅' if is_correct else '❌'} (act {act:+.1%})"

                fig.add_trace(go.Scatter(
                    x=[pd.to_datetime(r["asof_date"])],
                    y=[float(r["spot"])],
                    mode="markers+text",
                    text=[label],
                    textposition="top center",
                    marker=dict(size=9, color=color),
                    showlegend=False
                ))

        fig.update_layout(height=520, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(str(e))

    # summary metrics
    st.subheader("Summary Metrics")
    if Path(LOG_PATH_PARQUET).exists():
        done = pd.read_parquet(LOG_PATH_PARQUET)
        done = done[pd.notna(done["correct"])]
        if not done.empty:
            acc = done["correct"].mean()
            st.metric("Accuracy (resolved)", f"{acc:.1%}")
            # 20-observation rolling hit-rate by as-of date (across all symbols)
            roll = (
                done.sort_values("asof_date")
                    .groupby("asof_date")["correct"].mean()
                    .rolling(20, min_periods=5).mean()
            )
            st.line_chart(roll.rename("20-obs rolling hit-rate"))
        else:
            st.info("No resolved predictions yet—click “Update scores” after the horizon has passed.")
