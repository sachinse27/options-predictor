import os
from pathlib import Path
import pandas as pd
import yaml
import streamlit as st
import plotly.graph_objects as go

from src.predict import predict_symbol
from src.fetch_data import get_ohlcv
from src.auto_train import ensure_trained
from src.track import log_predictions, score_predictions, LOG_PATH_PARQUET
from src.options_utils import option_ev_over_h

st.set_page_config(page_title="Stock Direction Predictor", layout="wide")

cfg = yaml.safe_load(open("config.yaml"))
HORIZON = int(cfg.get("horizon_days", 3))
TRAIN_START = cfg.get("train_start", "2015-01-01")

def parse_symbols(text: str) -> list[str]:
    return [s.strip().upper() for s in text.split(",") if s.strip()]

def ensure_list_has_symbols(symbols: list[str]) -> list[str]:
    if symbols:
        return symbols
    return [s.upper() for s in cfg.get("symbols", ["SPY","QQQ","AAPL"])]

def load_log_df() -> pd.DataFrame:
    if Path(LOG_PATH_PARQUET).exists():
        return pd.read_parquet(LOG_PATH_PARQUET)
    return pd.DataFrame()

def get_logged_symbols(log_df: pd.DataFrame) -> list[str]:
    if log_df.empty:
        return []
    return sorted(log_df["symbol"].dropna().unique().tolist())

def build_symbol_chart(sym: str, train_start: str, log_df: pd.DataFrame) -> go.Figure:
    hist = get_ohlcv(sym, start=train_start).sort_values("date")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist["date"], y=hist["close"], mode="lines", name=f"{sym} Close"))
    if not log_df.empty:
        L = log_df[log_df["symbol"] == sym].sort_values("asof_date")
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
    return fig

tab1, tab2 = st.tabs(["🔮 Predict (one-off)", "📈 Tracker (log & chart)"])

# ===== TAB 1: Predict =====
with tab1:
    st.title("Stock Direction Predictor")
    st.caption(f"Predicts probability UP after {HORIZON} trading days + expected return. Educational demo.")
    symbol = st.text_input("Symbol (any ticker)", value="SPY").upper()

    # --- New: Option selection settings (manual controls) ---
    with st.expander("Option selection settings"):
        side_mode = st.selectbox("Side", ["Auto (by probability)", "Call", "Put"])
        tgt_delta = st.slider("Target absolute delta", 0.10, 0.60, 0.30, 0.05)
        dte_min = st.number_input("Min DTE", 1, 180, 20)
        dte_max = st.number_input("Max DTE", 7, 365, 45)
        min_oi = st.number_input("Min Open Interest", 0, 5000, 300, 50)
        max_spread = st.slider("Max spread %", 0.01, 0.25, 0.10, 0.01)

    if st.button("Predict"):
        try:
            with st.spinner(f"Training/Loading model for {symbol}…"):
                ensure_trained(symbol, horizon_days=HORIZON, start=TRAIN_START)
                p, mu = predict_symbol(symbol, start=TRAIN_START, horizon_days=HORIZON)
            st.metric(f"Probability of Up Move ({HORIZON} days)", f"{p:.1%}")
            st.metric(f"Expected {HORIZON}-day Return (μ̂)", f"{mu:+.2%}")

            side = ("call" if p >= 0.5 else "put") if side_mode.startswith("Auto") else side_mode.lower()

            with st.spinner(f"Selecting a ~{tgt_delta:.2f}Δ {side.upper()} and estimating EV…"):
                rec = option_ev_over_h(
                    symbol, side, mu_hat=mu, H_days=HORIZON,
                    target_delta=float(tgt_delta),
                    dte_min=int(dte_min), dte_max=int(dte_max),
                    min_oi=int(min_oi), max_spread_pct=float(max_spread)
                )
            if isinstance(rec, dict) and "error" in rec:
                st.warning(rec["error"])
            else:
                st.subheader("Suggested Option (approx. target Δ)")
                col1, col2, col3 = st.columns(3)
                col1.metric("Contract", rec.get("contractSymbol","?"))
                col2.metric("Strike", f"{rec['strike']:.2f}")
                col3.metric("IV (chain)", f"{rec['iv']:.1%}")
                col4, col5, col6 = st.columns(3)
                col4.metric("Bid / Ask", f"{rec['bid']:.2f} / {rec['ask']:.2f}")
                col5.metric("Δ (abs)", f"{rec['delta_abs']:.2f}")
                col6.metric("Spread %", f"{rec['spread_pct']:.2%}")
                st.metric("Estimated EV over horizon", f"{rec['EV_estimate']:+.2f}")
        except ValueError as ve:
            st.error(str(ve))
        except Exception as e:
            st.error(f"Unexpected error for {symbol}: {e}")

# ===== TAB 2: Tracker =====
with tab2:
    st.title("Prediction Tracker")
    st.caption("Log today's predictions; later, click Update to score them and visualize on charts.")

    default_syms_text = ",".join([s.upper() for s in cfg.get("symbols", ["SPY","QQQ","AAPL"])])
    symbols_text = st.text_input("Symbols to track (comma-separated)", value=default_syms_text)
    symbols = ensure_list_has_symbols(parse_symbols(symbols_text))

    skip_earn = st.checkbox("Skip symbols with earnings within 5 days (if earnings.csv present)", value=True)

    colA, colB, colC = st.columns([1,1,2])
    with colA:
        if st.button("Log today's predictions"):
            try:
                with st.spinner("Training/loading models & logging…"):
                    syms_to_log = symbols
                    if skip_earn:
                        try:
                            earn = pd.read_csv("data/earnings.csv", parse_dates=["date"])
                            today = pd.Timestamp.utcnow().normalize()
                            blocked = set()
                            for s in symbols:
                                dates = earn[earn["symbol"].str.upper() == s]["date"].dt.normalize()
                                if any((0 <= (d - today).days <= 5) for d in dates):
                                    blocked.add(s)
                            if blocked:
                                st.info("Skipping due to upcoming earnings: " + ", ".join(sorted(blocked)))
                                syms_to_log = [s for s in symbols if s not in blocked]
                        except Exception:
                            pass
                    for s in syms_to_log:
                        ensure_trained(s, horizon_days=HORIZON, start=TRAIN_START)
                    if syms_to_log:
                        log_predictions(syms_to_log, HORIZON, train_start=TRAIN_START)
                        st.success("Logged: " + ", ".join(syms_to_log))
                    else:
                        st.warning("No symbols to log after earnings filter.")
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
    log_df = load_log_df()
    if log_df.empty:
        st.info("No predictions logged yet. Enter symbols above and click “Log today's predictions”.")
    else:
        st.dataframe(log_df.sort_values(["symbol","asof_date"], ascending=[True, False]), use_container_width=True)

    st.subheader("Chart with Prediction Markers")
    logged_syms = get_logged_symbols(log_df)
    sym_opts = logged_syms or symbols
    sym_chart = st.selectbox("Symbol to visualize", options=sym_opts, index=0)
    try:
        fig = build_symbol_chart(sym_chart, TRAIN_START, log_df)
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(str(e))

    st.subheader("All symbols with logged predictions")
    if not log_df.empty and logged_syms:
        max_syms = st.number_input("Max symbols to show", min_value=1, max_value=50, value=min(12, len(logged_syms)))
        syms_to_show = logged_syms[: int(max_syms)]
        cols = st.columns(2)
        for i, sym in enumerate(syms_to_show):
            with cols[i % 2]:
                try:
                    small_fig = build_symbol_chart(sym, TRAIN_START, log_df)
                    small_fig.update_layout(height=380)
                    st.plotly_chart(small_fig, use_container_width=True)
                except Exception as e:
                    st.warning(f"{sym}: {e}")
    else:
        st.info("No logged symbols yet to render in a grid.")

    st.subheader("Summary Metrics")
    if Path(LOG_PATH_PARQUET).exists():
        log_all = pd.read_parquet(LOG_PATH_PARQUET)
        done = log_all[pd.notna(log_all["correct"])].copy()
        if done.empty:
            st.info("No resolved predictions yet — click “Update scores” after the horizon has passed.")
        else:
            overall_acc = done["correct"].mean()
            colN1, colN2 = st.columns([1,3])
            with colN1:
                N = st.number_input("Last N predictions", min_value=5, max_value=1000, value=30, step=5)
            lastN = done.sort_values("asof_date").tail(int(N))
            lastN_acc = lastN["correct"].mean()
            try:
                brier = ((done["p_up"] - done["correct"].astype(float))**2).mean()
            except Exception:
                brier = float("nan")
            colM1, colM2, colM3 = st.columns(3)
            colM1.metric("Overall Accuracy (resolved)", f"{overall_acc:.1%}")
            colM2.metric(f"Last {int(N)} Accuracy", f"{lastN_acc:.1%}")
            colM3.metric("Brier Score (lower is better)", f"{brier:.3f}")
            roll = (
                done.sort_values("asof_date")
                    .groupby("asof_date")["correct"].mean()
                    .rolling(20, min_periods=5).mean()
            )
            st.line_chart(roll.rename("20-obs rolling hit-rate"))
            st.subheader("Per-Symbol Accuracy (resolved)")
            sym_stats = (
                done.groupby("symbol")
                    .agg(n_resolved=("correct","size"),
                         accuracy=("correct","mean"),
                         avg_prob=("p_up","mean"),
                         avg_actual_ret=("actual_ret","mean"))
                    .sort_values("accuracy", ascending=False)
            )
            st.dataframe(sym_stats.style.format({
                "accuracy": "{:.1%}",
                "avg_prob": "{:.1%}",
                "avg_actual_ret": "{:+.2%}"
            }), use_container_width=True)
    else:
        st.info("No prediction log found yet — log some predictions first.")
