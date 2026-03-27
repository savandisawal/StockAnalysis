"""Stock Quant App -- Home Page.

Run: streamlit run ui_streamlit/Home.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

st.set_page_config(
    page_title="Stock Quant App",
    page_icon="Q",
    layout="wide",
    initial_sidebar_state="expanded",
)

from ui_streamlit.components.styles import (
    inject_global_css, styled_header, metric_card, confidence_badge,
    prediction_range_card, section_header, COLORS,
)
from data.fetch_ohlc import fetch_ohlc
from features.technicals import compute_all_technicals
from model.model_registry import list_models
from model.predict import predict_next_day
from model.train_model import train_quantile_models, get_feature_importance
from ui_streamlit.components.charts import candlestick_chart
from utils.sectors import SECTOR_STOCKS, get_sector

inject_global_css()

# ── Sidebar ──────────────────────────────────────────────────────

st.sidebar.markdown('<div style="text-align:center;padding:16px 0 8px 0"><div style="font-size:2rem;margin-bottom:4px">Q</div><div style="font-size:1.2rem;font-weight:800;background:linear-gradient(90deg,#6C63FF,#00D4AA);-webkit-background-clip:text;-webkit-text-fill-color:transparent">STOCK QUANT</div><div style="color:#78909C;font-size:0.75rem;margin-top:2px">NSE India Price Predictor</div></div>', unsafe_allow_html=True)
st.sidebar.divider()

all_tickers = sorted(set(
    ticker for tickers in SECTOR_STOCKS.values() for ticker in tickers
))
selected_ticker = st.sidebar.selectbox(
    "Select Stock",
    options=all_tickers,
    index=all_tickers.index("RELIANCE") if "RELIANCE" in all_tickers else 0,
)
st.session_state["selected_ticker"] = selected_ticker

sector = get_sector(selected_ticker)
if sector:
    st.sidebar.markdown(f'<div style="background:#6C63FF15;border:1px solid #6C63FF30;border-radius:8px;padding:8px 12px;text-align:center;margin:8px 0"><span style="color:#9E9E9E;font-size:0.7rem;text-transform:uppercase;letter-spacing:1px">Sector</span><br><span style="color:#BB86FC;font-weight:600">{sector}</span></div>', unsafe_allow_html=True)

st.sidebar.divider()
st.sidebar.markdown("**Model Controls**")
train_years = st.sidebar.slider("Training years", 1, 5, 2)
use_fundamentals = st.sidebar.checkbox("Include Fundamentals", value=False)
use_macro = st.sidebar.checkbox("Include Macro/Sentiment", value=False)

if st.sidebar.button("Train Model", type="primary", use_container_width=True):
    with st.sidebar.status(f"Training {selected_ticker}...", expanded=True) as status:
        st.write("Fetching data & computing features...")
        try:
            models, features, metrics = train_quantile_models(
                selected_ticker, years=train_years,
                include_fundamentals=use_fundamentals,
                include_macro=use_macro,
            )
            st.write(f"Direction accuracy: {metrics['direction_accuracy']:.1%}")
            st.write(f"Interval coverage: {metrics['interval_coverage_80']:.1%}")
            status.update(label="Training complete!", state="complete")
        except Exception as e:
            status.update(label=f"Failed: {e}", state="error")

try:
    existing_models = list_models(selected_ticker)
except Exception:
    existing_models = []

if existing_models:
    ts = existing_models[0].get('timestamp', '')[:8]
    st.sidebar.markdown(f'<div style="background:#00E67615;border:1px solid #00E67630;border-radius:8px;padding:8px 12px;margin-top:8px"><span style="color:#00E676;font-size:0.8rem">Model ready</span><span style="color:#78909C;font-size:0.7rem"> | {ts}</span></div>', unsafe_allow_html=True)
else:
    st.sidebar.warning("No model trained yet")

# ── Main Content ─────────────────────────────────────────────────

styled_header(selected_ticker, f"{sector} sector | NSE India" if sector else "NSE India")

try:
    with st.spinner("Loading market data..."):
        df = fetch_ohlc(selected_ticker, years=1)
except Exception as e:
    st.error(f"Failed to fetch data: {e}")
    st.stop()

if df.empty:
    st.error(f"Could not fetch data for {selected_ticker}")
    st.stop()

# Price metrics row
latest = df.iloc[-1]
prev = df.iloc[-2] if len(df) > 1 else latest
daily_change = ((latest["Close"] - prev["Close"]) / prev["Close"]) * 100
daily_change_str = f"{daily_change:+.2f}%"

col1, col2, col3, col4 = st.columns(4)
with col1:
    metric_card("Close", f"Rs.{latest['Close']:,.2f}", daily_change_str, COLORS["primary"])
with col2:
    metric_card("High", f"Rs.{latest['High']:,.2f}", color=COLORS["bullish"])
with col3:
    metric_card("Low", f"Rs.{latest['Low']:,.2f}", color=COLORS["bearish"])
with col4:
    metric_card("Volume", f"{latest['Volume']:,.0f}", color=COLORS["cyan"])

# ── Prediction Section ───────────────────────────────────────────

prediction = None
if existing_models:
    try:
        pred = predict_next_day(
            selected_ticker,
            include_fundamentals=use_fundamentals,
            include_macro=use_macro,
        )
        if pred:
            prediction = pred.to_dict()
            st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
            section_header("Next-Day Prediction")
            p_left, p_right = st.columns([3, 1])
            with p_left:
                prediction_range_card(pred.predicted_low, pred.predicted_mid, pred.predicted_high, pred.predicted_change_pct)
            with p_right:
                confidence_badge(pred.confidence, pred.direction)
    except Exception as e:
        st.caption(f"Prediction unavailable: {e}")

# ── Candlestick Chart ────────────────────────────────────────────

st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
section_header("Price Chart")

chart_days = st.slider("Chart period (days)", 30, 365, 90, label_visibility="collapsed")
chart_df = df.tail(chart_days)
try:
    fig = candlestick_chart(chart_df, prediction=prediction, title=f"{selected_ticker} - {chart_days}D")
    st.plotly_chart(fig, use_container_width=True)
except Exception as e:
    st.error(f"Chart rendering failed: {e}")

# ── Feature Importance ───────────────────────────────────────────

if existing_models:
    with st.expander("Feature Importance"):
        try:
            from model.model_registry import load_model_bundle
            bundle = load_model_bundle(selected_ticker)
            if bundle:
                models_loaded, feat_names, meta = bundle
                imp = get_feature_importance(models_loaded, feat_names, top_n=15)
                import plotly.express as px
                fig = px.bar(imp, x="importance", y="feature", orientation="h", color="importance", color_continuous_scale=["#6C63FF", "#00D4AA"])
                fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(14,17,23,0.8)", font=dict(color="#C0C0C0"), yaxis=dict(autorange="reversed", gridcolor="#1E2028"), xaxis=dict(gridcolor="#1E2028"), coloraxis_showscale=False, height=400, margin=dict(l=120, r=20, t=20, b=20))
                st.plotly_chart(fig, use_container_width=True)
        except Exception:
            st.caption("Train a model to see feature importance.")
