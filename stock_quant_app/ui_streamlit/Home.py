"""Stock Quant App -- Home Page.

Run: streamlit run ui_streamlit/Home.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
import pandas as pd

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
from model.model_registry import list_models
from model.predict import predict_next_day
from model.train_model import train_quantile_models, get_feature_importance
from ui_streamlit.components.charts import candlestick_chart
from utils.sectors import (
    SECTOR_STOCKS, get_sector, get_company_name, get_all_tickers, get_all_sectors,
)

inject_global_css()

# ── Initialize session state ────────────────────────────────────

if "search_history" not in st.session_state:
    st.session_state["search_history"] = []

# ── Sidebar ──────────────────────────────────────────────────────

st.sidebar.markdown('<div style="text-align:center;padding:16px 0 8px 0"><div style="font-size:2rem;margin-bottom:4px">Q</div><div style="font-size:1.2rem;font-weight:800;background:linear-gradient(90deg,#6C63FF,#00D4AA);-webkit-background-clip:text;-webkit-text-fill-color:transparent">STOCK QUANT</div><div style="color:#78909C;font-size:0.75rem;margin-top:2px">NSE India Price Predictor</div></div>', unsafe_allow_html=True)
st.sidebar.divider()

# Stock selection — sector filter + searchable dropdown + custom input
st.sidebar.markdown("**Select Stock**")

# Sector filter
all_sectors = ["All Sectors"] + get_all_sectors()
selected_sector = st.sidebar.selectbox(
    "Filter by Sector", options=all_sectors, label_visibility="collapsed",
)

# Build filtered options with company names
if selected_sector == "All Sectors":
    ticker_list = get_all_tickers()
else:
    ticker_list = [t for t, _ in SECTOR_STOCKS[selected_sector]]

display_options = [f"{t} — {get_company_name(t)}" for t in ticker_list]
ticker_map = dict(zip(display_options, ticker_list))

# Default selection
default_display = f"RELIANCE — {get_company_name('RELIANCE')}"
default_idx = display_options.index(default_display) if default_display in display_options else 0

selected_display = st.sidebar.selectbox(
    "Stock", options=display_options, index=default_idx, label_visibility="collapsed",
)
selected_ticker = ticker_map[selected_display]

# Custom ticker input
st.sidebar.markdown(
    '<div style="color:#78909C;font-size:0.75rem;margin:-8px 0 4px 0">'
    'Or enter any NSE ticker:</div>',
    unsafe_allow_html=True,
)
custom_ticker = st.sidebar.text_input(
    "Custom ticker", value="", placeholder="e.g. ZOMATO, PAYTM",
    label_visibility="collapsed",
)
if custom_ticker.strip():
    selected_ticker = custom_ticker.strip().upper().replace(".NS", "").replace(".BO", "")

st.session_state["selected_ticker"] = selected_ticker

sector = get_sector(selected_ticker)
company = get_company_name(selected_ticker)
if sector:
    st.sidebar.markdown(f'<div style="background:rgba(108,99,255,0.08);border:1px solid rgba(108,99,255,0.19);border-radius:8px;padding:8px 12px;text-align:center;margin:8px 0"><span style="color:#9E9E9E;font-size:0.7rem;text-transform:uppercase;letter-spacing:1px">Sector</span><br><span style="color:#BB86FC;font-weight:600">{sector}</span></div>', unsafe_allow_html=True)

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
    st.sidebar.markdown(f'<div style="background:rgba(0,230,118,0.08);border:1px solid rgba(0,230,118,0.19);border-radius:8px;padding:8px 12px;margin-top:8px"><span style="color:#00E676;font-size:0.8rem">Model ready</span><span style="color:#78909C;font-size:0.7rem"> | {ts}</span></div>', unsafe_allow_html=True)
else:
    st.sidebar.warning("No model trained yet")

# ── Main Content ─────────────────────────────────────────────────

header_subtitle = f"{company} | {sector} sector" if sector else company
styled_header(selected_ticker, header_subtitle)

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
pred = None
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

# ── Record search history ───────────────────────────────────────

history_entry = {
    "Ticker": selected_ticker,
    "Company": company,
    "Sector": sector or "—",
    "Date": str(df.index[-1].date()),
    "Close": round(latest["Close"], 2),
    "Change %": round(daily_change, 2),
    "High": round(latest["High"], 2),
    "Low": round(latest["Low"], 2),
    "Volume": int(latest["Volume"]),
    "Prediction": f"Rs.{pred.predicted_mid:,.2f}" if pred else "—",
    "Direction": pred.direction if pred else "—",
    "Confidence": f"{pred.confidence}%" if pred else "—",
}

# Avoid duplicate consecutive entries
history = st.session_state["search_history"]
if not history or history[-1]["Ticker"] != selected_ticker:
    history.append(history_entry)
    # Keep last 50
    st.session_state["search_history"] = history[-50:]
else:
    # Update the latest entry for same ticker
    history[-1] = history_entry

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

# ── Search History ──────────────────────────────────────────────

st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
section_header("Search History")

history = st.session_state["search_history"]
if history:
    hist_df = pd.DataFrame(reversed(history))

    # Style the change column
    st.dataframe(
        hist_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Close": st.column_config.NumberColumn(format="Rs.%.2f"),
            "High": st.column_config.NumberColumn(format="Rs.%.2f"),
            "Low": st.column_config.NumberColumn(format="Rs.%.2f"),
            "Change %": st.column_config.NumberColumn(format="%.2f%%"),
            "Volume": st.column_config.NumberColumn(format="%d"),
        },
    )

    if st.button("Clear History", type="secondary"):
        st.session_state["search_history"] = []
        st.rerun()
else:
    st.caption("Browse stocks to build your search history.")
