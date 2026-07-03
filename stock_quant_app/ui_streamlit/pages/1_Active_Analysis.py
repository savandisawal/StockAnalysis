"""Active Analysis -- deep technical analysis with indicators."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st

st.set_page_config(page_title="Active Analysis", page_icon="Q", layout="wide")

from ui_streamlit.components.styles import (
    inject_global_css,
    styled_header,
    metric_card,
    confidence_badge,
    prediction_range_card,
    section_header,
    COLORS,
)
from data.fetch_ohlc import fetch_ohlc
from features.technicals import compute_all_technicals
from features.macro_sentiment import compute_macro_sentiment_features
from services.prediction_service import get_prediction
from model.model_registry import list_models
from ui_streamlit.components.charts import candlestick_chart, indicator_chart, shap_bar_chart
from utils.sectors import get_sector

inject_global_css()

ticker = st.session_state.get("selected_ticker", "RELIANCE")
sector = get_sector(ticker)
styled_header(f"Active Analysis: {ticker}", f"{sector} sector" if sector else "")

try:
    with st.spinner("Computing indicators..."):
        df = fetch_ohlc(ticker, years=1)
        if df.empty:
            st.error(f"No data for {ticker}")
            st.stop()
        df = compute_all_technicals(df)
except Exception as e:
    st.error(f"Failed to load data: {e}")
    st.stop()

# ── Prediction Card ──────────────────────────────────────────────

try:
    existing_models = list_models(ticker)
except Exception:
    existing_models = []

pred = None
if existing_models:
    try:
        pred = get_prediction(ticker, include_fundamentals=True, include_macro=True, source="ui")
        if pred:
            section_header("Prediction")
            p_left, p_right = st.columns([3, 1])
            with p_left:
                prediction_range_card(
                    pred.predicted_low,
                    pred.predicted_mid,
                    pred.predicted_high,
                    pred.predicted_change_pct,
                )
            with p_right:
                confidence_badge(pred.confidence, pred.direction)
            st.markdown(
                f'<div style="text-align:right;color:#78909C;font-size:0.75rem;margin-top:4px">Model: <code>{pred.model_version}</code></div>',
                unsafe_allow_html=True,
            )
            if pred.has_warnings:
                for w in pred.warnings:
                    icon = {"critical": "🔴", "warning": "🟡", "info": "🔵"}.get(w.level, "")
                    if w.level == "critical":
                        st.error(f"{icon} {w.message}")
                    elif w.level == "warning":
                        st.warning(f"{icon} {w.message}")
                    else:
                        st.info(f"{icon} {w.message}")

            # ── Why this prediction — per-prediction SHAP ────────
            if pred.explanation and pred.explanation.get("top_features"):
                section_header("Why this prediction")
                st.plotly_chart(shap_bar_chart(pred.explanation), use_container_width=True)
                st.caption(
                    "SHAP contributions to the median (P50) forecast. Green bars pushed "
                    "the prediction up, red bars pushed it down; the number beside each "
                    "bar is that feature's input value today."
                )
    except Exception as e:
        st.caption(f"Prediction unavailable: {e}")
else:
    st.info("No trained model. Go to Home page and click 'Train Model' first.")

# ── Period Selector ──────────────────────────────────────────────

st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
period = st.radio("Period", ["30D", "60D", "90D", "180D", "1Y"], horizontal=True, index=2)
days = {"30D": 30, "60D": 60, "90D": 90, "180D": 180, "1Y": 365}[period]
chart_df = df.tail(days)

# ── Candlestick ──────────────────────────────────────────────────

prediction_dict = pred.to_dict() if pred else None
fig = candlestick_chart(chart_df, prediction=prediction_dict, title=f"{ticker} OHLCV")
st.plotly_chart(fig, use_container_width=True)

# ── Technical Indicators ─────────────────────────────────────────

section_header("Technical Indicators")

tab_rsi, tab_macd, tab_bb, tab_adx, tab_vol = st.tabs(
    ["RSI", "MACD", "Bollinger %B", "ADX", "Volume Z"]
)


def _info_box(text: str, color: str):
    st.markdown(
        f'<div style="background:{color}15;border:1px solid {color}40;border-radius:10px;padding:10px 16px;color:{color}">{text}</div>',
        unsafe_allow_html=True,
    )


with tab_rsi:
    fig = indicator_chart(chart_df, "rsi_14", title="RSI (14)", color="#BB86FC")
    st.plotly_chart(fig, use_container_width=True)
    v = chart_df["rsi_14"].iloc[-1]
    if v > 70:
        _info_box(f"<b>RSI = {v:.1f}</b> -- Overbought zone. Potential reversal down.", "#FF5252")
    elif v < 30:
        _info_box(f"<b>RSI = {v:.1f}</b> -- Oversold zone. Potential reversal up.", "#00E676")
    else:
        _info_box(f"<b>RSI = {v:.1f}</b> -- Neutral zone.", "#6C63FF")

with tab_macd:
    fig = indicator_chart(chart_df, "macd_hist_pct", title="MACD Histogram (%)", color="#18FFFF")
    st.plotly_chart(fig, use_container_width=True)
    v = chart_df["macd_hist_pct"].iloc[-1]
    c = "#00E676" if v > 0 else "#FF5252"
    _info_box(
        f"<b>MACD Histogram = {v:.3f}%</b> -- {'Bullish momentum' if v > 0 else 'Bearish momentum'}",
        c,
    )

with tab_bb:
    fig = indicator_chart(chart_df, "bb_pct_b", title="Bollinger %B", color="#FFD740")
    st.plotly_chart(fig, use_container_width=True)
    v = chart_df["bb_pct_b"].iloc[-1]
    _info_box(
        f"<b>%B = {v:.3f}</b> -- {'Near upper band (resistance)' if v > 0.8 else 'Near lower band (support)' if v < 0.2 else 'Mid-range'}",
        "#FFD740",
    )

with tab_adx:
    fig = indicator_chart(chart_df, "adx_14", title="ADX (14) - Trend Strength", color="#00D4AA")
    st.plotly_chart(fig, use_container_width=True)
    adx_v = chart_df["adx_14"].iloc[-1]
    regime = int(chart_df["regime"].iloc[-1])
    regime_label = {1: "TRENDING UP", -1: "TRENDING DOWN", 0: "SIDEWAYS"}
    regime_color = {1: "#00E676", -1: "#FF5252", 0: "#FFD740"}
    rc = regime_color.get(regime, "#78909C")
    _info_box(
        f"<b>ADX = {adx_v:.1f}</b> | Regime: <b>{regime_label.get(regime, 'Unknown')}</b>", rc
    )

with tab_vol:
    fig = indicator_chart(chart_df, "volume_zscore", title="Volume Z-Score (20D)", color="#FF4081")
    st.plotly_chart(fig, use_container_width=True)
    v = chart_df["volume_zscore"].iloc[-1]
    if abs(v) > 2:
        _info_box(f"<b>Volume Z = {v:.2f}</b> -- Unusual volume! Potential breakout.", "#FF4081")
    else:
        _info_box(f"<b>Volume Z = {v:.2f}</b> -- Normal volume range.", "#78909C")

# ── Key Metrics ──────────────────────────────────────────────────

st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
section_header("Key Metrics")

latest = chart_df.iloc[-1]
m1, m2, m3, m4, m5, m6 = st.columns(6)
with m1:
    metric_card("ATR %", f"{latest.get('atr_pct', 0):.2f}%", color=COLORS["orange"])
with m2:
    v = latest.get("ema_20_dist_pct", 0)
    metric_card("EMA 20", f"{v:+.2f}%", color=COLORS["bullish"] if v > 0 else COLORS["bearish"])
with m3:
    v = latest.get("ema_50_dist_pct", 0)
    metric_card("EMA 50", f"{v:+.2f}%", color=COLORS["bullish"] if v > 0 else COLORS["bearish"])
with m4:
    v = latest.get("ema_200_dist_pct", 0)
    metric_card("EMA 200", f"{v:+.2f}%", color=COLORS["bullish"] if v > 0 else COLORS["bearish"])
with m5:
    metric_card("BB Width", f"{latest.get('bb_width_pct', 0):.2f}%", color=COLORS["gold"])
with m6:
    v = latest.get("return_1d", 0)
    metric_card("1D Return", f"{v:+.2f}%", color=COLORS["bullish"] if v > 0 else COLORS["bearish"])

# ── Macro Mood ───────────────────────────────────────────────────

st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
with st.expander("Macro Mood"):
    try:
        macro = compute_macro_sentiment_features(stock=ticker, sector=sector)
        mood = macro.macro_mood or 0
        mood_label = "BULLISH" if mood > 0.3 else "BEARISH" if mood < -0.3 else "NEUTRAL"
        mood_color = (
            COLORS["bullish"]
            if mood > 0.3
            else COLORS["bearish"]
            if mood < -0.3
            else COLORS["warning"]
        )
        st.markdown(
            f'<div style="background:{mood_color}15;border:1px solid {mood_color}40;border-radius:12px;padding:16px;text-align:center;margin-bottom:16px"><div style="font-size:0.8rem;color:#9E9E9E;text-transform:uppercase;letter-spacing:1px">Macro Mood</div><div style="font-size:1.8rem;font-weight:800;color:{mood_color}">{mood_label}</div><div style="color:#78909C;font-size:0.85rem">Score: {mood:+.2f}</div></div>',
            unsafe_allow_html=True,
        )
        mc1, mc2, mc3 = st.columns(3)
        with mc1:
            metric_card("S&P 500", f"{macro.sp500_change or 0:+.2f}%", color=COLORS["primary"])
        with mc2:
            metric_card("Nasdaq", f"{macro.nasdaq_change or 0:+.2f}%", color=COLORS["cyan"])
        with mc3:
            metric_card("India VIX", f"{macro.vix_value or 0:+.2f}%", color=COLORS["warning"])
    except Exception as e:
        st.caption(f"Macro data unavailable: {e}")
