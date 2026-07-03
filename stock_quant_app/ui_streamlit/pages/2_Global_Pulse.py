"""Global Pulse -- live macro indicators dashboard with colorful cards."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st

st.set_page_config(page_title="Global Pulse", page_icon="Q", layout="wide")

from ui_streamlit.components.styles import (
    inject_global_css,
    styled_header,
    macro_indicator_card,
    section_header,
    COLORS,
)
from data.fetch_macro import fetch_macro_snapshot
from features.macro_sentiment import compute_macro_sentiment_features

inject_global_css()

styled_header("Global Pulse", "Real-time global macro indicators")

# ── Refresh Button ───────────────────────────────────────────────

refresh = st.button("Refresh Data", type="primary")

try:
    with st.spinner("Fetching global indicators..."):
        snapshots = fetch_macro_snapshot(use_cache=not refresh)
except Exception as e:
    st.error(f"Failed to fetch macro data: {e}")
    st.stop()

if not snapshots:
    st.error("Failed to fetch macro data. Check internet connection.")
    st.stop()

# ── Macro Mood Banner ────────────────────────────────────────────

try:
    macro = compute_macro_sentiment_features()
    mood = macro.macro_mood or 0
    if mood > 0.3:
        mood_label, mood_desc, mood_bg = (
            "BULLISH",
            "Positive global signals for Indian markets",
            "#00E676",
        )
    elif mood < -0.3:
        mood_label, mood_desc, mood_bg = (
            "BEARISH",
            "Negative global signals for Indian markets",
            "#FF5252",
        )
    else:
        mood_label, mood_desc, mood_bg = "NEUTRAL", "Mixed global signals", "#FFD740"
    st.markdown(
        f'<div style="background:linear-gradient(135deg,{mood_bg}12,{mood_bg}04);border:1px solid {mood_bg}35;border-radius:16px;padding:24px;margin:16px 0 28px 0;display:flex;justify-content:space-between;align-items:center"><div><div style="font-size:0.75rem;color:#9E9E9E;text-transform:uppercase;letter-spacing:2px">Market Mood</div><div style="font-size:2rem;font-weight:800;color:{mood_bg};margin-top:4px">{mood_label}</div><div style="color:#78909C;font-size:0.9rem;margin-top:2px">{mood_desc}</div></div><div style="text-align:right"><div style="font-size:3rem;font-weight:800;color:{mood_bg}">{mood:+.2f}</div><div style="color:#78909C;font-size:0.7rem">Composite Score</div></div></div>',
        unsafe_allow_html=True,
    )
except Exception:
    pass

# ── Equity Indices ───────────────────────────────────────────────

section_header("Equity Indices")

equity_names = ["S&P 500", "Nasdaq", "Nifty 50"]
eq_cols = st.columns(3)
for i, snap in enumerate(s for s in snapshots if s.name in equity_names):
    with eq_cols[i % 3]:
        macro_indicator_card(snap.name, snap.current_price, snap.change_pct)

st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)

# ── Commodities & Currency ───────────────────────────────────────

section_header("Commodities & Currency")

other_names = ["Brent Crude", "USD/INR", "India VIX"]
other_cols = st.columns(3)
for i, snap in enumerate(s for s in snapshots if s.name in other_names):
    with other_cols[i % 3]:
        macro_indicator_card(snap.name, snap.current_price, snap.change_pct)

# VIX interpretation
vix_snap = next((s for s in snapshots if s.name == "India VIX"), None)
if vix_snap and vix_snap.current_price:
    vix_val = vix_snap.current_price
    if vix_val > 25:
        vix_color, vix_label = (
            COLORS["bearish"],
            "HIGH FEAR -- Elevated volatility, markets may be stressed",
        )
    elif vix_val > 18:
        vix_color, vix_label = COLORS["warning"], "MODERATE -- Normal volatility range"
    else:
        vix_color, vix_label = COLORS["bullish"], "LOW -- Calm markets, complacency risk"
    st.markdown(
        f'<div style="background:{vix_color}10;border-left:4px solid {vix_color};border-radius:0 10px 10px 0;padding:12px 18px;margin-top:16px;color:{vix_color};font-size:0.85rem">VIX at <b>{vix_val:.1f}</b> -- {vix_label}</div>',
        unsafe_allow_html=True,
    )

# ── Impact Guide ─────────────────────────────────────────────────

st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)

with st.expander("How do these affect Indian markets?"):
    st.markdown(
        '<div style="padding:8px 0;color:#B0B0B0;line-height:1.8"><div style="margin-bottom:12px"><span style="color:#00E676;font-weight:600">S&P 500 / Nasdaq</span> -- Rising US markets = bullish for India (FII sentiment, global risk-on)</div><div style="margin-bottom:12px"><span style="color:#FF9100;font-weight:600">Brent Crude</span> -- Rising crude = bearish for India (net oil importer, fiscal pressure, inflation)</div><div style="margin-bottom:12px"><span style="color:#18FFFF;font-weight:600">USD/INR</span> -- Rising USD = bearish (rupee weakening, FII outflows, import costs up)</div><div><span style="color:#FF4081;font-weight:600">India VIX</span> -- High VIX = fear gauge elevated, expect wider intraday ranges and potential sell-offs</div></div>',
        unsafe_allow_html=True,
    )

# ── Raw Data ─────────────────────────────────────────────────────

with st.expander("Raw Data"):
    import pandas as pd

    data = [
        {
            "Indicator": s.name,
            "Ticker": s.ticker,
            "Current": s.current_price,
            "Prev Close": s.prev_close,
            "Change %": s.change_pct,
            "Date": s.fetch_date,
        }
        for s in snapshots
    ]
    st.dataframe(pd.DataFrame(data), use_container_width=True, hide_index=True)
