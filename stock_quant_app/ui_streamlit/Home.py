"""Stock Quant App -- Home Page.

Run: streamlit run ui_streamlit/Home.py
"""

import sys
from datetime import date
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
from data.cache import get_json, set_json
from data.fetch_ohlc import fetch_ohlc
from model.model_registry import list_models
from model.predict import predict_next_day
from model.train_model import train_quantile_models, get_feature_importance
from ui_streamlit.components.charts import candlestick_chart
from utils.sectors import (
    SECTOR_STOCKS, get_sector, get_company_name, get_all_tickers, get_all_sectors,
)

_HISTORY_KEY = "ui:search_history"
_HISTORY_MAX = 50
# 10 years TTL — effectively never expires
_HISTORY_TTL = 86400 * 365 * 10

inject_global_css()

# Auto-select text on click for sidebar inputs (selectbox search + text input)
st.markdown(
    """<script>
    const doc = window.parent.document;
    doc.addEventListener('focusin', function(e) {
        if (e.target.tagName === 'INPUT' && e.target.type === 'text') {
            setTimeout(() => e.target.select(), 0);
        }
    });
    </script>""",
    unsafe_allow_html=True,
)

# ── Initialize session state (load from DB) ─────────────────────

if "search_history" not in st.session_state:
    saved = get_json(_HISTORY_KEY, ttl=_HISTORY_TTL)
    st.session_state["search_history"] = saved if isinstance(saved, list) else []

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

_TRAIN_LIMIT_KEY = f"training:log:{date.today().isoformat()}"
_TRAIN_LIMIT_TTL = 86400 * 2  # 2 days — survives the full calendar day
_MAX_TRAINS_PER_DAY = 10

_today_trained: list = get_json(_TRAIN_LIMIT_KEY, ttl=_TRAIN_LIMIT_TTL) or []
_already_trained = selected_ticker in _today_trained
_limit_reached = len(_today_trained) >= _MAX_TRAINS_PER_DAY

if _limit_reached:
    st.sidebar.warning(f"Daily limit reached — {_MAX_TRAINS_PER_DAY} stocks trained today.", icon="🚫")
elif _already_trained:
    st.sidebar.info(f"{selected_ticker} already trained today.", icon="✅")

_train_disabled = _already_trained or _limit_reached

if st.sidebar.button("Train Model", type="primary", use_container_width=True, disabled=_train_disabled):
    # Re-read from cache at click time — guards against stale UI state
    _trained_now = get_json(_TRAIN_LIMIT_KEY, ttl=_TRAIN_LIMIT_TTL) or []
    if selected_ticker in _trained_now:
        st.sidebar.error(f"{selected_ticker} already trained today. Come back tomorrow.", icon="🚫")
    elif len(_trained_now) >= _MAX_TRAINS_PER_DAY:
        st.sidebar.error(f"Daily limit of {_MAX_TRAINS_PER_DAY} stocks reached.", icon="🚫")
    else:
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
                _trained_now.append(selected_ticker)
                set_json(_TRAIN_LIMIT_KEY, _trained_now)
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

            # Show prediction warnings/safeguards
            if pred.has_warnings:
                for w in pred.warnings:
                    icon = {"critical": "🔴", "warning": "🟡", "info": "🔵"}.get(w.level, "")
                    if w.level == "critical":
                        st.error(f"{icon} {w.message}")
                    elif w.level == "warning":
                        st.warning(f"{icon} {w.message}")
                    else:
                        st.info(f"{icon} {w.message}")
                if pred.guardrail_applied:
                    st.caption(f"Original predicted change: {pred.original_change_pct:+.2f}% (capped by guardrail)")
    except Exception as e:
        st.caption(f"Prediction unavailable: {e}")

# ── Prediction Explainer ─────────────────────────────────────────

if pred:
    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
    section_header("Prediction Explainer")

    # Signal Summary Table
    from model.predict import get_signal_summary
    signals = get_signal_summary(selected_ticker)
    if signals:
        with st.expander("Signal Summary", expanded=True):
            sig_df = pd.DataFrame(signals)
            # Color-code sentiment column
            def _color_sentiment(val):
                colors = {"Bullish": "#00E676", "Bearish": "#FF5252", "Neutral": "#FFD740"}
                return f"color: {colors.get(val, '#C0C0C0')}"
            styled_df = sig_df.style.map(_color_sentiment, subset=["sentiment"])
            st.dataframe(
                styled_df, use_container_width=True, hide_index=True,
                column_config={
                    "signal": st.column_config.TextColumn("Signal", width="medium"),
                    "value": st.column_config.TextColumn("Value", width="small"),
                    "interpretation": st.column_config.TextColumn("Interpretation", width="large"),
                    "sentiment": st.column_config.TextColumn("Sentiment", width="small"),
                },
            )

            # Sentiment summary counts
            bull_count = sum(1 for s in signals if s["sentiment"] == "Bullish")
            bear_count = sum(1 for s in signals if s["sentiment"] == "Bearish")
            neut_count = sum(1 for s in signals if s["sentiment"] == "Neutral")
            st.markdown(
                f'<div style="display:flex;gap:16px;justify-content:center;padding:8px 0">'
                f'<span style="color:#00E676;font-weight:600">{bull_count} Bullish</span>'
                f'<span style="color:#FFD740;font-weight:600">{neut_count} Neutral</span>'
                f'<span style="color:#FF5252;font-weight:600">{bear_count} Bearish</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

    # Feature Importance (inline — no need to open separate expander)
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

    # Confidence Breakdown
    with st.expander("Confidence Breakdown"):
        st.markdown(
            f'<div style="background:rgba(108,99,255,0.06);border:1px solid rgba(108,99,255,0.15);border-radius:12px;padding:16px">'
            f'<div style="font-size:0.85rem;color:#C0C0C0;line-height:1.7">'
            f'<b>Confidence: {pred.confidence}%</b><br>'
            f'Predicted range width: <b>{pred.range_width_pct:.2f}%</b> (P10 to P90)<br>'
            f'Direction: <b>{pred.direction}</b> (median change: {pred.predicted_change_pct:+.3f}%)<br><br>'
            f'<span style="color:#78909C">Confidence is derived from how narrow the prediction range is '
            f'relative to the stock\'s typical daily volatility (ATR). A narrow range means the model '
            f'is more certain about tomorrow\'s price action.</span>'
            f'</div></div>',
            unsafe_allow_html=True,
        )

# ── Corporate Actions ────────────────────────────────────────────

st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
with st.expander("Corporate Actions & Filings"):
    try:
        from data.fetch_corporate import fetch_corporate_announcements
        filings = fetch_corporate_announcements(selected_ticker, days=30)
        if filings.announcements:
            display_data = filings.to_display_list(max_items=15)
            filings_df = pd.DataFrame(display_data)
            st.dataframe(
                filings_df, use_container_width=True, hide_index=True,
                column_config={
                    "Date": st.column_config.TextColumn("Date", width="small"),
                    "Category": st.column_config.TextColumn("Category", width="medium"),
                    "Subject": st.column_config.TextColumn("Subject", width=400),
                    "Important": st.column_config.CheckboxColumn("Material", width="small"),
                },
            )
            imp_count = len(filings.important_announcements())
            st.caption(f"{filings.total_count} announcements in last 30 days ({imp_count} material)")
        else:
            st.caption("No corporate announcements found in the last 30 days.")
    except Exception as e:
        st.caption(f"Corporate filings unavailable: {e}")

# ── Fundamentals Overview ────────────────────────────────────────

st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
with st.expander("Fundamentals Overview"):
    try:
        from data.fetch_fundamentals import fetch_fundamentals
        from features.fundamentals import compute_fundamental_features
        raw = fetch_fundamentals(selected_ticker)
        feat = compute_fundamental_features(selected_ticker)

        if raw.is_valid:
            f1, f2, f3, f4, f5 = st.columns(5)

            # PE Ratio
            with f1:
                pe_val = f"{raw.pe_ratio:.1f}" if raw.pe_ratio else "N/A"
                pe_sub = ""
                if feat.pe_zscore is not None:
                    if feat.pe_zscore > 0.5:
                        pe_sub = "Expensive vs peers"
                    elif feat.pe_zscore < -0.5:
                        pe_sub = "Cheap vs peers"
                    else:
                        pe_sub = "Fair vs peers"
                pe_color = COLORS["bearish"] if (feat.pe_zscore or 0) > 0.5 else COLORS["bullish"] if (feat.pe_zscore or 0) < -0.5 else COLORS["primary"]
                metric_card("PE Ratio", pe_val, pe_sub, pe_color)

            # ROE
            with f2:
                roe_val = f"{raw.roe:.1f}%" if raw.roe else "N/A"
                roe_sub = ""
                if feat.roe_percentile is not None:
                    if feat.roe_percentile >= 0.75:
                        roe_sub = "Top quartile"
                    elif feat.roe_percentile >= 0.5:
                        roe_sub = "Above median"
                    else:
                        roe_sub = "Below median"
                roe_color = COLORS["bullish"] if (feat.roe_percentile or 0) >= 0.5 else COLORS["warning"]
                metric_card("ROE", roe_val, roe_sub, roe_color)

            # Debt/Equity
            with f3:
                de_val = f"{raw.debt_to_equity:.2f}" if raw.debt_to_equity is not None else "N/A"
                de_sub = ""
                if feat.de_percentile is not None:
                    if feat.de_percentile >= 0.75:
                        de_sub = "Low debt vs peers"
                    elif feat.de_percentile >= 0.5:
                        de_sub = "Moderate debt"
                    else:
                        de_sub = "High debt vs peers"
                de_color = COLORS["bullish"] if (feat.de_percentile or 0) >= 0.5 else COLORS["bearish"]
                metric_card("Debt/Equity", de_val, de_sub, de_color)

            # EPS Growth
            with f4:
                eps_val = f"{raw.eps_cagr_3y:+.1f}%" if raw.eps_cagr_3y is not None else "N/A"
                if raw.eps_cagr_3y is not None:
                    eps_color = COLORS["bullish"] if raw.eps_cagr_3y > 10 else COLORS["warning"] if raw.eps_cagr_3y > 0 else COLORS["bearish"]
                else:
                    eps_color = COLORS["primary"]
                metric_card("EPS CAGR 3Y", eps_val, color=eps_color)

            # Promoter Holding
            with f5:
                prom_val = f"{raw.promoter_holding:.1f}%" if raw.promoter_holding else "N/A"
                prom_sub = ""
                if raw.promoter_holding_change is not None:
                    prom_sub = f"{raw.promoter_holding_change:+.2f}% QoQ"
                prom_color = COLORS["bullish"] if (raw.promoter_holding_change or 0) >= 0 else COLORS["bearish"]
                metric_card("Promoter", prom_val, prom_sub, prom_color)

            # Sector context
            if raw.sector_pe and raw.pe_ratio:
                ratio = raw.pe_ratio / raw.sector_pe
                if ratio > 1.2:
                    verdict = f"Trading at {ratio:.1f}x sector PE ({raw.sector_pe:.1f}) — premium valuation"
                    v_color = COLORS["warning"]
                elif ratio < 0.8:
                    verdict = f"Trading at {ratio:.1f}x sector PE ({raw.sector_pe:.1f}) — discount valuation"
                    v_color = COLORS["bullish"]
                else:
                    verdict = f"Trading near sector PE ({raw.sector_pe:.1f}) — fair valuation"
                    v_color = COLORS["primary"]
                st.markdown(
                    f'<div style="background:{v_color}10;border-left:4px solid {v_color};'
                    f'border-radius:0 10px 10px 0;padding:10px 16px;margin-top:12px;'
                    f'color:{v_color};font-size:0.85rem">{verdict}</div>',
                    unsafe_allow_html=True,
                )
        else:
            st.caption(f"No fundamental data available for {selected_ticker} on Screener.in")
    except Exception as e:
        st.caption(f"Fundamentals unavailable: {e}")

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

# Avoid duplicate consecutive entries; rolling window of 50
history = st.session_state["search_history"]
if not history or history[-1]["Ticker"] != selected_ticker:
    if len(history) >= _HISTORY_MAX:
        history = history[1:]  # Drop oldest
    history.append(history_entry)
else:
    history[-1] = history_entry  # Update latest for same ticker
st.session_state["search_history"] = history
set_json(_HISTORY_KEY, history)

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
        set_json(_HISTORY_KEY, [])
        st.rerun()
else:
    st.caption("Browse stocks to build your search history.")
