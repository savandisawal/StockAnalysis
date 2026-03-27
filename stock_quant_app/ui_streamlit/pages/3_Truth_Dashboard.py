"""Truth Dashboard -- backtest results, accuracy tracking, honesty metrics."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st
import pandas as pd

st.set_page_config(page_title="Truth Dashboard", page_icon="Q", layout="wide")

from ui_streamlit.components.styles import (
    inject_global_css, styled_header, metric_card, section_header, COLORS,
)
from model.backtest import get_backtest_history, get_backtest_summary, run_backtest
from ui_streamlit.components.charts import mae_over_time_chart, prediction_vs_actual_chart

inject_global_css()

ticker = st.session_state.get("selected_ticker", "RELIANCE")
styled_header("Truth Dashboard", f"How accurate are the predictions for {ticker}?")

# ── Run Backtest ─────────────────────────────────────────────────

section_header("Run Backtest")

col_btn, col_years = st.columns([1, 1])
with col_years:
    bt_years = st.slider("Backtest years", 1, 3, 2, key="bt_years")
with col_btn:
    if st.button("Run Walk-Forward Backtest", type="primary", use_container_width=True):
        with st.status(f"Backtesting {ticker}...", expanded=True) as status:
            st.write(f"Building {bt_years}Y feature matrix...")
            try:
                metrics = run_backtest(
                    ticker, years=bt_years, min_train_days=200, retrain_every=10,
                    include_fundamentals=False, include_macro=False,
                )
                st.write(f"Predictions: {metrics.total_predictions}")
                st.write(f"MAE: {metrics.mae:.4f}%")
                st.write(f"Direction accuracy: {metrics.direction_accuracy:.1%}")
                status.update(label="Backtest complete!", state="complete")
            except Exception as e:
                status.update(label=f"Failed: {e}", state="error")

# ── Load History ─────────────────────────────────────────────────

try:
    history = get_backtest_history(ticker, limit=500)
except Exception as e:
    st.error(f"Failed to load backtest history: {e}")
    st.stop()

if history.empty:
    st.markdown(f'<div style="background:#6C63FF10;border:1px solid #6C63FF30;border-radius:16px;padding:40px;text-align:center;margin:32px 0"><div style="font-size:1.5rem;color:#6C63FF;margin-bottom:8px">No backtest results yet</div><div style="color:#78909C">Click "Run Walk-Forward Backtest" above to generate accuracy metrics for {ticker}</div></div>', unsafe_allow_html=True)
    st.stop()

# ── Summary Metrics ──────────────────────────────────────────────

try:
    summaries = get_backtest_summary(ticker)
except Exception:
    summaries = []
if summaries:
    latest = summaries[0]
    section_header("Performance Metrics")

    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        metric_card("MAE", f"{latest['mae']:.4f}%", color=COLORS["orange"])
    with m2:
        metric_card("RMSE", f"{latest['rmse']:.4f}%", color=COLORS["pink"])
    with m3:
        acc = latest["direction_accuracy"]
        acc_color = COLORS["bullish"] if acc > 0.55 else COLORS["warning"] if acc > 0.45 else COLORS["bearish"]
        metric_card("Direction Acc.", f"{acc:.1%}", color=acc_color)
    with m4:
        cov = latest["interval_coverage"]
        cov_color = COLORS["bullish"] if cov > 0.7 else COLORS["warning"] if cov > 0.5 else COLORS["bearish"]
        metric_card("80% Coverage", f"{cov:.1%}", color=cov_color)
    with m5:
        metric_card("Predictions", f"{latest['total_predictions']}", color=COLORS["purple"])

    # Model grade
    if acc > 0.58 and cov > 0.7:
        grade, grade_color, grade_desc = "A", COLORS["bullish"], "Strong model -- reliable for decision support"
    elif acc > 0.52 and cov > 0.6:
        grade, grade_color, grade_desc = "B", COLORS["accent"], "Good model -- useful with human oversight"
    elif acc > 0.48:
        grade, grade_color, grade_desc = "C", COLORS["warning"], "Moderate -- consider retraining with more data"
    else:
        grade, grade_color, grade_desc = "D", COLORS["bearish"], "Weak -- needs significant improvement"
    st.markdown(f'<div style="background:{grade_color}10;border:2px solid {grade_color}40;border-radius:16px;padding:20px;text-align:center;margin:20px 0"><div style="display:flex;align-items:center;justify-content:center;gap:16px"><div style="font-size:3rem;font-weight:900;color:{grade_color};width:70px;height:70px;border-radius:50%;border:3px solid {grade_color};display:flex;align-items:center;justify-content:center">{grade}</div><div style="text-align:left"><div style="font-size:1.1rem;font-weight:700;color:#E0E0E0">Model Grade: {grade}</div><div style="color:#9E9E9E;font-size:0.9rem">{grade_desc}</div></div></div></div>', unsafe_allow_html=True)

# ── Charts ───────────────────────────────────────────────────────

section_header("Visual Analysis")

history = history.sort_values("pred_date")

tab_scatter, tab_mae, tab_table = st.tabs(["Predicted vs Actual", "Error Trend", "Raw Data"])

with tab_scatter:
    fig = prediction_vs_actual_chart(history)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('<div style="display:flex;gap:24px;justify-content:center;margin-top:8px;font-size:0.8rem"><div><span style="color:#00E676">&#9679;</span> <span style="color:#9E9E9E">Correct direction</span></div><div><span style="color:#FF5252">&#9679;</span> <span style="color:#9E9E9E">Wrong direction</span></div><div><span style="color:#BB86FC">&#9671;</span> <span style="color:#9E9E9E">Predicted (P50)</span></div><div><span style="color:#6C63FF30">&#9632;</span> <span style="color:#9E9E9E">P10-P90 range</span></div></div>', unsafe_allow_html=True)

with tab_mae:
    fig = mae_over_time_chart(history)
    st.plotly_chart(fig, use_container_width=True)

with tab_table:
    display_df = history.copy()
    display_df["error"] = abs(display_df["actual_change"] - display_df["predicted_p50"])
    display_df = display_df.rename(columns={
        "pred_date": "Date", "actual_change": "Actual %",
        "predicted_p10": "P10 %", "predicted_p50": "P50 %",
        "predicted_p90": "P90 %", "confidence": "Confidence",
        "direction_correct": "Dir OK", "error": "Error %",
    })
    st.dataframe(
        display_df[["Date", "Actual %", "P10 %", "P50 %", "P90 %", "Confidence", "Dir OK", "Error %"]]
        .sort_values("Date", ascending=False),
        use_container_width=True, hide_index=True,
    )

# ── Cross-Stock Comparison ───────────────────────────────────────

st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
with st.expander("All Backtested Stocks"):
    all_summaries = get_backtest_summary()
    if all_summaries:
        summary_df = pd.DataFrame(all_summaries)
        cols = ["ticker", "mae", "rmse", "direction_accuracy", "interval_coverage", "total_predictions", "run_date"]
        available = [c for c in cols if c in summary_df.columns]
        st.dataframe(summary_df[available].sort_values("direction_accuracy", ascending=False), use_container_width=True, hide_index=True)
    else:
        st.caption("No backtest results yet.")
