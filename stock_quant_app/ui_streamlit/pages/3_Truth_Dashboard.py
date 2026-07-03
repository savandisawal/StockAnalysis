"""Truth Dashboard -- honest accuracy tracking.

Two views, clearly separated:
- Live predictions (served): real predictions the system actually served,
  scored against what happened the next day. This is the ground truth.
- Backtest (simulated): walk-forward simulation on history. Useful, but
  it is a simulation — labeled as such.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st
import pandas as pd

st.set_page_config(page_title="Truth Dashboard", page_icon="Q", layout="wide")

from ui_streamlit.components.styles import (
    inject_global_css,
    styled_header,
    metric_card,
    section_header,
    COLORS,
)
from model.backtest import get_backtest_history, get_backtest_summary, run_backtest
from services.prediction_store import backfill_outcomes, get_live_accuracy, get_live_history
from ui_streamlit.components.charts import mae_over_time_chart, prediction_vs_actual_chart

inject_global_css()

ticker = st.session_state.get("selected_ticker", "RELIANCE")
styled_header("Truth Dashboard", f"How accurate are the predictions for {ticker}?")

tab_live, tab_backtest = st.tabs(["Live predictions (served)", "Backtest (simulated)"])


# ═══════════════ Live predictions (served) ═══════════════════════

with tab_live:
    st.caption(
        "Real predictions this system served (API, UI, or scheduler), scored "
        "against the actual next-day close. No simulation."
    )

    col_refresh, _ = st.columns([1, 3])
    with col_refresh:
        if st.button("Backfill outcomes now", use_container_width=True):
            with st.spinner("Fetching actual closes..."):
                n = backfill_outcomes([ticker])
                st.success(f"Resolved {n} prediction(s)")

    try:
        live = get_live_history(ticker, limit=250)
    except Exception as e:
        st.error(f"Failed to load live predictions: {e}")
        live = pd.DataFrame()

    if live.empty:
        st.markdown(
            '<div style="background:#6C63FF10;border:1px solid #6C63FF30;border-radius:16px;'
            'padding:40px;text-align:center;margin:32px 0">'
            '<div style="font-size:1.5rem;color:#6C63FF;margin-bottom:8px">No served predictions yet</div>'
            f'<div style="color:#78909C">Predictions for {ticker} are recorded automatically '
            "whenever one is served (Active Analysis page, API, or the daily scheduler). "
            "Outcomes fill in after the next trading day.</div></div>",
            unsafe_allow_html=True,
        )
    else:
        resolved = live.dropna(subset=["actual_change"])

        try:
            acc_rows = get_live_accuracy(ticker)
        except Exception:
            acc_rows = []

        if acc_rows:
            acc = acc_rows[0]
            section_header("Live Performance (last 60 resolved)")
            m1, m2, m3, m4, m5 = st.columns(5)
            with m1:
                metric_card("MAE", f"{acc['mae']:.4f}%", color=COLORS["orange"])
            with m2:
                metric_card("Pinball P50", f"{acc['pinball_p50']:.4f}", color=COLORS["pink"])
            with m3:
                da = acc["direction_accuracy"]
                da_color = (
                    COLORS["bullish"]
                    if da > 0.55
                    else COLORS["warning"]
                    if da > 0.45
                    else COLORS["bearish"]
                )
                metric_card("Direction Acc.", f"{da:.1%}", color=da_color)
            with m4:
                cov = acc["coverage_80"]
                cov_color = (
                    COLORS["bullish"]
                    if cov > 0.7
                    else COLORS["warning"]
                    if cov > 0.5
                    else COLORS["bearish"]
                )
                metric_card("80% Coverage", f"{cov:.1%}", color=cov_color)
            with m5:
                metric_card("Resolved", f"{acc['n']}", color=COLORS["purple"])
        else:
            st.info(
                f"{len(live)} prediction(s) recorded, none resolved yet — outcomes "
                "appear after the next trading day (or click Backfill outcomes)."
            )

        if not resolved.empty:
            section_header("Predicted vs Actual (served)")
            chart_df = resolved.rename(
                columns={
                    "prediction_date": "pred_date",
                    "p10": "predicted_p10",
                    "p50": "predicted_p50",
                    "p90": "predicted_p90",
                }
            ).sort_values("pred_date")
            st.plotly_chart(prediction_vs_actual_chart(chart_df), use_container_width=True)

        with st.expander("All served predictions"):
            show = live.copy()
            show = show.rename(
                columns={
                    "prediction_date": "Date",
                    "p10": "P10 %",
                    "p50": "P50 %",
                    "p90": "P90 %",
                    "actual_change": "Actual %",
                    "confidence": "Confidence",
                    "direction_correct": "Dir OK",
                    "in_range": "In Range",
                    "model_version": "Model",
                    "source": "Source",
                }
            )
            cols = [
                "Date",
                "P10 %",
                "P50 %",
                "P90 %",
                "Actual %",
                "Confidence",
                "Dir OK",
                "In Range",
                "Source",
                "Model",
            ]
            st.dataframe(
                show[[c for c in cols if c in show.columns]],
                use_container_width=True,
                hide_index=True,
            )


# ═══════════════ Backtest (simulated) ════════════════════════════

with tab_backtest:
    st.caption(
        "Walk-forward simulation: models retrained on expanding history with "
        "CQR calibration, predicting one day ahead. A simulation — not served "
        "predictions."
    )

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
                        ticker,
                        years=bt_years,
                        min_train_days=200,
                        retrain_every=10,
                        include_fundamentals=True,
                        include_macro=True,
                    )
                    st.write(f"Predictions: {metrics.total_predictions}")
                    st.write(f"MAE: {metrics.mae:.4f}%")
                    st.write(f"Pinball P50: {metrics.pinball_p50:.4f}")
                    st.write(f"Direction accuracy: {metrics.direction_accuracy:.1%}")
                    status.update(label="Backtest complete!", state="complete")
                except Exception as e:
                    status.update(label=f"Failed: {e}", state="error")

    # ── Load History ─────────────────────────────────────────────

    try:
        history = get_backtest_history(ticker, limit=500)
    except Exception as e:
        st.error(f"Failed to load backtest history: {e}")
        history = pd.DataFrame()

    if history.empty:
        st.markdown(
            '<div style="background:#6C63FF10;border:1px solid #6C63FF30;border-radius:16px;'
            'padding:40px;text-align:center;margin:32px 0">'
            '<div style="font-size:1.5rem;color:#6C63FF;margin-bottom:8px">No backtest results yet</div>'
            f'<div style="color:#78909C">Click "Run Walk-Forward Backtest" above to generate '
            f"accuracy metrics for {ticker}</div></div>",
            unsafe_allow_html=True,
        )
        st.stop()

    # ── Summary Metrics ──────────────────────────────────────────

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
            acc_color = (
                COLORS["bullish"]
                if acc > 0.55
                else COLORS["warning"]
                if acc > 0.45
                else COLORS["bearish"]
            )
            metric_card("Direction Acc.", f"{acc:.1%}", color=acc_color)
        with m4:
            cov = latest["interval_coverage"]
            cov_color = (
                COLORS["bullish"]
                if cov > 0.7
                else COLORS["warning"]
                if cov > 0.5
                else COLORS["bearish"]
            )
            metric_card("80% Coverage", f"{cov:.1%}", color=cov_color)
        with m5:
            metric_card("Predictions", f"{latest['total_predictions']}", color=COLORS["purple"])

        # Model grade
        if acc > 0.58 and cov > 0.7:
            grade, grade_color, grade_desc = (
                "A",
                COLORS["bullish"],
                "Strong model -- reliable for decision support",
            )
        elif acc > 0.52 and cov > 0.6:
            grade, grade_color, grade_desc = (
                "B",
                COLORS["accent"],
                "Good model -- useful with human oversight",
            )
        elif acc > 0.48:
            grade, grade_color, grade_desc = (
                "C",
                COLORS["warning"],
                "Moderate -- consider retraining with more data",
            )
        else:
            grade, grade_color, grade_desc = (
                "D",
                COLORS["bearish"],
                "Weak -- needs significant improvement",
            )
        st.markdown(
            f'<div style="background:{grade_color}10;border:2px solid {grade_color}40;border-radius:16px;padding:20px;text-align:center;margin:20px 0"><div style="display:flex;align-items:center;justify-content:center;gap:16px"><div style="font-size:3rem;font-weight:900;color:{grade_color};width:70px;height:70px;border-radius:50%;border:3px solid {grade_color};display:flex;align-items:center;justify-content:center">{grade}</div><div style="text-align:left"><div style="font-size:1.1rem;font-weight:700;color:#E0E0E0">Model Grade: {grade}</div><div style="color:#9E9E9E;font-size:0.9rem">{grade_desc}</div></div></div></div>',
            unsafe_allow_html=True,
        )

    # ── Charts ───────────────────────────────────────────────────

    section_header("Visual Analysis")

    history = history.sort_values("pred_date")

    tab_scatter, tab_mae, tab_table = st.tabs(["Predicted vs Actual", "Error Trend", "Raw Data"])

    with tab_scatter:
        fig = prediction_vs_actual_chart(history)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(
            '<div style="display:flex;gap:24px;justify-content:center;margin-top:8px;font-size:0.8rem"><div><span style="color:#00E676">&#9679;</span> <span style="color:#9E9E9E">Correct direction</span></div><div><span style="color:#FF5252">&#9679;</span> <span style="color:#9E9E9E">Wrong direction</span></div><div><span style="color:#BB86FC">&#9671;</span> <span style="color:#9E9E9E">Predicted (P50)</span></div><div><span style="color:#6C63FF30">&#9632;</span> <span style="color:#9E9E9E">P10-P90 range</span></div></div>',
            unsafe_allow_html=True,
        )

    with tab_mae:
        fig = mae_over_time_chart(history)
        st.plotly_chart(fig, use_container_width=True)

    with tab_table:
        display_df = history.copy()
        display_df["error"] = abs(display_df["actual_change"] - display_df["predicted_p50"])
        display_df = display_df.rename(
            columns={
                "pred_date": "Date",
                "actual_change": "Actual %",
                "predicted_p10": "P10 %",
                "predicted_p50": "P50 %",
                "predicted_p90": "P90 %",
                "confidence": "Confidence",
                "direction_correct": "Dir OK",
                "error": "Error %",
            }
        )
        st.dataframe(
            display_df[
                ["Date", "Actual %", "P10 %", "P50 %", "P90 %", "Confidence", "Dir OK", "Error %"]
            ].sort_values("Date", ascending=False),
            use_container_width=True,
            hide_index=True,
        )

    # ── Cross-Stock Comparison ───────────────────────────────────

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    with st.expander("All Backtested Stocks"):
        all_summaries = get_backtest_summary()
        if all_summaries:
            summary_df = pd.DataFrame(all_summaries)
            cols = [
                "ticker",
                "mae",
                "rmse",
                "pinball_p50",
                "direction_accuracy",
                "interval_coverage",
                "total_predictions",
                "run_date",
            ]
            available = [c for c in cols if c in summary_df.columns]
            st.dataframe(
                summary_df[available].sort_values("direction_accuracy", ascending=False),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.caption("No backtest results yet.")
