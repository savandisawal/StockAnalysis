"""Reusable Plotly chart components — polished dark theme with vibrant colors."""

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# Shared layout defaults
_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(14,17,23,0.8)",
    font=dict(family="Inter, sans-serif", color="#C0C0C0"),
    xaxis=dict(gridcolor="#1E2028", zerolinecolor="#2D3139"),
    yaxis=dict(gridcolor="#1E2028", zerolinecolor="#2D3139"),
    showlegend=False,
)


def candlestick_chart(
    df: pd.DataFrame,
    prediction: dict | None = None,
    title: str = "",
    height: int = 600,
) -> go.Figure:
    """Candlestick chart with volume and prediction overlay."""
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.75, 0.25],
    )

    # Candlestick — green/red with glow
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name="OHLC",
            increasing=dict(line=dict(color="#00E676"), fillcolor="rgba(0,230,118,0.19)"),
            decreasing=dict(line=dict(color="#FF5252"), fillcolor="rgba(255,82,82,0.19)"),
        ),
        row=1,
        col=1,
    )

    # Volume bars with gradient colors
    colors = [
        "rgba(0,230,118,0.45)" if c >= o else "rgba(255,82,82,0.45)"
        for c, o in zip(df["Close"], df["Open"])
    ]
    fig.add_trace(
        go.Bar(x=df.index, y=df["Volume"], name="Volume", marker_color=colors),
        row=2,
        col=1,
    )

    # Prediction overlay — glowing band
    if prediction and prediction.get("predicted_low"):
        fig.add_hrect(
            y0=prediction["predicted_low"],
            y1=prediction["predicted_high"],
            fillcolor="rgba(108,99,255,0.12)",
            line=dict(color="rgba(108,99,255,0.4)", width=1),
            row=1,
            col=1,
        )
        fig.add_hline(
            y=prediction["predicted_mid"],
            line_dash="dash",
            line_color="#BB86FC",
            line_width=2,
            annotation_text=f"  Target: Rs.{prediction['predicted_mid']:,.0f}",
            annotation_font_color="#BB86FC",
            annotation_font_size=11,
            row=1,
            col=1,
        )

    fig.update_layout(
        **_LAYOUT,
        height=height,
        title=dict(text=title, font=dict(size=16, color="#E0E0E0"), x=0.01),
        xaxis_rangeslider_visible=False,
        margin=dict(l=50, r=20, t=45, b=20),
        xaxis2=dict(gridcolor="#1E2028"),
        yaxis2=dict(gridcolor="#1E2028"),
    )

    return fig


def indicator_chart(
    df: pd.DataFrame,
    indicator: str,
    title: str = "",
    color: str = "#BB86FC",
    height: int = 220,
) -> go.Figure:
    """Line chart with gradient fill for a single indicator."""
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df[indicator],
            mode="lines",
            name=indicator,
            line=dict(color=color, width=2.5),
            fill="tozeroy",
            fillcolor=f"rgba({int(color[1:3], 16)},{int(color[3:5], 16)},{int(color[5:7], 16)},0.08)",
        )
    )

    # RSI zones
    if "rsi" in indicator.lower():
        fig.add_hrect(y0=70, y1=100, fillcolor="rgba(255,82,82,0.06)", line_width=0)
        fig.add_hrect(y0=0, y1=30, fillcolor="rgba(0,230,118,0.06)", line_width=0)
        fig.add_hline(y=70, line_dash="dot", line_color="rgba(255,82,82,0.38)", line_width=1)
        fig.add_hline(y=30, line_dash="dot", line_color="rgba(0,230,118,0.38)", line_width=1)
        fig.add_hline(y=50, line_dash="dot", line_color="rgba(120,144,156,0.19)", line_width=1)

    # ADX reference
    if "adx" in indicator.lower():
        fig.add_hline(
            y=25,
            line_dash="dot",
            line_color="rgba(255,215,64,0.38)",
            line_width=1,
            annotation_text="Trend threshold",
            annotation_font_color="#FFD740",
            annotation_font_size=10,
        )

    fig.update_layout(
        **_LAYOUT,
        title=dict(text=title, font=dict(size=13, color="#B0B0B0"), x=0.01),
        height=height,
        margin=dict(l=50, r=20, t=35, b=20),
    )

    return fig


def mae_over_time_chart(df: pd.DataFrame, height: int = 380) -> go.Figure:
    """Rolling MAE with gradient fill and trend line."""
    if df.empty or "actual_change" not in df.columns:
        return go.Figure()

    df = df.copy()
    df["error"] = abs(df["actual_change"] - df["predicted_p50"])
    df["rolling_mae"] = df["error"].rolling(window=20, min_periods=5).mean()

    fig = go.Figure()

    # Individual errors as faded dots
    fig.add_trace(
        go.Scatter(
            x=df["pred_date"],
            y=df["error"],
            mode="markers",
            name="Daily Error",
            marker=dict(color="#6C63FF", size=3, opacity=0.25),
        )
    )

    # Rolling MAE line
    fig.add_trace(
        go.Scatter(
            x=df["pred_date"],
            y=df["rolling_mae"],
            mode="lines",
            name="Rolling MAE (20-day)",
            line=dict(color="#FF9100", width=2.5),
            fill="tozeroy",
            fillcolor="rgba(255,145,0,0.08)",
        )
    )

    fig.update_layout(
        **_LAYOUT,
        title=dict(text="Prediction Error Over Time", font=dict(size=15, color="#E0E0E0"), x=0.01),
        yaxis_title="MAE (%)",
        height=height,
        margin=dict(l=50, r=20, t=45, b=20),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, font=dict(size=10)),
    )

    return fig


def prediction_vs_actual_chart(df: pd.DataFrame, height: int = 420) -> go.Figure:
    """Predicted vs actual with glowing P10-P90 range."""
    if df.empty:
        return go.Figure()

    fig = go.Figure()

    # P10-P90 range band
    fig.add_trace(
        go.Scatter(
            x=list(df["pred_date"]) + list(df["pred_date"])[::-1],
            y=list(df["predicted_p90"]) + list(df["predicted_p10"])[::-1],
            fill="toself",
            fillcolor="rgba(108,99,255,0.1)",
            line=dict(color="rgba(108,99,255,0)"),
            name="P10-P90 Range",
            hoverinfo="skip",
        )
    )

    # Predicted P50 line
    fig.add_trace(
        go.Scatter(
            x=df["pred_date"],
            y=df["predicted_p50"],
            mode="lines",
            name="Predicted (P50)",
            line=dict(color="#BB86FC", width=2, dash="dot"),
        )
    )

    # Actual dots — color by direction correct
    colors = ["#00E676" if dc else "#FF5252" for dc in df["direction_correct"]]
    fig.add_trace(
        go.Scatter(
            x=df["pred_date"],
            y=df["actual_change"],
            mode="markers",
            name="Actual",
            marker=dict(color=colors, size=6, line=dict(width=1, color="rgba(255,255,255,0.13)")),
        )
    )

    # Zero line
    fig.add_hline(y=0, line_dash="solid", line_color="rgba(120,144,156,0.19)", line_width=1)

    fig.update_layout(
        **_LAYOUT,
        title=dict(text="Predicted vs Actual Changes", font=dict(size=15, color="#E0E0E0"), x=0.01),
        yaxis_title="% Change",
        height=height,
        margin=dict(l=50, r=20, t=45, b=20),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, font=dict(size=10)),
    )

    return fig


def shap_bar_chart(explanation: dict, height: int = 340) -> go.Figure:
    """Horizontal bar chart of top SHAP contributors for one prediction.

    Positive SHAP (pushes prediction up) in green, negative in red.
    Each bar is annotated with the feature's actual input value.
    """
    if not explanation or not explanation.get("top_features"):
        return go.Figure()

    feats = list(reversed(explanation["top_features"]))  # largest on top
    labels = [f["feature"] for f in feats]
    shap_vals = [f["shap"] for f in feats]
    hover = [
        f"{f['feature']}: value={f['value']:+.3f}, contribution={f['shap']:+.4f}%" for f in feats
    ]
    colors = ["#00E676" if v > 0 else "#FF5252" for v in shap_vals]

    fig = go.Figure(
        go.Bar(
            x=shap_vals,
            y=labels,
            orientation="h",
            marker_color=colors,
            marker_line=dict(width=0),
            hovertext=hover,
            hoverinfo="text",
            text=[f"{f['value']:+.2f}" for f in feats],
            textposition="outside",
            textfont=dict(size=10, color="#9E9E9E"),
        )
    )

    fig.add_vline(x=0, line_color="rgba(120,144,156,0.4)", line_width=1)

    fig.update_layout(
        **_LAYOUT,
        title=dict(
            text="Why this prediction — SHAP contributions to P50 (% change)",
            font=dict(size=14, color="#E0E0E0"),
            x=0.01,
        ),
        xaxis_title="Contribution to predicted % change",
        height=height,
        margin=dict(l=140, r=40, t=45, b=35),
    )

    return fig


def macro_gauge(name: str, value: float | None, height: int = 150) -> go.Figure:
    """Compact gauge for a macro metric."""
    if value is None:
        value = 0.0

    color = "#00E676" if value > 0 else "#FF5252" if value < 0 else "#78909C"

    fig = go.Figure(
        go.Indicator(
            mode="number+delta",
            value=value,
            number={"suffix": "%", "font": {"size": 32, "color": color, "family": "Inter"}},
            title={"text": name, "font": {"size": 13, "color": "#9E9E9E"}},
        )
    )

    fig.update_layout(
        **_LAYOUT,
        height=height,
        margin=dict(l=10, r=10, t=45, b=10),
    )

    return fig
