"""Shared CSS styles and UI helpers for the Streamlit app."""

import streamlit as st

# ── Color Palette ────────────────────────────────────────────────

COLORS = {
    "primary": "#6C63FF",
    "accent": "#00D4AA",
    "bullish": "#00E676",
    "bearish": "#FF5252",
    "warning": "#FFD740",
    "neutral": "#78909C",
    "card_bg": "#1E2028",
    "card_border": "#2D3139",
    "surface": "#161921",
    "gold": "#FFD700",
    "purple": "#BB86FC",
    "cyan": "#18FFFF",
    "orange": "#FF9100",
    "pink": "#FF4081",
}


def inject_global_css():
    """Inject global custom CSS into the Streamlit page."""
    css = """<style>
.stApp {background: linear-gradient(180deg, #0E1117 0%, #131620 50%, #0E1117 100%);}
section[data-testid="stSidebar"] {background: linear-gradient(180deg, #12141D 0%, #1A1D26 100%);border-right: 1px solid #2D3139;}
div[data-testid="stMetric"] {background: linear-gradient(135deg, #1E2028 0%, #252830 100%);border: 1px solid #2D3139;border-radius: 12px;padding: 16px 20px;box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);transition: transform 0.2s, box-shadow 0.2s;}
div[data-testid="stMetric"]:hover {transform: translateY(-2px);box-shadow: 0 6px 20px rgba(108, 99, 255, 0.15);border-color: #6C63FF40;}
div[data-testid="stMetric"] label {color: #9E9E9E !important;font-size: 0.8rem !important;text-transform: uppercase;letter-spacing: 0.5px;}
div[data-testid="stMetric"] div[data-testid="stMetricValue"] {font-size: 1.5rem !important;font-weight: 700 !important;}
.stTabs [data-baseweb="tab-list"] {gap: 8px;background: #1A1D23;border-radius: 12px;padding: 4px;}
.stTabs [data-baseweb="tab"] {border-radius: 8px;padding: 8px 20px;color: #9E9E9E;font-weight: 500;}
.stTabs [aria-selected="true"] {background: linear-gradient(135deg, #6C63FF 0%, #5B54E6 100%) !important;color: white !important;}
div[data-testid="stExpander"] {background: #1A1D23;border: 1px solid #2D3139;border-radius: 12px;}
div[data-testid="stDataFrame"] {border-radius: 12px;overflow: hidden;}
.stButton > button[kind="primary"] {background: linear-gradient(135deg, #6C63FF 0%, #5B54E6 100%);border: none;border-radius: 10px;font-weight: 600;transition: all 0.3s;}
.stButton > button[kind="primary"]:hover {background: linear-gradient(135deg, #7B73FF 0%, #6C63FF 100%);box-shadow: 0 4px 15px rgba(108, 99, 255, 0.4);}
hr {border-color: #2D3139 !important;margin: 1.5rem 0 !important;}
.js-plotly-plot {border-radius: 12px;overflow: hidden;}
</style>"""
    st.markdown(css, unsafe_allow_html=True)


def styled_header(title: str, subtitle: str = "", icon: str = ""):
    """Render a styled page header with gradient text."""
    icon_html = f'<span style="font-size:2rem;margin-right:12px">{icon}</span>' if icon else ""
    sub_html = f'<p style="color:#9E9E9E;font-size:0.95rem;margin-top:4px">{subtitle}</p>' if subtitle else ""
    html = f'<div style="margin-bottom:24px"><div style="display:flex;align-items:center">{icon_html}<h1 style="background:linear-gradient(90deg,#6C63FF,#00D4AA);-webkit-background-clip:text;-webkit-text-fill-color:transparent;font-size:2.2rem;font-weight:800;margin:0">{title}</h1></div>{sub_html}</div>'
    st.markdown(html, unsafe_allow_html=True)


def metric_card(label: str, value: str, delta: str = "", color: str = "#6C63FF", icon: str = ""):
    """Render a custom styled metric card with color accent."""
    delta_color = COLORS["bullish"] if delta.startswith("+") else COLORS["bearish"] if delta.startswith("-") else COLORS["neutral"]
    delta_html = f'<span style="color:{delta_color};font-size:0.9rem;font-weight:600">{delta}</span>' if delta else ""
    icon_html = f'<span style="font-size:1.3rem;margin-right:8px">{icon}</span>' if icon else ""
    html = f'<div style="background:linear-gradient(135deg,#1E2028,#252830);border:1px solid #2D3139;border-radius:14px;padding:18px 22px;border-left:4px solid {color};box-shadow:0 4px 15px rgba(0,0,0,0.25);margin-bottom:8px"><div style="color:#9E9E9E;font-size:0.75rem;text-transform:uppercase;letter-spacing:1px;margin-bottom:6px">{icon_html}{label}</div><div style="font-size:1.6rem;font-weight:700;color:#FFFFFF;margin-bottom:2px">{value}</div>{delta_html}</div>'
    st.markdown(html, unsafe_allow_html=True)


def confidence_badge(confidence: float, direction: str):
    """Render a large confidence + direction badge."""
    if direction == "Bullish":
        bg, border, dir_color, arrow = "linear-gradient(135deg, #00E67620, #00E67605)", "#00E676", COLORS["bullish"], "^"
    elif direction == "Bearish":
        bg, border, dir_color, arrow = "linear-gradient(135deg, #FF525220, #FF525205)", "#FF5252", COLORS["bearish"], "v"
    else:
        bg, border, dir_color, arrow = "linear-gradient(135deg, #78909C20, #78909C05)", "#78909C", COLORS["neutral"], "-"
    conf_color = COLORS["bullish"] if confidence > 60 else COLORS["warning"] if confidence > 35 else COLORS["bearish"]
    html = f'<div style="background:{bg};border:2px solid {border};border-radius:16px;padding:20px 28px;text-align:center;box-shadow:0 4px 20px rgba(0,0,0,0.3)"><div style="font-size:0.8rem;color:#9E9E9E;text-transform:uppercase;letter-spacing:1.5px;margin-bottom:8px">Confidence</div><div style="font-size:2.8rem;font-weight:800;color:{conf_color};line-height:1">{confidence}%</div><div style="font-size:1.1rem;font-weight:700;color:{dir_color};margin-top:8px">{arrow} {direction}</div></div>'
    st.markdown(html, unsafe_allow_html=True)


def prediction_range_card(low: float, mid: float, high: float, change_pct: float):
    """Render a visual prediction range bar."""
    change_color = COLORS["bullish"] if change_pct > 0 else COLORS["bearish"] if change_pct < 0 else COLORS["neutral"]
    html = f'<div style="background:linear-gradient(135deg,#1E2028,#252830);border:1px solid #2D3139;border-radius:16px;padding:24px;box-shadow:0 4px 20px rgba(0,0,0,0.3)"><div style="color:#9E9E9E;font-size:0.75rem;text-transform:uppercase;letter-spacing:1.5px;margin-bottom:16px">Predicted Price Range</div><div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px"><div style="text-align:center"><div style="color:#FF5252;font-size:0.7rem;text-transform:uppercase">Low (P10)</div><div style="color:#FF8A80;font-size:1.3rem;font-weight:700">Rs.{low:,.2f}</div></div><div style="text-align:center"><div style="color:#6C63FF;font-size:0.7rem;text-transform:uppercase">Mid (P50)</div><div style="color:#FFFFFF;font-size:1.8rem;font-weight:800">Rs.{mid:,.2f}</div><div style="color:{change_color};font-size:0.95rem;font-weight:600">{change_pct:+.2f}%</div></div><div style="text-align:center"><div style="color:#00E676;font-size:0.7rem;text-transform:uppercase">High (P90)</div><div style="color:#69F0AE;font-size:1.3rem;font-weight:700">Rs.{high:,.2f}</div></div></div><div style="background:#2D3139;border-radius:8px;height:8px;position:relative;margin-top:8px"><div style="background:linear-gradient(90deg,#FF5252,#6C63FF,#00E676);border-radius:8px;height:100%;width:100%;opacity:0.7"></div></div></div>'
    st.markdown(html, unsafe_allow_html=True)


def macro_indicator_card(name: str, price: float | None, change: float | None, icon: str = ""):
    """Render a colored macro indicator card."""
    if change is None:
        change = 0.0
    price_str = f"{price:,.2f}" if price is not None else "N/A"
    if change > 0:
        change_color, bg_accent, border_color = COLORS["bullish"], "rgba(0, 230, 118, 0.06)", "rgba(0, 230, 118, 0.3)"
    elif change < 0:
        change_color, bg_accent, border_color = COLORS["bearish"], "rgba(255, 82, 82, 0.06)", "rgba(255, 82, 82, 0.3)"
    else:
        change_color, bg_accent, border_color = COLORS["neutral"], "rgba(120, 144, 156, 0.06)", "rgba(120, 144, 156, 0.3)"
    icon_html = f'<span style="font-size:1.5rem;margin-right:8px">{icon}</span>' if icon else ""
    html = f'<div style="background:{bg_accent};border:1px solid {border_color};border-radius:16px;padding:20px;text-align:center;box-shadow:0 4px 15px rgba(0,0,0,0.2);min-height:160px;display:flex;flex-direction:column;justify-content:center"><div style="font-size:0.8rem;color:#9E9E9E;text-transform:uppercase;letter-spacing:1px;margin-bottom:8px">{icon_html}{name}</div><div style="font-size:1.6rem;font-weight:700;color:#FFFFFF">{price_str}</div><div style="font-size:1.3rem;font-weight:700;color:{change_color};margin-top:4px">{change:+.2f}%</div></div>'
    st.markdown(html, unsafe_allow_html=True)


def section_header(title: str, icon: str = ""):
    """Render a styled section header."""
    icon_html = f'<span style="margin-right:8px">{icon}</span>' if icon else ""
    html = f'<div style="margin:24px 0 16px 0"><h3 style="color:#E0E0E0;font-weight:700;font-size:1.3rem;margin:0">{icon_html}{title}</h3><div style="width:40px;height:3px;background:linear-gradient(90deg,#6C63FF,#00D4AA);border-radius:2px;margin-top:6px"></div></div>'
    st.markdown(html, unsafe_allow_html=True)
