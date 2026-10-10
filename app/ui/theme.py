"""
Visual theme: a dark, minimal research-terminal look.

Colors are defined once here and reused by CSS and Plotly so every chart and
panel matches. Number-heavy text uses a monospaced font so columns of
figures line up like a terminal.
"""

from __future__ import annotations

import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

# Surfaces and ink
BG = "#0f1012"
PANEL = "#17181a"
PANEL_2 = "#1d1e21"
BORDER = "#2a2b2f"
TEXT = "#e8e6e1"
TEXT_2 = "#b4b2aa"
MUTED = "#7d7c76"

# Categorical series (fixed order, dark-mode steps of a CVD-validated palette)
SERIES = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#9085e9"]
BLUE, ORANGE, AQUA, YELLOW = SERIES[:4]

# Status colors: always paired with an arrow/icon/label, never color alone.
GOOD = "#0ca30c"
WARN = "#fab219"
CRIT = "#d03b3b"

# Sequential (one hue, dark -> bright on a dark surface) for IV magnitude
SEQ_BLUE = ["#0d366b", "#184f95", "#256abf", "#3987e5", "#6da7ec", "#9ec5f4", "#cde2fb"]
# Diverging (cheap = blue, neutral gray midpoint, rich = red)
DIVERGING = [[0.0, "#184f95"], [0.25, "#3987e5"], [0.5, "#383835"], [0.75, "#e66767"], [1.0, "#b8302f"]]

PROVENANCE_COLORS = {
    "LIVE": GOOD,
    "COLLECTED": BLUE,
    "MODELED": WARN,
    "SYNTHETIC": CRIT,
}

FONT_SANS = "'IBM Plex Sans', system-ui, sans-serif"
FONT_MONO = "'IBM Plex Mono', ui-monospace, monospace"

_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600&display=swap');

html, body, [class*="css"], .stMarkdown, .stText, p, li, label {{
  font-family: {FONT_SANS};
}}
.stApp {{ background: {BG}; }}
section[data-testid="stSidebar"] {{ background: {PANEL}; border-right: 1px solid {BORDER}; }}
header[data-testid="stHeader"] {{ background: transparent; }}
.block-container {{ padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1400px; }}

h1, h2, h3, h4 {{ font-family: {FONT_SANS}; font-weight: 600; letter-spacing: -0.01em; color: {TEXT}; }}
h1 {{ font-size: 1.45rem !important; }}
h2 {{ font-size: 1.15rem !important; }}
h3 {{ font-size: 1.08rem !important; margin-top: 0.6rem; }}

/* Metric tiles */
div[data-testid="stMetric"] {{
  background: {PANEL}; border: 1px solid {BORDER}; border-radius: 6px; padding: 10px 14px;
}}
div[data-testid="stMetricLabel"] p {{
  font-family: {FONT_MONO}; font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.06em; color: {MUTED};
}}
div[data-testid="stMetricValue"] {{ font-family: {FONT_MONO}; font-size: 1.35rem; color: {TEXT}; }}
div[data-testid="stMetricDelta"] {{ font-family: {FONT_MONO}; }}

/* Tables */
div[data-testid="stDataFrame"] {{ border: 1px solid {BORDER}; border-radius: 6px; }}

/* Expanders as quiet "learn" panels */
div[data-testid="stExpander"] details {{ background: {PANEL}; border: 1px solid {BORDER}; border-radius: 6px; }}
div[data-testid="stExpander"] summary p {{ font-size: 0.85rem; color: {TEXT_2}; }}

/* Tabs */
button[data-baseweb="tab"] p {{ font-family: {FONT_MONO}; font-size: 0.8rem; letter-spacing: 0.03em; }}

.ovrp-topbar {{
  display:flex; align-items:center; justify-content:space-between; gap:16px; flex-wrap:wrap;
  border-bottom: 1px solid {BORDER}; padding: 0 0 10px 0; margin-bottom: 14px;
}}
.ovrp-brand {{ font-family:{FONT_MONO}; font-size:0.78rem; color:{MUTED}; letter-spacing:0.08em; text-transform:uppercase; }}
.ovrp-quote {{ font-family:{FONT_MONO}; display:flex; gap:18px; align-items:baseline; flex-wrap:wrap; }}
.ovrp-quote .tk {{ font-size:1.25rem; font-weight:600; color:{TEXT}; }}
.ovrp-quote .px {{ font-size:1.25rem; color:{TEXT}; }}
.ovrp-quote .chg-up {{ color:{GOOD}; }}
.ovrp-quote .chg-dn {{ color:{CRIT}; }}
.ovrp-quote .meta {{ font-size:0.75rem; color:{MUTED}; }}

.ovrp-badge {{
  display:inline-block; font-family:{FONT_MONO}; font-size:0.68rem; font-weight:600; letter-spacing:0.06em;
  padding:2px 8px; border-radius:3px; border:1px solid; margin-right:6px; vertical-align:middle;
}}
.ovrp-page-title {{ display:flex; align-items:center; gap:10px; flex-wrap:wrap; margin: 4px 0 2px 0; }}
.ovrp-page-title h1 {{ margin:0; padding:0; }}
.ovrp-sub {{ color:{TEXT_2}; font-size:0.9rem; margin: 0 0 14px 0; max-width: 900px; }}

.ovrp-card {{ background:{PANEL}; border:1px solid {BORDER}; border-radius:6px; padding:14px 16px; margin-bottom:12px; }}
.ovrp-card .label {{ font-family:{FONT_MONO}; font-size:0.72rem; text-transform:uppercase; letter-spacing:0.06em; color:{MUTED}; }}
.ovrp-card .value {{ font-family:{FONT_MONO}; font-size:1.3rem; color:{TEXT}; margin:2px 0 6px 0; }}
.ovrp-card .explain {{ font-size:0.84rem; color:{TEXT_2}; line-height:1.45; }}

.ovrp-callout {{ border-left:3px solid {BLUE}; background:{PANEL}; padding:10px 14px; border-radius:0 6px 6px 0; margin:8px 0 14px 0; color:{TEXT_2}; font-size:0.9rem; }}
.ovrp-callout.warn {{ border-left-color:{WARN}; }}
.ovrp-callout.crit {{ border-left-color:{CRIT}; }}
.ovrp-callout.ai {{ border-left-color:#9085e9; }}
.ovrp-callout b {{ color:{TEXT}; }}

.ovrp-synthetic {{
  background:{CRIT}; color:#fff; font-family:{FONT_MONO}; font-weight:600; letter-spacing:0.05em;
  padding:8px 14px; border-radius:4px; margin-bottom:12px; text-align:center;
}}
.ovrp-mono {{ font-family:{FONT_MONO}; }}
.ovrp-pnl-stack {{ font-family:{FONT_MONO}; background:{PANEL}; border:1px solid {BORDER}; border-radius:6px; padding:12px 16px; }}
.ovrp-pnl-stack .row {{ display:flex; justify-content:space-between; gap:16px; padding:3px 0; color:{TEXT_2}; font-size:0.88rem; }}
.ovrp-pnl-stack .row span:last-child {{ white-space:nowrap; }}
.ovrp-pnl-stack .row.total {{ border-top:1px solid {BORDER}; margin-top:6px; padding-top:8px; color:{TEXT}; font-weight:600; }}
</style>
"""


def inject_css() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)


def _register_plotly_template() -> None:
    axis = dict(
        gridcolor="#24252a", zerolinecolor="#34353a", linecolor=BORDER,
        tickfont=dict(family=FONT_MONO, size=11, color=MUTED),
        title=dict(font=dict(family=FONT_SANS, size=12, color=TEXT_2)),
    )
    tmpl = go.layout.Template(
        layout=go.Layout(
            paper_bgcolor=PANEL, plot_bgcolor=PANEL,
            font=dict(family=FONT_SANS, color=TEXT_2, size=12),
            colorway=SERIES,
            xaxis=axis, yaxis=axis,
            legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=11, color=TEXT_2), orientation="h",
                        yanchor="bottom", y=1.01, xanchor="right", x=1),
            margin=dict(l=56, r=20, t=64, b=48),
            hoverlabel=dict(bgcolor=PANEL_2, bordercolor=BORDER, font=dict(family=FONT_MONO, size=12, color=TEXT)),
            title=dict(font=dict(family=FONT_SANS, size=14, color=TEXT), x=0, xanchor="left"),
            hovermode="x unified",
        )
    )
    pio.templates["ovrp"] = tmpl
    pio.templates.default = "ovrp"


_register_plotly_template()
