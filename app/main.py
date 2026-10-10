"""
Entry point:  streamlit run app/main.py

Builds the sidebar (ticker / expiry / type / strike), a top bar with the
current quote, and the navigation between the nine sections.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make `import app...` work when Streamlit runs this file directly.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

from app import config  # noqa: E402
from app.ui import components, state, theme  # noqa: E402
from app.ui.pages import (  # noqa: E402
    anomaly_scanner,
    backtest,
    market_overview,
    methodology,
    options_chain,
    risk,
    trade_simulator,
    vol_analysis,
    vol_surface,
)

st.set_page_config(page_title=config.APP_NAME, page_icon="◧", layout="wide", initial_sidebar_state="expanded")
theme.inject_css()

state.render_sidebar()
ctx = state.get_context()
if ctx is not None:
    if ctx.synthetic:
        components.synthetic_banner()
    components.topbar(ctx)

pages = {
    "Market": [
        st.Page(market_overview.render, title="Overview", url_path="overview", default=True),
        st.Page(options_chain.render, title="Options Chain", url_path="chain"),
    ],
    "Volatility": [
        st.Page(vol_analysis.render, title="IV vs Realized", url_path="volatility"),
        st.Page(vol_surface.render, title="Surface", url_path="surface"),
        st.Page(anomaly_scanner.render, title="Anomaly Scanner", url_path="scanner"),
    ],
    "Trading": [
        st.Page(trade_simulator.render, title="Trade Simulator", url_path="simulator"),
        st.Page(backtest.render, title="Backtest", url_path="backtest"),
        st.Page(risk.render, title="Risk", url_path="risk"),
    ],
    "Docs": [st.Page(methodology.render, title="Methodology", url_path="methodology")],
}
st.navigation(pages, position="top").run()
