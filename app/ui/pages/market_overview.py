"""Section 1: Market Overview. Underlying price and the volatility environment."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.analytics.chain import atm_iv
from app.analytics.realized_vol import rolling_realized_vol
from app.ui import components as c
from app.ui import learn, state, theme
from app.utils.dates import market_now
from app.utils.provenance import DataKind


def nearest_expiry(expirations: list[str], target_days: int = 30) -> str:
    today = market_now().date()
    future = [e for e in expirations if (pd.Timestamp(e).date() - today).days >= 7] or expirations
    return min(future, key=lambda e: abs((pd.Timestamp(e).date() - today).days - target_days))


def environment_sentence(iv: float, rv: float) -> str:
    if not (pd.notna(iv) and pd.notna(rv)) or rv <= 0:
        return "Not enough clean data to compare implied and realized volatility right now."
    ratio = iv / rv
    if ratio > 1.25:
        return (f"The options market is pricing <b>noticeably more</b> movement ({iv:.1%}) than the stock has "
                f"recently delivered ({rv:.1%} over 30 days). That is common before scheduled events or after a calm "
                f"stretch, and it is also the normal state of markets on average (see 'Why implied vol is usually "
                f"above realized vol'). It is not by itself evidence of mispricing.")
    if ratio < 0.9:
        return (f"Options are pricing <b>less</b> movement ({iv:.1%}) than the stock recently delivered "
                f"({rv:.1%}). This often happens right after a volatile spike, when traders expect things to calm down.")
    return (f"Implied ({iv:.1%}) and recent realized volatility ({rv:.1%}) are broadly in line: the market expects "
            f"roughly the kind of movement the stock has been showing.")


def render() -> None:
    ctx = state.require_context()
    kinds = [DataKind.SYNTHETIC if ctx.synthetic else DataKind.LIVE]
    c.page_header("Market Overview", "Where the stock is, how much it has been moving, and how much movement the "
                  "options market is pricing for the next month.", kinds)

    exp30 = nearest_expiry(ctx.expirations)
    try:
        chain30 = state.load_enriched_chain(ctx.ticker, exp30, ctx.spot.price, ctx.rate.value, ctx.dividend.value, ctx.rv30)
        iv30 = atm_iv(chain30, ctx.spot.price)
    except Exception as exc:  # noqa: BLE001
        st.warning(f"Could not load the {exp30} chain for ATM IV: {exc}")
        iv30 = float("nan")
    rv = ctx.rv_table.set_index("window_days")["realized_vol"]
    rv30 = rv.get(30, float("nan"))

    cols = st.columns(6)
    cols[0].metric("Spot", f"{ctx.spot.price:,.2f}", help=learn.GLOSSARY["spot"])
    cols[1].metric("ATM IV ~30d", c.pct(iv30), help=f"Implied volatility of the at-the-money option expiring {exp30} "
                   "(the expiration closest to 30 days). " + learn.GLOSSARY["iv"])
    cols[2].metric("Realized 30d", c.pct(rv30), help=learn.GLOSSARY["rv"])
    ratio = iv30 / rv30 if pd.notna(iv30) and pd.notna(rv30) and rv30 > 0 else float("nan")
    cols[3].metric("IV / RV", c.num(ratio, 2) + "×", help="Implied divided by 30-day realized volatility. Above 1 = "
                   "options price more movement than recently delivered.")
    cols[4].metric("Risk-free rate", c.pct(ctx.rate.value, 2), help=f"Source: {ctx.rate.source}")
    cols[5].metric("Dividend yield", c.pct(ctx.dividend.value, 2), help=f"Source: {ctx.dividend.source}")

    c.callout(environment_sentence(iv30, rv30))

    left, right = st.columns([3, 2])
    with left:
        hist = ctx.history.iloc[-252:]
        fig = go.Figure(go.Scatter(x=hist.index, y=hist.values, mode="lines", name="Close",
                                   line=dict(color=theme.BLUE, width=2),
                                   hovertemplate="%{x|%b %d, %Y}<br>Close %{y:,.2f}<extra></extra>"))
        fig.update_layout(title=f"{ctx.ticker} · 1-year price (adjusted close)", height=340, showlegend=False,
                          yaxis_title="Price", hovermode="x")
        st.plotly_chart(fig, use_container_width=True)
    with right:
        tbl = ctx.rv_table.copy()
        tbl["window"] = tbl["window_days"].astype(str) + " trading days"
        tbl["realized vol"] = tbl["realized_vol"].map(lambda v: c.pct(v))
        st.markdown("**Realized volatility by window**")
        st.dataframe(tbl[["window", "realized vol"]], hide_index=True, use_container_width=True)
        st.caption("Short windows react fast but are noisy; long windows are stable but slow. "
                   "All annualized with √252.")

    roll20 = rolling_realized_vol(ctx.history, 20).iloc[-252:]
    roll60 = rolling_realized_vol(ctx.history, 60).iloc[-252:]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=roll20.index, y=roll20 * 100, name="Realized 20d", mode="lines",
                             line=dict(color=theme.SERIES[0], width=2)))
    fig.add_trace(go.Scatter(x=roll60.index, y=roll60 * 100, name="Realized 60d", mode="lines",
                             line=dict(color=theme.SERIES[1], width=2)))
    if pd.notna(iv30):
        fig.add_hline(y=iv30 * 100, line=dict(color=theme.SERIES[2], width=2, dash="dash"),
                      annotation_text=f"Today's ATM IV ({exp30}) {iv30:.1%}", annotation_position="top left",
                      annotation_font=dict(color=theme.TEXT_2, size=11))
    fig.update_layout(title="Rolling realized volatility vs today's implied volatility", height=340,
                      yaxis_title="Annualized volatility (%)", yaxis_ticksuffix="%")
    fig.update_traces(hovertemplate="%{y:.1f}%")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Realized lines are history. The dashed line is a single number observed today and is drawn across "
               "the chart only as a reference level.")

    c.learn_panel("realized_vol")
    c.learn_panel("variance_risk_premium")
    c.learn_panel("options_basics")
