"""Section 3: Volatility Analysis. Implied vs historical (realized) volatility."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.analytics.interpret import iv_vs_rv_sentences
from app.analytics.realized_vol import rolling_realized_vol
from app.analytics.surface import atm_iv_history, term_structure
from app.ui import components as c
from app.ui import state, theme
from app.utils.provenance import DataKind


def _iv_rv_bars(iv: float, rv: pd.Series, label: str) -> go.Figure:
    names = [f"RV {w}d" for w in rv.index] + [label]
    vals = [v * 100 for v in rv.values] + [iv * 100]
    colors = [theme.SERIES[0]] * len(rv) + [theme.SERIES[1]]
    fig = go.Figure(go.Bar(x=names, y=vals, marker=dict(color=colors, cornerradius=4),
                           text=[f"{v:.1f}%" for v in vals], textposition="outside",
                           textfont=dict(family=theme.FONT_MONO, color=theme.TEXT_2),
                           hovertemplate="%{x}: %{y:.2f}%<extra></extra>", width=0.55))
    fig.update_layout(title="Historical volatility vs implied volatility", height=360, showlegend=False,
                      yaxis_title="Annualized volatility (%)", yaxis_ticksuffix="%", hovermode="closest",
                      yaxis_range=[0, max(vals) * 1.25 if vals else 1])
    return fig


def render() -> None:
    ctx, sel = state.require_selection()
    kinds = [DataKind.SYNTHETIC if ctx.synthetic else DataKind.LIVE]
    c.page_header("Implied vs Realized Volatility",
                  "Compares what the options market expects (implied) with what the stock has actually done "
                  "(realized). This comparison is the core question of the platform.", kinds)

    chain = state.selected_chain(ctx, sel)
    row = chain[(chain["option_type"] == sel.option_type) & (chain["strike"].sub(sel.strike).abs() < 1e-9)]
    iv = float(row["computed_iv"].iloc[0]) if not row.empty else float("nan")
    rv = ctx.rv_table.set_index("window_days")["realized_vol"]

    label = f"IV {sel.strike:g}{sel.option_type[0].upper()}"
    left, right = st.columns([3, 2])
    with left:
        st.plotly_chart(_iv_rv_bars(iv, rv, label), use_container_width=True)
        st.caption("Blue bars: realized volatility over different look-back windows (history). "
                   "Orange bar: the selected option's implied volatility (today's market expectation to expiry).")
    with right:
        st.markdown("**What this means**")
        for s in iv_vs_rv_sentences(iv, rv.to_dict(), sel.expiration, ctx.earnings_date):
            st.markdown(f"- {s}")
        if pd.isna(iv):
            c.callout("The selected contract has no usable implied volatility (see Options Chain for the reason). "
                      "Try an at-the-money strike.", "warn")

    # Term structure across expirations
    st.markdown("#### Term structure: ATM implied volatility by expiration")
    exps = state.default_surface_expirations(ctx, max_days=370, max_count=12)
    multi = state.load_multi_chain(ctx.ticker, exps, ctx.spot.price, ctx.rate.value, ctx.dividend.value, ctx.rv30)
    ts = term_structure(multi, ctx.spot.price) if not multi.empty else pd.DataFrame()
    if ts.empty:
        st.info("Not enough liquid quotes to build a term structure.")
    else:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ts["dte"], y=ts["atm_iv"] * 100, mode="lines+markers", name="ATM implied vol",
                                 line=dict(color=theme.SERIES[1], width=2), marker=dict(size=8),
                                 customdata=ts[["expiration", "atm_strike"]],
                                 hovertemplate="%{customdata[0]} (%{x}d)<br>ATM strike %{customdata[1]:,.2f}"
                                               "<br>IV %{y:.2f}%<extra></extra>"))
        if pd.notna(rv.get(30)):
            fig.add_hline(y=rv[30] * 100, line=dict(color=theme.SERIES[0], width=2, dash="dot"),
                          annotation_text=f"30d realized {rv[30]:.1%}", annotation_position="bottom right",
                          annotation_font=dict(color=theme.TEXT_2, size=11))
        if ctx.earnings_date is not None:
            edte = (pd.Timestamp(ctx.earnings_date) - ctx.fetched_at.tz_localize(None).normalize()).days
            fig.add_vline(x=edte, line=dict(color=theme.MUTED, width=1, dash="dash"),
                          annotation_text="earnings", annotation_font=dict(color=theme.TEXT_2, size=11))
        fig.update_layout(height=340, xaxis_title="Days to expiration", yaxis_title="Implied volatility (%)",
                          yaxis_ticksuffix="%", hovermode="closest", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Upward slope (longer-dated IV higher) is typical in calm markets; downward slope often appears "
                   "in stressed markets or right before a known event. A bump at one expiry frequently marks earnings.")

    # Collected IV history, if we have snapshots
    st.markdown("#### Implied volatility history from collected snapshots")
    snaps = state.get_repo().load_snapshots(ctx.ticker)
    hist = atm_iv_history(snaps) if not snaps.empty else pd.DataFrame()
    if hist.empty or len(hist) < 2:
        n = 0 if snaps.empty else snaps["snapshot_date"].nunique()
        c.callout(f"<b>{n} day(s) of {ctx.ticker} snapshots collected.</b> Free data has no historical option "
                  "quotes, so this chart fills in as you run <span class='ovrp-mono'>scripts/collect_snapshot.py</span> "
                  "each trading day (or press 'Collect today's snapshot' on the Backtest page).")
    else:
        prov = DataKind(snaps["provenance"].iloc[-1])
        st.markdown(c.badge(prov), unsafe_allow_html=True)
        rv20 = rolling_realized_vol(ctx.history, 30)
        rv20 = rv20[rv20.index >= hist["date"].min() - pd.Timedelta(days=5)]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hist["date"], y=hist["atm_iv"] * 100, name="~30d ATM IV (collected)",
                                 mode="lines+markers", line=dict(color=theme.SERIES[1], width=2)))
        fig.add_trace(go.Scatter(x=rv20.index, y=rv20 * 100, name="30d realized", mode="lines",
                                 line=dict(color=theme.SERIES[0], width=2)))
        fig.update_layout(height=320, yaxis_title="Annualized volatility (%)", yaxis_ticksuffix="%")
        st.plotly_chart(fig, use_container_width=True)

    c.learn_panel("realized_vol")
    c.learn_panel("implied_vol")
    c.learn_panel("variance_risk_premium")
