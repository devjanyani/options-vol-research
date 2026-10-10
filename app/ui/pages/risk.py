"""Section 8: Risk dashboard. Greeks, stress scenarios, maximum loss, drawdown, position size."""

from __future__ import annotations

import math

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from app.risk.position_risk import Position, max_loss_description, position_greeks, scenario_grid
from app.ui import components as c
from app.ui import state, theme
from app.utils.provenance import DataKind


def _default_position(ctx, sel):
    chain = state.selected_chain(ctx, sel)
    m = chain[(chain["option_type"] == sel.option_type) & (chain["strike"].sub(sel.strike).abs() < 1e-9)]
    if m.empty or m.iloc[0]["iv_status"] != "ok":
        return None
    row = m.iloc[0]
    pos = Position(sel.option_type, sel.strike, float(row["T"]), 1, 1, ctx.spot.price, float(row["computed_iv"]),
                   ctx.rate.value, ctx.dividend.value)
    return {"position": pos, "mid": float(row["mid"]), "bid": float(row["bid"]), "ask": float(row["ask"]), "label": f"Buy 1× {ctx.ticker} {sel.expiration} {sel.strike:g} "
            f"{sel.option_type} (default: set a position on the Trade Simulator page)"}


def render() -> None:
    ctx, sel = state.require_selection()
    kinds = [DataKind.SYNTHETIC if ctx.synthetic else DataKind.LIVE]
    c.page_header("Risk", "How the position reacts to the stock, to volatility and to time, and how bad things can get.",
                  kinds)
    sim = st.session_state.get("sim_position")
    if not sim or sim.get("key") != (ctx.ticker, sel.expiration, sel.option_type, sel.strike):
        sim = _default_position(ctx, sel)
    if sim is None:
        st.info("The selected contract has no usable implied volatility. Pick another strike.")
        return
    p: Position = sim["position"]
    g = position_greeks(p)
    st.markdown(f"**Position:** {sim['label']}")

    cols = st.columns(4)
    with cols[0]:
        c.explain_card("Delta (share-equivalent)", f"{g['delta_shares']:+,.1f} sh",
                       f"The position currently moves like {g['delta_shares']:+,.0f} shares of {ctx.ticker} "
                       f"(≈ {c.money(g['dollar_delta'], 0)} of stock). A $1 rise in the stock changes P&L by about "
                       f"{c.money(g['delta_shares'])}.")
    with cols[1]:
        c.explain_card("Gamma", f"{g['gamma_shares']:+,.2f} sh/$",
                       f"For each $1 the stock moves, the position's delta changes by {g['gamma_shares']:+,.2f} shares. "
                       "Positive gamma (long options) gains from big moves; negative gamma (short options) is hurt by them.")
    with cols[2]:
        c.explain_card("Vega", f"{c.money(g['vega_dollars'])}/pt",
                       f"Vega measures how much the option's value changes when implied volatility changes. If IV rises "
                       f"1 point, this position makes {c.money(g['vega_dollars'])}.")
    with cols[3]:
        c.explain_card("Theta", f"{c.money(g['theta_dollars'])}/day",
                       f"Time decay: each calendar day that passes changes the position's value by "
                       f"{c.money(g['theta_dollars'])} if nothing else moves.")

    # Premium actually exchanged: buyers pay the ask, sellers receive the bid (mid if unknown).
    premium_ps = sim.get("ask" if p.side > 0 else "bid", sim["mid"])
    ml, why = max_loss_description(p, premium_ps)
    cols = st.columns(4)
    with cols[0]:
        txt = "Unlimited" if ml == -math.inf else c.money(ml, 0)
        c.explain_card("Maximum loss (theoretical)", txt, why)
    max_days = max(int(p.T * 365) - 1, 0)
    days = st.slider("Stress-test horizon (days forward)", 0, max_days, min(5, max_days), key="risk_days") if max_days else 0
    grid = scenario_grid(p, days_forward=days)
    worst = float(grid.to_numpy().min())
    with cols[1]:
        c.explain_card("Maximum simulated loss", c.money(worst, 0),
                       f"Worst P&L across the stress grid below (stock −20% to +20%, IV −10 to +10 pts, {days} days "
                       "forward). Real losses can exceed this in a gap or crash.")
    with cols[2]:
        c.explain_card("Underlying volatility", c.pct(ctx.rv30),
                       f"{ctx.ticker}'s 30-day realized volatility. A typical one-day move is about "
                       f"{c.pct(ctx.rv30 / np.sqrt(252), 2)} (annual vol ÷ √252).")
    with cols[3]:
        c.explain_card("Position size", f"{p.contracts} contract{'s' if p.contracts > 1 else ''}",
                       f"Premium {c.money(abs(g['premium']), 0)} · controls {p.contracts * 100:,} shares "
                       f"(notional ≈ {c.money(g['option_notional'], 0)}). Small premiums can carry large notional exposure.")

    st.markdown("#### Stress test: P&L under stock and volatility shocks")
    lim = max(abs(grid.to_numpy()).max(), 1.0)
    z = grid.to_numpy()
    fig = go.Figure(go.Heatmap(
        z=z, x=[f"{m:+.0%}" for m in grid.columns], y=[f"{v * 100:+.0f} pts" for v in grid.index],
        colorscale=[[0, "#b8302f"], [0.25, "#e66767"], [0.5, "#383835"], [0.75, "#3987e5"], [1, "#184f95"]],
        zmin=-lim, zmax=lim, xgap=2, ygap=2, text=[[c.money(v, 0) for v in r] for r in z], texttemplate="%{text}",
        textfont=dict(family=theme.FONT_MONO, size=11, color=theme.TEXT),
        colorbar=dict(title=dict(text="P&L $"), tickprefix="$"),
        hovertemplate="Stock %{x}, IV %{y}<br>P&L %{text}<extra></extra>"))
    fig.update_layout(height=340, xaxis_title="Stock price move", yaxis_title="Implied vol change", hovermode="closest",
                      xaxis_type="category", yaxis_type="category")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Red = loss, blue = gain, gray ≈ unchanged. Each cell fully re-prices the position with Black-Scholes, "
               "so it captures gamma and vega effects that a single Greek misses.")

    st.markdown("#### Drawdown from backtests")
    last = st.session_state.get("last_backtest")
    if last:
        n = last["net"]
        st.markdown(f"{c.badge(last['kind'])} {last['title']}", unsafe_allow_html=True)
        cc = st.columns(3)
        cc[0].metric("Max drawdown (net)", c.money(n["max_drawdown"], 0))
        cc[1].metric("Worst single trade (net)", c.money(n["worst_trade"], 0))
        cc[2].metric("Trades", f"{n['trades']:,}")
        st.caption("From the most recent backtest you ran on the Backtest page. Historical simulation, not live results.")
    else:
        st.info("Run a backtest on the Backtest page to see its drawdown here.")
    c.learn_panel("greeks")
    c.learn_panel("sharpe_drawdown")
