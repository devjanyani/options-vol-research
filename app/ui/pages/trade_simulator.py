"""Section 6: Trade Simulator. A hypothetical what-if on the selected contract (no orders placed)."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from app import config
from app.backtesting.costs import option_leg_cost, stock_hedge_cost
from app.risk.position_risk import Position, expiry_payoff_curve, hedge_shares_for
from app.ui import components as c
from app.ui import state, theme
from app.utils.provenance import DataKind


def render() -> None:
    ctx, sel = state.require_selection()
    kinds = [DataKind.SYNTHETIC if ctx.synthetic else DataKind.LIVE]
    c.page_header("Trade Simulator", "A hypothetical what-if: pick a side, a horizon and a market scenario, and see "
                  "the projected P&L before and after trading costs. Nothing is sent to any broker.", kinds)

    chain = state.selected_chain(ctx, sel)
    m = chain[(chain["option_type"] == sel.option_type) & (chain["strike"].sub(sel.strike).abs() < 1e-9)]
    if m.empty:
        st.info("Selected contract not found.")
        return
    row = m.iloc[0]
    if row["iv_status"] != "ok" or not (row["bid"] > 0 and row["ask"] > 0):
        c.callout("This contract has no two-sided quote or no solvable implied volatility, so a realistic simulation "
                  "is not possible. Pick a more liquid strike (near the money).", "warn")
        return

    st.markdown(f"**Contract:** {ctx.ticker} {sel.expiration} {sel.strike:g} {sel.option_type} · bid "
                f"{row['bid']:.2f} / ask {row['ask']:.2f} · mid {row['mid']:.3f} · IV {row['computed_iv']:.1%} · "
                f"{int(row['dte'])} days to expiry")
    a, b, cc, d = st.columns(4)
    side_label = a.radio("Side", ["Buy", "Sell"], horizontal=True)
    side = 1 if side_label == "Buy" else -1
    contracts = int(b.number_input("Contracts", 1, 100, 1))
    hedge = cc.checkbox("Delta-hedge with stock", value=False,
                        help="Buy/sell shares so the position starts with zero delta: you then mostly trade volatility.")
    max_h = max(int(row["dte"]) - 1, 1)
    horizon = int(d.number_input("Holding days", 1, max_h, min(5, max_h)))
    e, f = st.columns(2)
    move = e.slider("Stock move by the end (%)", -20.0, 20.0, 0.0, 0.5) / 100
    dvol = f.slider("Change in implied volatility (vol pts)", -15.0, 15.0, 0.0, 0.5) / 100

    S, K, T, r, q, iv = ctx.spot.price, sel.strike, float(row["T"]), ctx.rate.value, ctx.dividend.value, float(row["computed_iv"])
    hs = hedge_shares_for(sel.option_type, K, T, side, contracts, S, iv, r, q) if hedge else 0.0
    pos = Position(sel.option_type, K, T, side, contracts, S, iv, r, q, hs)
    st.session_state["sim_position"] = {"key": (ctx.ticker, sel.expiration, sel.option_type, sel.strike), "position": pos, "mid": float(row["mid"]), "bid": float(row["bid"]),
                                        "ask": float(row["ask"]), "label": f"{side_label} {contracts}× {ctx.ticker} "
                                        f"{sel.expiration} {K:g} {sel.option_type}" + (" (delta-hedged)" if hedge else "")}

    T_h = max(T - horizon / config.CALENDAR_DAYS_PER_YEAR, 0.0)
    S_h, iv_h = S * (1 + move), max(iv + dvol, 0.01)
    v0, v_h = pos.value(), pos.value(spot=S_h, iv=iv_h, T=T_h)
    gross = v_h - v0
    cin = option_leg_cost(row["bid"], row["ask"], contracts)
    cout = option_leg_cost(row["bid"], row["ask"], contracts)   # assumption: same $ spread at exit
    hcost = 2 * stock_hedge_cost(hs)
    net = gross - cin.total - cout.total - hcost

    left, right = st.columns([2, 3])
    with left:
        fill = row["ask"] if side > 0 else row["bid"]
        c.explain_card("Fill price you would get", c.money(fill, 2),
                       f"{'Buying pays the ask' if side > 0 else 'Selling receives the bid'}, not the mid "
                       f"({row['mid']:.3f}). Premium {'paid' if side > 0 else 'received'}: "
                       f"{c.money(fill * 100 * contracts)}.")
        c.pnl_stack(gross, [("Bid/ask spread", cin.half_spread + cout.half_spread),
                            ("Slippage", cin.slippage + cout.slippage), ("Commissions & fees", cin.fees + cout.fees),
                            ("Stock hedge costs", hcost)], net)
        st.caption(f"Scenario: after {horizon} days, stock {move:+.1%} to {S_h:,.2f}, IV {iv:.1%} → {iv_h:.1%}. "
                   "Assumes the exit spread equals today's spread (it can widen in volatile markets).")
        if hedge:
            st.caption(f"Hedge: {hs:+,.0f} shares of {ctx.ticker} at {S:,.2f}.")

    with right:
        grid = np.linspace(S * 0.8, S * 1.2, 161)
        at_h = [pos.value(spot=x, iv=iv_h, T=T_h) - v0 for x in grid]
        at_exp = expiry_payoff_curve(pos, float(row["mid"]), grid)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=grid, y=at_h, name=f"In {horizon} days (IV {iv_h:.1%})", mode="lines",
                                 line=dict(color=theme.SERIES[0], width=2)))
        fig.add_trace(go.Scatter(x=grid, y=at_exp, name="At expiration", mode="lines",
                                 line=dict(color=theme.SERIES[1], width=2, dash="dash")))
        fig.add_hline(y=0, line=dict(color=theme.MUTED, width=1))
        fig.add_vline(x=S, line=dict(color=theme.MUTED, width=1, dash="dot"), annotation_text="today",
                      annotation_font=dict(color=theme.TEXT_2, size=11))
        fig.update_layout(title="Gross P&L vs stock price (before costs)", height=380, xaxis_title=f"{ctx.ticker} price",
                          yaxis_title="P&L ($)", yaxis_tickprefix="$", hovermode="x unified")
        fig.update_traces(hovertemplate="%{y:$,.0f}")
        st.plotly_chart(fig, use_container_width=True)
    c.callout("Hypothetical simulation using Black-Scholes with your assumptions. Not a forecast or a recommendation. "
              "The Risk page shows the Greeks and stress scenarios for this same position.")
    c.learn_panel("greeks")
    c.learn_panel("costs")
