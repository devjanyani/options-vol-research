"""Section 9: Methodology. How everything works, in one place. Needs no market data."""

from __future__ import annotations

import streamlit as st

from app import config
from app.ui import components as c
from app.ui import learn
from app.utils.provenance import DESCRIPTIONS, DataKind


def render() -> None:
    c.page_header("Methodology", "Exactly how every number in this app is produced, what is assumed, and where it can "
                  "be wrong. Written for someone learning quantitative finance.")

    st.markdown("### The question")
    st.markdown("**Are some options unusually expensive or cheap compared with what their recent market behavior "
                "would suggest?** The app answers in stages: get quotes → solve each option's implied volatility → "
                "compare with realized volatility and with neighboring options → flag unusual ones → explain them → "
                "test simple trading rules historically, after costs.")

    st.markdown("### Data and its labels")
    for k in (DataKind.LIVE, DataKind.COLLECTED, DataKind.MODELED):
        st.markdown(f"{c.badge(k)} {DESCRIPTIONS[k]}", unsafe_allow_html=True)
    st.markdown(
        "- Source: Yahoo Finance via `yfinance` (free, ~15-minute delayed, sometimes rate-limited, no SLA).\n"
        "- Risk-free rate: 13-week T-bill yield (^IRX). If unavailable, an assumed "
        f"{config.FALLBACK_RISK_FREE_RATE:.1%} is used and the UI says so.\n"
        "- Dividend yield: trailing 12-month dividends ÷ price, computed from the dividend history.\n"
        "- No historical option quotes are available for free. The app never fabricates them: it collects its own "
        "daily snapshots, and separately runs a clearly labeled **modeled** backtest from SPY and VIX.")

    st.markdown("### 1 · Cleaning quotes")
    lf = config.SETTINGS.liquidity
    st.markdown(f"Each option is priced at its **bid/ask midpoint**. A quote is *usable* only if: bid ≥ "
                f"${lf.min_bid:.2f}, spread ≤ {lf.max_relative_spread:.0%} of mid, open interest ≥ "
                f"{lf.min_open_interest}, at least {lf.min_dte} days to expiry, and its IV can be solved. Unusable "
                "quotes are shown in the chain (with the reason) but excluded from the surface, scanner and backtests.")

    st.markdown("### 2 · Black-Scholes and implied volatility")
    st.latex(r"d_1=\frac{\ln(S/K)+(r-q+\tfrac12\sigma^2)T}{\sigma\sqrt T},\qquad d_2=d_1-\sigma\sqrt T")
    st.latex(r"C=Se^{-qT}N(d_1)-Ke^{-rT}N(d_2),\qquad P=Ke^{-rT}N(-d_2)-Se^{-qT}N(-d_1)")
    st.markdown("Time *T* is measured in calendar years to the 4:00 pm ET close on expiry. Implied volatility is the σ "
                "that makes the formula equal the mid price, found with Brent's root-finding method between 0.01% "
                "and 500%. Prices outside no-arbitrage bounds, or with under half a cent of time value, get **no IV** "
                "(with a reason) rather than a forced number.")
    c.learn_panel("black_scholes")
    c.learn_panel("implied_vol")

    st.markdown("### 3 · Greeks")
    st.latex(r"\Delta_C=e^{-qT}N(d_1),\quad \Gamma=\frac{e^{-qT}n(d_1)}{S\sigma\sqrt T},\quad "
             r"\text{Vega}=Se^{-qT}n(d_1)\sqrt T,\quad \Theta=\frac{\partial V}{\partial t}")
    st.markdown("Vega is reported per 1 volatility point, Theta per calendar day. Greeks use the option's own IV. "
                "Tests check them against finite differences and Hull's textbook examples.")
    c.learn_panel("greeks")

    st.markdown("### 4 · Realized volatility")
    st.latex(r"\sigma_{\text{realized}}=\sqrt{252}\cdot\operatorname{stdev}\big(\ln(P_t/P_{t-1})\big)_{\text{last }N}")
    st.markdown("Computed for N = 10, 20, 30 and 60 trading days from split- and dividend-adjusted closes.")
    c.learn_panel("realized_vol")
    c.learn_panel("variance_risk_premium")

    st.markdown("### 5 · Volatility surface")
    st.markdown("Out-of-the-money options only (puts below spot, calls above). Strikes are bucketed to a 'nice' spacing "
                "for display; empty cells are left blank, never interpolated.")
    c.learn_panel("surface")

    st.markdown("### 6 · Anomaly score")
    a = config.SETTINGS.anomaly
    st.latex(r"z=\frac{\text{IV}-\text{comparison}}{\max(1.4826\cdot\text{MAD},\;0.5\text{ pts})},\qquad "
             r"\text{score}=100\cdot\min\!\Big(\frac{\sum w_i|z_i|}{\sum w_i\cdot 4},1\Big)")
    st.markdown(
        f"- **Nearby strikes ({a.weight_strike_neighbors:.0%})**: trend line through up to {a.neighbor_count} strikes "
        "on each side (self excluded), read at this strike.\n"
        f"- **Neighboring expirations ({a.weight_expiry_neighbors:.0%})**: compares smile *shape* (IV minus ATM IV) "
        "with the adjacent expirations' smoothed smiles at the same moneyness.\n"
        f"- **Rest of chain ({a.weight_rest_of_chain:.0%})**: robust z-score of log(IV ÷ 30-day realized vol) across "
        "all scored options. Realized vol is identical for every option on a stock, so it cannot separate them; the "
        "IV − RV gap is shown as context, not as the main signal.\n"
        f"- Flag threshold: {a.flag_threshold:.0f}. A flag is a **potential volatility anomaly**, never a guaranteed mispricing.")
    c.learn_panel("anomaly")

    st.markdown("### 7 · AI explanations")
    st.markdown(f"Optional. The model ({config.LLM_MODEL}) receives only a fact sheet of numbers computed by the app and "
                "is told not to add numbers. Its answer is then checked: if it contains **any** number that is not in "
                "the fact sheet, it is discarded and the rule-based explanation is shown with a note. AI text is always "
                "labeled. Without an API key the app uses rule-based explanations only.")

    st.markdown("### 8 · Backtests")
    st.markdown(
        "**Collected-snapshot backtest (real quotes).** Walk forward through stored daily snapshots; on each day score "
        "anomalies using only that day's data; sell 'rich' / buy 'cheap' options (top 3 per day), optional one-time "
        "delta hedge; exit after 5/10/20 business days at that day's stored quote. Trades without an exit quote are "
        "listed as unresolved, never estimated.\n\n"
        "**Modeled proxy backtest (SPY + VIX).** Every *h* trading days: if VIX − 20-day realized vol ≥ threshold, "
        "sell a 30-day at-the-money SPY straddle priced with Black-Scholes at VIX, delta-hedged; re-price at exit with "
        "the new SPY price and VIX. Compared against an 'always sell' baseline and shown across a full sensitivity "
        "grid, so no single favorable setting is cherry-picked.\n\n"
        "**No look-ahead** is enforced by construction and checked by tests that truncate the data or inject a future "
        "shock and verify that earlier decisions do not change.")
    c.learn_panel("backtest")

    st.markdown("### 9 · Costs and metrics")
    cm = config.SETTINGS.costs
    st.markdown(f"Net P&L = gross P&L at mid − half the bid/ask spread on entry and on exit − slippage "
                f"({cm.slippage_fraction_of_spread:.0%} of the spread per fill) − ${cm.option_fee_per_contract:.2f} per "
                f"contract per fill − ${cm.stock_cost_per_share:.2f}/share for hedges. The modeled backtest assumes a "
                f"spread of {cm.modeled_spread_fraction:.0%} of the option price. All editable in the app.")
    st.markdown("Return per trade = P&L ÷ premium. P&L is booked at exit; Sharpe = mean ÷ stdev of daily booked P&L × √252 "
                "(optimistic, because in-trade swings are not marked). Max drawdown = largest fall from a running peak "
                "of cumulative booked P&L.")
    c.learn_panel("costs")
    c.learn_panel("sharpe_drawdown")

    st.markdown("### Known limitations")
    st.markdown(
        "- Free, delayed data; quotes outside market hours are often stale or zero.\n"
        "- European Black-Scholes applied to American options; no discrete dividends.\n"
        "- Anomalies may be explained by events, borrow costs, dividends or bad quotes.\n"
        "- The collected backtest starts empty and needs weeks of daily collection to mean anything.\n"
        "- The modeled backtest uses VIX (not ATM IV), no term structure, assumed costs, and a static hedge.\n"
        "- Nothing here is investment advice, and simulated results are not actual returns.")

    st.markdown("### Glossary")
    st.dataframe({"Term": list(learn.GLOSSARY), "Meaning": list(learn.GLOSSARY.values())}, hide_index=True,
                 use_container_width=True)
