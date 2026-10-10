"""Section 2: Options Chain + selected-contract analytics (Black-Scholes, IV, Greeks)."""

from __future__ import annotations


import pandas as pd
import streamlit as st

from app.ui import components as c
from app.ui import learn, state
from app.utils.provenance import DataKind

COLS = {
    "strike": st.column_config.NumberColumn("Strike", format="%.2f", help=learn.GLOSSARY["strike"]),
    "option_type": st.column_config.TextColumn("Type", help=learn.GLOSSARY["option_type"]),
    "bid": st.column_config.NumberColumn("Bid", format="%.2f", help=learn.GLOSSARY["bid"]),
    "ask": st.column_config.NumberColumn("Ask", format="%.2f", help=learn.GLOSSARY["ask"]),
    "mid": st.column_config.NumberColumn("Mid", format="%.3f", help=learn.GLOSSARY["mid"]),
    "last_price": st.column_config.NumberColumn("Last", format="%.2f", help=learn.GLOSSARY["last_price"]),
    "volume": st.column_config.NumberColumn("Volume", format="%d", help=learn.GLOSSARY["volume"]),
    "open_interest": st.column_config.NumberColumn("Open int.", format="%d", help=learn.GLOSSARY["open_interest"]),
    "rel_spread_pct": st.column_config.NumberColumn("Spread %", format="%.1f%%", help=learn.GLOSSARY["rel_spread"]),
    "yahoo_iv_pct": st.column_config.NumberColumn("Yahoo IV", format="%.1f%%", help=learn.GLOSSARY["yahoo_iv"]),
    "computed_iv_pct": st.column_config.NumberColumn("Our IV", format="%.1f%%", help=learn.GLOSSARY["computed_iv"]),
    "delta": st.column_config.NumberColumn("Delta", format="%.3f", help=learn.GLOSSARY["delta"]),
    "dte": st.column_config.NumberColumn("DTE", format="%d", help=learn.GLOSSARY["dte"]),
    "is_liquid": st.column_config.CheckboxColumn("Usable", help=learn.GLOSSARY["is_liquid"]),
    "liquidity_note": st.column_config.TextColumn("Why excluded", help="Reason a quote failed the liquidity filters."),
}


def _contract_panel(row: pd.Series, ctx: state.MarketContext) -> None:
    opt = row["option_type"]
    st.markdown(f"### Selected contract · {ctx.ticker} {row['expiration']} {row['strike']:,.2f} {opt}")
    st.caption(f"{row['contract_symbol']} · {int(row['dte'])} days to expiry · "
               f"T = {row['T']:.4f} years · priced from the {row['price_basis']} price")
    if not row["is_liquid"]:
        c.callout(f"<b>Use with caution:</b> this quote fails the liquidity filters ({row['liquidity_note']}). "
                  "Its implied volatility may not reflect a tradable price.", "warn")

    mkt, model, iv, rv30 = row["price_used"], row["model_price_rv"], row["computed_iv"], ctx.rv30
    a, b, cc = st.columns(3)
    with a:
        c.explain_card("Market option price", c.money(mkt, 3),
                       f"The bid/ask midpoint: what the market currently charges per share "
                       f"(×100 = {c.money(mkt * 100 if pd.notna(mkt) else float('nan'))} per contract).")
    with b:
        c.explain_card("Model price (BS @ 30d realized vol)", c.money(model, 3),
                       f"Black-Scholes value if the stock keeps moving like it did over the last 30 days "
                       f"({c.pct(rv30)} annualized). A reference point, not a 'true' value.")
    with cc:
        yahoo = row["yahoo_iv"]
        c.explain_card("Implied volatility", c.pct(iv),
                       f"The volatility that makes Black-Scholes match the market price. "
                       f"Yahoo's own estimate: {c.pct(yahoo)}. We use ours because we know exactly which price went in.")

    if pd.notna(mkt) and pd.notna(model) and model > 0 and pd.notna(iv):
        gap = mkt / model - 1
        word = "more" if gap > 0 else "less"
        c.callout(f"The market charges <b>{abs(gap):.0%} {word}</b> than the realized-vol model value. "
                  f"Equivalently: the option implies {c.pct(iv)} volatility vs {c.pct(rv30)} recently realized. "
                  f"This is a difference in <i>expectations</i>, not proof of mispricing: the market may be "
                  f"pricing an upcoming event, or the general premium sellers demand for bearing volatility risk.")

    if row["iv_status"] != "ok":
        from app.models.implied_vol import STATUS_MESSAGES
        c.callout(f"<b>No IV for this quote:</b> {STATUS_MESSAGES.get(row['iv_status'], row['iv_status'])}. "
                  "Greeks are not shown because they depend on the volatility.", "warn")
        return

    d, g, v, th = row["delta"], row["gamma"], row["vega"], row["theta"]
    cols = st.columns(4)
    with cols[0]:
        c.explain_card("Delta", c.num(d, 3), learn.delta_sentence(d, opt))
    with cols[1]:
        c.explain_card("Gamma", c.num(g, 4), learn.gamma_sentence(g, d))
    with cols[2]:
        c.explain_card("Vega (per vol pt)", c.num(v, 3), learn.vega_sentence(v))
    with cols[3]:
        c.explain_card("Theta (per day)", c.num(th, 3), learn.theta_sentence(th))
    st.caption("Greeks are evaluated at the option's own implied volatility, which is the market convention.")


def render() -> None:
    ctx, sel = state.require_selection()
    kinds = [DataKind.SYNTHETIC if ctx.synthetic else DataKind.LIVE]
    c.page_header("Options Chain", f"Every listed {ctx.ticker} contract for {sel.expiration}, with our own implied "
                  "volatility and Greeks. Hover a column header for a definition.", kinds)
    chain = state.selected_chain(ctx, sel)
    if chain.empty:
        st.warning("The data source returned an empty chain for this expiration.")
        return

    f1, f2, f3, f4 = st.columns([1.2, 2.4, 1, 1.4])
    show = f1.selectbox("Show", ["Calls", "Puts", "Both"], index=0 if sel.option_type == "call" else 1)
    spot = ctx.spot.price
    kmin, kmax = float(chain["strike"].min()), float(chain["strike"].max())
    lo_default, hi_default = max(kmin, spot * 0.85), min(kmax, spot * 1.15)
    lo, hi = f2.slider("Strike range", kmin, kmax, (lo_default, hi_default))
    liquid_only = f3.checkbox("Usable only", value=False, help=learn.GLOSSARY["is_liquid"])
    search = f4.text_input("Search", placeholder="strike or symbol", label_visibility="visible")

    df = chain[(chain["strike"] >= lo) & (chain["strike"] <= hi)].copy()
    if show != "Both":
        df = df[df["option_type"] == show[:-1].lower()]
    if liquid_only:
        df = df[df["is_liquid"]]
    if search.strip():
        s = search.strip().upper()
        df = df[df["contract_symbol"].str.upper().str.contains(s, regex=False) |
                df["strike"].map(lambda k: f"{k:g}").str.startswith(s)]
    df["rel_spread_pct"] = df["rel_spread"] * 100
    df["yahoo_iv_pct"] = df["yahoo_iv"] * 100
    df["computed_iv_pct"] = df["computed_iv"] * 100

    n_bad = int((~chain["is_liquid"]).sum())
    st.caption(f"{len(df)} contracts shown · spot {spot:,.2f} · {n_bad} of {len(chain)} contracts in this expiry fail "
               f"the liquidity filters and are excluded from the scanner and surface.")
    st.dataframe(df[list(COLS)], column_config=COLS, hide_index=True, use_container_width=True, height=420)

    match = chain[(chain["option_type"] == sel.option_type) & (chain["strike"].sub(sel.strike).abs() < 1e-9)]
    if match.empty:
        st.info("The selected strike is not in this chain.")
    else:
        _contract_panel(match.iloc[0], ctx)

    c.learn_panel("options_basics")
    c.learn_panel("black_scholes")
    c.learn_panel("implied_vol")
    c.learn_panel("greeks")
