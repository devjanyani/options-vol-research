"""
Turns a raw option chain into an analysis-ready table.

For every contract we add:
- days to expiry and time in years
- mid price, spread, relative spread
- moneyness (strike / spot) and log-moneyness ln(K/S)
- a liquidity flag with the reason a quote was rejected
- OUR OWN implied volatility (solved from the mid price)
- Greeks evaluated at that implied volatility
- a "model price" = Black-Scholes price using recent REALIZED volatility

Why compute IV ourselves instead of using Yahoo's IV column?
Yahoo's number is computed from the last trade price, which can be hours or
days old, and it is often obviously wrong outside market hours. Solving from
the current bid/ask midpoint is reproducible and we know exactly what went in.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from app import config
from app.models import black_scholes as bs
from app.models.implied_vol import implied_volatility
from app.utils.dates import days_to_expiry, time_to_expiry_years


def _liquidity_reason(row, lf: config.LiquidityFilter) -> str:
    if row["price_basis"] != "mid":
        return "no two-sided quote"
    if row["bid"] < lf.min_bid:
        return f"bid < ${lf.min_bid:.2f}"
    if row["rel_spread"] > lf.max_relative_spread:
        return f"spread > {lf.max_relative_spread:.0%} of mid"
    if row["open_interest"] < lf.min_open_interest:
        return f"open interest < {lf.min_open_interest}"
    if row["dte"] < lf.min_dte:
        return f"expires in < {lf.min_dte} days"
    if row["iv_status"] != "ok":
        return "IV could not be solved"
    return ""


def enrich_chain(
    chain: pd.DataFrame,
    spot: float,
    r: float,
    q: float,
    as_of: pd.Timestamp | None = None,
    model_vol: float | None = None,
    liquidity: config.LiquidityFilter = config.SETTINGS.liquidity,
) -> pd.DataFrame:
    """
    `model_vol` is the volatility used for the "model price" column. We pass
    30-day realized volatility, so the model price answers: "what would this
    option be worth if the stock keeps moving the way it recently has?"
    """
    if chain is None or chain.empty:
        return pd.DataFrame()
    df = chain.copy()
    as_of_date = (as_of.date() if as_of is not None else None)
    df["dte"] = [days_to_expiry(e, as_of_date) for e in df["expiration"]]
    df["T"] = [time_to_expiry_years(e, as_of) for e in df["expiration"]]

    bid, ask = df["bid"].fillna(0.0), df["ask"].fillna(0.0)
    two_sided = (bid > 0) & (ask > 0) & (ask >= bid)
    df["mid"] = np.where(two_sided, (bid + ask) / 2.0, np.nan)
    df["spread"] = np.where(two_sided, ask - bid, np.nan)
    df["rel_spread"] = df["spread"] / df["mid"]
    # Price used for IV: mid when there is a real two-sided quote, otherwise
    # the last trade (flagged, and such rows are never treated as liquid).
    df["price_basis"] = np.where(two_sided, "mid", np.where(df["last_price"] > 0, "last", "none"))
    df["price_used"] = np.where(two_sided, df["mid"], np.where(df["last_price"] > 0, df["last_price"], np.nan))

    df["moneyness"] = df["strike"] / spot
    df["log_moneyness"] = np.log(df["moneyness"])
    df["is_otm"] = np.where(df["option_type"] == "call", df["strike"] >= spot, df["strike"] <= spot)

    ivs, statuses, deltas, gammas, vegas, thetas, model_prices = [], [], [], [], [], [], []
    for opt, K, T, p in zip(df["option_type"], df["strike"], df["T"], df["price_used"], strict=True):
        res = implied_volatility(float(p) if pd.notna(p) else float("nan"), opt, spot, K, T, r, q)
        ivs.append(res.value)
        statuses.append(res.status)
        if res.ok:
            g = bs.all_greeks(opt, spot, K, T, r, res.value, q)
            deltas.append(g.delta); gammas.append(g.gamma); vegas.append(g.vega); thetas.append(g.theta)
        else:
            deltas.append(np.nan); gammas.append(np.nan); vegas.append(np.nan); thetas.append(np.nan)
        if model_vol is not None and math.isfinite(model_vol) and T > 0:
            model_prices.append(bs.price(opt, spot, K, T, r, model_vol, q))
        else:
            model_prices.append(np.nan)

    df["computed_iv"] = ivs
    df["iv_status"] = statuses
    df["delta"], df["gamma"], df["vega"], df["theta"] = deltas, gammas, vegas, thetas
    df["model_price_rv"] = model_prices
    df["liquidity_note"] = df.apply(lambda row: _liquidity_reason(row, liquidity), axis=1)
    df["is_liquid"] = df["liquidity_note"] == ""
    return df


def atm_iv(enriched: pd.DataFrame, spot: float) -> float:
    """
    At-the-money implied volatility for one expiration: average of the call
    and put IV at the strike closest to spot (using liquid quotes only).
    This is the single number most people mean by "the IV" of an expiry.
    """
    if enriched.empty:
        return float("nan")
    liq = enriched[enriched["is_liquid"]]
    if liq.empty:
        liq = enriched[enriched["iv_status"] == "ok"]
    if liq.empty:
        return float("nan")
    k = liq.loc[(liq["strike"] - spot).abs().idxmin(), "strike"]
    return float(liq.loc[liq["strike"] == k, "computed_iv"].mean())
