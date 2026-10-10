"""
MODELED proxy backtest: SPY prices + the VIX index.

!!! The option prices in this backtest are COMPUTED with Black-Scholes. They
!!! are NOT historical option quotes. Results are labeled MODELED everywhere.

WHY THIS EXISTS
---------------
Free data has no history of option prices, but it does have:
  - SPY daily prices (going back decades)
  - the VIX index: the market's 30-day implied volatility for the S&P 500,
    computed by Cboe from real SPX option prices every day.
So we can ask a simplified version of our core question historically:
"When implied volatility (VIX) was well above recent realized volatility,
did SELLING volatility pay off after costs?"

THE STRATEGY (deliberately simple, no fitted parameters)
--------------------------------------------------------
Every `holding_days` trading days (non-overlapping), at the close of day t:
  1. implied  = VIX_t / 100            (minus an optional adjustment, see below)
     realized = 20-day realized vol of SPY using closes up to and including t
  2. premium  = implied - realized
  3. Signal:
       short-only : SELL a straddle if premium >= threshold
       long/short : also BUY a straddle if premium <= -threshold
       always-short: sell every period (a baseline, to measure whether the
                     signal adds anything over just selling volatility)
  4. The straddle = 1 call + 1 put, strike = SPY close (at the money),
     30 calendar days to expiry, priced with Black-Scholes at `implied`.
  5. Delta hedge once at entry with SPY shares (so we are trading volatility,
     not market direction).
  6. Exit at the close of t + holding_days: re-price the same straddle with
     the remaining time, the new SPY price and the new VIX.

NO LOOK-AHEAD
-------------
The decision on day t only reads SPY closes <= t, VIX_t and the T-bill rate
of day t. Future data is used only to measure the outcome of a trade that
was already decided. tests/test_backtest.py checks this by truncating the
data and verifying past decisions do not change.

MODEL LIMITATIONS (also shown in the UI)
----------------------------------------
- VIX is not the at-the-money IV. It is a variance-swap style measure that is
  usually a few points ABOVE ATM IV because it includes expensive OTM puts.
  Pricing a short straddle at VIX therefore overstates the premium collected.
  `vix_adjustment_pts` lets you subtract points to test sensitivity.
- At exit we re-price a ~20-day option with the 30-day VIX (no term structure).
- Black-Scholes with flat volatility: no skew, no jumps.
- European pricing; SPY options are American.
- Costs are assumed (bid/ask = a fixed % of the model price).
- Adjusted SPY closes with q = 0 approximate dividends.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from app import config
from app.analytics.realized_vol import rolling_realized_vol
from app.backtesting.costs import modeled_option_leg_cost, stock_hedge_cost
from app.models import black_scholes as bs


@dataclass(frozen=True)
class ProxyParams:
    holding_days: int = 10
    rv_window: int = 20
    threshold_pts: float = 4.0          # vol points, e.g. 4.0 = 4%
    mode: str = "short_only"            # short_only | long_short | always_short
    tenor_days: int = 30
    delta_hedge: bool = True
    vix_adjustment_pts: float = 0.0     # subtract from VIX before pricing
    contracts: int = 1
    start: str = "2010-01-01"


def align_inputs(spy: pd.Series, vix: pd.Series, rates: pd.Series | None) -> pd.DataFrame:
    df = pd.DataFrame({"spy": spy.astype(float), "vix": vix.astype(float)})
    df.index = pd.to_datetime(df.index).normalize()
    df = df.dropna()
    if rates is not None and not rates.empty:
        r = rates.copy()
        r.index = pd.to_datetime(r.index).normalize()
        df["r"] = (r / 100.0).reindex(df.index).ffill()  # ^IRX is quoted in percent
    else:
        df["r"] = np.nan
    df["r_is_assumption"] = df["r"].isna()
    df["r"] = df["r"].fillna(config.FALLBACK_RISK_FREE_RATE)
    return df


def straddle_value(S: float, K: float, T: float, r: float, sigma: float) -> tuple[float, float, float, float]:
    c = bs.call_price(S, K, T, r, sigma)
    p = bs.put_price(S, K, T, r, sigma)
    d = bs.delta("call", S, K, T, r, sigma) + bs.delta("put", S, K, T, r, sigma)
    return c + p, c, p, d


def decide(premium: float, mode: str, threshold: float) -> int:
    """+1 = buy volatility (long straddle), -1 = sell volatility, 0 = no trade."""
    if mode == "always_short":
        return -1
    if premium >= threshold:
        return -1
    if mode == "long_short" and premium <= -threshold:
        return 1
    return 0


def run_proxy_backtest(spy: pd.Series, vix: pd.Series, rates: pd.Series | None,
                       p: ProxyParams = ProxyParams(),
                       cm: config.CostModel = config.SETTINGS.costs) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (trades, decisions). `decisions` logs every check, traded or not."""
    df = align_inputs(spy, vix, rates)
    df["rv"] = rolling_realized_vol(df["spy"], p.rv_window).reindex(df.index)
    df = df[df.index >= pd.Timestamp(p.start)].dropna(subset=["rv"])
    dates = df.index
    thr, adj = p.threshold_pts / 100.0, p.vix_adjustment_pts / 100.0
    m = config.CONTRACT_MULTIPLIER * p.contracts
    trades, decisions = [], []

    for i in range(0, len(dates) - p.holding_days, p.holding_days):
        t, j = dates[i], dates[i + p.holding_days]
        row, out = df.loc[t], df.loc[j]
        implied = max(row.vix / 100.0 - adj, 0.01)
        premium = implied - row.rv
        side = decide(premium, p.mode, thr)
        decisions.append({"date": t, "implied": implied, "realized": row.rv, "premium": premium, "side": side})
        if side == 0:
            continue

        T0 = p.tenor_days / config.CALENDAR_DAYS_PER_YEAR
        K = row.spy
        v0, c0, p0, d0 = straddle_value(row.spy, K, T0, row.r, implied)
        elapsed = (j - t).days / config.CALENDAR_DAYS_PER_YEAR
        T1 = max(T0 - elapsed, 0.0)
        implied_exit = max(out.vix / 100.0 - adj, 0.01)
        v1, c1, p1, _ = straddle_value(out.spy, K, T1, out.r, implied_exit)

        hedge_shares = -side * d0 * m if p.delta_hedge else 0.0
        option_pnl = side * (v1 - v0) * m
        hedge_pnl = hedge_shares * (out.spy - row.spy)
        gross = option_pnl + hedge_pnl

        legs = [modeled_option_leg_cost(x, p.contracts, cm) for x in (c0, p0, c1, p1)]  # 2 legs in, 2 legs out
        hedge_cost = 2 * stock_hedge_cost(hedge_shares, cm)
        cost_spread = sum(l.half_spread for l in legs)
        cost_slip = sum(l.slippage for l in legs)
        cost_fees = sum(l.fees for l in legs)
        costs = cost_spread + cost_slip + cost_fees + hedge_cost
        trades.append({
            "entry_date": t, "exit_date": j,
            "description": f"{'Sell' if side < 0 else 'Buy'} {p.tenor_days}d ATM SPY straddle K={K:,.2f}",
            "side": "short vol" if side < 0 else "long vol",
            "spot_entry": row.spy, "spot_exit": out.spy, "implied_entry": implied, "implied_exit": implied_exit,
            "realized_entry": row.rv, "premium_entry": premium,
            "entry_price": v0, "exit_price": v1, "option_pnl": option_pnl, "hedge_pnl": hedge_pnl,
            "gross_pnl": gross, "cost_spread": cost_spread, "cost_slippage": cost_slip, "cost_fees": cost_fees,
            "cost_hedge": hedge_cost,
            "costs": costs, "net_pnl": gross - costs, "capital": v0 * m,
            "rate_is_assumption": bool(row.r_is_assumption),
        })
    return pd.DataFrame(trades), pd.DataFrame(decisions)


def sensitivity_grid(spy: pd.Series, vix: pd.Series, rates: pd.Series | None, base: ProxyParams,
                     thresholds=(0.0, 2.0, 4.0, 6.0, 8.0), holdings=(5, 10, 20)) -> pd.DataFrame:
    """
    Net results for EVERY combination, so nobody (including us) is tempted to
    show only the best one. Choosing the best cell after looking would be
    data snooping.
    """
    from app.backtesting.metrics import summarize

    rows = []
    for h in holdings:
        for thr in thresholds:
            params = ProxyParams(**{**asdict(base), "holding_days": h, "threshold_pts": thr})
            tr, _ = run_proxy_backtest(spy, vix, rates, params)
            s = summarize(tr, "net_pnl")
            rows.append({"holding_days": h, "threshold_pts": thr, "trades": s["trades"],
                         "net_total_pnl": s["total_pnl"], "net_sharpe": s["sharpe"], "win_rate": s["win_rate"],
                         "max_drawdown": s["max_drawdown"]})
    return pd.DataFrame(rows)
