"""
Performance metrics for simulated trades.

Every trade row needs: entry_date, exit_date, gross_pnl, costs, net_pnl,
capital (the option premium paid/received, our "capital at stake").

CONVENTIONS (state them in an interview!)
- Return of a trade = P&L / capital, where capital = entry option premium x 100
  x contracts. This is "return on premium", not return on a whole account.
- P&L is booked on the EXIT date. The daily P&L series is zero on days with no
  exits. Ignoring day-to-day mark-to-market inside a trade makes the series
  look smoother than reality, so Sharpe here is optimistic and drawdown is
  understated. Both are labeled as such in the UI.
- Sharpe = mean(daily P&L) / std(daily P&L) x sqrt(252). No risk-free rate is
  subtracted (the P&L is from a hedged, self-financing-ish position).
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from app import config


def max_drawdown(cum_pnl: pd.Series) -> float:
    """Largest peak-to-trough fall of a cumulative P&L curve (a negative number, or 0)."""
    if cum_pnl.empty:
        return 0.0
    running_peak = np.maximum.accumulate(np.r_[0.0, cum_pnl.to_numpy()])  # start from 0 before trading
    dd = np.r_[0.0, cum_pnl.to_numpy()] - running_peak
    return float(dd.min())


def sharpe_ratio(period_pnl: pd.Series, periods_per_year: float = config.TRADING_DAYS_PER_YEAR) -> float:
    x = period_pnl.dropna()
    if len(x) < 2:
        return float("nan")
    sd = float(x.std(ddof=1))
    if sd == 0 or not math.isfinite(sd):
        return float("nan")
    return float(x.mean() / sd * math.sqrt(periods_per_year))


def daily_pnl(trades: pd.DataFrame, column: str) -> pd.Series:
    """P&L booked by exit date on a business-day calendar (zeros on idle days)."""
    if trades.empty:
        return pd.Series(dtype=float)
    by_day = trades.groupby(pd.to_datetime(trades["exit_date"]))[column].sum()
    start = pd.to_datetime(trades["entry_date"]).min()
    cal = pd.bdate_range(start, by_day.index.max())
    return by_day.reindex(cal, fill_value=0.0)


def summarize(trades: pd.DataFrame, column: str = "net_pnl") -> dict:
    if trades.empty:
        return {"trades": 0, "win_rate": float("nan"), "avg_return": float("nan"), "total_pnl": 0.0,
                "max_drawdown": 0.0, "sharpe": float("nan"), "worst_trade": float("nan"), "best_trade": float("nan")}
    pnl = trades[column].astype(float)
    ret = pnl / trades["capital"].astype(float)
    d = daily_pnl(trades, column)
    return {
        "trades": int(len(trades)),
        "win_rate": float((pnl > 0).mean()),
        "avg_return": float(ret.replace([np.inf, -np.inf], np.nan).mean()),
        "total_pnl": float(pnl.sum()),
        "max_drawdown": max_drawdown(d.cumsum()),
        "sharpe": sharpe_ratio(d),
        "worst_trade": float(pnl.min()),
        "best_trade": float(pnl.max()),
    }


def equity_curves(trades: pd.DataFrame) -> pd.DataFrame:
    """Cumulative gross and net P&L by business day."""
    if trades.empty:
        return pd.DataFrame(columns=["gross", "net"])
    return pd.DataFrame({"gross": daily_pnl(trades, "gross_pnl").cumsum(),
                         "net": daily_pnl(trades, "net_pnl").cumsum()})
