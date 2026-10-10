"""
Historical (realized) volatility.

PLAIN-ENGLISH SUMMARY
---------------------
Realized volatility measures how much the stock ACTUALLY moved over a recent
window. Implied volatility is what the options market EXPECTS it to move.
Comparing the two is the core of this project.

FORMULA
-------
1. Daily log return:  r_t = ln(P_t / P_{t-1})
   Log returns add up over time and treat up and down moves symmetrically.
2. Sample standard deviation of the last N returns (ddof=1).
3. Annualize: daily_std * sqrt(252).
   Why sqrt? If daily moves are independent, VARIANCE grows linearly with
   time (252 days -> 252x the variance), so standard deviation grows with
   the square root of time.

Example: a daily std of 1.0% -> 1.0% * sqrt(252) = 15.9% annualized.

ASSUMPTIONS / LIMITATIONS
-------------------------
- Close-to-close only: overnight gaps are counted, intraday swings are not.
- Backward-looking: it tells you nothing directly about scheduled events
  (earnings) in the future. IV can legitimately be higher for that reason.
- Short windows (10 days) are noisy; long windows (60 days) are slow to react.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from app import config


def log_returns(prices: pd.Series) -> pd.Series:
    prices = prices.astype(float)
    return np.log(prices / prices.shift(1)).dropna()


def annualize(daily_std: float | pd.Series, periods_per_year: int = config.TRADING_DAYS_PER_YEAR):
    return daily_std * np.sqrt(periods_per_year)


def realized_vol(prices: pd.Series, window: int) -> float:
    """Annualized realized volatility over the most recent `window` returns."""
    rets = log_returns(prices)
    if len(rets) < window:
        return float("nan")
    return float(annualize(rets.iloc[-window:].std(ddof=1)))


def rolling_realized_vol(prices: pd.Series, window: int) -> pd.Series:
    """
    Realized vol on every date, using ONLY returns up to and including that
    date. That property matters for backtests: no future data leaks in.
    """
    return annualize(log_returns(prices).rolling(window).std(ddof=1)).dropna()


def realized_vol_table(prices: pd.Series, windows=config.RV_WINDOWS) -> pd.DataFrame:
    return pd.DataFrame(
        {"window_days": list(windows), "realized_vol": [realized_vol(prices, w) for w in windows]}
    )
