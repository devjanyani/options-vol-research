"""
Risk of a single option position (optionally delta-hedged with stock).

Per-share Greeks become POSITION Greeks by multiplying by
    side (+1 long, -1 short) x contracts x 100 shares per contract.

    Position delta (shares)  : how many shares of stock the position behaves like
    Dollar delta             : position delta x stock price
    Position gamma (shares)  : change in position delta for a $1 stock move
    Position vega ($/vol pt) : P&L if implied vol rises 1 point
    Position theta ($/day)   : P&L from one day passing, all else equal

The scenario grid re-prices the position with Black-Scholes under shocks to
the stock price and to implied volatility, which captures the non-linear
effects that single Greeks miss.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from app import config
from app.models import black_scholes as bs

M = config.CONTRACT_MULTIPLIER


@dataclass(frozen=True)
class Position:
    option_type: str
    strike: float
    T: float              # years to expiry
    side: int             # +1 long, -1 short
    contracts: int
    spot: float
    iv: float
    r: float
    q: float = 0.0
    hedge_shares: float = 0.0   # stock held against the option (negative = short stock)

    @property
    def units(self) -> float:
        return self.side * self.contracts * M

    def value(self, spot: float | None = None, iv: float | None = None, T: float | None = None) -> float:
        S = self.spot if spot is None else spot
        sigma = self.iv if iv is None else iv
        t = self.T if T is None else T
        return self.units * bs.price(self.option_type, S, self.strike, t, self.r, sigma, self.q) + self.hedge_shares * S


def hedge_shares_for(option_type: str, strike: float, T: float, side: int, contracts: int,
                     spot: float, iv: float, r: float, q: float = 0.0) -> float:
    """Shares that make the position delta-neutral right now."""
    return -side * contracts * M * bs.delta(option_type, spot, strike, T, r, iv, q)


def position_greeks(p: Position) -> dict[str, float]:
    g = bs.all_greeks(p.option_type, p.spot, p.strike, p.T, p.r, p.iv, p.q)
    delta_sh = p.units * g.delta + p.hedge_shares
    return {
        "premium": p.units * g.price,                       # + = paid, - = received
        "delta_shares": delta_sh,
        "dollar_delta": delta_sh * p.spot,
        "gamma_shares": p.units * g.gamma,
        "vega_dollars": p.units * g.vega,
        "theta_dollars": p.units * g.theta,
        "option_notional": abs(p.contracts) * M * p.spot,
    }


def scenario_grid(p: Position, spot_moves=(-0.20, -0.10, -0.05, -0.02, 0.0, 0.02, 0.05, 0.10, 0.20),
                  vol_shifts=(-0.10, -0.05, 0.0, 0.05, 0.10), days_forward: int = 0) -> pd.DataFrame:
    """P&L in dollars: rows = vol shift (vol points), columns = stock move (%)."""
    base = p.value()
    T = max(p.T - days_forward / config.CALENDAR_DAYS_PER_YEAR, 0.0)
    rows = {}
    for dv in vol_shifts:
        rows[dv] = [p.value(spot=p.spot * (1 + m), iv=max(p.iv + dv, 0.01), T=T) - base for m in spot_moves]
    return pd.DataFrame(rows, index=list(spot_moves)).T


def max_loss_description(p: Position, premium_per_share: float) -> tuple[float, str]:
    """
    Theoretical worst case at expiry for an unhedged position. Returns
    (loss in dollars as a negative number, or -inf, explanation).
    """
    n = p.contracts * M
    if p.hedge_shares != 0:
        return float("nan"), "Hedged positions have no simple closed-form maximum loss; see the scenario grid."
    if p.side > 0:
        return -premium_per_share * n, "Buying an option: the most you can lose is the premium paid."
    if p.option_type == "call":
        return -math.inf, "Selling a call without owning the stock: losses are unlimited if the stock keeps rising."
    return -(p.strike - premium_per_share) * n, "Selling a put: worst case is the stock going to $0 (strike minus premium received)."


def expiry_payoff_curve(p: Position, entry_premium_per_share: float, spots: np.ndarray) -> np.ndarray:
    """P&L at expiry vs stock price (includes the stock hedge)."""
    intrinsic = np.maximum(spots - p.strike, 0) if p.option_type == "call" else np.maximum(p.strike - spots, 0)
    return p.units * (intrinsic - entry_premium_per_share) + p.hedge_shares * (spots - p.spot)
