"""
Implied volatility (IV): Black-Scholes run in reverse.

PLAIN-ENGLISH SUMMARY
---------------------
Black-Scholes turns a volatility into a price. The market gives us the
price (we use the bid/ask midpoint), so we ask the opposite question:
"Which volatility, plugged into Black-Scholes, reproduces this market price?"
That volatility is the option's implied volatility. It is the market's
price of future movement, expressed in the same units as historical
volatility, which is what lets us compare the two.

HOW WE SOLVE IT
---------------
There is no algebraic formula for sigma, so we search numerically. Because a
higher sigma always gives a higher option price (vega > 0), there is exactly
one answer, and we can trap it between a low guess (0.01%) and a high guess
(500%) and keep narrowing the bracket. We use Brent's method
(scipy.optimize.brentq): it combines safe bisection ("cut the interval in
half") with faster interpolation steps, and it cannot diverge as long as the
answer is inside the bracket. Newton's method is faster but can fly off when
vega is tiny (deep in/out-of-the-money options), so we prefer robustness.

WHEN NO ANSWER EXISTS
---------------------
Some quotes cannot be produced by ANY volatility, for example a call priced
below its intrinsic value. That usually means a stale or bad quote (or an
American early-exercise effect). We return NaN with a reason instead of a
fake number.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from scipy.optimize import brentq

from app.models import black_scholes as bs

SIGMA_LOW = 1e-4
SIGMA_HIGH = 5.0
MIN_TIME_VALUE = 0.005  # dollars


@dataclass(frozen=True)
class IVResult:
    value: float      # implied volatility (decimal) or NaN
    status: str       # "ok" or a reason it could not be solved

    @property
    def ok(self) -> bool:
        return self.status == "ok"


STATUS_MESSAGES = {
    "ok": "Solved",
    "invalid_input": "Missing or non-positive price, strike, spot or time",
    "below_intrinsic": "Price is below the no-arbitrage minimum (often a stale quote)",
    "above_maximum": "Price is above the no-arbitrage maximum",
    "no_time_value": "Price is (almost) exactly intrinsic value, so volatility cannot be inferred",
    "no_convergence": "Solver could not find a volatility between 0.01% and 500%",
}


def no_arbitrage_bounds(option_type: str, S: float, K: float, T: float, r: float, q: float) -> tuple[float, float]:
    """
    Every European option price must sit inside these bounds, for any
    volatility. Lower bound = value at zero volatility; upper bound = value at
    infinite volatility.
    """
    fwd_s = S * math.exp(-q * T)   # present value of the stock (net of dividends)
    pv_k = K * math.exp(-r * T)    # present value of the strike
    if option_type == "call":
        return max(fwd_s - pv_k, 0.0), fwd_s
    return max(pv_k - fwd_s, 0.0), pv_k


def implied_volatility(
    market_price: float,
    option_type: str,
    S: float,
    K: float,
    T: float,
    r: float,
    q: float = 0.0,
    tol: float = 1e-8,
) -> IVResult:
    option_type = option_type.lower()
    vals = (market_price, S, K, T)
    if any(v is None or not math.isfinite(v) for v in vals) or market_price <= 0 or S <= 0 or K <= 0 or T <= 0:
        return IVResult(float("nan"), "invalid_input")

    lo_bound, hi_bound = no_arbitrage_bounds(option_type, S, K, T, r, q)
    # A tiny tolerance so prices exactly at the bound (rounding) still solve.
    if market_price < lo_bound - 1e-6:
        return IVResult(float("nan"), "below_intrinsic")
    if market_price >= hi_bound:
        return IVResult(float("nan"), "above_maximum")
    # Option prices are quoted in cents. If the price is within half a cent
    # of intrinsic value, there is no time value left to "read" a volatility
    # from: many very different volatilities give the same rounded price.
    if market_price - lo_bound < MIN_TIME_VALUE:
        return IVResult(float("nan"), "no_time_value")

    def pricing_error(sigma: float) -> float:
        return bs.price(option_type, S, K, T, r, sigma, q) - market_price

    f_lo, f_hi = pricing_error(SIGMA_LOW), pricing_error(SIGMA_HIGH)
    if f_lo > 0:
        # Even 0.01% volatility gives a price above the market: price is at
        # (or within rounding of) intrinsic value. IV is effectively ~0.
        return IVResult(float("nan"), "below_intrinsic")
    if f_hi < 0:
        return IVResult(float("nan"), "no_convergence")
    try:
        sigma = brentq(pricing_error, SIGMA_LOW, SIGMA_HIGH, xtol=tol, maxiter=200)
    except (ValueError, RuntimeError):
        return IVResult(float("nan"), "no_convergence")
    return IVResult(float(sigma), "ok")
