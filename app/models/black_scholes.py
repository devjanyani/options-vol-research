"""
Black-Scholes(-Merton) option pricing, written out from the formulas.

PLAIN-ENGLISH SUMMARY
---------------------
Black-Scholes answers: "If the stock moves randomly with a constant
volatility sigma, what is a fair price for this option today?"

Inputs:
    S      current stock price
    K      strike price
    T      time to expiration, in YEARS (e.g. 30 days = 30/365)
    r      risk-free interest rate (decimal, continuously compounded)
    q      dividend yield (decimal, continuous). SPY/QQQ/AAPL/MSFT pay
           dividends; dividends make calls cheaper and puts dearer because
           the stock price drops when the dividend is paid out.
    sigma  volatility: annualized standard deviation of log returns

The two helper numbers:
    d1 = [ ln(S/K) + (r - q + sigma^2 / 2) * T ] / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)

N(x) is the standard normal cumulative distribution (probability a standard
normal variable is below x). Loosely, N(d2) is the risk-neutral probability
that a call finishes in the money.

    Call = S e^{-qT} N(d1)  - K e^{-rT} N(d2)
    Put  = K e^{-rT} N(-d2) - S e^{-qT} N(-d1)

KEY ASSUMPTIONS (and where they fail)
-------------------------------------
1. Volatility is constant over the option's life.
   Fails: real IV differs by strike and expiry (the "smile"/"skew"), which is
   exactly what the volatility surface page shows.
2. Stock returns are log-normal with no jumps.
   Fails: earnings announcements and crashes cause jumps.
3. European exercise (only at expiry).
   Fails slightly: US single-stock and ETF options are American (can be
   exercised early). For most strikes the difference is small; it matters
   most for deep in-the-money puts and for calls just before a dividend.
4. Continuous trading with no transaction costs.
   Fails: there are bid/ask spreads and fees (see Phase 9).

We implement the formulas ourselves (no options library) so every number
can be traced. The normal CDF uses math.erf, which is exact to machine
precision and much faster than scipy for single numbers.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

SQRT_2 = math.sqrt(2.0)
SQRT_2PI = math.sqrt(2.0 * math.pi)


def norm_cdf(x: float) -> float:
    """Standard normal cumulative distribution N(x)."""
    return 0.5 * (1.0 + math.erf(x / SQRT_2))


def norm_pdf(x: float) -> float:
    """Standard normal density n(x) = exp(-x^2/2) / sqrt(2*pi)."""
    return math.exp(-0.5 * x * x) / SQRT_2PI


def _check_type(option_type: str) -> str:
    t = option_type.lower()
    if t not in ("call", "put"):
        raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")
    return t


def d1_d2(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> tuple[float, float]:
    """The two standardized distances used by every Black-Scholes formula."""
    vol_sqrt_t = sigma * math.sqrt(T)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / vol_sqrt_t
    return d1, d1 - vol_sqrt_t


def _degenerate(S: float, K: float, T: float, sigma: float) -> bool:
    """True when the formula's sqrt(T)*sigma would be zero (expired or zero vol)."""
    return T <= 0.0 or sigma <= 0.0


def call_price(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> float:
    if _degenerate(S, K, T, sigma):
        # No time or no uncertainty left: worth the discounted intrinsic value.
        T = max(T, 0.0)
        return max(S * math.exp(-q * T) - K * math.exp(-r * T), 0.0)
    d1, d2 = d1_d2(S, K, T, r, sigma, q)
    return S * math.exp(-q * T) * norm_cdf(d1) - K * math.exp(-r * T) * norm_cdf(d2)


def put_price(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> float:
    if _degenerate(S, K, T, sigma):
        T = max(T, 0.0)
        return max(K * math.exp(-r * T) - S * math.exp(-q * T), 0.0)
    d1, d2 = d1_d2(S, K, T, r, sigma, q)
    return K * math.exp(-r * T) * norm_cdf(-d2) - S * math.exp(-q * T) * norm_cdf(-d1)


def price(option_type: str, S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> float:
    """Black-Scholes price of a call or a put."""
    if _check_type(option_type) == "call":
        return call_price(S, K, T, r, sigma, q)
    return put_price(S, K, T, r, sigma, q)


# ---------------------------------------------------------------- Greeks
# Greeks are sensitivities: "if one input moves a little, how much does the
# option price move?" Traders use them to understand and hedge risk.


def delta(option_type: str, S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> float:
    """
    dPrice/dS. Call delta is between 0 and 1, put delta between -1 and 0.
    Delta 0.42 => option gains ~$0.42 when the stock rises $1 (all else equal).
    """
    t = _check_type(option_type)
    if _degenerate(S, K, T, sigma):
        itm = S > K if t == "call" else S < K
        return (1.0 if t == "call" else -1.0) if itm else 0.0
    d1, _ = d1_d2(S, K, T, r, sigma, q)
    disc_q = math.exp(-q * T)
    return disc_q * norm_cdf(d1) if t == "call" else disc_q * (norm_cdf(d1) - 1.0)


def gamma(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> float:
    """
    d(Delta)/dS, identical for calls and puts. How fast delta changes when the
    stock moves $1. Highest for at-the-money options close to expiry.
    """
    if _degenerate(S, K, T, sigma):
        return 0.0
    d1, _ = d1_d2(S, K, T, r, sigma, q)
    return math.exp(-q * T) * norm_pdf(d1) / (S * sigma * math.sqrt(T))


def vega(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> float:
    """
    dPrice/dSigma, reported PER 1 VOLATILITY POINT (sigma 0.30 -> 0.31).
    Identical for calls and puts. Vega 0.15 => price rises ~$0.15 if implied
    volatility rises by 1 percentage point.
    """
    if _degenerate(S, K, T, sigma):
        return 0.0
    d1, _ = d1_d2(S, K, T, r, sigma, q)
    return S * math.exp(-q * T) * norm_pdf(d1) * math.sqrt(T) / 100.0


def theta(option_type: str, S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> float:
    """
    dPrice/dTime, reported PER CALENDAR DAY. Usually negative: an option
    loses value as time passes (time decay). Theta -0.05 => the option loses
    ~$0.05 per day if nothing else changes.
    """
    t = _check_type(option_type)
    if _degenerate(S, K, T, sigma):
        return 0.0
    d1, d2 = d1_d2(S, K, T, r, sigma, q)
    disc_q, disc_r = math.exp(-q * T), math.exp(-r * T)
    decay = -S * disc_q * norm_pdf(d1) * sigma / (2.0 * math.sqrt(T))
    if t == "call":
        annual = decay - r * K * disc_r * norm_cdf(d2) + q * S * disc_q * norm_cdf(d1)
    else:
        annual = decay + r * K * disc_r * norm_cdf(-d2) - q * S * disc_q * norm_cdf(-d1)
    return annual / 365.0


@dataclass(frozen=True)
class Greeks:
    price: float
    delta: float
    gamma: float
    vega: float   # per 1 vol point
    theta: float  # per calendar day


def all_greeks(option_type: str, S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> Greeks:
    return Greeks(
        price=price(option_type, S, K, T, r, sigma, q),
        delta=delta(option_type, S, K, T, r, sigma, q),
        gamma=gamma(S, K, T, r, sigma, q),
        vega=vega(S, K, T, r, sigma, q),
        theta=theta(option_type, S, K, T, r, sigma, q),
    )
