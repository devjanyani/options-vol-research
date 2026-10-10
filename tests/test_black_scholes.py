"""
Black-Scholes tests against textbook values (Hull, "Options, Futures, and
Other Derivatives") and against mathematical identities that must hold.
"""

import math

import pytest

from app.models import black_scholes as bs


def test_hull_example_call_and_put_prices():
    # Hull Example 15.6: S=42, K=40, r=10%, sigma=20%, T=0.5 -> c=4.76, p=0.81
    assert bs.call_price(42, 40, 0.5, 0.10, 0.20) == pytest.approx(4.76, abs=0.005)
    assert bs.put_price(42, 40, 0.5, 0.10, 0.20) == pytest.approx(0.81, abs=0.005)


def test_hull_greeks_example():
    # Hull Ch.19: S=49, K=50, r=5%, sigma=20%, T=20 weeks (0.3846 yr)
    S, K, r, sigma, T = 49, 50, 0.05, 0.20, 0.3846
    assert bs.delta("call", S, K, T, r, sigma) == pytest.approx(0.522, abs=0.001)
    assert bs.gamma(S, K, T, r, sigma) == pytest.approx(0.066, abs=0.001)
    # Hull: vega = 12.1 per 1.00 of vol -> 0.121 per vol point
    assert bs.vega(S, K, T, r, sigma) == pytest.approx(0.121, abs=0.001)
    # Hull: theta = -4.31 per year -> per calendar day
    assert bs.theta("call", S, K, T, r, sigma) == pytest.approx(-4.31 / 365, abs=0.0002)


@pytest.mark.parametrize("S,K,T,r,q,sigma", [
    (100, 100, 0.25, 0.04, 0.0, 0.2),
    (100, 120, 1.0, 0.05, 0.02, 0.35),
    (500, 450, 0.08, 0.043, 0.013, 0.15),
    (50, 60, 2.0, 0.01, 0.03, 0.6),
])
def test_put_call_parity_with_dividends(S, K, T, r, q, sigma):
    # C - P = S e^{-qT} - K e^{-rT} must hold for any volatility.
    c = bs.call_price(S, K, T, r, sigma, q)
    p = bs.put_price(S, K, T, r, sigma, q)
    assert c - p == pytest.approx(S * math.exp(-q * T) - K * math.exp(-r * T), abs=1e-10)


@pytest.mark.parametrize("opt", ["call", "put"])
def test_greeks_match_finite_differences(opt):
    S, K, T, r, q, sigma = 100, 105, 0.4, 0.03, 0.01, 0.27
    h = 1e-3
    p = lambda **kw: bs.price(opt, kw.get("S", S), K, kw.get("T", T), r, kw.get("sigma", sigma), q)
    fd_delta = (p(S=S + h) - p(S=S - h)) / (2 * h)
    fd_gamma = (p(S=S + h) - 2 * p() + p(S=S - h)) / h**2
    fd_vega = (p(sigma=sigma + h) - p(sigma=sigma - h)) / (2 * h) / 100
    fd_theta = -(p(T=T + h) - p(T=T - h)) / (2 * h) / 365
    assert bs.delta(opt, S, K, T, r, sigma, q) == pytest.approx(fd_delta, abs=1e-5)
    assert bs.gamma(S, K, T, r, sigma, q) == pytest.approx(fd_gamma, abs=1e-4)
    assert bs.vega(S, K, T, r, sigma, q) == pytest.approx(fd_vega, abs=1e-6)
    assert bs.theta(opt, S, K, T, r, sigma, q) == pytest.approx(fd_theta, abs=1e-6)


def test_delta_ranges_and_put_call_delta_relation():
    S, K, T, r, q, sigma = 100, 100, 0.5, 0.04, 0.02, 0.25
    dc, dp = bs.delta("call", S, K, T, r, sigma, q), bs.delta("put", S, K, T, r, sigma, q)
    assert 0 < dc < 1 and -1 < dp < 0
    assert dc - dp == pytest.approx(math.exp(-q * T))


def test_price_increases_with_volatility():
    prices = [bs.call_price(100, 100, 0.5, 0.03, s) for s in (0.1, 0.2, 0.3, 0.5)]
    assert prices == sorted(prices)


def test_expired_and_zero_vol_edge_cases():
    assert bs.call_price(110, 100, 0.0, 0.05, 0.2) == pytest.approx(10.0)
    assert bs.put_price(90, 100, 0.0, 0.05, 0.2) == pytest.approx(10.0)
    assert bs.call_price(90, 100, 0.0, 0.05, 0.2) == 0.0
    assert bs.delta("call", 110, 100, 0.0, 0.05, 0.2) == 1.0
    assert bs.gamma(100, 100, 0.0, 0.05, 0.2) == 0.0


def test_invalid_option_type():
    with pytest.raises(ValueError):
        bs.price("straddle", 100, 100, 1, 0.01, 0.2)
