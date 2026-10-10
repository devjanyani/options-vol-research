import math

import pytest

from app.models import black_scholes as bs
from app.models.implied_vol import implied_volatility, no_arbitrage_bounds


@pytest.mark.parametrize("opt", ["call", "put"])
@pytest.mark.parametrize("K", [70, 90, 100, 110, 140])
@pytest.mark.parametrize("sigma", [0.08, 0.2, 0.45, 1.2])
@pytest.mark.parametrize("T", [7 / 365, 0.25, 1.5])
def test_roundtrip_recovers_volatility(opt, K, sigma, T):
    S, r, q = 100, 0.04, 0.015
    p = bs.price(opt, S, K, T, r, sigma, q)
    lo, _ = no_arbitrage_bounds(opt, S, K, T, r, q)
    if p - lo < 0.005:
        # Deep in/out of the money: (almost) no time value, IV not identifiable.
        assert implied_volatility(p, opt, S, K, T, r, q).status in ("no_time_value", "invalid_input")
        return
    res = implied_volatility(p, opt, S, K, T, r, q)
    assert res.ok
    assert res.value == pytest.approx(sigma, abs=1e-5)


def test_hull_market_price_example():
    # Hull: call price 1.875 with S=21, K=20, r=0.1, T=0.25 -> IV ~ 23.5%
    res = implied_volatility(1.875, "call", 21, 20, 0.25, 0.10)
    assert res.ok and res.value == pytest.approx(0.235, abs=0.001)


def test_price_below_intrinsic_is_flagged_not_forced():
    # Call worth at least S - K e^{-rT} ~ 20.4. A price of 15 is impossible.
    res = implied_volatility(15.0, "call", 120, 100, 0.5, 0.04)
    assert not res.ok and res.status == "below_intrinsic" and math.isnan(res.value)


def test_price_above_maximum_is_flagged():
    res = implied_volatility(150.0, "call", 100, 100, 0.5, 0.04)
    assert res.status == "above_maximum"


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan")])
def test_invalid_inputs(bad):
    assert implied_volatility(bad, "call", 100, 100, 0.5, 0.04).status == "invalid_input"
    assert implied_volatility(5.0, "call", 100, 100, bad if bad != 0 else 0.0, 0.04).status == "invalid_input"


def test_no_time_value_is_flagged():
    # Deep ITM call priced exactly at intrinsic: volatility is not identifiable.
    S, K, T, r = 100, 70, 0.25, 0.04
    intrinsic = S - K * math.exp(-r * T)
    assert implied_volatility(intrinsic + 0.001, "call", S, K, T, r).status == "no_time_value"
