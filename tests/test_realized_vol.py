import numpy as np
import pandas as pd
import pytest

from app.analytics.realized_vol import annualize, log_returns, realized_vol, rolling_realized_vol


def _prices_from_returns(rets):
    return pd.Series(100 * np.exp(np.cumsum(np.r_[0.0, rets])),
                     index=pd.bdate_range("2024-01-01", periods=len(rets) + 1))


def test_constant_growth_has_zero_vol():
    p = _prices_from_returns([0.001] * 40)
    assert realized_vol(p, 30) == pytest.approx(0.0, abs=1e-12)


def test_alternating_returns_known_answer():
    # Returns +1%, -1%, ... have sample std = 0.01 * sqrt(n/(n-1)) for even n.
    n = 20
    p = _prices_from_returns([0.01, -0.01] * (n // 2))
    expected = 0.01 * np.sqrt(n / (n - 1)) * np.sqrt(252)
    assert realized_vol(p, n) == pytest.approx(expected, rel=1e-10)


def test_annualization_uses_sqrt_252():
    assert annualize(0.01) == pytest.approx(0.01 * np.sqrt(252))
    assert annualize(0.01) == pytest.approx(0.1587, abs=1e-4)


def test_log_returns():
    p = pd.Series([100.0, 110.0, 99.0])
    assert log_returns(p).tolist() == pytest.approx([np.log(1.1), np.log(0.9)])


def test_not_enough_data_returns_nan():
    assert np.isnan(realized_vol(_prices_from_returns([0.01, -0.02]), 30))


def test_rolling_vol_uses_only_past_data():
    rng = np.random.default_rng(0)
    p = _prices_from_returns(rng.normal(0, 0.01, 200))
    full = rolling_realized_vol(p, 20)
    truncated = rolling_realized_vol(p.iloc[:120], 20)
    pd.testing.assert_series_equal(full.loc[truncated.index], truncated)
    # And the last rolling value equals the point estimate on the same window
    assert full.iloc[-1] == pytest.approx(realized_vol(p, 20))
