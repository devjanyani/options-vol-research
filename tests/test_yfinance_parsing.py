"""
Checks that YFinanceProvider parses data shaped exactly like yfinance's
responses (column names, tz-aware indexes). Uses an in-memory fake of the
yfinance module, so no network is needed. Values are arbitrary test inputs.
"""

from datetime import date, timedelta
from types import SimpleNamespace

import pandas as pd
import pytest

from app.data.market_data import CHAIN_COLUMNS, DataUnavailableError, YFinanceProvider


def _hist(n=30, start=100.0):
    idx = pd.date_range(end=pd.Timestamp.now(tz="America/New_York").normalize(), periods=n, freq="B")
    close = pd.Series([start + i for i in range(n)], index=idx, dtype=float)
    return pd.DataFrame({"Open": close, "High": close, "Low": close, "Close": close, "Volume": 1000})


def _chain_df(kind):
    return pd.DataFrame({
        "contractSymbol": [f"SPY261120{kind}00500000", f"SPY261120{kind}00510000"],
        "lastTradeDate": pd.to_datetime(["2026-10-07 19:59:00+00:00", "2026-10-07 19:58:00+00:00"]),
        "strike": [500.0, 510.0], "lastPrice": [10.1, 6.2], "bid": [10.0, 6.1], "ask": [10.2, 6.3],
        "change": [0.0, 0.0], "percentChange": [0.0, 0.0], "volume": [100.0, float("nan")],
        "openInterest": [1000, 500], "impliedVolatility": [0.2, 0.19], "inTheMoney": [True, False],
        "contractSize": ["REGULAR"] * 2, "currency": ["USD"] * 2,
    })


class FakeTicker:
    def __init__(self, sym):
        self.sym = sym
        self.options = () if sym == "NOOPT" else ("2026-11-20", "2026-12-18")
        self.fast_info = {"lastPrice": 505.5, "previousClose": 500.0}
        idx = pd.DatetimeIndex([pd.Timestamp.now(tz="America/New_York") - pd.Timedelta(days=d) for d in (300, 200, 100, 10, 500)])
        self.dividends = pd.Series([1.5, 1.6, 1.7, 1.8, 9.9], index=idx)
        self.calendar = {"Earnings Date": [date.today() + timedelta(days=20)]}

    def history(self, period="1y", auto_adjust=True):
        if self.sym == "BAD":
            return pd.DataFrame()
        if self.sym == "^IRX":
            return _hist(5, 4.3)
        return _hist()

    def option_chain(self, exp):
        return SimpleNamespace(calls=_chain_df("C"), puts=_chain_df("P"))


@pytest.fixture
def provider():
    p = YFinanceProvider.__new__(YFinanceProvider)
    p._yf = SimpleNamespace(Ticker=FakeTicker)
    p._pause = 0
    return p


def test_spot(provider):
    s = provider.get_spot("spy")
    assert s.price == 505.5 and s.previous_close == 500.0


def test_history_is_tz_naive_daily(provider):
    h = provider.get_price_history("SPY")
    assert h.index.tz is None and len(h) == 30 and h.iloc[-1] == 129.0


def test_chain_columns_and_types(provider):
    ch = provider.get_option_chain("SPY", "2026-11-20")
    assert list(ch.columns) == CHAIN_COLUMNS
    assert set(ch["option_type"]) == {"call", "put"} and len(ch) == 4
    assert ch["volume"].tolist() == [100, 0, 100, 0]  # NaN volume -> 0


def test_rates_dividends_earnings(provider):
    r = provider.get_risk_free_rate()
    assert r.value == pytest.approx(0.083) and not r.is_assumption   # last ^IRX close 8.3 (percent) -> 0.083
    q = provider.get_dividend_yield("SPY", 500.0)
    assert q.value == pytest.approx((1.5 + 1.6 + 1.7 + 1.8) / 500.0)   # the 500-day-old dividend is excluded
    assert provider.get_next_earnings_date("AAPL") == date.today() + timedelta(days=20)


def test_errors_are_explicit(provider):
    with pytest.raises(DataUnavailableError):
        provider.get_expirations("NOOPT")
    with pytest.raises(DataUnavailableError):
        provider.get_price_history("BAD")
