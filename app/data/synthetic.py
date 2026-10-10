"""
SYNTHETIC TEST DATA PROVIDER. NOT MARKET DATA.

This exists only so automated tests and the developer UI harness can render
every page without an internet connection. It is activated only by setting
the environment variable OVRP_PROVIDER=synthetic, and the app then shows a
large red warning banner on every page.

Numbers are generated from a fixed formula (and a fixed random seed for the
price path) so tests are reproducible. Nothing here should ever be presented
as a real result.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

from app import config
from app.data.market_data import CHAIN_COLUMNS, DataUnavailableError, RateQuote, SpotQuote
from app.models import black_scholes as bs

_BASE_SPOT = {"SPY": 500.0, "QQQ": 430.0, "AAPL": 210.0, "NVDA": 120.0, "MSFT": 420.0, "TSLA": 250.0}
_EXPIRY_DAYS = [7, 14, 30, 45, 60, 90, 120]


def synthetic_iv(log_m: float, days: int) -> float:
    """A smooth, typical-looking equity smile: downside skew plus curvature."""
    term = 0.02 * np.exp(-days / 30.0)
    return float(0.20 + term - 0.25 * log_m + 0.60 * log_m * log_m)


class SyntheticProvider:
    name = "SYNTHETIC TEST DATA (not market data)"
    is_synthetic = True

    def __init__(self, today: date | None = None, expiry_anchor: date | None = None):
        self.today = today or date.today()
        # Expirations are fixed calendar dates counted from the anchor, so
        # tests that simulate several days see the same contracts each day.
        self.anchor = expiry_anchor or self.today

    def _spot(self, ticker: str) -> float:
        return _BASE_SPOT.get(ticker.upper(), 100.0)

    def get_spot(self, ticker: str) -> SpotQuote:
        hist = self.get_price_history(ticker)
        return SpotQuote(float(hist.iloc[-1]), float(hist.iloc[-2]),
                         pd.Timestamp.now(tz=config.MARKET_TZ).to_pydatetime(), self.name)

    def get_price_history(self, ticker: str, period: str = "2y") -> pd.Series:
        ticker = ticker.upper()
        idx = pd.bdate_range(end=pd.Timestamp(self.today), periods=520)
        if ticker == "^IRX":
            return pd.Series(4.0, index=idx, name="^IRX")
        if ticker == "^VIX":
            spy = self.get_price_history("SPY")
            rv = np.log(spy).diff().rolling(20).std() * np.sqrt(252) * 100
            return (rv * 1.15 + 2.0).bfill().rename("^VIX")
        seed = sum(ord(c) for c in ticker)
        rng = np.random.default_rng(seed)
        rets = rng.normal(0.0003, 0.012, len(idx))
        path = self._spot(ticker) * np.exp(np.cumsum(rets) - np.sum(rets))
        return pd.Series(path, index=idx, name=ticker)

    def get_expirations(self, ticker: str) -> list[str]:
        if ticker.upper() not in _BASE_SPOT:
            raise DataUnavailableError(f"No listed options found for '{ticker}' (synthetic provider).")
        exps = [(self.anchor + timedelta(days=d)) for d in _EXPIRY_DAYS]
        return [e.isoformat() for e in exps if e > self.today]

    def get_option_chain(self, ticker: str, expiration: str) -> pd.DataFrame:
        S = float(self.get_price_history(ticker).iloc[-1])
        days = (date.fromisoformat(expiration) - self.today).days
        anchor_days = (date.fromisoformat(expiration) - self.anchor).days
        T = max(days, 1) / 365.0
        step = 5.0 if S > 200 else 2.5
        strikes = np.arange(np.floor(S * 0.7 / step) * step, S * 1.3 + step, step)
        rows = []
        for k in strikes:
            log_m = float(np.log(k / S))
            for opt in ("call", "put"):
                iv = synthetic_iv(log_m, days)
                # one deliberate bump so the anomaly scanner has something to find
                if opt == "put" and anchor_days == 30 and abs(k - np.round(S * 0.95 / step) * step) < 1e-9:
                    iv += 0.04
                mid = bs.price(opt, S, k, T, 0.04, iv, 0.013)
                half = max(0.01 * mid, 0.01)
                bid = round(max(mid - half, 0.0), 2)
                ask = round(mid + half, 2)
                if mid < 0.03:
                    bid = 0.0
                dist = abs(log_m)
                rows.append({
                    "contract_symbol": f"{ticker.upper()}{expiration.replace('-', '')[2:]}{opt[0].upper()}{int(k * 1000):08d}",
                    "option_type": opt, "expiration": expiration, "strike": float(k),
                    "bid": bid, "ask": ask, "last_price": round(mid, 2),
                    "volume": int(2000 * np.exp(-dist * 12)), "open_interest": int(8000 * np.exp(-dist * 10)),
                    "yahoo_iv": iv, "last_trade_date": pd.Timestamp.now(tz="UTC"),
                })
        return pd.DataFrame(rows)[CHAIN_COLUMNS]

    def get_risk_free_rate(self) -> RateQuote:
        return RateQuote(0.04, "synthetic", True)

    def get_dividend_yield(self, ticker: str, spot: float) -> RateQuote:
        return RateQuote(0.013, "synthetic", True)

    def get_next_earnings_date(self, ticker: str):
        return None
