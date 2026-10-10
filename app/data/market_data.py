"""
Market-data access.

All calls to Yahoo Finance (through the `yfinance` package) live in this file
and nowhere else. The rest of the app talks to a `MarketDataProvider`, so the
data source can be swapped (e.g. for a paid vendor) without touching any
calculation or UI code.

Honesty rules enforced here:
- If Yahoo returns nothing, we raise `DataUnavailableError` with a clear
  message. We never fill gaps with made-up numbers.
- When we fall back to an assumption (e.g. a default risk-free rate), the
  returned object says so in its `source` field so the UI can display it.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import date, datetime
from typing import Protocol

import numpy as np
import pandas as pd

from app import config


class DataUnavailableError(RuntimeError):
    """Raised when the data source cannot provide what was asked for."""


@dataclass(frozen=True)
class SpotQuote:
    price: float
    previous_close: float | None
    as_of: datetime
    source: str


@dataclass(frozen=True)
class RateQuote:
    value: float          # decimal, e.g. 0.043 = 4.3%
    source: str           # where the number came from (or "assumption")
    is_assumption: bool


# Standard column names used everywhere else in the app.
CHAIN_COLUMNS = [
    "contract_symbol",
    "option_type",      # "call" or "put"
    "expiration",       # "YYYY-MM-DD"
    "strike",
    "bid",
    "ask",
    "last_price",
    "volume",
    "open_interest",
    "yahoo_iv",         # Yahoo's own IV estimate (decimal). We do not trust it blindly.
    "last_trade_date",
]


class MarketDataProvider(Protocol):
    """The interface every data source must implement."""

    name: str

    def get_spot(self, ticker: str) -> SpotQuote: ...

    def get_price_history(self, ticker: str, period: str = "2y") -> pd.Series: ...

    def get_expirations(self, ticker: str) -> list[str]: ...

    def get_option_chain(self, ticker: str, expiration: str) -> pd.DataFrame: ...

    def get_risk_free_rate(self) -> RateQuote: ...

    def get_dividend_yield(self, ticker: str, spot: float) -> RateQuote: ...

    def get_next_earnings_date(self, ticker: str) -> date | None: ...


def _clean_float(x) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return float("nan")
    return v


class YFinanceProvider:
    """Free data from Yahoo Finance via yfinance. Quotes may be delayed."""

    name = "Yahoo Finance (yfinance)"

    def __init__(self, request_pause_s: float = 0.25):
        import yfinance as yf  # imported lazily so tests don't need network

        self._yf = yf
        self._pause = request_pause_s

    def _ticker(self, ticker: str):
        return self._yf.Ticker(ticker.upper().strip())

    def _history(self, ticker: str, **kw) -> pd.DataFrame:
        """yfinance.history with network failures turned into a clear message."""
        try:
            return self._ticker(ticker).history(**kw)
        except Exception as exc:  # noqa: BLE001 - yfinance raises many types when offline
            raise DataUnavailableError(
                f"Could not reach Yahoo Finance for '{ticker}' ({type(exc).__name__}). Check your internet "
                "connection; Yahoo may also be rate-limiting, so wait a minute and press 'Refresh data'."
            ) from exc

    # ------------------------------------------------------------------ spot
    def get_spot(self, ticker: str) -> SpotQuote:
        t = self._ticker(ticker)
        price = prev = None
        try:
            fi = t.fast_info
            price = _clean_float(fi["lastPrice"])
            prev = _clean_float(fi["previousClose"])
        except Exception:
            pass
        if price is None or not np.isfinite(price) or price <= 0:
            hist = self._history(ticker, period="5d", auto_adjust=False)
            if hist is None or hist.empty:
                raise DataUnavailableError(f"No price data returned for '{ticker}'. Check the ticker symbol.")
            price = float(hist["Close"].iloc[-1])
            prev = float(hist["Close"].iloc[-2]) if len(hist) > 1 else None
        return SpotQuote(
            price=float(price),
            previous_close=prev if prev is not None and np.isfinite(prev) else None,
            as_of=pd.Timestamp.now(tz=config.MARKET_TZ).to_pydatetime(),
            source=self.name,
        )

    # --------------------------------------------------------- price history
    def get_price_history(self, ticker: str, period: str = "2y") -> pd.Series:
        """
        Daily closing prices, adjusted for splits and dividends.

        Adjusted prices matter for volatility: on a dividend date the raw price
        drops by the dividend amount, which is not a real "move" of the stock.
        """
        hist = self._history(ticker, period=period, auto_adjust=True)
        if hist is None or hist.empty or "Close" not in hist:
            raise DataUnavailableError(f"No price history returned for '{ticker}'.")
        s = hist["Close"].astype(float).dropna()
        s.index = pd.to_datetime(s.index).tz_localize(None).normalize()
        s.name = ticker.upper()
        return s

    # ---------------------------------------------------------- expirations
    def get_expirations(self, ticker: str) -> list[str]:
        try:
            exps = list(self._ticker(ticker).options or [])
        except Exception as exc:  # noqa: BLE001
            raise DataUnavailableError(f"Could not download the option expirations for '{ticker}' "
                                       f"({type(exc).__name__}). Yahoo may be rate-limiting.") from exc
        if not exps:
            raise DataUnavailableError(
                f"No listed options found for '{ticker}'. It may not have options, or Yahoo is rate-limiting."
            )
        return exps

    # --------------------------------------------------------- option chain
    def get_option_chain(self, ticker: str, expiration: str) -> pd.DataFrame:
        time.sleep(self._pause)  # be polite to the free API
        chain = self._ticker(ticker).option_chain(expiration)
        frames = []
        for option_type, df in (("call", chain.calls), ("put", chain.puts)):
            if df is None or df.empty:
                continue
            out = pd.DataFrame(
                {
                    "contract_symbol": df["contractSymbol"].astype(str),
                    "option_type": option_type,
                    "expiration": expiration,
                    "strike": df["strike"].astype(float),
                    "bid": pd.to_numeric(df["bid"], errors="coerce"),
                    "ask": pd.to_numeric(df["ask"], errors="coerce"),
                    "last_price": pd.to_numeric(df["lastPrice"], errors="coerce"),
                    "volume": pd.to_numeric(df["volume"], errors="coerce").fillna(0).astype(int),
                    "open_interest": pd.to_numeric(df["openInterest"], errors="coerce").fillna(0).astype(int),
                    "yahoo_iv": pd.to_numeric(df["impliedVolatility"], errors="coerce"),
                    "last_trade_date": pd.to_datetime(df["lastTradeDate"], errors="coerce", utc=True),
                }
            )
            frames.append(out)
        if not frames:
            raise DataUnavailableError(f"Empty option chain for {ticker} {expiration}.")
        return pd.concat(frames, ignore_index=True)[CHAIN_COLUMNS]

    # ------------------------------------------------------- risk-free rate
    def get_risk_free_rate(self) -> RateQuote:
        """
        The 13-week US Treasury bill yield (Yahoo symbol ^IRX), quoted in
        percent. It is the standard short-term "risk-free" rate proxy.
        """
        try:
            hist = self._ticker("^IRX").history(period="10d")
            v = float(hist["Close"].dropna().iloc[-1]) / 100.0
            if 0.0 <= v < 0.25:
                return RateQuote(v, "13-week T-bill yield (^IRX)", False)
        except Exception:
            pass
        return RateQuote(config.FALLBACK_RISK_FREE_RATE, "assumption (^IRX unavailable)", True)

    # -------------------------------------------------------- dividend yield
    def get_dividend_yield(self, ticker: str, spot: float) -> RateQuote:
        """
        Trailing-12-month dividends divided by the current price.
        Computed by us from the dividend history, which is more transparent
        than Yahoo's pre-computed 'dividendYield' field.
        """
        try:
            divs = self._ticker(ticker).dividends
            if divs is None or divs.empty:
                return RateQuote(0.0, "no dividends in history", False)
            idx = pd.to_datetime(divs.index).tz_localize(None)
            cutoff = pd.Timestamp.now().normalize() - pd.Timedelta(days=365)
            ttm = float(divs[idx >= cutoff].sum())
            return RateQuote(ttm / spot if spot > 0 else 0.0, "trailing 12-month dividends / price", False)
        except Exception:
            return RateQuote(0.0, "assumption (dividend data unavailable)", True)

    # -------------------------------------------------------- earnings date
    def get_next_earnings_date(self, ticker: str) -> date | None:
        """Next scheduled earnings date if Yahoo lists one. ETFs have none."""
        try:
            cal = self._ticker(ticker).calendar
            dates = cal.get("Earnings Date") if isinstance(cal, dict) else None
            if not dates:
                return None
            today = date.today()
            future = sorted(d for d in dates if isinstance(d, date) and d >= today)
            return future[0] if future else None
        except Exception:
            return None


def get_default_provider() -> MarketDataProvider:
    """
    Returns the provider the app should use.

    OVRP_PROVIDER=synthetic is ONLY for the developer UI test harness (it lets
    the UI be rendered without internet). The app shows a large red
    "SYNTHETIC TEST DATA" banner whenever it is active.
    """
    import os

    if os.getenv("OVRP_PROVIDER", "").lower() == "synthetic":
        from app.data.synthetic import SyntheticProvider

        return SyntheticProvider()
    return YFinanceProvider()


def provider_is_synthetic(provider: MarketDataProvider) -> bool:
    return getattr(provider, "is_synthetic", False)
