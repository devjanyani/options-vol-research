"""Date and time-to-expiry helpers."""

from __future__ import annotations

from datetime import date, datetime

import pandas as pd

from app import config


def market_now() -> pd.Timestamp:
    return pd.Timestamp.now(tz=config.MARKET_TZ)


def expiry_timestamp(expiration: str) -> pd.Timestamp:
    """Options expire at the 4:00 pm New York close on expiration day."""
    d = pd.Timestamp(expiration)
    return pd.Timestamp(year=d.year, month=d.month, day=d.day, hour=config.EXPIRY_HOUR, tz=config.MARKET_TZ)


def time_to_expiry_years(expiration: str, as_of: pd.Timestamp | datetime | None = None) -> float:
    """
    Time to expiration in years, measured in calendar time to the 4 pm close.
    Example: 30 days -> 30/365 = 0.0822 years.
    """
    as_of = pd.Timestamp(as_of) if as_of is not None else market_now()
    if as_of.tzinfo is None:
        as_of = as_of.tz_localize(config.MARKET_TZ)
    seconds = (expiry_timestamp(expiration) - as_of).total_seconds()
    return max(seconds, 0.0) / (config.CALENDAR_DAYS_PER_YEAR * 24 * 3600)


def days_to_expiry(expiration: str, as_of: date | None = None) -> int:
    """Whole calendar days until expiration (0 on expiration day)."""
    as_of = as_of or market_now().date()
    return (pd.Timestamp(expiration).date() - as_of).days


def is_market_open(now: pd.Timestamp | None = None) -> bool:
    """Regular US session, weekdays 9:30-16:00 ET (ignores exchange holidays)."""
    now = now or market_now()
    if now.weekday() >= 5:
        return False
    minutes = now.hour * 60 + now.minute
    return 9 * 60 + 30 <= minutes < 16 * 60
