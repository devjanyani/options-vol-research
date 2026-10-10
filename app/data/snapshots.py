"""
Daily option-chain snapshot collection.

Free data sources only give TODAY's option chain. To ever backtest on real
option quotes, we have to save them ourselves, day by day. Each run stores
every contract for the chosen tickers and expirations, together with the
inputs known AT THAT MOMENT (spot, rates, 30-day realized vol). The backtest
later uses only those stored values, so it can never peek at the future.

Best time to collect: shortly before the 4:00 pm ET close on a trading day,
when quotes are live and two-sided.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from app.analytics.chain import enrich_chain
from app.analytics.realized_vol import realized_vol
from app.data.market_data import MarketDataProvider
from app.data.repository import SNAPSHOT_COLUMNS, Repository
from app.utils.dates import days_to_expiry, market_now
from app.utils.provenance import DataKind


@dataclass
class CollectionReport:
    ticker: str
    snapshot_date: str
    expirations_saved: int
    contracts_saved: int
    liquid_contracts: int
    errors: list[str]


def build_snapshot_frame(enriched: pd.DataFrame, ticker: str, spot: float, r: float, q: float, rv30: float,
                         collected_at: pd.Timestamp, provenance: str) -> pd.DataFrame:
    df = enriched.copy()
    df["snapshot_date"] = collected_at.strftime("%Y-%m-%d")
    df["collected_at"] = collected_at.isoformat()
    df["ticker"] = ticker.upper()
    df["spot"], df["risk_free_rate"], df["dividend_yield"], df["rv30"] = spot, r, q, rv30
    df["t_years"] = df["T"]
    df["mid"] = df["mid"]
    df["is_liquid"] = df["is_liquid"].astype(int)
    df["provenance"] = provenance
    return df[SNAPSHOT_COLUMNS]


def collect_snapshot(provider: MarketDataProvider, repo: Repository, ticker: str,
                     max_expirations: int = 8, max_days: int = 120) -> CollectionReport:
    now = market_now()
    provenance = DataKind.SYNTHETIC.value if getattr(provider, "is_synthetic", False) else DataKind.COLLECTED.value
    errors: list[str] = []
    spot = provider.get_spot(ticker).price
    hist = provider.get_price_history(ticker, period="1y")
    rv30 = realized_vol(hist, 30)
    r = provider.get_risk_free_rate().value
    q = provider.get_dividend_yield(ticker, spot).value
    exps = [e for e in provider.get_expirations(ticker) if 0 <= days_to_expiry(e, now.date()) <= max_days]
    exps = exps[:max_expirations]
    saved = liquid = n_exp = 0
    for e in exps:
        try:
            raw = provider.get_option_chain(ticker, e)
            enriched = enrich_chain(raw, spot, r, q, as_of=now, model_vol=rv30)
            frame = build_snapshot_frame(enriched, ticker, spot, r, q, rv30, now, provenance)
            saved += repo.save_snapshot(frame)
            liquid += int(frame["is_liquid"].sum())
            n_exp += 1
        except Exception as exc:  # noqa: BLE001 - keep collecting other expirations
            errors.append(f"{e}: {type(exc).__name__}: {exc}")
    return CollectionReport(ticker.upper(), now.strftime("%Y-%m-%d"), n_exp, saved, liquid, errors)
