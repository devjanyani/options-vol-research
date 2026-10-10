"""
Central configuration.

Every assumption the app makes (risk-free fallback, cost model, liquidity
filters, anomaly thresholds) lives in this one file, so it is easy to find,
explain, and change. Nothing here is market data.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

try:  # optional: load a local .env file if python-dotenv is installed
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover
    pass

APP_NAME = "Options Volatility Research Platform"
APP_SHORT = "OVRP"

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Liquid, widely traded underlyings. Users can also type any other ticker.
DEFAULT_TICKERS = ["SPY", "QQQ", "AAPL", "NVDA", "MSFT", "TSLA"]

# There are ~252 trading days in a year. We use this to annualize volatility.
TRADING_DAYS_PER_YEAR = 252

# Option time-to-expiry is measured in calendar time (365 days), because
# interest accrues and time decay happens on weekends too.
CALENDAR_DAYS_PER_YEAR = 365.0

# US equity options stop trading at 4:00 pm New York time on expiration day.
MARKET_TZ = "America/New_York"
EXPIRY_HOUR = 16

# Standard US equity option contract = 100 shares.
CONTRACT_MULTIPLIER = 100

# Realized-volatility look-back windows (trading days).
RV_WINDOWS = (10, 20, 30, 60)

# If the 13-week T-bill yield (^IRX) cannot be downloaded we fall back to this
# number. The UI always shows which one was used.
FALLBACK_RISK_FREE_RATE = 0.04

DB_PATH = Path(os.getenv("OVRP_DB_PATH", PROJECT_ROOT / "data_store" / "options_research.db"))

# LLM explanations (Phase 7). Optional: the app works without a key.
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
LLM_MODEL = os.getenv("OVRP_LLM_MODEL", "claude-haiku-5-5")


@dataclass(frozen=True)
class LiquidityFilter:
    """
    Free option quotes contain a lot of junk: zero bids, huge spreads,
    contracts nobody trades. An "anomaly" in a junk quote is not information,
    it is noise. These thresholds decide what counts as a usable quote.
    """

    min_bid: float = 0.05            # dollars; a zero bid means no real buyer
    max_relative_spread: float = 0.35  # (ask - bid) / mid
    min_open_interest: int = 50       # contracts outstanding
    min_dte: int = 3                  # very short-dated IV is extremely noisy


@dataclass(frozen=True)
class CostModel:
    """
    Transaction-cost assumptions (Phase 9). These are assumptions, not
    measurements, and are shown in the UI next to every net P&L number.

    - Crossing the spread: buying pays the ask, selling receives the bid.
      Relative to the mid price, each trade loses half the spread.
    - Slippage: extra adverse price movement beyond the quoted bid/ask,
      expressed as a fraction of the full spread.
    - Option fee: per-contract commission + exchange/regulatory fees.
      $0.65/contract is a common retail-broker commission level.
    - Stock hedge cost: cost per share to buy/sell the delta hedge.
    """

    option_fee_per_contract: float = 0.65
    slippage_fraction_of_spread: float = 0.10
    stock_cost_per_share: float = 0.01
    # Used only in the modeled (proxy) backtest, where there is no real bid/ask:
    # assumed full bid/ask spread as a fraction of the option's modeled price.
    modeled_spread_fraction: float = 0.02


@dataclass(frozen=True)
class AnomalySettings:
    """Weights and thresholds of the Volatility Anomaly Score (Phase 6)."""

    neighbor_count: int = 3          # strikes on each side used as "nearby"
    weight_strike_neighbors: float = 0.55
    weight_expiry_neighbors: float = 0.30
    weight_rest_of_chain: float = 0.15
    z_cap: float = 4.0               # a 4-sigma combined outlier maps to 100
    min_scale_vol_points: float = 0.005  # floor for the robust scale (0.5 vol pts)
    flag_threshold: float = 50.0
    max_abs_log_moneyness: float = 0.30  # ignore extremely far OTM strikes


@dataclass(frozen=True)
class Settings:
    liquidity: LiquidityFilter = field(default_factory=LiquidityFilter)
    costs: CostModel = field(default_factory=CostModel)
    anomaly: AnomalySettings = field(default_factory=AnomalySettings)


SETTINGS = Settings()
