"""
Data loading for the UI, with caching, plus the sidebar selector.

Streamlit re-runs the script on every click; caching stops us from
re-downloading the same chain from Yahoo each time (and from getting
rate-limited). Cached data expires after a few minutes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import pandas as pd
import streamlit as st

from app import config
from app.analytics.chain import atm_iv, enrich_chain
from app.analytics.realized_vol import realized_vol, realized_vol_table
from app.data.market_data import DataUnavailableError, RateQuote, SpotQuote, get_default_provider
from app.utils.dates import market_now


@dataclass
class MarketContext:
    ticker: str
    spot: SpotQuote
    history: pd.Series
    expirations: list[str]
    rate: RateQuote
    dividend: RateQuote
    earnings_date: date | None
    provider_name: str
    synthetic: bool
    fetched_at: pd.Timestamp
    rv_table: pd.DataFrame = field(default_factory=pd.DataFrame)

    @property
    def rv30(self) -> float:
        return realized_vol(self.history, 30)


@dataclass
class Selection:
    ticker: str
    expiration: str
    option_type: str
    strike: float


def _friendly_error(exc: Exception) -> str:
    msg = str(exc)
    name = type(exc).__name__
    if "RateLimit" in name or "Too Many Requests" in msg or "429" in msg:
        return "Yahoo Finance is rate-limiting requests. Wait a minute and press 'Refresh data'."
    if isinstance(exc, DataUnavailableError):
        return msg
    return f"Could not download market data ({name}: {msg[:160]}). Check your internet connection or try another ticker."


@st.cache_data(ttl=300, show_spinner="Downloading market data…")
def load_market_context(ticker: str) -> MarketContext:
    p = get_default_provider()
    spot = p.get_spot(ticker)
    history = p.get_price_history(ticker, period="2y")
    exps = p.get_expirations(ticker)
    rate = p.get_risk_free_rate()
    div = p.get_dividend_yield(ticker, spot.price)
    earn = p.get_next_earnings_date(ticker)
    return MarketContext(
        ticker=ticker.upper(), spot=spot, history=history, expirations=exps, rate=rate, dividend=div,
        earnings_date=earn, provider_name=p.name, synthetic=getattr(p, "is_synthetic", False),
        fetched_at=market_now(), rv_table=realized_vol_table(history),
    )


@st.cache_data(ttl=300, show_spinner="Downloading option chain…")
def load_raw_chain(ticker: str, expiration: str) -> pd.DataFrame:
    return get_default_provider().get_option_chain(ticker, expiration)


@st.cache_data(ttl=300, show_spinner=False)
def load_enriched_chain(ticker: str, expiration: str, spot: float, r: float, q: float, model_vol: float) -> pd.DataFrame:
    raw = load_raw_chain(ticker, expiration)
    return enrich_chain(raw, spot, r, q, model_vol=model_vol)


@st.cache_data(ttl=600, show_spinner="Downloading option chains across expirations…")
def load_multi_chain(ticker: str, expirations: tuple[str, ...], spot: float, r: float, q: float, model_vol: float) -> pd.DataFrame:
    frames, failed = [], []
    for e in expirations:
        try:
            frames.append(load_enriched_chain(ticker, e, spot, r, q, model_vol))
        except Exception:
            failed.append(e)
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    df.attrs["failed_expirations"] = failed
    return df


@st.cache_data(ttl=3600, show_spinner="Downloading price history…")
def load_history(ticker: str, period: str = "max") -> pd.Series:
    return get_default_provider().get_price_history(ticker, period=period)


def get_context() -> MarketContext | None:
    return st.session_state.get("ctx")


def get_selection() -> Selection | None:
    return st.session_state.get("selection")


def require_context() -> MarketContext:
    ctx = get_context()
    if ctx is None:
        st.info("Market data is not loaded. Pick a ticker in the sidebar (and check the error message there).")
        st.stop()
    return ctx


def require_selection() -> tuple[MarketContext, Selection]:
    ctx = require_context()
    sel = get_selection()
    if sel is None:
        st.info("Select an expiration, option type and strike in the sidebar.")
        st.stop()
    return ctx, sel


def selected_chain(ctx: MarketContext, sel: Selection) -> pd.DataFrame:
    return load_enriched_chain(ctx.ticker, sel.expiration, ctx.spot.price, ctx.rate.value, ctx.dividend.value, ctx.rv30)


def default_surface_expirations(ctx: MarketContext, max_days: int = 200, max_count: int = 10) -> tuple[str, ...]:
    today = market_now().date()
    keep = [e for e in ctx.expirations if 0 < (pd.Timestamp(e).date() - today).days <= max_days]
    return tuple(keep[:max_count])


def render_sidebar() -> None:
    """Ticker / expiration / call-put / strike pickers. Stores results in session_state."""
    sb = st.sidebar
    sb.markdown(f'<div class="ovrp-brand">{config.APP_NAME}</div>', unsafe_allow_html=True)
    sb.caption("Research and education only. No orders are placed.")

    choices = config.DEFAULT_TICKERS + ["Other…"]
    pick = sb.selectbox("Underlying", choices, key="ticker_pick", help="Liquid names give the cleanest option data.")
    ticker = sb.text_input("Ticker symbol", value="", key="ticker_custom").upper().strip() if pick == "Other…" else pick
    if not ticker:
        st.session_state["ctx"] = None
        st.session_state["selection"] = None
        return

    if sb.button("Refresh data", use_container_width=True, help="Clear the cache and download fresh quotes."):
        st.cache_data.clear()

    try:
        ctx = load_market_context(ticker)
    except Exception as exc:  # noqa: BLE001 - show any data error to the user
        sb.error(_friendly_error(exc))
        st.session_state["ctx"] = None
        st.session_state["selection"] = None
        return
    st.session_state["ctx"] = ctx

    today = market_now().date()
    exp_labels = {e: f"{e}  ·  {(pd.Timestamp(e).date() - today).days}d" for e in ctx.expirations}
    default_exp = next((i for i, e in enumerate(ctx.expirations) if (pd.Timestamp(e).date() - today).days >= 25), 0)
    expiration = sb.selectbox("Expiration", ctx.expirations, index=default_exp,
                              format_func=lambda e: exp_labels[e], key=f"exp_{ticker}")
    option_type = sb.radio("Type", ["call", "put"], horizontal=True, key="opt_type",
                           format_func=str.capitalize)

    try:
        chain = load_raw_chain(ticker, expiration)
    except Exception as exc:  # noqa: BLE001
        sb.error(_friendly_error(exc))
        st.session_state["selection"] = None
        return
    strikes = sorted(chain.loc[chain["option_type"] == option_type, "strike"].unique().tolist())
    if not strikes:
        sb.warning("No strikes listed for this expiration and type.")
        st.session_state["selection"] = None
        return
    atm_idx = min(range(len(strikes)), key=lambda i: abs(strikes[i] - ctx.spot.price))
    strike = sb.selectbox("Strike", strikes, index=atm_idx, key=f"strike_{ticker}_{expiration}_{option_type}",
                          format_func=lambda k: f"{k:,.2f}" + ("  ·  ATM" if strikes.index(k) == atm_idx else ""))
    st.session_state["selection"] = Selection(ticker, expiration, option_type, float(strike))

    sb.divider()
    sb.caption(f"Data: {ctx.provider_name}")
    sb.caption(f"Risk-free rate: {ctx.rate.value:.2%} ({ctx.rate.source})")
    sb.caption(f"Dividend yield: {ctx.dividend.value:.2%} ({ctx.dividend.source})")
    sb.caption(f"Loaded {ctx.fetched_at:%Y-%m-%d %H:%M} ET")


__all__ = [
    "MarketContext", "Selection", "atm_iv", "load_market_context", "load_raw_chain", "load_enriched_chain",
    "load_multi_chain", "load_history", "get_context", "get_selection", "require_context", "require_selection",
    "selected_chain", "default_surface_expirations", "render_sidebar",
]


@st.cache_resource
def get_repo():
    """One shared SQLite repository for the whole app session."""
    import os

    from app.data.repository import Repository

    return Repository(os.getenv("OVRP_DB_PATH", str(config.DB_PATH)))
