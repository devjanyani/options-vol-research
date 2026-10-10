"""
Backtest on COLLECTED option-chain snapshots (real historical quotes that this
app saved day by day).

WALK-FORWARD, NO LOOK-AHEAD
---------------------------
For each snapshot date d, in chronological order:
  1. Score anomalies using ONLY the rows stored on date d (their IVs, quotes,
     and the 30-day realized vol that was known on d).
  2. Take the top `max_trades_per_day` options with score >= threshold.
  3. Trade direction: IV above neighbors ("rich") -> SELL the option;
     IV below neighbors ("cheap") -> BUY it. Delta-hedge once with stock.
  4. Exit `holding_days` business days later at that day's stored quote.
     While a contract is held, new signals on the same contract are skipped.
Future snapshots are used only to look up the exit price of a trade that was
already decided.

Trades whose exit day has not happened yet, or whose contract has no
two-sided quote on the exit day, are reported as "unresolved" and excluded
from P&L (never guessed). Options expiring before the planned exit are not
traded, so there is no expiry settlement to model.

Gross P&L uses mid prices. Net P&L buys at the ask / sells at the bid and
adds slippage, fees and the stock-hedge cost.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from app import config
from app.analytics.anomaly import score_anomalies
from app.backtesting.costs import option_leg_cost, stock_hedge_cost


@dataclass(frozen=True)
class SnapshotParams:
    holding_days: int = 5
    threshold: float = 50.0
    max_trades_per_day: int = 3
    delta_hedge: bool = True
    contracts: int = 1
    exit_search_days: int = 3   # if the exact exit day is missing, use the next snapshot within N days
    allow_reentry: bool = False  # if False, skip a contract that already has an open trade


def prepare_snapshot_day(day: pd.DataFrame) -> pd.DataFrame:
    """Add the columns the anomaly scorer needs, from stored values only."""
    d = day.copy()
    d["is_liquid"] = d["is_liquid"].astype(bool)
    d["log_moneyness"] = np.log(d["strike"] / d["spot"])
    d["moneyness"] = d["strike"] / d["spot"]
    d["is_otm"] = np.where(d["option_type"] == "call", d["strike"] >= d["spot"], d["strike"] <= d["spot"])
    d["rel_spread"] = (d["ask"] - d["bid"]) / d["mid"]
    return d


def signals_for_day(day: pd.DataFrame, p: SnapshotParams) -> pd.DataFrame:
    d = prepare_snapshot_day(day)
    rv30 = float(d["rv30"].dropna().iloc[0]) if d["rv30"].notna().any() else float("nan")
    scored = score_anomalies(d, rv30, str(d["ticker"].iloc[0]))
    if scored.empty:
        return scored
    min_cal_days = int(np.ceil(p.holding_days * 7 / 5)) + 3
    sig = scored[(scored["score"] >= p.threshold) & (scored["dte"] > min_cal_days)]
    return sig.head(p.max_trades_per_day)


def generate_signals(snapshots: pd.DataFrame, p: SnapshotParams) -> pd.DataFrame:
    out = []
    for d, day in snapshots.groupby("snapshot_date", sort=True):
        s = signals_for_day(day, p)
        if not s.empty:
            out.append(s.assign(signal_date=d))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def run_snapshot_backtest(snapshots: pd.DataFrame, p: SnapshotParams = SnapshotParams(),
                          cm: config.CostModel = config.SETTINGS.costs) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Returns (trades, unresolved, signals)."""
    if snapshots.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    snaps = snapshots.copy()
    snaps["snapshot_date"] = pd.to_datetime(snaps["snapshot_date"])
    dates = sorted(snaps["snapshot_date"].unique())
    by_contract = snaps.set_index(["contract_symbol", "snapshot_date"]).sort_index()

    signals = generate_signals(snaps.assign(snapshot_date=snaps["snapshot_date"].dt.strftime("%Y-%m-%d")), p)
    trades, unresolved = [], []
    open_until: dict[str, pd.Timestamp] = {}
    m = config.CONTRACT_MULTIPLIER * p.contracts
    for s in signals.itertuples():
        entry = pd.Timestamp(s.signal_date)
        if not p.allow_reentry and s.contract_symbol in open_until and entry < open_until[s.contract_symbol]:
            continue  # already holding this contract
        target = entry + pd.offsets.BDay(p.holding_days)
        candidates = [d for d in dates if target <= d <= target + pd.Timedelta(days=p.exit_search_days)]
        desc = f"{'Sell' if s.direction == 'rich' else 'Buy'} {s.ticker} {s.expiration} {s.strike:g} {s.option_type}"
        if not candidates:
            unresolved.append({"signal_date": s.signal_date, "description": desc, "reason": "exit date not reached / no snapshot"})
            continue
        exit_day = candidates[0]
        open_until[s.contract_symbol] = exit_day
        try:
            ex = by_contract.loc[(s.contract_symbol, exit_day)]
        except KeyError:
            unresolved.append({"signal_date": s.signal_date, "description": desc, "reason": "contract missing on exit day"})
            continue
        if not (pd.notna(ex["mid"]) and ex["bid"] > 0 and ex["ask"] > 0):
            unresolved.append({"signal_date": s.signal_date, "description": desc, "reason": "no two-sided quote on exit day"})
            continue

        side = -1 if s.direction == "rich" else 1     # sell rich, buy cheap
        hedge_shares = -side * s.delta * m if p.delta_hedge else 0.0
        option_pnl = side * (ex["mid"] - s.mid) * m
        hedge_pnl = hedge_shares * (ex["spot"] - s.spot)
        gross = option_pnl + hedge_pnl
        c_in = option_leg_cost(s.bid, s.ask, p.contracts, cm)
        c_out = option_leg_cost(ex["bid"], ex["ask"], p.contracts, cm)
        c_hedge = 2 * stock_hedge_cost(hedge_shares, cm)
        costs = c_in.total + c_out.total + c_hedge
        trades.append({
            "entry_date": entry, "exit_date": exit_day, "description": desc,
            "side": "short option" if side < 0 else "long option", "score": s.score,
            "iv_entry": s.computed_iv, "iv_exit": ex["computed_iv"], "spot_entry": s.spot, "spot_exit": ex["spot"],
            "entry_price": s.mid, "exit_price": ex["mid"], "option_pnl": option_pnl, "hedge_pnl": hedge_pnl,
            "gross_pnl": gross, "cost_spread": c_in.half_spread + c_out.half_spread,
            "cost_slippage": c_in.slippage + c_out.slippage, "cost_fees": c_in.fees + c_out.fees,
            "cost_hedge": c_hedge, "costs": costs, "net_pnl": gross - costs, "capital": s.mid * m,
        })
    return pd.DataFrame(trades), pd.DataFrame(unresolved), signals
