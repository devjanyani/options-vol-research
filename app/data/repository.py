"""
Storage layer (SQLite).

Everything that reads or writes the database goes through `Repository`.
To move to PostgreSQL later, only this file (and the connection setup)
needs to change; analytics and UI code never see SQL.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pandas as pd

from app import config
from app.utils.dates import market_now

SCHEMA_PATH = Path(__file__).with_name("schema.sql")

SNAPSHOT_COLUMNS = [
    "snapshot_date", "collected_at", "ticker", "contract_symbol", "option_type", "expiration", "strike",
    "bid", "ask", "last_price", "volume", "open_interest", "yahoo_iv", "spot", "risk_free_rate",
    "dividend_yield", "rv30", "dte", "t_years", "mid", "computed_iv", "iv_status", "delta", "gamma",
    "vega", "theta", "is_liquid", "provenance",
]


class Repository:
    def __init__(self, db_path: str | Path = config.DB_PATH):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as con:
            con.executescript(SCHEMA_PATH.read_text())

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.db_path)
        con.execute("PRAGMA foreign_keys = ON")
        return con

    # ----------------------------------------------------------- prices
    def save_prices(self, ticker: str, prices: pd.Series, provenance: str) -> int:
        rows = [(ticker.upper(), pd.Timestamp(d).strftime("%Y-%m-%d"), float(v), provenance)
                for d, v in prices.dropna().items()]
        with self._connect() as con:
            con.executemany("INSERT OR REPLACE INTO price_history VALUES (?,?,?,?)", rows)
        return len(rows)

    def load_prices(self, ticker: str) -> pd.Series:
        with self._connect() as con:
            df = pd.read_sql_query("SELECT date, close FROM price_history WHERE ticker=? ORDER BY date",
                                   con, params=(ticker.upper(),))
        return pd.Series(df["close"].values, index=pd.to_datetime(df["date"]), name=ticker.upper())

    # -------------------------------------------------------- snapshots
    def save_snapshot(self, snapshot: pd.DataFrame) -> int:
        """`snapshot` must contain SNAPSHOT_COLUMNS (built by data/snapshots.py)."""
        df = snapshot[SNAPSHOT_COLUMNS].copy()
        df = df.astype(object).where(pd.notna(df), None)
        placeholders = ",".join("?" * len(SNAPSHOT_COLUMNS))
        with self._connect() as con:
            con.executemany(
                f"INSERT OR REPLACE INTO option_snapshots ({','.join(SNAPSHOT_COLUMNS)}) VALUES ({placeholders})",
                df.itertuples(index=False, name=None),
            )
        return len(df)

    def load_snapshots(self, ticker: str | None = None, provenance: str | None = None) -> pd.DataFrame:
        q, params = "SELECT * FROM option_snapshots WHERE 1=1", []
        if ticker:
            q += " AND ticker=?"; params.append(ticker.upper())
        if provenance:
            q += " AND provenance=?"; params.append(provenance)
        q += " ORDER BY snapshot_date, expiration, option_type, strike"
        with self._connect() as con:
            df = pd.read_sql_query(q, con, params=params)
        if not df.empty:
            df["is_liquid"] = df["is_liquid"].astype(bool)
        return df

    def snapshot_summary(self) -> pd.DataFrame:
        with self._connect() as con:
            return pd.read_sql_query(
                """SELECT ticker, provenance, COUNT(DISTINCT snapshot_date) AS days,
                          MIN(snapshot_date) AS first_date, MAX(snapshot_date) AS last_date,
                          COUNT(*) AS contracts_stored
                   FROM option_snapshots GROUP BY ticker, provenance ORDER BY ticker""", con)

    # ---------------------------------------------------------- signals
    def save_signals(self, signals: pd.DataFrame, provenance: str) -> int:
        if signals.empty:
            return 0
        now = market_now().isoformat()
        rows = [(now, r.signal_date, r.ticker, r.contract_symbol, r.option_type, r.expiration, float(r.strike),
                 float(r.iv), None if pd.isna(r.rv30) else float(r.rv30), float(r.score), r.direction, provenance)
                for r in signals.itertuples()]
        with self._connect() as con:
            con.executemany(
                """INSERT INTO signals (created_at, signal_date, ticker, contract_symbol, option_type, expiration,
                   strike, iv, rv30, score, direction, provenance) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""", rows)
        return len(rows)

    def load_signals(self, ticker: str | None = None) -> pd.DataFrame:
        q, params = "SELECT * FROM signals", []
        if ticker:
            q += " WHERE ticker=?"; params.append(ticker.upper())
        with self._connect() as con:
            return pd.read_sql_query(q + " ORDER BY id DESC", con, params=params)

    # --------------------------------------------------------- backtests
    def save_backtest(self, kind: str, provenance: str, params: dict, metrics: dict, trades: pd.DataFrame) -> int:
        with self._connect() as con:
            cur = con.execute(
                "INSERT INTO backtest_runs (created_at, kind, provenance, params_json, metrics_json) VALUES (?,?,?,?,?)",
                (market_now().isoformat(), kind, provenance, json.dumps(params, default=str),
                 json.dumps(metrics, default=float)),
            )
            run_id = cur.lastrowid
            if not trades.empty:
                rows = [(run_id, i, str(t.entry_date)[:10], str(t.exit_date)[:10], t.description, t.side,
                         float(t.entry_price), float(t.exit_price), float(t.gross_pnl), float(t.costs),
                         float(t.net_pnl), float(t.capital) if pd.notna(t.capital) else None)
                        for i, t in enumerate(trades.itertuples(), start=1)]
                con.executemany("INSERT INTO backtest_trades VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", rows)
        return run_id

    def list_backtest_runs(self) -> pd.DataFrame:
        with self._connect() as con:
            return pd.read_sql_query("SELECT * FROM backtest_runs ORDER BY id DESC", con)

    def load_backtest_trades(self, run_id: int) -> pd.DataFrame:
        with self._connect() as con:
            return pd.read_sql_query("SELECT * FROM backtest_trades WHERE run_id=? ORDER BY trade_no",
                                     con, params=(run_id,))
