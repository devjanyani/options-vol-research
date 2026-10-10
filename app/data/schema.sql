-- SQLite schema. Every table that holds prices or results carries a
-- `provenance` column: LIVE, COLLECTED, MODELED or SYNTHETIC.
-- Types are kept ANSI-friendly so the same schema ports to PostgreSQL.

CREATE TABLE IF NOT EXISTS price_history (
    ticker      TEXT NOT NULL,
    date        TEXT NOT NULL,          -- YYYY-MM-DD
    close       REAL NOT NULL,          -- split/dividend adjusted close
    provenance  TEXT NOT NULL,
    PRIMARY KEY (ticker, date)
);

-- One row per option contract per collection day.
CREATE TABLE IF NOT EXISTS option_snapshots (
    snapshot_date    TEXT NOT NULL,      -- New York trading date of collection
    collected_at     TEXT NOT NULL,      -- full timestamp (ET)
    ticker           TEXT NOT NULL,
    contract_symbol  TEXT NOT NULL,
    option_type      TEXT NOT NULL,
    expiration       TEXT NOT NULL,
    strike           REAL NOT NULL,
    bid              REAL,
    ask              REAL,
    last_price       REAL,
    volume           INTEGER,
    open_interest    INTEGER,
    yahoo_iv         REAL,
    spot             REAL NOT NULL,
    risk_free_rate   REAL NOT NULL,
    dividend_yield   REAL NOT NULL,
    rv30             REAL,               -- 30d realized vol known on snapshot_date
    dte              INTEGER,
    t_years          REAL,
    mid              REAL,
    computed_iv      REAL,
    iv_status        TEXT,
    delta            REAL,
    gamma            REAL,
    vega             REAL,
    theta            REAL,
    is_liquid        INTEGER,
    provenance       TEXT NOT NULL,
    PRIMARY KEY (snapshot_date, contract_symbol)
);
CREATE INDEX IF NOT EXISTS ix_snap_ticker_date ON option_snapshots (ticker, snapshot_date);

CREATE TABLE IF NOT EXISTS signals (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at       TEXT NOT NULL,
    signal_date      TEXT NOT NULL,
    ticker           TEXT NOT NULL,
    contract_symbol  TEXT,
    option_type      TEXT,
    expiration       TEXT,
    strike           REAL,
    iv               REAL,
    rv30             REAL,
    score            REAL,
    direction        TEXT,               -- 'rich' or 'cheap'
    provenance       TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS backtest_runs (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at   TEXT NOT NULL,
    kind         TEXT NOT NULL,          -- 'collected_snapshots' or 'modeled_proxy'
    provenance   TEXT NOT NULL,
    params_json  TEXT NOT NULL,
    metrics_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS backtest_trades (
    run_id        INTEGER NOT NULL REFERENCES backtest_runs(id),
    trade_no      INTEGER NOT NULL,
    entry_date    TEXT NOT NULL,
    exit_date     TEXT NOT NULL,
    description   TEXT NOT NULL,
    side          TEXT NOT NULL,
    entry_price   REAL,
    exit_price    REAL,
    gross_pnl     REAL NOT NULL,
    costs         REAL NOT NULL,
    net_pnl       REAL NOT NULL,
    capital       REAL,
    PRIMARY KEY (run_id, trade_no)
);
