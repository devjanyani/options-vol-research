"""Section 7: Backtest. Collected-snapshot backtest and the MODELED SPY/VIX proxy backtest."""

from __future__ import annotations

import html
import json
from dataclasses import asdict

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app import config
from app.backtesting.metrics import equity_curves, summarize
from app.backtesting.proxy_backtest import ProxyParams, run_proxy_backtest, sensitivity_grid
from app.backtesting.snapshot_backtest import SnapshotParams, run_snapshot_backtest
from app.data.market_data import get_default_provider
from app.data.snapshots import collect_snapshot
from app.ui import components as c
from app.ui import state, theme
from app.utils.dates import is_market_open, market_now
from app.utils.provenance import DataKind

METRIC_ROWS = [
    ("trades", "Number of trades", lambda v: f"{v:,}"),
    ("win_rate", "Win rate", lambda v: c.pct(v)),
    ("avg_return", "Average return per trade (on premium)", lambda v: c.pct(v, 2)),
    ("total_pnl", "Total simulated P&L", lambda v: c.money(v, 0)),
    ("max_drawdown", "Maximum drawdown", lambda v: c.money(v, 0)),
    ("sharpe", "Sharpe ratio (annualized)", lambda v: c.num(v, 2)),
    ("worst_trade", "Worst single trade", lambda v: c.money(v, 0)),
]


def _cost_model_inputs(key: str) -> config.CostModel:
    d = config.SETTINGS.costs
    with st.expander("Transaction-cost assumptions (edit to test sensitivity)"):
        a, b, cc, dd = st.columns(4)
        fee = a.number_input("Fee per option contract ($)", 0.0, 5.0, d.option_fee_per_contract, 0.05, key=f"{key}_fee")
        slip = b.number_input("Slippage (% of spread)", 0.0, 100.0, d.slippage_fraction_of_spread * 100, 5.0,
                              key=f"{key}_slip") / 100
        stock = cc.number_input("Stock hedge cost ($/share)", 0.0, 0.10, d.stock_cost_per_share, 0.005,
                                format="%.3f", key=f"{key}_stk")
        spread = dd.number_input("Modeled bid/ask (% of option price)", 0.0, 20.0, d.modeled_spread_fraction * 100, 0.5,
                                 key=f"{key}_spr", help="Only used by the modeled backtest, which has no real quotes.") / 100
    return config.CostModel(fee, slip, stock, spread)


def _metrics_table(trades: pd.DataFrame) -> pd.DataFrame:
    g, n = summarize(trades, "gross_pnl"), summarize(trades, "net_pnl")
    return pd.DataFrame({"Metric": [lbl for _, lbl, _ in METRIC_ROWS],
                         "Gross (before costs)": [f(g[k]) for k, _, f in METRIC_ROWS],
                         "Net (after costs)": [f(n[k]) for k, _, f in METRIC_ROWS]})


def _results(trades: pd.DataFrame, kind: DataKind, title: str) -> None:
    if trades.empty:
        st.info("No trades were generated with these settings.")
        return
    st.markdown(f"{c.badge(kind)} <b>{html.escape(title)}</b>", unsafe_allow_html=True)
    left, right = st.columns([3, 2])
    with left:
        st.dataframe(_metrics_table(trades), hide_index=True, use_container_width=True)
    with right:
        cost_lines = [(lbl, trades[col].sum()) for col, lbl in
                      [("cost_spread", "Bid/ask spread"),
                       ("cost_slippage", "Slippage"), ("cost_fees", "Commissions & fees"),
                       ("cost_hedge", "Stock hedge costs")] if col in trades]
        c.pnl_stack(trades["gross_pnl"].sum(), cost_lines, trades["net_pnl"].sum())
        gross, net = trades["gross_pnl"].sum(), trades["net_pnl"].sum()
        if gross > 0 >= net:
            c.callout("<b>Profitable before costs, unprofitable after.</b> This is exactly the gap between theoretical "
                      "and executable profits.", "warn")
        elif gross > 0:
            c.callout(f"Costs consumed <b>{(gross - net) / gross:.0%}</b> of the gross profit.")

    st.session_state["last_backtest"] = {"kind": kind.value, "title": title, "net": summarize(trades, "net_pnl")}
    curves = equity_curves(trades)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=curves.index, y=curves["gross"], name="Gross (mid prices)", mode="lines",
                             line=dict(color=theme.SERIES[0], width=2)))
    fig.add_trace(go.Scatter(x=curves.index, y=curves["net"], name="Net (after costs)", mode="lines",
                             line=dict(color=theme.SERIES[1], width=2)))
    fig.add_hline(y=0, line=dict(color=theme.MUTED, width=1))
    fig.update_layout(title="Cumulative simulated P&L, booked at each trade's exit", height=340,
                      yaxis_title="P&L ($, 1 contract per signal)", yaxis_tickprefix="$")
    fig.update_traces(hovertemplate="%{y:$,.0f}")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("P&L is booked on exit dates only, so day-to-day swings inside a trade are not shown. This makes the "
               "curve look smoother than reality: drawdown is understated and Sharpe is optimistic.")
    if len(trades) < 30:
        c.callout(f"Only {len(trades)} trades. Win rate, Sharpe and drawdown from so few trades are statistically "
                  "unreliable.", "warn")

    with st.expander(f"All {len(trades)} trades"):
        st.dataframe(trades, hide_index=True, use_container_width=True)
        st.download_button("Download trades (CSV)", trades.to_csv(index=False).encode(), "trades.csv", "text/csv")


@st.cache_data(show_spinner="Running modeled backtest…")
def _run_proxy(spy: pd.Series, vix: pd.Series, irx: pd.Series | None, params: ProxyParams, cm: config.CostModel):
    return run_proxy_backtest(spy, vix, irx, params, cm)


def _proxy_tab(ctx) -> None:
    c.callout("<b>MODELED RESULTS.</b> Option prices here are computed with Black-Scholes from SPY's price and the "
              "VIX index. They are <b>not</b> historical option quotes, and these are <b>not</b> actual investment "
              "returns. Use this to study how a volatility-selling rule behaves across market regimes.", "warn")
    try:
        spy = state.load_history("SPY", "max")
        vix = state.load_history("^VIX", "max")
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not download SPY/VIX history: {exc}")
        return
    try:
        irx = state.load_history("^IRX", "max")
    except Exception:  # noqa: BLE001
        irx = None
        st.caption(f"T-bill history unavailable; using an assumed {config.FALLBACK_RISK_FREE_RATE:.1%} rate.")

    kind = DataKind.SYNTHETIC if ctx and ctx.synthetic else DataKind.MODELED
    first_year = max(spy.index.min().year, vix.index.min().year) + 1
    a, b, cc, d = st.columns(4)
    start = a.number_input("Start year", first_year, pd.Timestamp.now().year - 1, max(first_year, 2010))
    hold = b.selectbox("Holding period", [5, 10, 20], index=1, format_func=lambda h: f"{h} trading days")
    thr = cc.slider("Signal threshold (VIX − realized, vol pts)", 0.0, 10.0, 4.0, 0.5)
    mode_label = d.selectbox("Rule", ["Sell vol when IV ≫ RV", "Sell when IV ≫ RV, buy when IV ≪ RV", "Always sell (baseline)"])
    mode = {"Sell vol when IV ≫ RV": "short_only", "Sell when IV ≫ RV, buy when IV ≪ RV": "long_short",
            "Always sell (baseline)": "always_short"}[mode_label]
    e, f, _, _ = st.columns(4)
    hedge = e.checkbox("Delta-hedge at entry", value=True)
    adj = f.slider("Subtract from VIX (vol pts)", 0.0, 5.0, 0.0, 0.5,
                   help="VIX usually sits above at-the-money IV. Subtracting points makes the model more conservative.")
    cm = _cost_model_inputs("proxy")
    params = ProxyParams(holding_days=hold, threshold_pts=thr, mode=mode, delta_hedge=hedge,
                         vix_adjustment_pts=adj, start=f"{int(start)}-01-01")

    trades, decisions = _run_proxy(spy, vix, irx, params, cm)
    st.caption(f"{len(decisions)} decision dates checked (every {hold} trading days from {params.start}); "
               f"{len(trades)} trades taken. Data: SPY & VIX daily closes {spy.index.min():%Y}–{spy.index.max():%Y}.")
    _results(trades, kind, f"Modeled straddle strategy · {mode_label} · {hold}-day hold")

    if mode != "always_short" and not trades.empty:
        base, _ = _run_proxy(spy, vix, irx, ProxyParams(**{**asdict(params), "mode": "always_short"}), cm)
        sb, ss = summarize(base, "net_pnl"), summarize(trades, "net_pnl")
        st.markdown("**Does the signal add anything over always selling?**")
        st.dataframe(pd.DataFrame({
            "Rule": [mode_label, "Always sell (baseline)"],
            "Trades": [ss["trades"], sb["trades"]],
            "Net P&L": [c.money(ss["total_pnl"], 0), c.money(sb["total_pnl"], 0)],
            "Net P&L per trade": [c.money(ss["total_pnl"] / max(ss["trades"], 1), 1), c.money(sb["total_pnl"] / max(sb["trades"], 1), 1)],
            "Net Sharpe": [c.num(ss["sharpe"]), c.num(sb["sharpe"])],
            "Max drawdown": [c.money(ss["max_drawdown"], 0), c.money(sb["max_drawdown"], 0)],
        }), hide_index=True, use_container_width=True)

    if st.button("Run sensitivity grid (15 backtests: 3 holding periods × 5 thresholds)"):
        grid = sensitivity_grid(spy, vix, irx, params)
        st.session_state["proxy_grid"] = grid
    if "proxy_grid" in st.session_state:
        g = st.session_state["proxy_grid"]
        st.dataframe(g.assign(net_total_pnl=g["net_total_pnl"].map(lambda v: c.money(v, 0)),
                              max_drawdown=g["max_drawdown"].map(lambda v: c.money(v, 0)),
                              net_sharpe=g["net_sharpe"].map(lambda v: c.num(v)),
                              win_rate=g["win_rate"].map(lambda v: c.pct(v))),
                     hide_index=True, use_container_width=True)
        st.caption("All combinations are shown on purpose. Picking the best cell after seeing this table would be "
                   "data snooping: its result would overstate what you could expect going forward.")

    if not trades.empty and st.button("Save this modeled run to the database"):
        run_id = state.get_repo().save_backtest("modeled_proxy", kind.value, asdict(params) | {"costs": asdict(cm)},
                                                summarize(trades, "net_pnl"), trades)
        st.success(f"Saved as run #{run_id} (provenance {kind.value}).")

    with st.expander("Model limitations (read before quoting any number)"):
        st.markdown(
            "- **VIX is not ATM implied volatility.** It is a variance-swap style index that usually sits a few points "
            "above at-the-money IV, so pricing a short straddle at VIX overstates the premium collected. Use the "
            "'Subtract from VIX' slider to see how sensitive results are.\n"
            "- **No term structure:** the exit price of a ~20-day option uses the 30-day VIX.\n"
            "- **Black-Scholes with one volatility:** no skew, no jumps, European exercise.\n"
            "- **Costs are assumed,** not observed. Real spreads widen sharply in crises, exactly when this strategy loses.\n"
            "- **Static hedge:** delta is hedged once at entry, not rebalanced.\n"
            "- **Adjusted prices, zero dividend yield** approximate SPY's dividends.\n"
            "- **One straddle per signal,** no position sizing or margin modeling.")


def _snapshot_tab(ctx) -> None:
    repo = state.get_repo()
    summary = repo.snapshot_summary()
    st.markdown("Real option quotes this app has saved, one snapshot per trading day. "
                "This is the only fully realistic backtest possible with free data, and it grows over time.")

    a, b = st.columns([2, 3])
    with a:
        if ctx is not None and st.button(f"Collect today's snapshot for {ctx.ticker}", use_container_width=True):
            now = market_now()
            if now.weekday() >= 5 and not ctx.synthetic:
                st.warning("Markets are closed on weekends. Weekend quotes are stale copies of Friday's, so nothing "
                           "was saved. Collect on a trading day, ideally 3:30-3:55 pm ET.")
                return
            if not is_market_open(now):
                st.warning("The market is closed, so quotes may be stale or zero. Best practice: collect between "
                           "3:30 and 3:55 pm ET on trading days.")
            with st.spinner("Downloading and saving chains…"):
                rep = collect_snapshot(get_default_provider(), repo, ctx.ticker)
            st.success(f"Saved {rep.contracts_saved:,} contracts across {rep.expirations_saved} expirations "
                       f"({rep.liquid_contracts:,} liquid) for {rep.snapshot_date}.")
            for err in rep.errors:
                st.caption(f"Skipped {err}")
            summary = repo.snapshot_summary()
        st.caption("Automate it: run `python scripts/collect_snapshot.py` daily (see README for cron / Task Scheduler).")
    with b:
        if summary.empty:
            st.info("No snapshots collected yet.")
        else:
            st.dataframe(summary, hide_index=True, use_container_width=True)

    if summary.empty or ctx is None:
        return
    snaps = repo.load_snapshots(ctx.ticker)
    n_days = snaps["snapshot_date"].nunique() if not snaps.empty else 0
    f1, f2, f3, f4 = st.columns(4)
    hold = f1.selectbox("Holding period", [5, 10, 20], format_func=lambda h: f"{h} trading days", key="snap_hold")
    thr = f2.slider("Score threshold", 20, 90, 50, 5, key="snap_thr")
    per_day = f3.number_input("Max new trades per day", 1, 10, 3)
    hedge = f4.checkbox("Delta-hedge at entry", value=True, key="snap_hedge")
    cm = _cost_model_inputs("snap")
    if n_days < hold + 1:
        c.callout(f"<b>{n_days} day(s) of {ctx.ticker} snapshots collected.</b> A {hold}-day holding period needs at "
                  f"least {hold + 1} trading days of snapshots before a single trade can be evaluated. Keep collecting "
                  "daily. No results are shown until real data supports them.")
        return

    trades, unresolved, signals = run_snapshot_backtest(
        snaps, SnapshotParams(holding_days=hold, threshold=thr, max_trades_per_day=int(per_day), delta_hedge=hedge), cm)
    prov = DataKind(snaps["provenance"].iloc[0])
    st.caption(f"{n_days} snapshot days · {len(signals)} signals · {len(trades)} resolved trades · "
               f"{len(unresolved)} unresolved (exit not yet reached or no clean exit quote)")
    _results(trades, prov, f"Collected-snapshot backtest · {ctx.ticker} · {hold}-day hold")
    if not unresolved.empty:
        with st.expander("Unresolved signals (excluded from P&L, never guessed)"):
            st.dataframe(unresolved, hide_index=True, use_container_width=True)
    if not trades.empty and st.button("Save this run to the database", key="save_snap"):
        rid = repo.save_backtest("collected_snapshots", prov.value,
                                 {"ticker": ctx.ticker, "holding_days": hold, "threshold": thr}, summarize(trades), trades)
        st.success(f"Saved as run #{rid}.")


def render() -> None:
    ctx = state.get_context()
    c.page_header("Backtest", "Historical simulations of trading on volatility signals. Two clearly separated kinds "
                  "of history: real quotes we collected, and a longer modeled proxy.",
                  [DataKind.COLLECTED, DataKind.MODELED])
    c.callout("These are <b>backtested, hypothetical</b> results, not live observations and not actual investment "
              "returns. Every decision uses only information available on its date.")
    tab1, tab2 = st.tabs(["Collected snapshots (real quotes)", "Modeled proxy · SPY + VIX"])
    with tab1:
        _snapshot_tab(ctx)
    with tab2:
        _proxy_tab(ctx)

    runs = state.get_repo().list_backtest_runs()
    if not runs.empty:
        with st.expander(f"Saved runs ({len(runs)})"):
            show = runs.assign(metrics=runs["metrics_json"].map(lambda s: {k: round(v, 3) if isinstance(v, float) else v
                                                                          for k, v in json.loads(s).items()}))
            st.dataframe(show[["id", "created_at", "kind", "provenance", "metrics"]], hide_index=True,
                         use_container_width=True)
    c.learn_panel("backtest")
    c.learn_panel("costs")
    c.learn_panel("sharpe_drawdown")
