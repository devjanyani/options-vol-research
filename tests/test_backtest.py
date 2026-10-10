
import numpy as np
import pandas as pd
import pytest

from app import config
from app.backtesting import snapshot_backtest as sb
from app.backtesting.costs import modeled_option_leg_cost, option_leg_cost, stock_hedge_cost
from app.backtesting.metrics import max_drawdown, sharpe_ratio, summarize
from app.backtesting.proxy_backtest import ProxyParams, decide, run_proxy_backtest, straddle_value
from app.models import black_scholes as bs


# ------------------------------------------------------------------ metrics
def test_max_drawdown_known_series():
    cum = pd.Series(np.cumsum([100, -50, -80, 200, -300]))
    assert max_drawdown(cum) == pytest.approx(-300)


def test_max_drawdown_counts_losses_from_zero():
    assert max_drawdown(pd.Series([-10.0, -40.0, -20.0])) == pytest.approx(-40)


def test_sharpe_known_value():
    x = pd.Series([1.0, -1.0, 1.0, -1.0, 2.0])
    expected = x.mean() / x.std(ddof=1) * np.sqrt(252)
    assert sharpe_ratio(x) == pytest.approx(expected)
    assert np.isnan(sharpe_ratio(pd.Series([5.0, 5.0, 5.0])))


def test_summarize():
    tr = pd.DataFrame({"entry_date": pd.to_datetime(["2024-01-01", "2024-01-08"]),
                       "exit_date": pd.to_datetime(["2024-01-05", "2024-01-12"]),
                       "gross_pnl": [100.0, -40.0], "costs": [10.0, 10.0], "net_pnl": [90.0, -50.0],
                       "capital": [1000.0, 500.0]})
    s = summarize(tr, "net_pnl")
    assert s["trades"] == 2 and s["win_rate"] == 0.5
    assert s["total_pnl"] == pytest.approx(40.0)
    assert s["avg_return"] == pytest.approx((0.09 - 0.10) / 2)
    assert s["max_drawdown"] == pytest.approx(-50.0)


# -------------------------------------------------------------------- costs
def test_option_leg_cost_arithmetic():
    c = option_leg_cost(1.00, 1.10, 2)
    assert c.half_spread == pytest.approx(0.05 * 200)
    assert c.slippage == pytest.approx(0.10 * 0.10 * 200)
    assert c.fees == pytest.approx(0.65 * 2)
    assert c.total == pytest.approx(10 + 2 + 1.3)


def test_modeled_cost_and_hedge_cost():
    c = modeled_option_leg_cost(5.0, 1)
    assert c.half_spread == pytest.approx(0.5 * 0.02 * 5.0 * 100)
    assert stock_hedge_cost(-37) == pytest.approx(0.37)


# ------------------------------------------------------------ proxy backtest
def _series(n=400, seed=1):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2015-01-01", periods=n)
    spy = pd.Series(200 * np.exp(np.cumsum(rng.normal(0, 0.01, n))), index=idx)
    vix = pd.Series(15 + 5 * np.sin(np.arange(n) / 20) + rng.normal(0, 1, n), index=idx)
    return spy, vix


def test_decide_rules():
    assert decide(0.05, "short_only", 0.04) == -1
    assert decide(0.01, "short_only", 0.04) == 0
    assert decide(-0.05, "short_only", 0.04) == 0
    assert decide(-0.05, "long_short", 0.04) == 1
    assert decide(-0.50, "always_short", 0.04) == -1


def test_proxy_no_lookahead_truncation_invariance():
    """Decisions made before a cut-off must not change when later data is removed."""
    spy, vix = _series()
    p = ProxyParams(start="2015-01-01", threshold_pts=2.0, holding_days=5)
    tr_full, dec_full = run_proxy_backtest(spy, vix, None, p)
    cut = spy.index[300]
    tr_cut, dec_cut = run_proxy_backtest(spy[:cut], vix[:cut], None, p)
    common = dec_cut["date"]
    pd.testing.assert_frame_equal(dec_full[dec_full["date"].isin(common)].reset_index(drop=True),
                                  dec_cut.reset_index(drop=True))
    done = tr_cut["exit_date"] <= cut
    pd.testing.assert_frame_equal(tr_full[tr_full["entry_date"].isin(tr_cut.loc[done, "entry_date"])].reset_index(drop=True),
                                  tr_cut[done].reset_index(drop=True))


def test_proxy_future_shock_does_not_change_past_decisions():
    spy, vix = _series()
    p = ProxyParams(start="2015-01-01", threshold_pts=2.0, holding_days=10)
    _, dec_a = run_proxy_backtest(spy, vix, None, p)
    vix2 = vix.copy()
    vix2.iloc[350:] = 80.0   # a crash in the "future"
    _, dec_b = run_proxy_backtest(spy, vix2, None, p)
    before = dec_a["date"] < spy.index[350]
    pd.testing.assert_frame_equal(dec_a[before].reset_index(drop=True), dec_b[dec_b["date"] < spy.index[350]].reset_index(drop=True))


def test_proxy_trade_arithmetic_matches_black_scholes():
    spy, vix = _series()
    p = ProxyParams(start="2015-01-01", mode="always_short", holding_days=10)
    tr, dec = run_proxy_backtest(spy, vix, None, p)
    assert len(tr) == len(dec)
    t = tr.iloc[0]
    r = config.FALLBACK_RISK_FREE_RATE
    T0 = 30 / 365
    v0, *_ , d0 = straddle_value(t.spot_entry, t.spot_entry, T0, r, t.implied_entry)
    assert t.entry_price == pytest.approx(v0)
    T1 = T0 - (t.exit_date - t.entry_date).days / 365
    v1 = bs.call_price(t.spot_exit, t.spot_entry, T1, r, t.implied_exit) + bs.put_price(t.spot_exit, t.spot_entry, T1, r, t.implied_exit)
    assert t.exit_price == pytest.approx(v1)
    expected_gross = -(v1 - v0) * 100 + (d0 * 100) * (t.spot_exit - t.spot_entry)
    assert t.gross_pnl == pytest.approx(expected_gross)
    assert t.net_pnl == pytest.approx(t.gross_pnl - t.costs)
    assert t.costs > 0


def test_short_only_trades_subset_of_always_short():
    spy, vix = _series()
    a, _ = run_proxy_backtest(spy, vix, None, ProxyParams(start="2015-01-01", mode="always_short"))
    s, _ = run_proxy_backtest(spy, vix, None, ProxyParams(start="2015-01-01", mode="short_only", threshold_pts=4))
    assert set(s["entry_date"]).issubset(set(a["entry_date"]))


# --------------------------------------------------------- snapshot backtest
def _snapshot_history(n_days=12):
    """SYNTHETIC snapshots on consecutive business days (test data only)."""
    from app.analytics.chain import enrich_chain
    from app.data.snapshots import build_snapshot_frame
    from app.data.synthetic import SyntheticProvider

    frames = []
    days = pd.bdate_range("2026-03-02", periods=n_days)
    for d in days:
        prov = SyntheticProvider(today=d.date(), expiry_anchor=days[0].date())
        spot = prov.get_spot("SPY").price
        as_of = pd.Timestamp(d.date()).tz_localize(config.MARKET_TZ) + pd.Timedelta(hours=15, minutes=45)
        for e in prov.get_expirations("SPY")[1:5]:
            enriched = enrich_chain(prov.get_option_chain("SPY", e), spot, 0.04, 0.013, as_of=as_of, model_vol=0.18)
            frames.append(build_snapshot_frame(enriched, "SPY", spot, 0.04, 0.013, 0.18, as_of, "SYNTHETIC"))
    return pd.concat(frames, ignore_index=True)


def test_snapshot_signals_no_lookahead():
    snaps = _snapshot_history()
    p = sb.SnapshotParams(holding_days=3)
    full = sb.generate_signals(snaps, p)
    dates = sorted(snaps["snapshot_date"].unique())
    cut = snaps[snaps["snapshot_date"] <= dates[5]]
    part = sb.generate_signals(cut, p)
    assert not part.empty
    cols = ["signal_date", "contract_symbol", "score", "direction"]
    pd.testing.assert_frame_equal(full[full["signal_date"] <= dates[5]][cols].reset_index(drop=True),
                                  part[cols].reset_index(drop=True))


def test_snapshot_backtest_arithmetic_and_unresolved():
    snaps = _snapshot_history()
    p = sb.SnapshotParams(holding_days=3, delta_hedge=True)
    trades, unresolved, signals = sb.run_snapshot_backtest(snaps, p)
    assert not trades.empty
    # The last few signal days cannot have an exit inside the data: they must be unresolved, not guessed.
    assert not unresolved.empty
    assert (unresolved["reason"] == "exit date not reached / no snapshot").any()
    t = trades.iloc[0]
    assert t.side == "short option"  # synthetic bump is 'rich' -> sell
    assert t.gross_pnl == pytest.approx(t.option_pnl + t.hedge_pnl)
    assert t.costs == pytest.approx(t.cost_spread + t.cost_slippage + t.cost_fees + t.cost_hedge)
    assert t.net_pnl == pytest.approx(t.gross_pnl - t.costs)
    assert t.exit_date >= t.entry_date + pd.offsets.BDay(3)


def test_snapshot_backtest_empty():
    tr, un, sig = sb.run_snapshot_backtest(pd.DataFrame())
    assert tr.empty and un.empty and sig.empty
