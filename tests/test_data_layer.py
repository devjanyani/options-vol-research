import numpy as np
import pandas as pd
import pytest

from app.analytics.chain import atm_iv, enrich_chain
from app.analytics.surface import surface_grid, surface_points, term_structure
from app.data.repository import Repository
from app.data.snapshots import collect_snapshot
from app.risk.position_risk import Position, hedge_shares_for, max_loss_description, position_greeks, scenario_grid


def test_enrich_chain_recovers_synthetic_iv(synthetic_provider):
    p = synthetic_provider
    spot = p.get_spot("SPY").price
    exp = p.get_expirations("SPY")[2]
    ch = enrich_chain(p.get_option_chain("SPY", exp), spot, 0.04, 0.013)
    otm = ch[ch["is_liquid"] & ch["is_otm"]]
    # Synthetic quotes are rounded to cents and priced with a slightly different T,
    # so IVs match only approximately.
    assert np.nanmax(np.abs(otm["computed_iv"] - otm["yahoo_iv"])) < 0.02
    assert 0.15 < atm_iv(ch, spot) < 0.25


def test_liquidity_reasons(synthetic_provider):
    p = synthetic_provider
    spot = p.get_spot("SPY").price
    raw = p.get_option_chain("SPY", p.get_expirations("SPY")[3])
    raw.loc[0, ["bid", "ask"]] = [0.0, 0.0]
    raw.loc[1, "open_interest"] = 0
    ch = enrich_chain(raw, spot, 0.04, 0.013)
    assert ch.loc[0, "liquidity_note"] == "no two-sided quote"
    assert not ch.loc[0, "is_liquid"]


def test_surface_and_term_structure(synthetic_chains):
    chains, spot, _ = synthetic_chains
    pts = surface_points(chains)
    grid, step = surface_grid(pts)
    assert step in (5.0, 10.0) and grid.shape[0] == chains["expiration"].nunique()
    ts = term_structure(chains, spot)
    assert ts["dte"].is_monotonic_increasing and ts["atm_iv"].between(0.1, 0.4).all()


def test_repository_roundtrip(tmp_path, synthetic_provider):
    repo = Repository(tmp_path / "t.db")
    rep = collect_snapshot(synthetic_provider, repo, "SPY", max_expirations=2)
    assert rep.contracts_saved > 0 and not rep.errors
    snaps = repo.load_snapshots("SPY")
    assert len(snaps) == rep.contracts_saved and set(snaps["provenance"]) == {"SYNTHETIC"}
    # Collecting twice on the same day replaces rows instead of duplicating them.
    collect_snapshot(synthetic_provider, repo, "SPY", max_expirations=2)
    assert len(repo.load_snapshots("SPY")) == rep.contracts_saved
    prices = pd.Series([1.0, 2.0], index=pd.to_datetime(["2024-01-02", "2024-01-03"]))
    repo.save_prices("SPY", prices, "LIVE")
    assert repo.load_prices("SPY").tolist() == [1.0, 2.0]
    trades = pd.DataFrame({"entry_date": ["2024-01-02"], "exit_date": ["2024-01-09"], "description": ["x"],
                           "side": ["short vol"], "entry_price": [1.0], "exit_price": [0.5], "gross_pnl": [50.0],
                           "costs": [3.0], "net_pnl": [47.0], "capital": [100.0]})
    rid = repo.save_backtest("modeled_proxy", "MODELED", {"a": 1}, {"total_pnl": 47.0}, trades)
    assert repo.load_backtest_trades(rid)["net_pnl"].tolist() == [47.0]


def test_position_greeks_and_hedge():
    S, K, T, r, iv = 100.0, 100.0, 0.25, 0.04, 0.2
    hs = hedge_shares_for("call", K, T, -1, 2, S, iv, r)
    p = Position("call", K, T, -1, 2, S, iv, r, 0.0, hs)
    g = position_greeks(p)
    assert g["delta_shares"] == pytest.approx(0.0, abs=1e-9)    # delta-neutral
    assert g["vega_dollars"] < 0 and g["theta_dollars"] > 0     # short option: short vega, collects theta
    grid = scenario_grid(p)
    assert grid.loc[0.0, 0.0] == pytest.approx(0.0)
    assert grid.loc[0.10, 0.0] < 0                               # short vol loses when IV rises


def test_max_loss_rules():
    long_call = Position("call", 100, 0.25, 1, 1, 100, 0.2, 0.04)
    assert max_loss_description(long_call, 4.0)[0] == pytest.approx(-400)
    short_call = Position("call", 100, 0.25, -1, 1, 100, 0.2, 0.04)
    assert max_loss_description(short_call, 4.0)[0] == -np.inf
    short_put = Position("put", 100, 0.25, -1, 1, 100, 0.2, 0.04)
    assert max_loss_description(short_put, 4.0)[0] == pytest.approx(-9600)
