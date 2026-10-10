import pandas as pd
import pytest

from app.analytics.anomaly import robust_scale, score_anomalies, scoring_universe


def test_robust_scale_ignores_outlier():
    x = pd.Series([0.0, 0.01, -0.01, 0.005, -0.005, 0.5])
    assert robust_scale(x, floor=0.0) < 0.05   # std would be ~0.2


def test_robust_scale_floor():
    assert robust_scale(pd.Series([0.0] * 10), floor=0.005) == 0.005


def test_planted_bump_is_top_anomaly(synthetic_chains):
    chains, spot, rv = synthetic_chains
    scored = score_anomalies(chains, rv, "SPY")
    top = scored.iloc[0]
    # The synthetic provider adds +4 vol points to one 30-day put near 95% of spot.
    assert top["option_type"] == "put" and top["dte"] in (29, 30, 31)
    assert top["strike"] == pytest.approx(475.0)
    assert top["direction"] == "rich"
    assert top["score"] == pytest.approx(100.0)
    assert (scored["score"] >= 50).sum() == 1     # nothing else on a smooth smile


def test_scores_bounded_and_universe_filters(synthetic_chains):
    chains, spot, rv = synthetic_chains
    scored = score_anomalies(chains, rv, "SPY")
    assert scored["score"].between(0, 100).all()
    assert scored["is_liquid"].all() and scored["is_otm"].all()
    assert (scored["iv_status"] == "ok").all()


def test_illiquid_bump_is_excluded(synthetic_chains):
    chains, spot, rv = synthetic_chains
    c = chains.copy()
    idx = c[(c["option_type"] == "call") & c["is_otm"] & c["is_liquid"]].index[3]
    c.loc[idx, "computed_iv"] += 0.10
    c.loc[idx, "is_liquid"] = False
    assert idx not in scoring_universe(c).index or c.loc[idx, "contract_symbol"] not in set(
        score_anomalies(c, rv, "SPY")["contract_symbol"])


def test_cheap_direction(synthetic_chains):
    chains, spot, rv = synthetic_chains
    c = chains.copy()
    target = c[(c["option_type"] == "call") & (c["dte"].between(44, 46)) & (c["strike"] == 525.0)].index[0]
    c.loc[target, "computed_iv"] -= 0.04
    scored = score_anomalies(c, rv, "SPY")
    row = scored[scored["contract_symbol"] == c.loc[target, "contract_symbol"]].iloc[0]
    assert row["direction"] == "cheap" and row["score"] >= 50


def test_empty_input():
    assert score_anomalies(pd.DataFrame(), 0.2).empty
