from datetime import date

import pytest

from app.analytics import explain
from app.analytics.anomaly import score_anomalies


@pytest.fixture
def facts(synthetic_chains):
    chains, spot, rv = synthetic_chains
    top = score_anomalies(chains, rv, "SPY").iloc[0]
    return explain.build_fact_sheet(top, spot, date(2099, 1, 1))


def test_template_uses_only_fact_numbers(facts):
    text = explain.template_explanation(facts)
    assert explain.find_unverified_numbers(text, facts) == []
    assert "potential volatility anomaly" in text and "guaranteed" in text


def test_detects_invented_numbers(facts):
    bad = f"The option has IV of {facts['implied_vol']} but similar options trade at 31.7% and earnings are on 2031-05-05."
    found = explain.find_unverified_numbers(bad, facts)
    assert "31.7" in found


def test_accepts_numbers_copied_from_facts(facts):
    ok = (f"{facts['ticker']} {facts['strike']} {facts['option_type']} expiring {facts['expiration']} shows "
          f"{facts['implied_vol']} implied vol vs {facts['realized_vol_30d']} 30-day realized; score "
          f"{facts['anomaly_score']} out of 100.")
    assert explain.find_unverified_numbers(ok, facts) == []


def test_llm_without_key_falls_back(facts, monkeypatch):
    monkeypatch.setattr(explain.config, "ANTHROPIC_API_KEY", "")
    ex = explain.llm_explanation(facts, api_key="")
    assert ex.source == "rule-based" and "No ANTHROPIC_API_KEY" in ex.note


def test_llm_output_with_invented_number_is_rejected(facts, monkeypatch):
    class FakeBlock:
        type = "text"
        text = "Implied vol is 99.9%, far above neighbors."

    class FakeClient:
        def __init__(self, api_key):
            self.messages = self

        def create(self, **kw):
            assert "Never introduce any number" in kw["system"]
            return type("M", (), {"content": [FakeBlock()]})()

    import anthropic
    monkeypatch.setattr(anthropic, "Anthropic", FakeClient)
    ex = explain.llm_explanation(facts, api_key="test-key")
    assert ex.source == "rule-based" and "99.9" in ex.note


def test_llm_output_with_only_fact_numbers_is_accepted(facts, monkeypatch):
    class FakeBlock:
        type = "text"
        text = f"Implied vol is {facts['implied_vol']}, a potential volatility anomaly with score {facts['anomaly_score']}."

    class FakeClient:
        def __init__(self, api_key):
            self.messages = self

        def create(self, **kw):
            return type("M", (), {"content": [FakeBlock()]})()

    import anthropic
    monkeypatch.setattr(anthropic, "Anthropic", FakeClient)
    ex = explain.llm_explanation(facts, api_key="test-key")
    assert ex.source == "ai"
