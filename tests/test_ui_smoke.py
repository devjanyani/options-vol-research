"""
UI smoke tests: render every page headlessly with the SYNTHETIC provider and
assert no exceptions. These check wiring, not numbers.
"""

import os

import pytest

os.environ["OVRP_PROVIDER"] = "synthetic"

from streamlit.testing.v1 import AppTest  # noqa: E402

PAGES = ["market_overview", "options_chain", "vol_analysis", "vol_surface", "anomaly_scanner",
         "trade_simulator", "backtest", "risk", "methodology"]


def _page_script(page: str) -> None:
    import importlib

    from app.ui import state, theme

    theme.inject_css()
    state.render_sidebar()
    importlib.import_module(f"app.ui.pages.{page}").render()


@pytest.fixture(autouse=True)
def _tmp_db(tmp_path_factory, monkeypatch):
    monkeypatch.setenv("OVRP_DB_PATH", str(tmp_path_factory.getbasetemp() / "ui_smoke.db"))
    monkeypatch.setenv("OVRP_PROVIDER", "synthetic")


def _run(page: str) -> AppTest:
    at = AppTest.from_function(_page_script, args=(page,), default_timeout=120)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    return at


@pytest.mark.parametrize("page", PAGES)
def test_page_renders(page):
    _run(page)


def test_main_entry_renders():
    at = AppTest.from_file("../app/main.py", default_timeout=120)
    at.run()
    assert not at.exception, [e.value for e in at.exception]


def test_anomaly_scanner_finds_planted_synthetic_bump():
    at = _run("anomaly_scanner")
    assert any("score 100" in e.label for e in at.expander)


def test_backtest_sensitivity_grid_and_save():
    at = _run("backtest")
    grid_btn = next(b for b in at.button if b.label.startswith("Run sensitivity grid"))
    grid_btn.click().run()
    assert not at.exception
    save = next(b for b in at.button if b.label.startswith("Save this modeled run"))
    save.click().run()
    assert not at.exception
    assert any("Saved as run" in s.value for s in at.success)


def test_collect_snapshot_button():
    at = _run("backtest")
    btn = next(b for b in at.button if b.label.startswith("Collect today's snapshot"))
    btn.click().run()
    assert not at.exception
    assert any("Saved" in s.value for s in at.success)


def test_trade_simulator_sell_hedged_then_risk():
    at = _run("trade_simulator")
    side = next(r for r in at.radio if r.label == "Side")
    side.set_value("Sell").run()
    next(c for c in at.checkbox if c.label.startswith("Delta-hedge")).check().run()
    assert not at.exception
