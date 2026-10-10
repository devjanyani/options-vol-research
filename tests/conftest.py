import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


@pytest.fixture
def synthetic_provider():
    from app.data.synthetic import SyntheticProvider

    return SyntheticProvider()


@pytest.fixture
def synthetic_chains(synthetic_provider):
    """Enriched SYNTHETIC chains for all expirations (test data, not market data)."""
    from app.analytics.chain import enrich_chain
    from app.analytics.realized_vol import realized_vol

    p = synthetic_provider
    spot = p.get_spot("SPY").price
    rv = realized_vol(p.get_price_history("SPY"), 30)
    frames = [enrich_chain(p.get_option_chain("SPY", e), spot, 0.04, 0.013, model_vol=rv) for e in p.get_expirations("SPY")]
    return pd.concat(frames, ignore_index=True), spot, rv
