"""Shared test fixtures — offline, deterministic.

- sample_ohlcv: committed synthetic OHLCV fixture (520 trading days)
- patch_ohlc: routes all OHLC fetches to the fixture (no network)
- tmp_model_dir / tmp_db: isolate model bundles and SQLite state
- _test_auth (autouse): bcrypt-hashed test credentials admin:changeme
- _no_rate_limit (autouse): disables slowapi limits inside tests
"""

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from app.auth import hash_password
from app.config import settings

FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "sample_ohlcv.csv"

# Hash once per session — bcrypt is deliberately slow
_TEST_HASH = hash_password("changeme")


@pytest.fixture(autouse=True)
def _test_auth(monkeypatch):
    """All tests run with admin:changeme (bcrypt-hashed, as production requires)."""
    monkeypatch.setattr(settings, "auth_users", f"admin:{_TEST_HASH}")


@pytest.fixture(autouse=True)
def _no_rate_limit():
    """Disable slowapi limits so repeated train/backtest calls don't 429."""
    try:
        from app.main import limiter

        previous = limiter.enabled
        limiter.enabled = False
        yield
        limiter.enabled = previous
    except ImportError:
        yield


@pytest.fixture
def sample_ohlcv() -> pd.DataFrame:
    """The committed 520-day synthetic OHLCV fixture."""
    df = pd.read_csv(FIXTURE_PATH, index_col="Date", parse_dates=True)
    df.index.name = "Date"
    return df


@pytest.fixture
def patch_ohlc(monkeypatch, sample_ohlcv):
    """Route every fetch_ohlc call (any ticker) to the fixture.

    Patches both the source module and the name imported into
    feature_builder. Respects years and as_of_date like the real fetcher.
    """

    def fake_fetch_ohlc(ticker, years=3, as_of_date=None, use_cache=True):
        df = sample_ohlcv.copy()
        if as_of_date is not None:
            cutoff = as_of_date if isinstance(as_of_date, date) else pd.Timestamp(as_of_date).date()
            df = df.loc[[d.date() <= cutoff for d in df.index]]
        n = int(years * 252)
        return df.tail(n) if len(df) > n else df

    import data.fetch_ohlc
    import features.feature_builder

    monkeypatch.setattr(data.fetch_ohlc, "fetch_ohlc", fake_fetch_ohlc)
    monkeypatch.setattr(features.feature_builder, "fetch_ohlc", fake_fetch_ohlc)
    return fake_fetch_ohlc


@pytest.fixture
def tmp_model_dir(monkeypatch, tmp_path):
    """Save/load model bundles under a temp directory."""
    import model.model_registry

    model_dir = tmp_path / "saved_models"
    model_dir.mkdir()
    monkeypatch.setattr(model.model_registry, "MODEL_DIR", model_dir)
    return model_dir


@pytest.fixture
def tmp_db(monkeypatch, tmp_path):
    """Point all SQLite state (main DB + cache) at a temp directory."""
    monkeypatch.setattr(settings, "data_dir", tmp_path)
    return tmp_path
