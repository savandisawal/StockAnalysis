"""Look-ahead leakage regression tests.

The historical bug: today's fundamental/macro values were broadcast
across all past training rows — every historical row saw the future.
These tests pin the fix:

1. Features built as-of date D must be identical to the same rows built
   with 30 extra days of data (past rows can't change when the future
   arrives).
2. Fundamental/macro columns must vary through time (point-in-time
   series, not a constant broadcast of today's value).
"""

from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

from data.fetch_fundamentals import FundamentalHistory


@pytest.fixture
def patch_pillars(monkeypatch, sample_ohlcv):
    """Deterministic synthetic macro + fundamental histories (no network)."""
    idx = sample_ohlcv.index
    rng = np.random.default_rng(7)

    macro = pd.DataFrame(
        {
            "sp500_change": rng.normal(0, 1, len(idx)).round(3),
            "nasdaq_change": rng.normal(0, 1.2, len(idx)).round(3),
            "nifty_change": rng.normal(0, 0.9, len(idx)).round(3),
            "brent_change": rng.normal(0, 1.5, len(idx)).round(3),
            "usdinr_change": rng.normal(0, 0.3, len(idx)).round(3),
            "vix_value": rng.normal(0, 3, len(idx)).round(3),
        },
        index=idx,
    )

    import features.macro_sentiment

    monkeypatch.setattr(
        features.macro_sentiment,
        "fetch_macro_history_df",
        lambda years=4, use_cache=True: macro,
    )

    # Quarterly/annual fundamental series spanning the fixture period
    q_ends = pd.date_range(idx[0] - timedelta(days=400), idx[-1], freq="QE")
    y_ends = pd.date_range(idx[0] - timedelta(days=1500), idx[-1], freq="YE-MAR")

    def make_history(ticker: str) -> FundamentalHistory:
        seed = sum(ord(c) for c in ticker)
        r = np.random.default_rng(seed)
        eps_q = pd.Series(
            np.linspace(8, 16, len(q_ends)) + r.normal(0, 0.5, len(q_ends)), index=q_ends
        )
        eps_y = pd.Series(
            np.linspace(30, 60, len(y_ends)) + r.normal(0, 2, len(y_ends)), index=y_ends
        )
        return FundamentalHistory(
            ticker=ticker,
            quarterly_eps=eps_q,
            promoter_pct=pd.Series(np.linspace(48, 55, len(q_ends)), index=q_ends),
            annual_eps=eps_y,
            annual_net_profit=eps_y * 10,
            annual_equity=pd.Series(np.linspace(500, 900, len(y_ends)), index=y_ends),
            annual_borrowings=pd.Series(np.linspace(250, 150, len(y_ends)), index=y_ends),
        )

    import data.fetch_fundamentals
    import features.fundamentals

    monkeypatch.setattr(
        data.fetch_fundamentals,
        "fetch_fundamentals_history",
        lambda ticker, use_cache=True: make_history(ticker),
    )
    monkeypatch.setattr(
        features.fundamentals,
        "fetch_fundamentals_history",
        lambda ticker, use_cache=True: make_history(ticker),
    )
    monkeypatch.setattr(
        features.fundamentals, "get_sector_peers", lambda ticker: ["PEERA", "PEERB", "PEERC"]
    )


def _build(as_of=None):
    from features.feature_builder import build_features_for_training

    return build_features_for_training(
        "TESTSTOCK",
        years=3,
        as_of_date=as_of,
        include_fundamentals=True,
        include_macro=True,
    )


class TestNoLookAhead:
    def test_past_rows_unchanged_by_future_data(
        self, patch_ohlc, patch_pillars, tmp_db, sample_ohlcv
    ):
        """Rows as-of D are identical when built with 30 more days of data."""
        cutoff = sample_ohlcv.index[-45].date()

        df_early = _build(as_of=cutoff)
        df_late = _build(as_of=cutoff + timedelta(days=30))

        assert not df_early.empty and not df_late.empty
        common = df_early.index.intersection(df_late.index)
        assert len(common) > 100

        feature_cols = [c for c in df_early.columns if c not in ("target",)]
        pd.testing.assert_frame_equal(
            df_early.loc[common, feature_cols],
            df_late.loc[common, feature_cols],
            check_exact=False,
            rtol=1e-9,
        )

    def test_pillar_columns_vary_through_time(self, patch_ohlc, patch_pillars, tmp_db):
        """Point-in-time joins produce time-varying columns, not constants."""
        df = _build()
        assert len(df) > 200

        for col in ("sp500_change", "nifty_change", "vix_value", "macro_mood"):
            assert df[col].nunique() > 10, f"{col} is (near-)constant — broadcast leak?"

        # PE z-score: with the shared fixture price path the common price
        # factor cancels in the cross-sectional z-score, so it steps only
        # when a quarterly TTM EPS lands — the broadcast bug would give
        # exactly 1 unique value
        assert df["pe_zscore"].nunique() >= 4, "pe_zscore is (near-)constant — broadcast leak?"
        # Annual step features: a ~15-month window spans >= 2 fiscal disclosures
        assert df["eps_cagr_3y"].nunique() >= 2, "eps_cagr_3y never steps — broadcast leak?"
        # Percentile rank can legitimately be stable, but must be a real value
        assert df["roe_percentile"].between(0, 1).all()

    def test_target_is_next_day_return(self, patch_ohlc, patch_pillars, tmp_db, sample_ohlcv):
        """target[t] must equal close-to-close % change realized at t+1."""
        df = _build()
        closes = sample_ohlcv["Close"]
        sample_dates = df.index[:50]
        for d in sample_dates:
            pos = closes.index.get_loc(d)
            expected = (closes.iloc[pos + 1] / closes.iloc[pos] - 1) * 100
            assert df.loc[d, "target"] == pytest.approx(expected, rel=1e-9)
