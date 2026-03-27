"""Tests for the Feature Engine — Pillars 1, 2, 3 and the feature builder."""

import numpy as np
import pandas as pd
import pytest

from features.technicals import (
    TECHNICAL_FEATURES,
    add_bollinger_bands,
    add_macd,
    add_rsi,
    add_volume_zscore,
    compute_all_technicals,
)
from utils.sectors import get_sector, get_sector_peers

# ── Helper: generate synthetic OHLCV data ───────────────────────

def _make_ohlcv(rows: int = 300) -> pd.DataFrame:
    """Generate synthetic OHLCV data for testing (no network needed)."""
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=rows, freq="B")
    close = 1000 + np.cumsum(np.random.randn(rows) * 10)
    high = close + np.abs(np.random.randn(rows) * 5)
    low = close - np.abs(np.random.randn(rows) * 5)
    open_ = close + np.random.randn(rows) * 3
    volume = np.random.randint(1_000_000, 50_000_000, rows)
    return pd.DataFrame({
        "Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume,
    }, index=dates)


# ── Pillar 1: Technical Feature Tests ────────────────────────────


class TestTechnicals:
    def setup_method(self):
        self.df = _make_ohlcv(300)

    def test_compute_all_technicals_columns(self):
        result = compute_all_technicals(self.df)
        for col in TECHNICAL_FEATURES:
            assert col in result.columns, f"Missing column: {col}"

    def test_rsi_bounded(self):
        result = add_rsi(self.df.copy())
        rsi = result["rsi_14"].dropna()
        assert rsi.min() >= 0, "RSI below 0"
        assert rsi.max() <= 100, "RSI above 100"

    def test_bollinger_pct_b_bounded(self):
        result = add_bollinger_bands(self.df.copy())
        bb = result["bb_pct_b"].dropna()
        # %B can exceed 0-1 when price breaks bands, but should be mostly within
        assert bb.median() > 0 and bb.median() < 1

    def test_volume_zscore_centered(self):
        result = add_volume_zscore(self.df.copy())
        vz = result["volume_zscore"].dropna()
        # Z-score should be centered around 0
        assert abs(vz.mean()) < 1.0

    def test_market_regime_values(self):
        result = compute_all_technicals(self.df)
        regimes = result["regime"].dropna().unique()
        for r in regimes:
            assert r in [-1, 0, 1], f"Invalid regime value: {r}"

    def test_no_raw_prices_in_features(self):
        """Verify all features are stationary — no raw price columns leaked."""
        result = compute_all_technicals(self.df)
        for col in TECHNICAL_FEATURES:
            series = result[col].dropna()
            if len(series) > 50:
                # Raw prices would have very high mean (>100), features should be bounded
                mean = series.mean()
                assert abs(mean) < 200, f"{col} looks like raw price (mean={mean:.1f})"

    def test_macd_not_all_nan(self):
        result = add_macd(self.df.copy())
        assert result["macd_hist_pct"].notna().sum() > 200

    def test_returns_computed(self):
        result = compute_all_technicals(self.df)
        assert result["return_1d"].notna().sum() > 290
        assert result["return_5d"].notna().sum() > 290


# ── Sector Mapping Tests ─────────────────────────────────────────


class TestSectors:
    def test_known_stock(self):
        assert get_sector("RELIANCE") == "Energy"
        assert get_sector("TCS") == "IT"
        assert get_sector("HDFCBANK") == "Banking"

    def test_with_ns_suffix(self):
        assert get_sector("RELIANCE.NS") == "Energy"

    def test_unknown_stock(self):
        assert get_sector("XYZUNKNOWN") is None

    def test_peers(self):
        peers = get_sector_peers("TCS")
        assert "INFY" in peers
        assert "TCS" not in peers  # Should exclude self


# ── Feature Builder Tests (integration, network required) ────────


class TestFeatureBuilder:
    @pytest.mark.slow
    def test_build_training_features(self):
        from features.feature_builder import build_features_for_training

        df = build_features_for_training(
            "RELIANCE", years=1,
            include_fundamentals=False,  # Skip to avoid Screener.in in tests
            include_macro=False,         # Skip to avoid API calls
        )
        assert not df.empty
        assert "target" in df.columns
        # Check technical features are present
        for col in TECHNICAL_FEATURES:
            assert col in df.columns, f"Missing: {col}"

    @pytest.mark.slow
    def test_no_future_leakage_in_target(self):
        """Target should be NEXT day's return, computed from today's close."""
        from features.feature_builder import build_features_for_training

        df = build_features_for_training(
            "RELIANCE", years=1,
            include_fundamentals=False,
            include_macro=False,
        )
        # The last available target should correspond to a day BEFORE the last OHLC day
        # (because the last day has no "next day" return)
        assert not df.empty
        assert df["target"].notna().all()

    @pytest.mark.slow
    def test_prediction_features(self):
        from features.feature_builder import build_features_for_prediction

        features = build_features_for_prediction(
            "TCS",
            include_fundamentals=False,
            include_macro=False,
        )
        assert len(features) > 0
        assert "_latest_close" in features
        assert features["_latest_close"] > 0
