"""Tests for model pipeline — training, prediction, registry, backtest."""

import lightgbm as lgb
import numpy as np
import pytest

from model.model_registry import load_model_bundle, save_model_bundle
from model.predict import _classify_direction, _compute_confidence

# ── Unit Tests (no network) ──────────────────────────────────────


class TestConfidence:
    def test_narrow_range_high_confidence(self):
        # Range < ATR → high confidence
        conf = _compute_confidence(range_width_pct=1.0, atr_pct=2.0)
        assert conf > 60

    def test_wide_range_low_confidence(self):
        # Range >> ATR → low confidence
        conf = _compute_confidence(range_width_pct=5.0, atr_pct=2.0)
        assert conf < 40

    def test_zero_atr_returns_default(self):
        conf = _compute_confidence(range_width_pct=1.0, atr_pct=0)
        assert conf == 50.0

    def test_confidence_bounded(self):
        conf_low = _compute_confidence(range_width_pct=100.0, atr_pct=1.0)
        conf_high = _compute_confidence(range_width_pct=0.01, atr_pct=2.0)
        assert 5 <= conf_low <= 95
        assert 5 <= conf_high <= 95


class TestDirection:
    def test_bullish(self):
        assert _classify_direction(1.5) == "Bullish"

    def test_bearish(self):
        assert _classify_direction(-0.8) == "Bearish"

    def test_neutral(self):
        assert _classify_direction(0.1) == "Neutral"


class TestModelRegistry:
    def test_save_and_load(self, tmp_path, monkeypatch):
        """Test model save/load roundtrip using a temp directory."""
        monkeypatch.setattr("model.model_registry.MODEL_DIR", tmp_path)

        # Create dummy models
        dummy_models = {}
        for name in ["p10", "p50", "p90"]:
            model = lgb.LGBMRegressor(n_estimators=5, verbose=-1)
            X = np.random.randn(50, 3)
            y = np.random.randn(50)
            model.fit(X, y)
            dummy_models[name] = model

        features = ["feat_a", "feat_b", "feat_c"]

        # Save
        version = save_model_bundle(
            ticker="TEST",
            models=dummy_models,
            feature_names=features,
            metrics={"mae": 0.5},
        )
        assert "TEST" in version

        # Load
        result = load_model_bundle("TEST")
        assert result is not None
        models, feat_names, metadata = result
        assert set(models.keys()) == {"p10", "p50", "p90"}
        assert feat_names == features
        assert metadata["metrics"]["mae"] == 0.5

    def test_load_nonexistent(self, tmp_path, monkeypatch):
        monkeypatch.setattr("model.model_registry.MODEL_DIR", tmp_path)
        assert load_model_bundle("NONEXISTENT") is None


# ── Integration Tests (network required) ─────────────────────────


class TestTrainPredict:
    @pytest.mark.slow
    def test_end_to_end_train_and_predict(self):
        """Full pipeline: train on RELIANCE, then predict."""
        from model.predict import predict_next_day
        from model.train_model import train_quantile_models

        # Train with minimal data (technicals only for speed)
        models, features, metrics = train_quantile_models(
            "RELIANCE", years=2,
            include_fundamentals=False,
            include_macro=False,
        )

        assert "p10" in models and "p50" in models and "p90" in models
        assert metrics["direction_accuracy"] > 0.3  # Better than random noise
        assert metrics["interval_coverage_80"] > 0.5  # Reasonable coverage

        # Predict
        result = predict_next_day(
            "RELIANCE",
            include_fundamentals=False,
            include_macro=False,
        )
        assert result is not None
        assert result.predicted_low < result.predicted_mid < result.predicted_high
        assert result.current_close > 0
        assert 5 <= result.confidence <= 95


class TestBacktest:
    @pytest.mark.slow
    def test_backtest_runs(self):
        from model.backtest import run_backtest

        metrics = run_backtest(
            "RELIANCE", years=2,
            min_train_days=200,
            retrain_every=20,
            include_fundamentals=False,
            include_macro=False,
        )

        assert metrics.total_predictions > 10
        assert 0 < metrics.mae < 5  # MAE should be reasonable
        assert 0 < metrics.direction_accuracy < 1
        assert 0 < metrics.interval_coverage < 1
