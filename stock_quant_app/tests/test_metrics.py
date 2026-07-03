"""Unit tests for shared quantile metrics."""

import numpy as np
import pytest

from model.metrics import (
    crossing_rate,
    interval_metrics,
    pinball_loss,
    quantile_coverage,
    quantile_metrics_summary,
)


class TestPinballLoss:
    def test_perfect_prediction_is_zero(self):
        y = np.array([1.0, -2.0, 0.5])
        assert pinball_loss(y, y, 0.5) == 0.0

    def test_median_pinball_is_half_mae(self):
        y = np.array([1.0, 2.0, 3.0])
        pred = np.array([0.0, 0.0, 0.0])
        assert pinball_loss(y, pred, 0.5) == pytest.approx(np.mean(np.abs(y)) / 2)

    def test_asymmetric_penalty(self):
        y = np.array([1.0])
        # Under-predicting the 90th percentile hurts 9x more than over
        under = pinball_loss(y, np.array([0.0]), 0.9)
        over = pinball_loss(y, np.array([2.0]), 0.9)
        assert under == pytest.approx(0.9)
        assert over == pytest.approx(0.1)


class TestCoverage:
    def test_quantile_coverage(self):
        y = np.arange(100, dtype=float)
        pred = np.full(100, 49.5)
        assert quantile_coverage(y, pred, 0.5) == pytest.approx(0.5)

    def test_interval_metrics(self):
        y = np.array([0.0, 5.0, -5.0, 1.0])
        p10 = np.full(4, -2.0)
        p90 = np.full(4, 2.0)
        m = interval_metrics(y, p10, p90)
        assert m["coverage_80"] == pytest.approx(0.5)
        assert m["mean_width"] == pytest.approx(4.0)


class TestCrossingRate:
    def test_no_crossing(self):
        p10, p50, p90 = np.array([-1.0]), np.array([0.0]), np.array([1.0])
        assert crossing_rate(p10, p50, p90) == 0.0

    def test_full_crossing(self):
        p10, p50, p90 = np.array([1.0]), np.array([0.0]), np.array([-1.0])
        assert crossing_rate(p10, p50, p90) == 1.0


class TestSummary:
    def test_summary_keys_and_finiteness(self):
        rng = np.random.default_rng(0)
        y = rng.normal(0, 1, 100)
        summary = quantile_metrics_summary(y, y - 1, y * 0, y + 1)
        expected_keys = {
            "pinball_p10",
            "pinball_p50",
            "pinball_p90",
            "coverage_p10",
            "coverage_p90",
            "mae",
            "rmse",
            "direction_accuracy",
            "crossing_rate",
            "coverage_80",
            "mean_width",
        }
        assert expected_keys <= set(summary.keys())
        assert all(np.isfinite(v) for v in summary.values())
