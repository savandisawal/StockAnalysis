"""Unit tests for CQR conformal calibration."""

import numpy as np
import pytest

from model.conformal import apply_conformal, compute_conformal_offsets


class TestComputeOffsets:
    def test_narrow_intervals_get_widened(self):
        rng = np.random.default_rng(0)
        y = rng.normal(0, 2.0, 300)
        # Model intervals are far too narrow (±0.5 for sigma=2 data)
        q10, q50, q90 = np.full(300, -0.5), np.zeros(300), np.full(300, 0.5)
        off = compute_conformal_offsets(y, q10, q50, q90)
        assert off["d10"] > 0  # widen downward
        assert off["d90"] > 0  # widen upward
        assert off["coverage_post"] > off["coverage_pre"]
        assert off["coverage_post"] >= 0.75

    def test_wide_intervals_get_tightened(self):
        rng = np.random.default_rng(1)
        y = rng.normal(0, 0.5, 300)
        # Model intervals are far too wide (±5 for sigma=0.5 data)
        q10, q90 = np.full(300, -5.0), np.full(300, 5.0)
        off = compute_conformal_offsets(y, q10, np.zeros(300), q90)
        assert off["d10"] < 0  # negative offset = tighten
        assert off["d90"] < 0

    def test_median_bias_correction(self):
        y = np.full(200, 1.0)
        off = compute_conformal_offsets(y, y - 1, y - 1, y + 1)  # q50 biased low by 1
        assert off["d50"] == pytest.approx(1.0)

    def test_metadata_fields(self):
        y = np.random.default_rng(2).normal(0, 1, 120)
        off = compute_conformal_offsets(y, y - 1, y, y + 1)
        assert off["method"] == "cqr_asymmetric_v1"
        assert off["cal_size"] == 120


class TestApplyConformal:
    def test_identity_without_offsets(self):
        assert apply_conformal(-1.0, 0.0, 1.0, None) == (-1.0, 0.0, 1.0)
        assert apply_conformal(-1.0, 0.0, 1.0, {}) == (-1.0, 0.0, 1.0)

    def test_offsets_applied_with_correct_signs(self):
        off = {"d10": 0.5, "d50": 0.1, "d90": 0.3}
        p10, p50, p90 = apply_conformal(-1.0, 0.0, 1.0, off)
        assert p10 == pytest.approx(-1.5)  # lower bound moves down
        assert p50 == pytest.approx(0.1)
        assert p90 == pytest.approx(1.3)  # upper bound moves up
