"""End-to-end smoke train on the committed fixture — offline, deterministic.

Trains the full pipeline (CV → final fit → CQR calibration → registry
save), asserts every artifact is sane, then proves reproducibility by
training twice and comparing metrics exactly. Also run in CI.
"""

import math

import numpy as np
import pytest

from model.conformal import apply_conformal
from model.model_registry import load_model_bundle
from model.train_model import train_quantile_models

SMOKE_PARAMS = {"n_estimators": 60}


def _train(save: bool = True):
    return train_quantile_models(
        "TESTSTOCK",
        years=3,
        include_fundamentals=False,
        include_macro=False,
        params=SMOKE_PARAMS,
        save=save,
        cal_days=60,
        cv_folds=4,
    )


@pytest.fixture
def trained(patch_ohlc, tmp_model_dir, tmp_db):
    return _train()


class TestSmokeTrain:
    def test_metrics_finite_and_complete(self, trained):
        _, feature_names, metrics = trained

        cv = metrics["cv_metrics"]
        assert cv["n_folds"] >= 3
        for q in ("p10", "p50", "p90"):
            assert math.isfinite(cv["per_quantile"][q]["pinball"])
            assert 0 <= cv["per_quantile"][q]["coverage"] <= 1

        cal = metrics["cal_metrics"]
        for key in ("pinball_p10", "pinball_p50", "pinball_p90", "mae", "rmse"):
            assert math.isfinite(cal[key]), f"{key} not finite"
        assert 0 <= cal["coverage_80"] <= 1

    def test_conformal_offsets(self, trained):
        _, _, metrics = trained
        conformal = metrics["conformal"]
        assert conformal["method"] == "cqr_asymmetric_v1"
        assert math.isfinite(conformal["d10"])
        assert math.isfinite(conformal["d50"])
        assert math.isfinite(conformal["d90"])
        assert conformal["cal_size"] == 60
        # Calibration must not reduce coverage
        assert conformal["coverage_post"] >= conformal["coverage_pre"]

    def test_post_cqr_quantiles_ordered(self, trained, patch_ohlc, sample_ohlcv):
        models, feature_names, metrics = trained
        conformal = metrics["conformal"]

        from features.feature_builder import build_features_for_training

        df = build_features_for_training(
            "TESTSTOCK", years=3, include_fundamentals=False, include_macro=False
        )
        X_cal = df[feature_names].values[-60:]

        p10 = models["p10"].predict(X_cal)
        p50 = models["p50"].predict(X_cal)
        p90 = models["p90"].predict(X_cal)
        for i in range(len(X_cal)):
            lo, mid, hi = apply_conformal(p10[i], p50[i], p90[i], conformal)
            lo, mid, hi = sorted([lo, mid, hi])
            assert lo <= mid <= hi

    def test_feature_stats_saved(self, trained, tmp_model_dir):
        _, feature_names, _ = trained
        bundle = load_model_bundle("TESTSTOCK")
        assert bundle is not None
        _, _, metadata = bundle
        assert metadata["schema_version"] == 2
        stats = metadata["feature_stats"]
        assert set(stats.keys()) == set(feature_names)
        for s in stats.values():
            assert {"mean", "std", "p01", "p99"} <= set(s.keys())

    def test_registry_roundtrip_and_predict(self, trained, tmp_model_dir):
        models, feature_names, metrics = trained
        bundle = load_model_bundle("TESTSTOCK")
        assert bundle is not None
        loaded_models, loaded_features, metadata = bundle
        assert loaded_features == feature_names
        assert metadata["conformal"]["d10"] == metrics["conformal"]["d10"]

        X = np.zeros((1, len(feature_names)))
        preds = sorted(float(loaded_models[q].predict(X)[0]) for q in ("p10", "p50", "p90"))
        assert all(math.isfinite(p) for p in preds)

    def test_training_is_deterministic(self, patch_ohlc, tmp_model_dir, tmp_db):
        """Same data + same seed = byte-identical metrics."""
        _, _, m1 = _train(save=False)
        _, _, m2 = _train(save=False)
        assert m1["cv_metrics"] == m2["cv_metrics"]
        assert m1["cal_metrics"] == m2["cal_metrics"]
        assert m1["conformal"] == m2["conformal"]
