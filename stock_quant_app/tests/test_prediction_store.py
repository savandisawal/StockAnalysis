"""Unit tests for prediction persistence, outcome backfill, and live accuracy."""

import sqlite3

import pandas as pd
import pytest

from model.predict import PredictionResult, PredictionWarning
from services.prediction_store import (
    backfill_outcomes,
    get_live_accuracy,
    get_live_history,
    init_prediction_tables,
    save_prediction,
    save_sentiment_snapshot,
)


def _make_result(ticker="TESTSTOCK", pred_date="2025-12-30", close=1000.0) -> PredictionResult:
    return PredictionResult(
        ticker=ticker,
        prediction_date=pred_date,
        current_close=close,
        predicted_low=round(close * 0.99, 2),
        predicted_mid=round(close * 1.002, 2),
        predicted_high=round(close * 1.012, 2),
        predicted_change_pct=0.2,
        range_width_pct=2.2,
        confidence=62.0,
        direction="Bullish",
        model_version="TESTSTOCK_20251201_abcd1234",
        warnings=[PredictionWarning(level="info", code="test", message="test warning")],
        guardrail_applied=False,
        explanation={"quantile": "p50", "base_value": 0.01, "top_features": []},
        features={"rsi_14": 55.0, "atr_pct": 1.5},
        features_hash="deadbeef00000000",
    )


class TestSaveAndLoad:
    def test_save_and_history_roundtrip(self, tmp_db):
        row_id = save_prediction(_make_result())
        assert row_id is not None

        df = get_live_history("TESTSTOCK")
        assert len(df) == 1
        row = df.iloc[0]
        assert row["model_version"] == "TESTSTOCK_20251201_abcd1234"
        assert row["source"] == "api"
        assert pd.isna(row["actual_change"])  # not resolved yet

    def test_upsert_on_same_day_and_source(self, tmp_db):
        save_prediction(_make_result())
        updated = _make_result()
        updated.confidence = 80.0
        save_prediction(updated)

        df = get_live_history("TESTSTOCK")
        assert len(df) == 1
        assert df.iloc[0]["confidence"] == 80.0

    def test_distinct_sources_kept_separate(self, tmp_db):
        save_prediction(_make_result(), source="api")
        save_prediction(_make_result(), source="scheduler")
        assert len(get_live_history("TESTSTOCK")) == 2

    def test_provenance_stored(self, tmp_db):
        from app.config import settings

        save_prediction(_make_result())
        conn = sqlite3.connect(str(settings.db_path))
        try:
            features_json, shap_json, warnings_json = conn.execute(
                "SELECT features_json, shap_json, warnings_json FROM predictions"
            ).fetchone()
        finally:
            conn.close()
        assert "rsi_14" in features_json
        assert "base_value" in shap_json
        assert "test warning" in warnings_json


class TestBackfill:
    def test_outcome_backfill(self, tmp_db, patch_ohlc, sample_ohlcv):
        # Predict for a known fixture day; current_close = previous day's close
        pred_day = sample_ohlcv.index[-1]
        prev_close = float(sample_ohlcv["Close"].iloc[-2])
        actual_close = float(sample_ohlcv["Close"].iloc[-1])

        save_prediction(_make_result(pred_date=str(pred_day.date()), close=prev_close))
        updated = backfill_outcomes(["TESTSTOCK"])
        assert updated == 1

        df = get_live_history("TESTSTOCK")
        expected_change = (actual_close / prev_close - 1) * 100
        assert df.iloc[0]["actual_change"] == pytest.approx(expected_change, abs=1e-3)
        assert df.iloc[0]["in_range"] in (0, 1)

    def test_live_accuracy_after_backfill(self, tmp_db, patch_ohlc, sample_ohlcv):
        prev_close = float(sample_ohlcv["Close"].iloc[-2])
        pred_day = sample_ohlcv.index[-1]
        save_prediction(_make_result(pred_date=str(pred_day.date()), close=prev_close))
        backfill_outcomes(["TESTSTOCK"])

        rows = get_live_accuracy("TESTSTOCK")
        assert len(rows) == 1
        assert rows[0]["n"] == 1
        assert "pinball_p50" in rows[0]

    def test_unresolved_prediction_not_scored(self, tmp_db, patch_ohlc):
        # Future prediction date — nothing to backfill
        save_prediction(_make_result(pred_date="2099-01-04"))
        assert backfill_outcomes(["TESTSTOCK"]) == 0
        assert get_live_accuracy("TESTSTOCK") == []


class TestSentimentHistory:
    def test_snapshot_roundtrip(self, tmp_db):
        from features.macro_sentiment import get_sentiment_history

        init_prediction_tables()
        save_sentiment_snapshot("TESTSTOCK", 0.4)
        save_sentiment_snapshot("TESTSTOCK", 0.5)  # same day upsert

        s = get_sentiment_history("TESTSTOCK")
        assert len(s) == 1
        assert s.iloc[0] == pytest.approx(0.5)

    def test_none_score_ignored(self, tmp_db):
        from features.macro_sentiment import get_sentiment_history

        init_prediction_tables()
        save_sentiment_snapshot("TESTSTOCK", None)
        assert get_sentiment_history("TESTSTOCK").empty
