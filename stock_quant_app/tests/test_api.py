"""Tests for FastAPI endpoints — uses TestClient with mocked services.

No network calls. All external dependencies are mocked.
Run: pytest tests/test_api.py -v
"""

import base64
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from app.main import app

# ── Auth helper ──────────────────────────────────────────────────


def _auth_header(user: str = "admin", password: str = "changeme") -> dict:
    token = base64.b64encode(f"{user}:{password}".encode()).decode()
    return {"Authorization": f"Basic {token}"}


VALID_AUTH = _auth_header()
BAD_AUTH = _auth_header("admin", "wrong")

client = TestClient(app, raise_server_exceptions=False)


# ── Health ───────────────────────────────────────────────────────


class TestHealth:
    def test_health_no_auth(self):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}


# ── Auth ─────────────────────────────────────────────────────────


class TestAuth:
    def test_no_credentials(self):
        resp = client.get("/macro")
        assert resp.status_code == 401

    def test_bad_credentials(self):
        resp = client.get("/macro", headers=BAD_AUTH)
        assert resp.status_code == 401

    def test_valid_credentials(self):
        with patch("app.main.fetch_macro_snapshot") as mock:
            mock.return_value = []
            resp = client.get("/macro", headers=VALID_AUTH)
            # 503 because empty list, but auth passed
            assert resp.status_code == 503


# ── Ticker Validation ───────────────────────────────────────────


class TestTickerValidation:
    def test_invalid_ticker_format(self):
        resp = client.get("/predict/123BAD!", headers=VALID_AUTH)
        assert resp.status_code == 400
        assert "Invalid ticker" in resp.json()["detail"]

    def test_valid_ticker_uppercase(self):
        with patch("app.main.get_prediction", return_value=None):
            resp = client.get("/predict/reliance", headers=VALID_AUTH)
            # 404 because no model, but ticker validation passed
            assert resp.status_code == 404


# ── Predict ─────────────────────────────────────────────────────


class TestPredict:
    def test_no_model_returns_404(self):
        with patch("app.main.get_prediction", return_value=None):
            resp = client.get("/predict/TCS", headers=VALID_AUTH)
            assert resp.status_code == 404
            assert "Train first" in resp.json()["detail"]

    def test_successful_prediction(self):
        mock_result = MagicMock()
        mock_result.to_dict.return_value = {
            "ticker": "TCS",
            "predicted_low": 3800.0,
            "predicted_mid": 3850.0,
            "predicted_high": 3900.0,
            "confidence": 72.5,
            "direction": "Bullish",
        }
        with patch("app.main.get_prediction", return_value=mock_result):
            resp = client.get("/predict/TCS", headers=VALID_AUTH)
            assert resp.status_code == 200
            data = resp.json()
            assert data["ticker"] == "TCS"
            assert data["predicted_mid"] == 3850.0


# ── Train ────────────────────────────────────────────────────────


class TestTrain:
    def test_successful_train(self):
        mock_metrics = {
            "direction_accuracy": 0.58,
            "interval_coverage_80": 0.75,
        }
        with patch("app.main.train_model", return_value=mock_metrics):
            resp = client.post(
                "/train/RELIANCE?years=2", headers=VALID_AUTH
            )
            assert resp.status_code == 200
            assert resp.json()["direction_accuracy"] == 0.58

    def test_train_error(self):
        with patch(
            "app.main.train_model",
            return_value={"error": "Not enough data"},
        ):
            resp = client.post("/train/INFY", headers=VALID_AUTH)
            assert resp.status_code == 500
            assert "Not enough data" in resp.json()["detail"]

    def test_train_invalid_years(self):
        resp = client.post("/train/RELIANCE?years=20", headers=VALID_AUTH)
        assert resp.status_code == 400


# ── Backtest ─────────────────────────────────────────────────────


class TestBacktest:
    def test_successful_backtest(self):
        mock_metrics = {
            "mae": 0.45,
            "direction_accuracy": 0.56,
        }
        with patch(
            "app.main.run_stock_backtest", return_value=mock_metrics
        ):
            resp = client.post(
                "/backtest/RELIANCE?years=2", headers=VALID_AUTH
            )
            assert resp.status_code == 200
            assert resp.json()["mae"] == 0.45

    def test_backtest_error(self):
        with patch(
            "app.main.run_stock_backtest",
            return_value={"error": "Insufficient data"},
        ):
            resp = client.post("/backtest/SBIN", headers=VALID_AUTH)
            assert resp.status_code == 500

    def test_backtest_history(self):
        with patch(
            "app.main.get_prediction_history", return_value=[]
        ):
            resp = client.get(
                "/backtest/RELIANCE/history", headers=VALID_AUTH
            )
            assert resp.status_code == 200
            assert resp.json() == {"predictions": []}


# ── Models List ──────────────────────────────────────────────────


class TestModelsList:
    def test_list_all(self):
        with patch("app.main.list_models", return_value=[]):
            resp = client.get("/models", headers=VALID_AUTH)
            assert resp.status_code == 200
            assert resp.json() == {"models": []}

    def test_list_filtered(self):
        mock_models = [{"ticker": "TCS", "version": "v1"}]
        with patch("app.main.list_models", return_value=mock_models):
            resp = client.get("/models?ticker=TCS", headers=VALID_AUTH)
            assert resp.status_code == 200
            assert len(resp.json()["models"]) == 1


# ── Macro ────────────────────────────────────────────────────────


class TestMacro:
    def test_macro_empty_returns_503(self):
        with patch("app.main.fetch_macro_snapshot", return_value=[]):
            resp = client.get("/macro", headers=VALID_AUTH)
            assert resp.status_code == 503

    def test_macro_success(self):
        mock_snap = MagicMock()
        mock_snap.name = "S&P 500"
        mock_snap.current_price = 5200.0
        mock_snap.prev_close = 5180.0
        mock_snap.change_pct = 0.39
        mock_snap.fetch_date = "2026-03-27"
        with patch(
            "app.main.fetch_macro_snapshot", return_value=[mock_snap]
        ):
            resp = client.get("/macro", headers=VALID_AUTH)
            assert resp.status_code == 200
            data = resp.json()["indicators"]
            assert len(data) == 1
            assert data[0]["name"] == "S&P 500"
