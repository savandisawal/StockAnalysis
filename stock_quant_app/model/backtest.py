"""Walk-forward backtesting engine.

Simulates real-world prediction by training on expanding windows
and predicting one day ahead. Tracks MAE, RMSE, direction accuracy,
and interval coverage over time.

Results are stored in SQLite for the Truth Dashboard.
"""

import sqlite3
from dataclasses import dataclass
from datetime import datetime

import lightgbm as lgb
import numpy as np
import pandas as pd

from app.config import PROJECT_ROOT
from features.feature_builder import ALL_FEATURES, build_features_for_training
from model.train_model import DEFAULT_PARAMS, QUANTILES
from utils.logger import logger

_BACKTEST_DB = PROJECT_ROOT / "stock_quant.db"


@dataclass
class BacktestMetrics:
    """Aggregate backtest results."""
    ticker: str
    mae: float                    # Mean Absolute Error (%)
    rmse: float                   # Root Mean Squared Error (%)
    direction_accuracy: float     # % of correct up/down calls
    interval_coverage: float      # % of actuals within P10-P90 range
    total_predictions: int
    avg_confidence: float
    run_date: str

    def to_dict(self) -> dict:
        return {
            "ticker": self.ticker,
            "mae": self.mae,
            "rmse": self.rmse,
            "direction_accuracy": self.direction_accuracy,
            "interval_coverage": self.interval_coverage,
            "total_predictions": self.total_predictions,
            "avg_confidence": self.avg_confidence,
            "run_date": self.run_date,
        }


def _init_backtest_tables():
    """Create backtest results tables if they don't exist."""
    conn = sqlite3.connect(str(_BACKTEST_DB))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS backtest_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            pred_date TEXT NOT NULL,
            actual_change REAL,
            predicted_p10 REAL,
            predicted_p50 REAL,
            predicted_p90 REAL,
            confidence REAL,
            direction_correct INTEGER,
            run_date TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS backtest_summary (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            mae REAL,
            rmse REAL,
            direction_accuracy REAL,
            interval_coverage REAL,
            total_predictions INTEGER,
            avg_confidence REAL,
            run_date TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()


def run_backtest(
    ticker: str,
    years: int = 3,
    min_train_days: int = 252,
    retrain_every: int = 5,
    include_fundamentals: bool = False,
    include_macro: bool = False,
) -> BacktestMetrics:
    """Run walk-forward backtest for a stock.

    Trains on expanding window, predicts one day ahead, compares to actual.

    Args:
        ticker: NSE ticker symbol.
        years: Total years of data to use.
        min_train_days: Minimum training samples before first prediction.
        retrain_every: Retrain model every N days (rolling).
        include_fundamentals: Include Pillar 2 (slow — hits Screener.in).
        include_macro: Include Pillar 3 (uses current macro as proxy).

    Returns:
        BacktestMetrics with aggregate performance stats.
    """
    logger.info(f"Starting walk-forward backtest for {ticker}")
    _init_backtest_tables()

    # Build full feature matrix
    df = build_features_for_training(
        ticker, years=years,
        include_fundamentals=include_fundamentals,
        include_macro=include_macro,
    )

    if df.empty or len(df) < min_train_days + 50:
        raise ValueError(
            f"Insufficient data for {ticker}: {len(df)} rows "
            f"(need {min_train_days + 50}+)"
        )

    feature_cols = [c for c in ALL_FEATURES if c in df.columns]
    X = df[feature_cols].values
    y = df["target"].values
    dates = df.index

    logger.info(f"Backtest data: {len(df)} days, {len(feature_cols)} features")

    # Walk-forward
    predictions = []
    current_models = None
    run_date = datetime.now().isoformat()

    for i in range(min_train_days, len(X)):
        # Retrain periodically
        if current_models is None or (i - min_train_days) % retrain_every == 0:
            X_train = X[:i]
            y_train = y[:i]

            current_models = {}
            params = {**DEFAULT_PARAMS, "n_estimators": 200, "verbose": -1}

            for name, alpha in QUANTILES.items():
                model = lgb.LGBMRegressor(objective="quantile", alpha=alpha, **params)
                model.fit(X_train, y_train)
                current_models[name] = model

        # Predict
        X_test = X[i : i + 1]
        p10 = float(current_models["p10"].predict(X_test)[0])
        p50 = float(current_models["p50"].predict(X_test)[0])
        p90 = float(current_models["p90"].predict(X_test)[0])

        # Ensure ordering
        p10, p50, p90 = sorted([p10, p50, p90])

        actual = float(y[i])
        direction_correct = int(np.sign(actual) == np.sign(p50))
        in_range = int(p10 <= actual <= p90)

        range_width = p90 - p10
        confidence = round(float(100 * np.exp(-0.8 * max(range_width, 0.01))), 1)

        predictions.append({
            "date": str(dates[i].date()),
            "actual": actual,
            "p10": p10,
            "p50": p50,
            "p90": p90,
            "confidence": confidence,
            "direction_correct": direction_correct,
            "in_range": in_range,
        })

    # Compute aggregate metrics
    actuals = np.array([p["actual"] for p in predictions])
    p50_preds = np.array([p["p50"] for p in predictions])
    errors = actuals - p50_preds

    mae = round(float(np.mean(np.abs(errors))), 4)
    rmse = round(float(np.sqrt(np.mean(errors ** 2))), 4)
    direction_acc = round(float(np.mean([p["direction_correct"] for p in predictions])), 4)
    interval_cov = round(float(np.mean([p["in_range"] for p in predictions])), 4)
    avg_conf = round(float(np.mean([p["confidence"] for p in predictions])), 1)

    metrics = BacktestMetrics(
        ticker=ticker.upper().replace(".NS", "").replace(".BO", ""),
        mae=mae,
        rmse=rmse,
        direction_accuracy=direction_acc,
        interval_coverage=interval_cov,
        total_predictions=len(predictions),
        avg_confidence=avg_conf,
        run_date=run_date,
    )

    logger.info(
        f"Backtest complete for {ticker}: MAE={mae:.4f}%, "
        f"Direction={direction_acc:.2%}, Coverage={interval_cov:.2%}, "
        f"Predictions={len(predictions)}"
    )

    # Save to SQLite
    _save_backtest_results(ticker, predictions, metrics, run_date)

    return metrics


def _save_backtest_results(
    ticker: str,
    predictions: list[dict],
    metrics: BacktestMetrics,
    run_date: str,
):
    """Persist backtest results to SQLite."""
    clean_ticker = ticker.upper().replace(".NS", "").replace(".BO", "")
    conn = sqlite3.connect(str(_BACKTEST_DB))

    try:
        # Save individual predictions
        for p in predictions:
            conn.execute(
                """INSERT INTO backtest_results
                   (ticker, pred_date, actual_change, predicted_p10, predicted_p50,
                    predicted_p90, confidence, direction_correct, run_date)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (clean_ticker, p["date"], p["actual"], p["p10"], p["p50"],
                 p["p90"], p["confidence"], p["direction_correct"], run_date),
            )

        # Save summary
        conn.execute(
            """INSERT INTO backtest_summary
               (ticker, mae, rmse, direction_accuracy, interval_coverage,
                total_predictions, avg_confidence, run_date)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (clean_ticker, metrics.mae, metrics.rmse, metrics.direction_accuracy,
             metrics.interval_coverage, metrics.total_predictions,
             metrics.avg_confidence, run_date),
        )

        conn.commit()
    finally:
        conn.close()


def get_backtest_history(ticker: str, limit: int = 500) -> pd.DataFrame:
    """Load backtest prediction history for the Truth Dashboard."""
    _init_backtest_tables()
    clean_ticker = ticker.upper().replace(".NS", "").replace(".BO", "")
    conn = sqlite3.connect(str(_BACKTEST_DB))

    try:
        df = pd.read_sql_query(
            """SELECT pred_date, actual_change, predicted_p10, predicted_p50,
                      predicted_p90, confidence, direction_correct
               FROM backtest_results
               WHERE ticker = ?
               ORDER BY pred_date DESC
               LIMIT ?""",
            conn,
            params=(clean_ticker, limit),
        )
        return df
    finally:
        conn.close()


def get_backtest_summary(ticker: str | None = None) -> list[dict]:
    """Get latest backtest summary for one or all tickers."""
    _init_backtest_tables()
    conn = sqlite3.connect(str(_BACKTEST_DB))

    try:
        if ticker:
            clean = ticker.upper().replace(".NS", "").replace(".BO", "")
            rows = conn.execute(
                """SELECT * FROM backtest_summary
                   WHERE ticker = ? ORDER BY run_date DESC LIMIT 1""",
                (clean,),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM backtest_summary ORDER BY run_date DESC"
            ).fetchall()

        cols = ["id", "ticker", "mae", "rmse", "direction_accuracy",
                "interval_coverage", "total_predictions", "avg_confidence", "run_date"]
        return [dict(zip(cols, row)) for row in rows]
    finally:
        conn.close()
