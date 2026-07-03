"""Walk-forward backtesting engine.

Simulates real-world prediction by training on expanding windows
and predicting one day ahead, with the same CQR conformal calibration
the production model uses (offsets from a trailing held-out window).
Tracks pinball loss, MAE, RMSE, direction accuracy, and interval
coverage over time.

Results are stored in SQLite for the Truth Dashboard.
"""

import sqlite3
from dataclasses import dataclass
from datetime import datetime

import lightgbm as lgb
import numpy as np
import pandas as pd

from app.config import settings
from features.feature_builder import ALL_FEATURES, build_features_for_training
from model.conformal import apply_conformal, compute_conformal_offsets
from model.metrics import quantile_metrics_summary
from model.train_model import DEFAULT_PARAMS, QUANTILES
from utils.logger import logger


def _backtest_db():
    return settings.db_path


@dataclass
class BacktestMetrics:
    """Aggregate backtest results."""

    ticker: str
    mae: float  # Mean Absolute Error (%)
    rmse: float  # Root Mean Squared Error (%)
    direction_accuracy: float  # % of correct up/down calls
    interval_coverage: float  # % of actuals within P10-P90 range
    total_predictions: int
    avg_confidence: float
    run_date: str
    pinball_p10: float = 0.0  # Mean pinball loss per quantile
    pinball_p50: float = 0.0
    pinball_p90: float = 0.0

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
            "pinball_p10": self.pinball_p10,
            "pinball_p50": self.pinball_p50,
            "pinball_p90": self.pinball_p90,
        }


def _init_backtest_tables():
    """Create backtest results tables if they don't exist."""
    conn = sqlite3.connect(str(_backtest_db()), timeout=5)
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
            run_date TEXT NOT NULL,
            pinball_p10 REAL,
            pinball_p50 REAL,
            pinball_p90 REAL
        )
    """)
    # Migrate pre-existing DBs that lack the pinball columns
    existing = {row[1] for row in conn.execute("PRAGMA table_info(backtest_summary)")}
    for col in ("pinball_p10", "pinball_p50", "pinball_p90"):
        if col not in existing:
            conn.execute(f"ALTER TABLE backtest_summary ADD COLUMN {col} REAL")
    conn.commit()
    conn.close()


def run_backtest(
    ticker: str,
    years: int = 3,
    min_train_days: int = 252,
    retrain_every: int = 5,
    include_fundamentals: bool = True,
    include_macro: bool = True,
) -> BacktestMetrics:
    """Run walk-forward backtest for a stock.

    Trains on expanding window, predicts one day ahead, compares to actual.

    Args:
        ticker: NSE ticker symbol.
        years: Total years of data to use.
        min_train_days: Minimum training samples before first prediction.
        retrain_every: Retrain model every N days (rolling).
        include_fundamentals: Include Pillar 2 (point-in-time Screener.in history).
        include_macro: Include Pillar 3 (point-in-time macro history).

    Returns:
        BacktestMetrics with aggregate performance stats.
    """
    logger.info(f"Starting walk-forward backtest for {ticker}")
    _init_backtest_tables()

    # Build full feature matrix
    df = build_features_for_training(
        ticker,
        years=years,
        include_fundamentals=include_fundamentals,
        include_macro=include_macro,
    )

    if df.empty or len(df) < min_train_days + 50:
        raise ValueError(
            f"Insufficient data for {ticker}: {len(df)} rows (need {min_train_days + 50}+)"
        )

    feature_cols = [c for c in ALL_FEATURES if c in df.columns]
    X = df[feature_cols].values
    y = df["target"].values
    dates = df.index
    atr_col = df["atr_pct"].values if "atr_pct" in df.columns else None

    logger.info(f"Backtest data: {len(df)} days, {len(feature_cols)} features")

    # Walk-forward — mirrors production training: fit on all but the last
    # cal_n rows of each window, compute CQR offsets on those held-out rows,
    # apply the offsets to subsequent predictions.
    cal_n = min(120, max(30, min_train_days // 4))
    gap = 2  # purge/embargo rows between fit and calibration windows

    predictions = []
    current_models = None
    conformal = None
    run_date = datetime.now().isoformat()

    for i in range(min_train_days, len(X)):
        # Retrain periodically
        if current_models is None or (i - min_train_days) % retrain_every == 0:
            fit_end = i - cal_n - gap
            X_fit, y_fit = X[:fit_end], y[:fit_end]
            X_cal, y_cal = X[i - cal_n : i], y[i - cal_n : i]

            current_models = {}
            params = {**DEFAULT_PARAMS, "n_estimators": 200, "verbose": -1}

            for name, alpha in QUANTILES.items():
                model = lgb.LGBMRegressor(objective="quantile", alpha=alpha, **params)
                model.fit(X_fit, y_fit)
                current_models[name] = model

            cal_preds = {name: current_models[name].predict(X_cal) for name in QUANTILES}
            conformal = compute_conformal_offsets(
                y_cal, cal_preds["p10"], cal_preds["p50"], cal_preds["p90"]
            )

        # Predict
        X_test = X[i : i + 1]
        p10 = float(current_models["p10"].predict(X_test)[0])
        p50 = float(current_models["p50"].predict(X_test)[0])
        p90 = float(current_models["p90"].predict(X_test)[0])

        # Apply CQR offsets, then ensure ordering
        p10, p50, p90 = apply_conformal(p10, p50, p90, conformal)
        p10, p50, p90 = sorted([p10, p50, p90])

        actual = float(y[i])
        direction_correct = int(np.sign(actual) == np.sign(p50))
        in_range = int(p10 <= actual <= p90)

        # Same confidence formula as predict.py: interval width normalized
        # by that day's ATR
        range_width = p90 - p10
        atr_pct = float(atr_col[i]) if atr_col is not None and atr_col[i] > 0 else None
        if atr_pct:
            confidence = round(float(np.clip(100 * np.exp(-0.8 * range_width / atr_pct), 5, 95)), 1)
        else:
            confidence = 50.0

        predictions.append(
            {
                "date": str(dates[i].date()),
                "actual": actual,
                "p10": p10,
                "p50": p50,
                "p90": p90,
                "confidence": confidence,
                "direction_correct": direction_correct,
                "in_range": in_range,
            }
        )

    # Compute aggregate metrics (shared implementation with training/CV)
    actuals = np.array([p["actual"] for p in predictions])
    summary = quantile_metrics_summary(
        actuals,
        np.array([p["p10"] for p in predictions]),
        np.array([p["p50"] for p in predictions]),
        np.array([p["p90"] for p in predictions]),
    )
    avg_conf = round(float(np.mean([p["confidence"] for p in predictions])), 1)

    metrics = BacktestMetrics(
        ticker=ticker.upper().replace(".NS", "").replace(".BO", ""),
        mae=summary["mae"],
        rmse=summary["rmse"],
        direction_accuracy=summary["direction_accuracy"],
        interval_coverage=summary["coverage_80"],
        total_predictions=len(predictions),
        avg_confidence=avg_conf,
        run_date=run_date,
        pinball_p10=summary["pinball_p10"],
        pinball_p50=summary["pinball_p50"],
        pinball_p90=summary["pinball_p90"],
    )

    logger.info(
        f"Backtest complete for {ticker}: MAE={metrics.mae:.4f}%, "
        f"Pinball P50={metrics.pinball_p50:.4f}, "
        f"Direction={metrics.direction_accuracy:.2%}, "
        f"Coverage={metrics.interval_coverage:.2%}, "
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
    conn = sqlite3.connect(str(_backtest_db()), timeout=5)

    try:
        # Save individual predictions
        for p in predictions:
            conn.execute(
                """INSERT INTO backtest_results
                   (ticker, pred_date, actual_change, predicted_p10, predicted_p50,
                    predicted_p90, confidence, direction_correct, run_date)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    clean_ticker,
                    p["date"],
                    p["actual"],
                    p["p10"],
                    p["p50"],
                    p["p90"],
                    p["confidence"],
                    p["direction_correct"],
                    run_date,
                ),
            )

        # Save summary
        conn.execute(
            """INSERT INTO backtest_summary
               (ticker, mae, rmse, direction_accuracy, interval_coverage,
                total_predictions, avg_confidence, run_date,
                pinball_p10, pinball_p50, pinball_p90)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                clean_ticker,
                metrics.mae,
                metrics.rmse,
                metrics.direction_accuracy,
                metrics.interval_coverage,
                metrics.total_predictions,
                metrics.avg_confidence,
                run_date,
                metrics.pinball_p10,
                metrics.pinball_p50,
                metrics.pinball_p90,
            ),
        )

        conn.commit()
    finally:
        conn.close()


def get_backtest_history(ticker: str, limit: int = 500) -> pd.DataFrame:
    """Load backtest prediction history for the Truth Dashboard."""
    _init_backtest_tables()
    clean_ticker = ticker.upper().replace(".NS", "").replace(".BO", "")
    conn = sqlite3.connect(str(_backtest_db()), timeout=5)

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
    conn = sqlite3.connect(str(_backtest_db()), timeout=5)

    try:
        conn.row_factory = sqlite3.Row
        if ticker:
            clean = ticker.upper().replace(".NS", "").replace(".BO", "")
            rows = conn.execute(
                """SELECT * FROM backtest_summary
                   WHERE ticker = ? ORDER BY run_date DESC LIMIT 1""",
                (clean,),
            ).fetchall()
        else:
            rows = conn.execute("SELECT * FROM backtest_summary ORDER BY run_date DESC").fetchall()

        return [dict(row) for row in rows]
    finally:
        conn.close()
