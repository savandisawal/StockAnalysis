"""Persistence for served predictions and their real outcomes.

Every prediction the system serves is stored with full provenance
(model version, feature vector + hash, SHAP explanation, safeguard
flags). The next trading day, backfill_outcomes() fills in what actually
happened — giving honest live accuracy for the Truth Dashboard, measured
on real served predictions rather than backtest simulations.

Also owns the sentiment_history table: daily Claude sentiment snapshots
recorded by the scheduler so future retrains learn from real sentiment
instead of a neutral placeholder.
"""

import json
import sqlite3
from datetime import date, datetime

import numpy as np
import pandas as pd

from app.config import settings
from model.metrics import quantile_metrics_summary
from utils.logger import logger


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(str(settings.db_path), timeout=5)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    return conn


def init_prediction_tables() -> None:
    """Create the predictions and sentiment_history tables if missing."""
    conn = _connect()
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL,
                prediction_date TEXT NOT NULL,
                created_at TEXT NOT NULL,
                current_close REAL,
                p10 REAL, p50 REAL, p90 REAL,
                predicted_low REAL, predicted_mid REAL, predicted_high REAL,
                confidence REAL,
                direction TEXT,
                model_version TEXT NOT NULL,
                features_hash TEXT NOT NULL,
                features_json TEXT,
                shap_json TEXT,
                warnings_json TEXT,
                guardrail_applied INTEGER DEFAULT 0,
                source TEXT NOT NULL DEFAULT 'api',
                actual_change REAL,
                direction_correct INTEGER,
                in_range INTEGER,
                outcome_filled_at TEXT,
                UNIQUE (ticker, prediction_date, source)
            )
        """)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_predictions_ticker_date "
            "ON predictions(ticker, prediction_date)"
        )
        conn.execute("""
            CREATE TABLE IF NOT EXISTS sentiment_history (
                ticker TEXT NOT NULL,
                date TEXT NOT NULL,
                score REAL,
                PRIMARY KEY (ticker, date)
            )
        """)
        conn.commit()
    finally:
        conn.close()


def save_prediction(result, source: str = "api") -> int | None:
    """Persist a served PredictionResult. Upserts on (ticker, date, source).

    Returns the row id, or None if persistence failed (never raises —
    storage problems must not break serving).
    """
    try:
        init_prediction_tables()
        conn = _connect()
        try:
            # % changes relative to current close, post-CQR
            p10, p50, p90 = _changes_from_prices(result)
            cursor = conn.execute(
                """INSERT INTO predictions
                   (ticker, prediction_date, created_at, current_close,
                    p10, p50, p90, predicted_low, predicted_mid, predicted_high,
                    confidence, direction, model_version, features_hash,
                    features_json, shap_json, warnings_json, guardrail_applied, source)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT (ticker, prediction_date, source) DO UPDATE SET
                    created_at=excluded.created_at,
                    current_close=excluded.current_close,
                    p10=excluded.p10, p50=excluded.p50, p90=excluded.p90,
                    predicted_low=excluded.predicted_low,
                    predicted_mid=excluded.predicted_mid,
                    predicted_high=excluded.predicted_high,
                    confidence=excluded.confidence,
                    direction=excluded.direction,
                    model_version=excluded.model_version,
                    features_hash=excluded.features_hash,
                    features_json=excluded.features_json,
                    shap_json=excluded.shap_json,
                    warnings_json=excluded.warnings_json,
                    guardrail_applied=excluded.guardrail_applied""",
                (
                    result.ticker,
                    result.prediction_date,
                    datetime.now().isoformat(timespec="seconds"),
                    result.current_close,
                    p10,
                    p50,
                    p90,
                    result.predicted_low,
                    result.predicted_mid,
                    result.predicted_high,
                    result.confidence,
                    result.direction,
                    result.model_version,
                    result.features_hash,
                    json.dumps(result.features, default=float) if result.features else None,
                    json.dumps(result.explanation) if result.explanation else None,
                    json.dumps(
                        [
                            {"level": w.level, "code": w.code, "message": w.message}
                            for w in result.warnings
                        ]
                    ),
                    int(result.guardrail_applied),
                    source,
                ),
            )
            conn.commit()
            return cursor.lastrowid
        finally:
            conn.close()
    except Exception as e:
        logger.error(f"Failed to persist prediction for {result.ticker}: {e}")
        return None


def _changes_from_prices(result) -> tuple[float, float, float]:
    """Recover % changes from price levels (post-CQR, post-guardrail)."""
    close = result.current_close or 0
    if not close:
        return 0.0, 0.0, 0.0
    return (
        round((result.predicted_low / close - 1) * 100, 4),
        round((result.predicted_mid / close - 1) * 100, 4),
        round((result.predicted_high / close - 1) * 100, 4),
    )


def backfill_outcomes(tickers: list[str] | None = None) -> int:
    """Fill in realized outcomes for past predictions.

    For each prediction whose prediction_date has completed and whose
    outcome is still empty: fetch the actual close, compute the realized
    % change vs the stored current_close, and record direction/coverage
    hits. Returns the number of rows updated.
    """
    from data.fetch_ohlc import fetch_ohlc

    init_prediction_tables()
    today = date.today().isoformat()

    conn = _connect()
    try:
        query = """SELECT id, ticker, prediction_date, current_close, p10, p50, p90
                   FROM predictions
                   WHERE actual_change IS NULL AND prediction_date < ?"""
        params: list = [today]
        if tickers:
            placeholders = ",".join("?" * len(tickers))
            query += f" AND ticker IN ({placeholders})"
            params.extend(t.upper().replace(".NS", "").replace(".BO", "") for t in tickers)
        rows = conn.execute(query, params).fetchall()
    finally:
        conn.close()

    if not rows:
        return 0

    updated = 0
    ohlc_cache: dict[str, pd.DataFrame] = {}
    for row_id, ticker, pred_date, current_close, p10, p50, p90 in rows:
        try:
            if ticker not in ohlc_cache:
                ohlc_cache[ticker] = fetch_ohlc(ticker, years=1)
            df = ohlc_cache[ticker]
            if df.empty or not current_close:
                continue
            target_day = pd.Timestamp(pred_date)
            if target_day not in df.index:
                continue
            actual_close = float(df.loc[target_day, "Close"])
            actual_change = round((actual_close / current_close - 1) * 100, 4)
            direction_correct = int(np.sign(actual_change) == np.sign(p50)) if p50 else 0
            in_range = int(p10 <= actual_change <= p90)

            conn = _connect()
            try:
                conn.execute(
                    """UPDATE predictions
                       SET actual_change = ?, direction_correct = ?, in_range = ?,
                           outcome_filled_at = ?
                       WHERE id = ?""",
                    (
                        actual_change,
                        direction_correct,
                        in_range,
                        datetime.now().isoformat(timespec="seconds"),
                        row_id,
                    ),
                )
                conn.commit()
            finally:
                conn.close()
            updated += 1
        except Exception as e:
            logger.warning(f"Outcome backfill failed for {ticker} {pred_date}: {e}")

    logger.info(f"Backfilled outcomes for {updated}/{len(rows)} predictions")
    return updated


def get_live_history(ticker: str, limit: int = 250) -> pd.DataFrame:
    """Served-prediction history for a ticker (most recent first)."""
    init_prediction_tables()
    clean = ticker.upper().replace(".NS", "").replace(".BO", "")
    conn = _connect()
    try:
        return pd.read_sql_query(
            """SELECT prediction_date, created_at, current_close,
                      p10, p50, p90, predicted_low, predicted_mid, predicted_high,
                      confidence, direction, model_version, source,
                      actual_change, direction_correct, in_range
               FROM predictions
               WHERE ticker = ?
               ORDER BY prediction_date DESC
               LIMIT ?""",
            conn,
            params=(clean, limit),
        )
    finally:
        conn.close()


def get_live_accuracy(ticker: str | None = None, window: int = 60) -> list[dict]:
    """Live accuracy metrics over the last `window` resolved predictions.

    Uses the shared metrics implementation, so numbers are directly
    comparable to training CV and backtest results.
    """
    init_prediction_tables()
    conn = _connect()
    try:
        query = """SELECT ticker, actual_change, p10, p50, p90, confidence
                   FROM predictions
                   WHERE actual_change IS NOT NULL"""
        params: tuple = ()
        if ticker:
            query += " AND ticker = ?"
            params = (ticker.upper().replace(".NS", "").replace(".BO", ""),)
        query += " ORDER BY prediction_date DESC"
        df = pd.read_sql_query(query, conn, params=params)
    finally:
        conn.close()

    if df.empty:
        return []

    out = []
    for tkr, grp in df.groupby("ticker"):
        grp = grp.head(window)
        summary = quantile_metrics_summary(
            grp["actual_change"].values,
            grp["p10"].values,
            grp["p50"].values,
            grp["p90"].values,
        )
        summary["ticker"] = tkr
        summary["n"] = int(len(grp))
        summary["avg_confidence"] = round(float(grp["confidence"].mean()), 1)
        out.append(summary)
    return sorted(out, key=lambda s: s["ticker"])


def save_sentiment_snapshot(ticker: str, score: float | None) -> None:
    """Record today's Claude sentiment score for future point-in-time training."""
    if score is None:
        return
    clean = ticker.upper().replace(".NS", "").replace(".BO", "")
    try:
        init_prediction_tables()
        conn = _connect()
        try:
            conn.execute(
                """INSERT INTO sentiment_history (ticker, date, score)
                   VALUES (?, ?, ?)
                   ON CONFLICT (ticker, date) DO UPDATE SET score = excluded.score""",
                (clean, date.today().isoformat(), float(score)),
            )
            conn.commit()
        finally:
            conn.close()
    except Exception as e:
        logger.warning(f"Sentiment snapshot failed for {clean}: {e}")
