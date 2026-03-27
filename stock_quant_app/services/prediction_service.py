"""Prediction service — orchestrates the full prediction pipeline.

Single entry point that coordinates: data fetch, feature build,
model load, inference, and result formatting.
"""

from model.backtest import get_backtest_history, get_backtest_summary, run_backtest
from model.predict import PredictionResult, predict_next_day
from model.train_model import get_feature_importance, train_quantile_models
from utils.logger import logger


def get_prediction(
    ticker: str,
    include_fundamentals: bool = True,
    include_macro: bool = True,
) -> PredictionResult | None:
    """Get next-day prediction for a stock. Uses latest trained model."""
    return predict_next_day(
        ticker,
        include_fundamentals=include_fundamentals,
        include_macro=include_macro,
    )


def train_model(
    ticker: str,
    years: int = 3,
    include_fundamentals: bool = True,
    include_macro: bool = True,
) -> dict:
    """Train a new model for a stock and return metrics."""
    try:
        models, features, metrics = train_quantile_models(
            ticker, years=years,
            include_fundamentals=include_fundamentals,
            include_macro=include_macro,
        )
        # Add feature importance
        importance = get_feature_importance(models, features, top_n=15)
        metrics["feature_importance"] = importance.to_dict(orient="records")
        return metrics
    except Exception as e:
        logger.error(f"Training failed for {ticker}: {e}")
        return {"error": str(e)}


def run_stock_backtest(
    ticker: str,
    years: int = 3,
    include_fundamentals: bool = False,
    include_macro: bool = False,
) -> dict:
    """Run backtest for a stock and return metrics."""
    try:
        metrics = run_backtest(
            ticker, years=years,
            include_fundamentals=include_fundamentals,
            include_macro=include_macro,
        )
        return metrics.to_dict()
    except Exception as e:
        logger.error(f"Backtest failed for {ticker}: {e}")
        return {"error": str(e)}


def get_prediction_history(ticker: str, limit: int = 500) -> list[dict]:
    """Get past backtest predictions for Truth Dashboard."""
    df = get_backtest_history(ticker, limit=limit)
    if df.empty:
        return []
    return df.to_dict(orient="records")


def get_all_backtest_summaries() -> list[dict]:
    """Get latest backtest summaries for all tickers."""
    return get_backtest_summary()
