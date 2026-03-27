"""FastAPI application — async API layer for the quant system."""

import re
from contextlib import asynccontextmanager

from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from app.auth import verify_credentials
from app.database import init_db
from data.fetch_macro import fetch_macro_snapshot
from model.model_registry import list_models
from services.prediction_service import (
    get_all_backtest_summaries,
    get_prediction,
    get_prediction_history,
    run_stock_backtest,
    train_model,
)
from services.scheduler import start_scheduler, stop_scheduler
from utils.logger import logger

# ── Validation helpers ────────────────────────────────────────────

_TICKER_RE = re.compile(r"^[A-Z][A-Z0-9&_-]{0,19}(\.NS|\.BO)?$", re.IGNORECASE)


def _validate_ticker(ticker: str) -> str:
    """Validate and normalize a ticker symbol."""
    ticker = ticker.strip().upper()
    if not _TICKER_RE.match(ticker):
        raise HTTPException(status_code=400, detail=f"Invalid ticker format: {ticker}")
    return ticker


def _validate_years(years: int, min_val: int = 1, max_val: int = 10) -> int:
    if not min_val <= years <= max_val:
        raise HTTPException(
            status_code=400, detail=f"years must be between {min_val} and {max_val}"
        )
    return years


def _validate_limit(limit: int, max_val: int = 5000) -> int:
    if limit < 1 or limit > max_val:
        raise HTTPException(status_code=400, detail=f"limit must be between 1 and {max_val}")
    return limit


# ── App lifecycle ─────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    logger.info("Starting Stock Quant API")
    await init_db()
    start_scheduler()
    yield
    stop_scheduler()
    logger.info("Shutting down Stock Quant API")


app = FastAPI(
    title="Stock Quant API",
    description="NSE India next-day price range prediction system",
    version="0.1.0",
    lifespan=lifespan,
)


# ── Global exception handler ─────────────────────────────────────


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error on {request.method} {request.url.path}: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)},
    )


# ── Health check ─────────────────────────────────────────────────


@app.get("/health")
async def health():
    return {"status": "ok"}


# ── Macro data ───────────────────────────────────────────────────


@app.get("/macro")
async def macro_snapshot(user: str = Depends(verify_credentials)):
    """Get current global macro indicators with % change."""
    snapshots = fetch_macro_snapshot()
    if not snapshots:
        raise HTTPException(status_code=503, detail="Failed to fetch macro data")
    return {
        "indicators": [
            {
                "name": s.name,
                "price": s.current_price,
                "prev_close": s.prev_close,
                "change_pct": s.change_pct,
                "date": s.fetch_date,
            }
            for s in snapshots
        ]
    }


# ── Prediction ───────────────────────────────────────────────────


@app.get("/predict/{ticker}")
async def predict(
    ticker: str,
    include_fundamentals: bool = True,
    include_macro: bool = True,
    user: str = Depends(verify_credentials),
):
    """Get next-day price range prediction for a stock."""
    ticker = _validate_ticker(ticker)
    result = get_prediction(
        ticker,
        include_fundamentals=include_fundamentals,
        include_macro=include_macro,
    )
    if result is None:
        raise HTTPException(
            status_code=404,
            detail=f"No trained model for {ticker}. Train first via POST /train/{ticker}",
        )
    return result.to_dict()


# ── Training ─────────────────────────────────────────────────────


@app.post("/train/{ticker}")
async def train(
    ticker: str,
    years: int = 3,
    include_fundamentals: bool = True,
    include_macro: bool = True,
    background_tasks: BackgroundTasks = None,
    user: str = Depends(verify_credentials),
):
    """Train a new quantile regression model for a stock."""
    ticker = _validate_ticker(ticker)
    years = _validate_years(years, 1, 5)

    metrics = train_model(
        ticker, years=years,
        include_fundamentals=include_fundamentals,
        include_macro=include_macro,
    )
    if "error" in metrics:
        raise HTTPException(status_code=500, detail=metrics["error"])
    return metrics


# ── Backtest ─────────────────────────────────────────────────────


@app.post("/backtest/{ticker}")
async def backtest(
    ticker: str,
    years: int = 3,
    user: str = Depends(verify_credentials),
):
    """Run walk-forward backtest for a stock."""
    ticker = _validate_ticker(ticker)
    years = _validate_years(years, 1, 5)

    metrics = run_stock_backtest(ticker, years=years)
    if "error" in metrics:
        raise HTTPException(status_code=500, detail=metrics["error"])
    return metrics


@app.get("/backtest/{ticker}/history")
async def backtest_history(
    ticker: str,
    limit: int = 500,
    user: str = Depends(verify_credentials),
):
    """Get past backtest predictions for Truth Dashboard."""
    ticker = _validate_ticker(ticker)
    limit = _validate_limit(limit)
    return {"predictions": get_prediction_history(ticker, limit=limit)}


@app.get("/backtest/summary/all")
async def backtest_summaries(user: str = Depends(verify_credentials)):
    """Get latest backtest summaries for all tickers."""
    return {"summaries": get_all_backtest_summaries()}


# ── Model registry ───────────────────────────────────────────────


@app.get("/models")
async def models_list(
    ticker: str | None = None,
    user: str = Depends(verify_credentials),
):
    """List all trained models, optionally filtered by ticker."""
    if ticker:
        ticker = _validate_ticker(ticker)
    return {"models": list_models(ticker)}
