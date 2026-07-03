"""FastAPI application — async API layer for the quant system."""

import re
import uuid
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from app.auth import validate_auth_config, verify_credentials
from app.config import settings
from app.database import init_db
from data.fetch_macro import fetch_macro_snapshot
from model.model_registry import list_models
from services.prediction_service import (
    get_all_backtest_summaries,
    get_live_accuracy_summary,
    get_live_prediction_history,
    get_prediction,
    get_prediction_history,
    run_stock_backtest,
    train_model,
)
from services.prediction_store import init_prediction_tables
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
    logger.info(f"Starting Stock Quant API (environment={settings.environment})")
    validate_auth_config()  # fail loudly on misconfigured prod auth
    await init_db()
    init_prediction_tables()
    start_scheduler()
    yield
    stop_scheduler()
    logger.info("Shutting down Stock Quant API")


limiter = Limiter(
    key_func=get_remote_address,
    default_limits=[settings.rate_limit_default],
)

app = FastAPI(
    title="Stock Quant API",
    description="NSE India next-day price range prediction system",
    version="0.2.0",
    lifespan=lifespan,
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

if settings.get_cors_origins():
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.get_cors_origins(),
        allow_credentials=True,
        allow_methods=["GET", "POST"],
        allow_headers=["Authorization", "Content-Type", "X-Request-ID"],
    )


# ── Request correlation ──────────────────────────────────────────


@app.middleware("http")
async def request_context_middleware(request: Request, call_next):
    """Attach a request ID to every request, its logs, and its response."""
    rid = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:12]
    request.state.request_id = rid
    with logger.contextualize(request_id=rid, path=request.url.path, method=request.method):
        response = await call_next(request)
    response.headers["X-Request-ID"] = rid
    return response


# ── Global exception handler ─────────────────────────────────────


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    rid = getattr(request.state, "request_id", "unknown")
    logger.opt(exception=exc).error(
        f"Unhandled error on {request.method} {request.url.path} (request_id={rid})"
    )
    # Never leak internals to the client — the request_id links to server logs
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "request_id": rid},
    )


# ── Health check ─────────────────────────────────────────────────


@app.get("/health")
async def health():
    return {"status": "ok"}


# ── Macro data ───────────────────────────────────────────────────


@app.get("/macro")
async def macro_snapshot(user: str = Depends(verify_credentials)):
    """Get current global macro indicators with % change."""
    snapshots = await run_in_threadpool(fetch_macro_snapshot)
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
    """Get next-day price range prediction for a stock.

    The prediction is persisted with full provenance (model version,
    features hash, SHAP explanation) and its outcome is backfilled the
    next trading day.
    """
    ticker = _validate_ticker(ticker)
    result = await run_in_threadpool(
        get_prediction,
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


@app.get("/predictions/{ticker}/history")
async def live_prediction_history(
    ticker: str,
    limit: int = 250,
    user: str = Depends(verify_credentials),
):
    """History of real served predictions with outcomes where resolved."""
    ticker = _validate_ticker(ticker)
    limit = _validate_limit(limit)
    predictions = await run_in_threadpool(get_live_prediction_history, ticker, limit)
    return {"predictions": predictions}


@app.get("/predictions/summary/live")
async def live_accuracy(
    ticker: str | None = None,
    window: int = 60,
    user: str = Depends(verify_credentials),
):
    """Live accuracy metrics computed from served predictions."""
    if ticker:
        ticker = _validate_ticker(ticker)
    window = _validate_limit(window, max_val=1000)
    summaries = await run_in_threadpool(get_live_accuracy_summary, ticker, window)
    return {"summaries": summaries}


# ── Training ─────────────────────────────────────────────────────


@app.post("/train/{ticker}")
@limiter.limit(settings.rate_limit_heavy)
async def train(
    request: Request,
    ticker: str,
    years: int = 3,
    include_fundamentals: bool = True,
    include_macro: bool = True,
    user: str = Depends(verify_credentials),
):
    """Train a new quantile regression model for a stock."""
    ticker = _validate_ticker(ticker)
    years = _validate_years(years, 1, 5)

    metrics = await run_in_threadpool(
        train_model,
        ticker,
        years=years,
        include_fundamentals=include_fundamentals,
        include_macro=include_macro,
    )
    if "error" in metrics:
        raise HTTPException(status_code=500, detail=metrics["error"])
    return metrics


# ── Backtest ─────────────────────────────────────────────────────


@app.post("/backtest/{ticker}")
@limiter.limit(settings.rate_limit_heavy)
async def backtest(
    request: Request,
    ticker: str,
    years: int = 3,
    user: str = Depends(verify_credentials),
):
    """Run walk-forward backtest for a stock."""
    ticker = _validate_ticker(ticker)
    years = _validate_years(years, 1, 5)

    metrics = await run_in_threadpool(run_stock_backtest, ticker, years=years)
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
    predictions = await run_in_threadpool(get_prediction_history, ticker, limit)
    return {"predictions": predictions}


@app.get("/backtest/summary/all")
async def backtest_summaries(user: str = Depends(verify_credentials)):
    """Get latest backtest summaries for all tickers."""
    summaries = await run_in_threadpool(get_all_backtest_summaries)
    return {"summaries": summaries}


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
