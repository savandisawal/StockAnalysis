"""APScheduler jobs for daily data refresh, predictions, and retraining.

Schedule (IST):
- Daily 15:45: refresh macro data (market close)
- Daily 15:50: refresh OHLC for tracked stocks
- Daily 16:30 Mon-Fri: backfill yesterday's outcomes, then predict +
  persist for all tracked stocks (feeds the live Truth Dashboard and
  the sentiment history used by future retrains)
- Weekly Friday 16:00: retrain models (all three pillars)
- Daily 00:00: cache cleanup

Run standalone: python -m services.scheduler
Or integrate with FastAPI via lifespan.
"""

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

from app.config import settings
from data.cache import clear_expired
from data.fetch_macro import fetch_macro_snapshot
from utils.logger import logger


def _tracked_stocks() -> list[str]:
    return [t.strip().upper() for t in settings.tracked_stocks.split(",") if t.strip()]


# Kept for backward compatibility with existing imports
TRACKED_STOCKS = _tracked_stocks()

scheduler = BackgroundScheduler(timezone="Asia/Kolkata")


def refresh_macro_data():
    """Fetch latest macro indicators. Runs daily after market close."""
    logger.info("Scheduler: refreshing macro data")
    try:
        snapshots = fetch_macro_snapshot(use_cache=False)
        valid = sum(1 for s in snapshots if s.is_valid)
        logger.info(f"Scheduler: macro refresh done, {valid}/{len(snapshots)} valid")
    except Exception as e:
        logger.error(f"Scheduler: macro refresh failed: {e}")


def refresh_ohlc_data():
    """Refresh OHLC data for tracked stocks."""
    from data.fetch_ohlc import fetch_ohlc

    stocks = _tracked_stocks()
    logger.info(f"Scheduler: refreshing OHLC for {len(stocks)} stocks")
    for ticker in stocks:
        try:
            fetch_ohlc(ticker, use_cache=False)
        except Exception as e:
            logger.error(f"Scheduler: OHLC refresh failed for {ticker}: {e}")


def daily_predictions():
    """Backfill yesterday's outcomes, then predict + persist for all
    tracked stocks. Runs after market close on trading days."""
    from features.macro_sentiment import compute_macro_sentiment_features
    from services.prediction_service import get_prediction
    from services.prediction_store import backfill_outcomes, save_sentiment_snapshot
    from utils.sectors import get_sector

    stocks = _tracked_stocks()
    logger.info(f"Scheduler: daily predictions for {len(stocks)} stocks")

    try:
        backfill_outcomes(stocks)
    except Exception as e:
        logger.error(f"Scheduler: outcome backfill failed: {e}")

    for ticker in stocks:
        try:
            result = get_prediction(ticker, source="scheduler")
            if result is None:
                logger.warning(f"Scheduler: no prediction for {ticker} (model missing?)")
                continue
            # Record today's sentiment for future point-in-time training
            sentiment = compute_macro_sentiment_features(stock=ticker, sector=get_sector(ticker))
            save_sentiment_snapshot(ticker, sentiment.news_sentiment)
        except Exception as e:
            logger.error(f"Scheduler: daily prediction failed for {ticker}: {e}")


def retrain_models():
    """Weekly model retraining for tracked stocks."""
    from services.prediction_service import train_model

    stocks = _tracked_stocks()
    logger.info(f"Scheduler: retraining models for {len(stocks)} stocks")
    for ticker in stocks:
        try:
            metrics = train_model(ticker, years=3, include_fundamentals=True, include_macro=True)
            if "error" not in metrics:
                acc = metrics.get("direction_accuracy")
                logger.info(f"Scheduler: retrained {ticker}, acc={acc}")
            else:
                logger.error(f"Scheduler: retrain failed for {ticker}: {metrics['error']}")
        except Exception as e:
            logger.error(f"Scheduler: retrain error for {ticker}: {e}")


def cleanup_cache():
    """Remove old cache entries (>7 days)."""
    removed = clear_expired()
    logger.info(f"Scheduler: cleaned {removed} expired cache entries")


def start_scheduler():
    """Start the background scheduler with all jobs."""
    # Daily at 3:45 PM IST — after market close
    scheduler.add_job(
        refresh_macro_data,
        CronTrigger(hour=15, minute=45),
        id="refresh_macro",
        replace_existing=True,
    )
    scheduler.add_job(
        refresh_ohlc_data,
        CronTrigger(hour=15, minute=50),
        id="refresh_ohlc",
        replace_existing=True,
    )

    # Daily 4:30 PM IST Mon-Fri — backfill outcomes + predict + persist
    scheduler.add_job(
        daily_predictions,
        CronTrigger(day_of_week="mon-fri", hour=16, minute=30),
        id="daily_predictions",
        replace_existing=True,
    )

    # Weekly Friday 4 PM IST — retrain models
    scheduler.add_job(
        retrain_models,
        CronTrigger(day_of_week="fri", hour=16, minute=0),
        id="retrain_models",
        replace_existing=True,
    )

    # Daily midnight — cache cleanup
    scheduler.add_job(
        cleanup_cache,
        CronTrigger(hour=0, minute=0),
        id="cleanup_cache",
        replace_existing=True,
    )

    scheduler.start()
    logger.info("Scheduler started with 5 jobs")


def stop_scheduler():
    """Gracefully shut down the scheduler."""
    if scheduler.running:
        scheduler.shutdown()
        logger.info("Scheduler stopped")
