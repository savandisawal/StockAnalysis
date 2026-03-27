"""APScheduler jobs for daily data refresh and model retraining.

Schedule:
- Daily at 3:45 PM IST: refresh OHLC + macro data (market close)
- Weekly Friday 4 PM IST: retrain models for tracked stocks

Run standalone: python -m services.scheduler
Or integrate with FastAPI via lifespan.
"""

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

from data.cache import clear_expired
from data.fetch_macro import fetch_macro_snapshot
from utils.logger import logger

# Stocks to auto-refresh (extend as needed)
TRACKED_STOCKS = [
    "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK",
    "KOTAKBANK", "SBIN", "BHARTIARTL", "ITC", "TATAMOTORS",
]

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

    logger.info(f"Scheduler: refreshing OHLC for {len(TRACKED_STOCKS)} stocks")
    for ticker in TRACKED_STOCKS:
        try:
            fetch_ohlc(ticker, use_cache=False)
        except Exception as e:
            logger.error(f"Scheduler: OHLC refresh failed for {ticker}: {e}")


def retrain_models():
    """Weekly model retraining for tracked stocks."""
    from services.prediction_service import train_model

    logger.info(f"Scheduler: retraining models for {len(TRACKED_STOCKS)} stocks")
    for ticker in TRACKED_STOCKS:
        try:
            metrics = train_model(ticker, years=3, include_fundamentals=False, include_macro=False)
            if "error" not in metrics:
                acc = metrics.get('direction_accuracy')
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
    logger.info("Scheduler started with 4 jobs")


def stop_scheduler():
    """Gracefully shut down the scheduler."""
    if scheduler.running:
        scheduler.shutdown()
        logger.info("Scheduler stopped")
