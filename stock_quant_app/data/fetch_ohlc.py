"""Fetch OHLCV data from yfinance with retry logic and caching.

Supports NSE tickers (append .NS if not present) and returns clean
DataFrames with Date index and columns: Open, High, Low, Close, Volume.
"""

import time
from datetime import date, timedelta

import pandas as pd
import yfinance as yf

from app.config import settings
from data.cache import get_dataframe, set_dataframe
from utils.logger import logger


def _normalize_ticker(ticker: str) -> str:
    """Ensure NSE tickers have .NS suffix."""
    ticker = ticker.strip().upper()
    if not ticker.endswith((".NS", ".BO")):
        ticker += ".NS"
    return ticker


def fetch_ohlc(
    ticker: str,
    years: int | None = None,
    as_of_date: date | None = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Fetch OHLCV data for a given ticker.

    Args:
        ticker: NSE stock symbol (e.g. "RELIANCE" or "RELIANCE.NS")
        years: Number of years of history. Defaults to config value.
        as_of_date: Only return data up to this date (look-ahead protection).
        use_cache: Whether to check cache first.

    Returns:
        DataFrame with DatetimeIndex and columns [Open, High, Low, Close, Volume].
        Empty DataFrame if fetch fails after retries.
    """
    ticker = _normalize_ticker(ticker)
    years = years or settings.ohlc_history_years
    cache_key = f"ohlc:{ticker}:{years}"

    # Check cache
    if use_cache:
        cached = get_dataframe(cache_key, settings.cache_ttl_ohlc)
        if cached is not None and not cached.empty:
            logger.debug(f"Cache hit for {ticker} OHLC")
            if as_of_date:
                return cached.loc[cached.index.date <= as_of_date]
            return cached

    # Calculate date range
    end = date.today()
    start = end - timedelta(days=years * 365 + 30)  # Extra buffer for holidays

    # Fetch with retry
    df = pd.DataFrame()
    for attempt in range(1, settings.yfinance_max_retries + 1):
        try:
            max_retries = settings.yfinance_max_retries
            logger.info(f"Fetching {ticker} OHLC (attempt {attempt}/{max_retries})")
            stock = yf.Ticker(ticker)
            df = stock.history(start=str(start), end=str(end), auto_adjust=True)

            if df.empty:
                logger.warning(f"Empty response for {ticker}, attempt {attempt}")
                time.sleep(settings.yfinance_retry_delay)
                continue

            break  # Success

        except Exception as e:
            logger.error(f"Failed to fetch {ticker}: {e} (attempt {attempt})")
            if attempt < settings.yfinance_max_retries:
                time.sleep(settings.yfinance_retry_delay * attempt)

    if df.empty:
        logger.error(f"All retries exhausted for {ticker}")
        return pd.DataFrame()

    # Clean up columns — yfinance sometimes includes Dividends, Stock Splits
    keep_cols = ["Open", "High", "Low", "Close", "Volume"]
    df = df[[c for c in keep_cols if c in df.columns]].copy()

    # Remove timezone info from index for consistency
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    df.index.name = "Date"

    # Cache the result
    if use_cache:
        set_dataframe(cache_key, df)

    # Apply as_of_date filter
    if as_of_date:
        df = df.loc[df.index.date <= as_of_date]

    logger.info(f"Fetched {len(df)} rows for {ticker}")
    return df


def fetch_ohlc_multiple(
    tickers: list[str],
    years: int | None = None,
    as_of_date: date | None = None,
) -> dict[str, pd.DataFrame]:
    """Fetch OHLCV for multiple tickers. Returns {ticker: DataFrame}."""
    results = {}
    for ticker in tickers:
        df = fetch_ohlc(ticker, years=years, as_of_date=as_of_date)
        results[_normalize_ticker(ticker)] = df
        # Brief pause to avoid rate limiting
        time.sleep(0.5)
    return results
