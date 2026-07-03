"""Fetch OHLCV data from NSE via jugaad-data (primary) with yfinance fallback.

jugaad-data scrapes NSE directly — gives current-day data immediately after
market close. yfinance is used as fallback for bulk historical data.

Returns clean DataFrames with Date index and columns:
Open, High, Low, Close, Volume.
"""

import time
from datetime import date, timedelta

import pandas as pd

from app.config import settings
from data.cache import get_dataframe, set_dataframe
from utils.logger import logger


def _clean_ticker(ticker: str) -> str:
    """Strip exchange suffixes to get the bare NSE symbol."""
    return ticker.strip().upper().replace(".NS", "").replace(".BO", "")


def _normalize_ticker(ticker: str) -> str:
    """Ensure NSE tickers have .NS suffix (for yfinance)."""
    ticker = ticker.strip().upper()
    if not ticker.endswith((".NS", ".BO")):
        ticker += ".NS"
    return ticker


def _fetch_jugaad(symbol: str, from_date: date, to_date: date) -> pd.DataFrame:
    """Fetch OHLCV from NSE via jugaad-data. Returns standardized DataFrame."""
    from jugaad_data.nse import stock_df

    raw = stock_df(symbol=symbol, from_date=from_date, to_date=to_date)
    if raw.empty:
        return pd.DataFrame()

    # Filter to EQ series only (skip BE, BL, etc.)
    if "SERIES" in raw.columns:
        raw = raw[raw["SERIES"] == "EQ"]

    # jugaad-data returns dates in UTC (IST - 5:30), so a March 27 IST session
    # shows as March 26 18:30 UTC. Add IST offset before normalizing to get
    # the correct trading date.
    ist_dates = pd.to_datetime(raw["DATE"]) + pd.Timedelta(hours=5, minutes=30)

    df = pd.DataFrame(
        {
            "Open": raw["OPEN"].values,
            "High": raw["HIGH"].values,
            "Low": raw["LOW"].values,
            "Close": raw["CLOSE"].values,
            "Volume": raw["VOLUME"].values,
        },
        index=pd.DatetimeIndex(ist_dates.dt.normalize(), name="Date"),
    )

    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df


def _fetch_yfinance(ticker: str, start: date, end: date) -> pd.DataFrame:
    """Fallback: fetch from yfinance with retry."""
    import yfinance as yf

    for attempt in range(1, settings.yfinance_max_retries + 1):
        try:
            logger.info(f"yfinance fallback for {ticker} (attempt {attempt})")
            stock = yf.Ticker(ticker)
            # yfinance end is exclusive, so add 1 day
            df = stock.history(
                start=str(start),
                end=str(end + timedelta(days=1)),
                auto_adjust=True,
            )
            if not df.empty:
                keep = ["Open", "High", "Low", "Close", "Volume"]
                df = df[[c for c in keep if c in df.columns]].copy()
                df = df.dropna(subset=["Open", "High", "Low", "Close"])
                if df.index.tz is not None:
                    df.index = df.index.tz_localize(None)
                df.index.name = "Date"
                return df
        except Exception as e:
            logger.error(f"yfinance failed for {ticker}: {e} (attempt {attempt})")
        if attempt < settings.yfinance_max_retries:
            time.sleep(settings.yfinance_retry_delay * attempt)

    return pd.DataFrame()


def fetch_ohlc(
    ticker: str,
    years: int | None = None,
    as_of_date: date | None = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Fetch OHLCV data for a given ticker.

    Uses jugaad-data (NSE direct) as primary source, falls back to yfinance.

    Args:
        ticker: NSE stock symbol (e.g. "RELIANCE" or "RELIANCE.NS")
        years: Number of years of history. Defaults to config value.
        as_of_date: Only return data up to this date (look-ahead protection).
        use_cache: Whether to check cache first.

    Returns:
        DataFrame with DatetimeIndex and columns [Open, High, Low, Close, Volume].
        Empty DataFrame if fetch fails.
    """
    symbol = _clean_ticker(ticker)
    yf_ticker = _normalize_ticker(ticker)
    years = years or settings.ohlc_history_years
    cache_key = f"ohlc:{symbol}:{years}"

    # Check cache
    if use_cache:
        cached = get_dataframe(cache_key, settings.cache_ttl_ohlc)
        if cached is not None and not cached.empty:
            logger.debug(f"Cache hit for {symbol} OHLC")
            if as_of_date:
                return cached.loc[cached.index.date <= as_of_date]
            return cached

    # Date range
    end = date.today()
    start = end - timedelta(days=years * 365 + 30)

    # Primary: jugaad-data (NSE direct)
    df = pd.DataFrame()
    try:
        logger.info(f"Fetching {symbol} OHLC from NSE (jugaad-data)")
        df = _fetch_jugaad(symbol, from_date=start, to_date=end)
        if not df.empty:
            logger.info(f"NSE: fetched {len(df)} rows for {symbol}, latest={df.index[-1].date()}")
    except Exception as e:
        logger.warning(f"jugaad-data failed for {symbol}: {e}")

    # Fallback: yfinance
    if df.empty:
        logger.info(f"Falling back to yfinance for {symbol}")
        df = _fetch_yfinance(yf_ticker, start, end)
        if not df.empty:
            logger.info(f"yfinance: fetched {len(df)} rows for {symbol}")

    if df.empty:
        logger.error(f"All sources failed for {symbol}")
        return pd.DataFrame()

    # Cache the result
    if use_cache:
        set_dataframe(cache_key, df)

    # Apply as_of_date filter
    if as_of_date:
        df = df.loc[df.index.date <= as_of_date]

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
        results[_clean_ticker(ticker)] = df
        time.sleep(0.5)  # Brief pause to avoid rate limiting
    return results
