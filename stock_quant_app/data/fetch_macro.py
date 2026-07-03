"""Fetch global macro indicators — daily % change values.

Tickers:
    S&P 500     → ^GSPC
    Nasdaq      → ^IXIC
    Nifty 50    → ^NSEI  (proxy for GIFT Nifty in yfinance)
    Brent Crude → BZ=F
    USD/INR     → USDINR=X
    India VIX   → ^INDIAVIX
"""

import socket
import time
from dataclasses import dataclass
from datetime import date, timedelta

import pandas as pd
import yfinance as yf

from app.config import settings
from data.cache import get_dataframe, get_json, set_dataframe, set_json
from utils.logger import logger

# Set a global socket timeout so yfinance calls don't hang indefinitely
socket.setdefaulttimeout(30)

# Mapping of display name → yfinance ticker
MACRO_TICKERS: dict[str, str] = {
    "S&P 500": "^GSPC",
    "Nasdaq": "^IXIC",
    "Nifty 50": "^NSEI",
    "Brent Crude": "BZ=F",
    "USD/INR": "USDINR=X",
    "India VIX": "^INDIAVIX",
}

# yfinance ticker → (feature column name, lag in trading days when joined
# to the NSE calendar). US/overnight series get lag 1: a prediction made
# after the NSE close on day t can only see the US session completed on
# day t-1, so historical rows must carry the same information set.
MACRO_HISTORY_MAP: dict[str, tuple[str, int]] = {
    "^GSPC": ("sp500_change", 1),
    "^IXIC": ("nasdaq_change", 1),
    "BZ=F": ("brent_change", 1),
    "^NSEI": ("nifty_change", 0),
    "USDINR=X": ("usdinr_change", 0),
    "^INDIAVIX": ("vix_value", 0),
}


@dataclass
class MacroSnapshot:
    name: str
    ticker: str
    current_price: float | None
    prev_close: float | None
    change_pct: float | None
    fetch_date: str

    @property
    def is_valid(self) -> bool:
        return self.change_pct is not None


def _fetch_single_macro(name: str, ticker: str) -> MacroSnapshot:
    """Fetch current price and daily % change for one macro ticker."""
    today = date.today().isoformat()

    for attempt in range(1, settings.yfinance_max_retries + 1):
        try:
            info = yf.Ticker(ticker)
            # Use fast_info for speed — avoids full page scrape
            fast = info.fast_info

            current = getattr(fast, "last_price", None)
            prev = getattr(fast, "previous_close", None)

            if current is not None and prev is not None and prev != 0:
                change_pct = round(((current - prev) / prev) * 100, 2)
                return MacroSnapshot(
                    name=name,
                    ticker=ticker,
                    current_price=round(current, 2),
                    prev_close=round(prev, 2),
                    change_pct=change_pct,
                    fetch_date=today,
                )

            # Fallback: use history if fast_info fails
            hist = info.history(period="5d")
            if len(hist) >= 2:
                current = hist["Close"].iloc[-1]
                prev = hist["Close"].iloc[-2]
                change_pct = round(((current - prev) / prev) * 100, 2)
                return MacroSnapshot(
                    name=name,
                    ticker=ticker,
                    current_price=round(current, 2),
                    prev_close=round(prev, 2),
                    change_pct=change_pct,
                    fetch_date=today,
                )

            logger.warning(f"Insufficient data for {name} ({ticker}), attempt {attempt}")

        except Exception as e:
            logger.error(f"Error fetching {name} ({ticker}): {e}, attempt {attempt}")

        if attempt < settings.yfinance_max_retries:
            time.sleep(settings.yfinance_retry_delay)

    # All retries failed
    return MacroSnapshot(
        name=name,
        ticker=ticker,
        current_price=None,
        prev_close=None,
        change_pct=None,
        fetch_date=today,
    )


def fetch_macro_snapshot(use_cache: bool = True) -> list[MacroSnapshot]:
    """Fetch daily % change for all macro indicators.

    Returns list of MacroSnapshot objects. Failed fetches have change_pct=None.
    """
    cache_key = "macro:snapshot"

    if use_cache:
        cached = get_json(cache_key, settings.cache_ttl_macro)
        if cached is not None:
            logger.debug("Cache hit for macro snapshot")
            return [MacroSnapshot(**item) for item in cached]

    results: list[MacroSnapshot] = []
    for name, ticker in MACRO_TICKERS.items():
        logger.info(f"Fetching macro: {name} ({ticker})")
        snapshot = _fetch_single_macro(name, ticker)
        results.append(snapshot)
        time.sleep(0.3)  # Rate limiting

    # Cache valid results
    if use_cache and any(s.is_valid for s in results):
        set_json(
            cache_key,
            [
                {
                    "name": s.name,
                    "ticker": s.ticker,
                    "current_price": s.current_price,
                    "prev_close": s.prev_close,
                    "change_pct": s.change_pct,
                    "fetch_date": s.fetch_date,
                }
                for s in results
            ],
        )

    valid_count = sum(1 for s in results if s.is_valid)
    logger.info(f"Macro snapshot: {valid_count}/{len(results)} indicators fetched")
    return results


def fetch_macro_history_df(years: int = 4, use_cache: bool = True) -> pd.DataFrame:
    """Fetch point-in-time daily % changes for all macro tickers.

    Returns a DataFrame with a naive DatetimeIndex (dates) and one column
    per feature name in MACRO_HISTORY_MAP. US/overnight series are already
    shifted by their lag so a row at date t contains only information that
    was observable before the NSE close on day t.

    Used to build leakage-free historical macro features for training.
    """
    cache_key = f"macro:history_df:{years}"
    if use_cache:
        cached = get_dataframe(cache_key, settings.cache_ttl_ohlc)
        if cached is not None and not cached.empty:
            cached.index = pd.to_datetime(cached.index)
            return cached

    end = date.today() + timedelta(days=1)
    start = end - timedelta(days=int(years * 365.25) + 30)

    series: dict[str, pd.Series] = {}
    for ticker, (feature, lag) in MACRO_HISTORY_MAP.items():
        for attempt in range(1, settings.yfinance_max_retries + 1):
            try:
                hist = yf.Ticker(ticker).history(start=str(start), end=str(end))
                if hist.empty or len(hist) < 2:
                    raise ValueError("empty history")
                close = hist["Close"]
                if close.index.tz is not None:
                    close.index = close.index.tz_localize(None)
                close.index = close.index.normalize()
                close = close[~close.index.duplicated(keep="last")]
                pct = (close.pct_change() * 100).round(4)
                if lag:
                    pct = pct.shift(lag)
                series[feature] = pct
                break
            except Exception as e:
                logger.warning(f"Macro history fetch failed for {ticker} (attempt {attempt}): {e}")
                if attempt < settings.yfinance_max_retries:
                    time.sleep(settings.yfinance_retry_delay)
        time.sleep(0.3)  # Rate limiting

    if not series:
        logger.error("No macro history could be fetched")
        return pd.DataFrame()

    df = pd.DataFrame(series).sort_index()
    # Union calendar across markets — forward-fill so NSE trading days that
    # are US holidays carry the last observed US value (same as live behavior).
    df = df.ffill()

    if use_cache and not df.empty:
        set_dataframe(cache_key, df)
    return df


def fetch_macro_history(days: int = 60) -> dict[str, list[float]]:
    """Fetch historical daily % changes for macro indicators.

    Returns {indicator_name: [pct_change_values]} for the last N trading days.
    Used as features in the ML model.
    """
    from datetime import timedelta

    cache_key = f"macro:history:{days}"
    cached = get_json(cache_key, settings.cache_ttl_macro)
    if cached is not None:
        return cached

    end = date.today()
    start = end - timedelta(days=days + 10)  # Buffer for weekends/holidays

    result: dict[str, list[float]] = {}

    for name, ticker in MACRO_TICKERS.items():
        try:
            hist = yf.Ticker(ticker).history(start=str(start), end=str(end))
            if not hist.empty and len(hist) >= 2:
                pct = hist["Close"].pct_change().dropna().round(4).tolist()
                result[name] = pct[-days:]  # Trim to requested length
            else:
                result[name] = []
        except Exception as e:
            logger.error(f"Error fetching {name} history: {e}")
            result[name] = []
        time.sleep(0.3)

    set_json(cache_key, result)
    return result
