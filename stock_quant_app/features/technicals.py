"""Pillar 1 — Technical Features for short-term timing.

Computes all technical indicators from OHLCV data using pandas_ta.
Every function takes a DataFrame and returns it with new columns appended.
All features are stationary (ratios, z-scores, bounded oscillators) — never raw prices.

Features:
    - RSI (14)
    - MACD (12, 26, 9) — histogram normalized by close
    - Bollinger Bands — %B position (0-1 scale)
    - EMA 20/50/200 — price position relative to each (ratio)
    - ATR (14) — normalized by close (%)
    - ADX (14) — trend strength (0-100)
    - Volume Z-score (20-day rolling)
    - Market Regime classifier (trending vs sideways)
"""

import numpy as np
import pandas as pd
from utils.logger import logger

try:
    import pandas_ta as ta
    _PANDAS_TA_AVAILABLE = True
except Exception as _e:  # noqa: BLE001
    ta = None  # type: ignore[assignment]
    _PANDAS_TA_AVAILABLE = False
    logger.warning(f"pandas_ta unavailable — technical indicators disabled: {_e}")


def add_rsi(df: pd.DataFrame, length: int = 14) -> pd.DataFrame:
    """RSI oscillator (0-100). Already stationary."""
    df["rsi_14"] = ta.rsi(df["Close"], length=length)
    return df


def add_macd(df: pd.DataFrame) -> pd.DataFrame:
    """MACD histogram normalized by closing price (percentage scale)."""
    macd = ta.macd(df["Close"], fast=12, slow=26, signal=9)
    if macd is not None and not macd.empty:
        # Histogram normalized by close — makes it comparable across price levels
        df["macd_hist_pct"] = macd.iloc[:, 2] / df["Close"] * 100
        # MACD line - Signal line crossover direction
        df["macd_signal_diff"] = (macd.iloc[:, 0] - macd.iloc[:, 1]) / df["Close"] * 100
    return df


def add_bollinger_bands(df: pd.DataFrame, length: int = 20, std: float = 2.0) -> pd.DataFrame:
    """Bollinger %B — position within bands (0 = lower, 1 = upper)."""
    bb = ta.bbands(df["Close"], length=length, std=std)
    if bb is not None and not bb.empty:
        upper = bb.iloc[:, 2]  # BBU
        lower = bb.iloc[:, 0]  # BBL
        band_width = upper - lower
        # %B: where price sits within the bands
        df["bb_pct_b"] = np.where(
            band_width > 0,
            (df["Close"] - lower) / band_width,
            0.5,
        )
        # Band width normalized by close — volatility measure
        df["bb_width_pct"] = band_width / df["Close"] * 100
    return df


def add_ema_positions(df: pd.DataFrame) -> pd.DataFrame:
    """Price position relative to EMA 20/50/200 (percentage distance)."""
    for period in [20, 50, 200]:
        ema = ta.ema(df["Close"], length=period)
        if ema is not None:
            # % distance from EMA — positive = above, negative = below
            df[f"ema_{period}_dist_pct"] = (df["Close"] - ema) / ema * 100
    return df


def add_atr(df: pd.DataFrame, length: int = 14) -> pd.DataFrame:
    """ATR normalized by close (percentage). Measures volatility."""
    atr = ta.atr(df["High"], df["Low"], df["Close"], length=length)
    if atr is not None:
        df["atr_pct"] = atr / df["Close"] * 100
    return df


def add_adx(df: pd.DataFrame, length: int = 14) -> pd.DataFrame:
    """ADX trend strength (0-100). >25 = trending, <20 = sideways."""
    adx = ta.adx(df["High"], df["Low"], df["Close"], length=length)
    if adx is not None and not adx.empty:
        df["adx_14"] = adx.iloc[:, 0]  # ADX value
        df["di_plus"] = adx.iloc[:, 1]  # +DI
        df["di_minus"] = adx.iloc[:, 2]  # -DI
        # DI spread — positive = bullish trend, negative = bearish
        df["di_spread"] = df["di_plus"] - df["di_minus"]
    return df


def add_volume_zscore(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Volume z-score over rolling window. Detects unusual volume (breakouts)."""
    vol_mean = df["Volume"].rolling(window=window).mean()
    vol_std = df["Volume"].rolling(window=window).std()
    df["volume_zscore"] = np.where(vol_std > 0, (df["Volume"] - vol_mean) / vol_std, 0)
    return df


def add_market_regime(df: pd.DataFrame) -> pd.DataFrame:
    """Classify market regime: Trending (1) vs Sideways (0).

    Uses ADX > 25 AND price above/below EMA 50 as primary signal.
    Also encodes direction: +1 = trending up, -1 = trending down, 0 = sideways.
    """
    # Ensure ADX and EMA are computed
    if "adx_14" not in df.columns:
        df = add_adx(df)
    if "ema_50_dist_pct" not in df.columns:
        df = add_ema_positions(df)

    # Regime: ADX > 25 = trending
    is_trending = df["adx_14"] > 25

    # Direction: based on EMA 50 distance
    is_bullish = df["ema_50_dist_pct"] > 0

    df["regime"] = np.where(
        is_trending & is_bullish, 1,      # Trending up
        np.where(
            is_trending & ~is_bullish, -1,  # Trending down
            0,                              # Sideways
        ),
    )
    return df


def add_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Add return-based features for stationarity."""
    df["return_1d"] = df["Close"].pct_change(1) * 100
    df["return_5d"] = df["Close"].pct_change(5) * 100
    df["return_20d"] = df["Close"].pct_change(20) * 100

    # Intraday range as % of close
    df["intraday_range_pct"] = (df["High"] - df["Low"]) / df["Close"] * 100

    # Gap — overnight return
    df["gap_pct"] = (df["Open"] - df["Close"].shift(1)) / df["Close"].shift(1) * 100
    return df


def compute_all_technicals(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all Pillar 1 technical features to an OHLCV DataFrame.

    Args:
        df: DataFrame with columns [Open, High, Low, Close, Volume] and DatetimeIndex.

    Returns:
        Same DataFrame with ~20 new feature columns appended.
    """
    df = df.copy()

    df = add_returns(df)
    df = add_volume_zscore(df)

    if not _PANDAS_TA_AVAILABLE:
        logger.warning("pandas_ta not available — returning only return-based features")
        return df

    df = add_rsi(df)
    df = add_macd(df)
    df = add_bollinger_bands(df)
    df = add_ema_positions(df)
    df = add_atr(df)
    df = add_adx(df)
    df = add_market_regime(df)

    return df


# Column names for all technical features (used by feature_builder)
TECHNICAL_FEATURES = [
    "return_1d", "return_5d", "return_20d",
    "intraday_range_pct", "gap_pct",
    "rsi_14",
    "macd_hist_pct", "macd_signal_diff",
    "bb_pct_b", "bb_width_pct",
    "ema_20_dist_pct", "ema_50_dist_pct", "ema_200_dist_pct",
    "atr_pct",
    "adx_14", "di_spread",
    "volume_zscore",
    "regime",
]
