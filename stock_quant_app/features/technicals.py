"""Pillar 1 — Technical Features for short-term timing.

Each indicator uses pandas_ta when available, with a pure numpy/pandas
fallback so the app works even if pandas_ta fails to install.

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
    logger.warning(f"pandas_ta unavailable — using built-in fallback implementations: {_e}")


# ── Pure numpy/pandas fallback implementations ───────────────────


def _ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False).mean()


def _rsi_fallback(series: pd.Series, length: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / length, min_periods=length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / length, min_periods=length, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _atr_fallback(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / length, min_periods=length, adjust=False).mean()


def _adx_fallback(
    high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Returns (ADX, +DI, -DI)."""
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)

    up = high - high.shift(1)
    down = low.shift(1) - low
    pos_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=high.index)
    neg_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=high.index)

    alpha = 1 / length
    atr_s = tr.ewm(alpha=alpha, min_periods=length, adjust=False).mean()
    pos_di = 100 * pos_dm.ewm(alpha=alpha, min_periods=length, adjust=False).mean() / atr_s
    neg_di = 100 * neg_dm.ewm(alpha=alpha, min_periods=length, adjust=False).mean() / atr_s
    di_sum = pos_di + neg_di
    dx = 100 * (pos_di - neg_di).abs() / di_sum.replace(0, np.nan)
    adx = dx.ewm(alpha=alpha, min_periods=length, adjust=False).mean()
    return adx, pos_di, neg_di


# ── Indicator functions ──────────────────────────────────────────


def add_rsi(df: pd.DataFrame, length: int = 14) -> pd.DataFrame:
    """RSI oscillator (0-100). Already stationary."""
    if _PANDAS_TA_AVAILABLE:
        df["rsi_14"] = ta.rsi(df["Close"], length=length)
    else:
        df["rsi_14"] = _rsi_fallback(df["Close"], length)
    return df


def add_macd(df: pd.DataFrame) -> pd.DataFrame:
    """MACD histogram normalized by closing price (percentage scale)."""
    if _PANDAS_TA_AVAILABLE:
        macd = ta.macd(df["Close"], fast=12, slow=26, signal=9)
        if macd is not None and not macd.empty:
            df["macd_hist_pct"] = macd.iloc[:, 2] / df["Close"] * 100
            df["macd_signal_diff"] = (macd.iloc[:, 0] - macd.iloc[:, 1]) / df["Close"] * 100
    else:
        ema_fast = _ema(df["Close"], 12)
        ema_slow = _ema(df["Close"], 26)
        macd_line = ema_fast - ema_slow
        signal_line = _ema(macd_line, 9)
        df["macd_hist_pct"] = (macd_line - signal_line) / df["Close"] * 100
        df["macd_signal_diff"] = (macd_line - signal_line) / df["Close"] * 100
    return df


def add_bollinger_bands(df: pd.DataFrame, length: int = 20, std: float = 2.0) -> pd.DataFrame:
    """Bollinger %B — position within bands (0 = lower, 1 = upper)."""
    if _PANDAS_TA_AVAILABLE:
        bb = ta.bbands(df["Close"], length=length, std=std)
        if bb is not None and not bb.empty:
            upper = bb.iloc[:, 2]
            lower = bb.iloc[:, 0]
            band_width = upper - lower
            df["bb_pct_b"] = np.where(band_width > 0, (df["Close"] - lower) / band_width, 0.5)
            df["bb_width_pct"] = band_width / df["Close"] * 100
    else:
        mean = df["Close"].rolling(length).mean()
        stddev = df["Close"].rolling(length).std()
        upper = mean + std * stddev
        lower = mean - std * stddev
        band_width = upper - lower
        df["bb_pct_b"] = np.where(band_width > 0, (df["Close"] - lower) / band_width, 0.5)
        df["bb_width_pct"] = band_width / df["Close"] * 100
    return df


def add_ema_positions(df: pd.DataFrame) -> pd.DataFrame:
    """Price position relative to EMA 20/50/200 (percentage distance)."""
    for period in [20, 50, 200]:
        if _PANDAS_TA_AVAILABLE:
            ema = ta.ema(df["Close"], length=period)
        else:
            ema = _ema(df["Close"], period)
        if ema is not None:
            df[f"ema_{period}_dist_pct"] = (df["Close"] - ema) / ema * 100
    return df


def add_atr(df: pd.DataFrame, length: int = 14) -> pd.DataFrame:
    """ATR normalized by close (percentage). Measures volatility."""
    if _PANDAS_TA_AVAILABLE:
        atr = ta.atr(df["High"], df["Low"], df["Close"], length=length)
    else:
        atr = _atr_fallback(df["High"], df["Low"], df["Close"], length)
    if atr is not None:
        df["atr_pct"] = atr / df["Close"] * 100
    return df


def add_adx(df: pd.DataFrame, length: int = 14) -> pd.DataFrame:
    """ADX trend strength (0-100). >25 = trending, <20 = sideways."""
    if _PANDAS_TA_AVAILABLE:
        adx = ta.adx(df["High"], df["Low"], df["Close"], length=length)
        if adx is not None and not adx.empty:
            df["adx_14"] = adx.iloc[:, 0]
            df["di_plus"] = adx.iloc[:, 1]
            df["di_minus"] = adx.iloc[:, 2]
            df["di_spread"] = df["di_plus"] - df["di_minus"]
    else:
        adx_s, pos_di, neg_di = _adx_fallback(df["High"], df["Low"], df["Close"], length)
        df["adx_14"] = adx_s
        df["di_plus"] = pos_di
        df["di_minus"] = neg_di
        df["di_spread"] = pos_di - neg_di
    return df


def add_volume_zscore(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Volume z-score over rolling window. Detects unusual volume (breakouts)."""
    vol_mean = df["Volume"].rolling(window=window).mean()
    vol_std = df["Volume"].rolling(window=window).std()
    df["volume_zscore"] = np.where(vol_std > 0, (df["Volume"] - vol_mean) / vol_std, 0)
    return df


def add_market_regime(df: pd.DataFrame) -> pd.DataFrame:
    """Classify market regime: +1 trending up, -1 trending down, 0 sideways."""
    if "adx_14" not in df.columns:
        df = add_adx(df)
    if "ema_50_dist_pct" not in df.columns:
        df = add_ema_positions(df)

    is_trending = df["adx_14"] > 25
    is_bullish = df["ema_50_dist_pct"] > 0

    df["regime"] = np.where(
        is_trending & is_bullish, 1,
        np.where(is_trending & ~is_bullish, -1, 0),
    )
    return df


def add_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Add return-based features for stationarity."""
    df["return_1d"] = df["Close"].pct_change(1) * 100
    df["return_5d"] = df["Close"].pct_change(5) * 100
    df["return_20d"] = df["Close"].pct_change(20) * 100
    df["intraday_range_pct"] = (df["High"] - df["Low"]) / df["Close"] * 100
    df["gap_pct"] = (df["Open"] - df["Close"].shift(1)) / df["Close"].shift(1) * 100
    return df


def compute_all_technicals(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all Pillar 1 technical features to an OHLCV DataFrame."""
    df = df.copy()
    df = add_returns(df)
    df = add_rsi(df)
    df = add_macd(df)
    df = add_bollinger_bands(df)
    df = add_ema_positions(df)
    df = add_atr(df)
    df = add_adx(df)
    df = add_volume_zscore(df)
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
