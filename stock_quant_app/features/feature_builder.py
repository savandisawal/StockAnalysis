"""Feature Builder — combines all three pillars into a single feature matrix.

This is the central module that:
1. Fetches OHLCV data and computes Pillar 1 (technical) features
2. Joins point-in-time Pillar 2 (fundamental) features per date
3. Joins point-in-time Pillar 3 (macro + sentiment) features per date
4. Merges everything into one DataFrame ready for ML training/prediction
5. Enforces as_of_date and reporting lags to prevent look-ahead bias
6. Handles missing values with domain-appropriate neutral defaults

Training and prediction share one assembly path (_assemble_feature_frame),
so the served feature vector is constructed exactly like training rows.

The target variable (next-day close % change) is also computed here.
"""

from datetime import date

import pandas as pd

from data.fetch_ohlc import fetch_ohlc
from features.fundamentals import (
    FUNDAMENTAL_FEATURES,
    build_fundamental_history_features,
)
from features.macro_sentiment import (
    MACRO_SENTIMENT_FEATURES,
    build_macro_history_features,
    compute_macro_sentiment_features,
)
from features.technicals import TECHNICAL_FEATURES, compute_all_technicals
from utils.logger import logger
from utils.sectors import get_sector

# ── All feature column names ─────────────────────────────────────

ALL_FEATURES = TECHNICAL_FEATURES + FUNDAMENTAL_FEATURES + MACRO_SENTIMENT_FEATURES

# Neutral defaults per feature — used when a pillar fails or before a
# value first became publicly available.
# 0.0 is the neutral value for z-scores, % changes, and sentiment.
_FEATURE_DEFAULTS: dict[str, float] = {
    "pe_zscore": 0.0,
    "roe_percentile": 0.5,  # Median
    "de_percentile": 0.5,  # Median
    "eps_cagr_3y": 0.0,  # No growth
    "promoter_change": 0.0,  # No change QoQ
    "macro_mood": 0.0,  # Neutral
    "news_sentiment": 0.0,  # Neutral
    "sp500_change": 0.0,
    "nasdaq_change": 0.0,
    "brent_change": 0.0,
    "usdinr_change": 0.0,
    "vix_value": 0.0,  # Daily % change of India VIX, not the level
    "nifty_change": 0.0,
}


def _get_default(feature: str) -> float:
    """Return a domain-appropriate default value for a feature."""
    return _FEATURE_DEFAULTS.get(feature, 0.0)


def _assemble_feature_frame(
    ticker: str,
    years: int,
    as_of_date: date | None = None,
    include_fundamentals: bool = True,
    include_macro: bool = True,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Shared assembly: OHLCV + technicals + point-in-time pillar joins.

    Returns a DataFrame indexed by trading day with OHLCV columns, all
    feature columns, and df.attrs["fundamentals_point_in_time"] recording
    whether Pillar 2 used real historical data (None = pillar disabled).
    """
    df = fetch_ohlc(ticker, years=years, as_of_date=as_of_date)
    if df.empty:
        logger.error(f"No OHLCV data for {ticker}")
        return pd.DataFrame()

    df = compute_all_technicals(df)
    df.attrs["fundamentals_point_in_time"] = None

    # ── Pillar 2: point-in-time fundamentals ─────────────────
    if include_fundamentals:
        try:
            fund = build_fundamental_history_features(
                ticker, df.index, close=df["Close"], use_cache=use_cache
            )
            for col in FUNDAMENTAL_FEATURES:
                df[col] = fund[col]
            df.attrs["fundamentals_point_in_time"] = bool(fund.attrs.get("point_in_time", False))
        except Exception as e:
            logger.warning(f"Fundamental features failed for {ticker}, using defaults: {e}")
            df.attrs["fundamentals_point_in_time"] = False
            for col in FUNDAMENTAL_FEATURES:
                df[col] = _get_default(col)
    else:
        for col in FUNDAMENTAL_FEATURES:
            df[col] = _get_default(col)

    # ── Pillar 3: point-in-time macro + recorded sentiment ───
    if include_macro:
        try:
            clean_ticker = ticker.replace(".NS", "").replace(".BO", "")
            macro = build_macro_history_features(df.index, ticker=clean_ticker, use_cache=use_cache)
            for col in MACRO_SENTIMENT_FEATURES:
                df[col] = macro[col]
        except Exception as e:
            logger.warning(f"Macro features failed for {ticker}, using defaults: {e}")
            for col in MACRO_SENTIMENT_FEATURES:
                df[col] = _get_default(col)
    else:
        for col in MACRO_SENTIMENT_FEATURES:
            df[col] = _get_default(col)

    return df


def _fill_feature_gaps(df: pd.DataFrame) -> pd.DataFrame:
    """Forward-fill then apply per-feature neutral defaults to remaining NaN."""
    feature_cols = [c for c in ALL_FEATURES if c in df.columns]
    df[feature_cols] = df[feature_cols].ffill()
    for col in feature_cols:
        df[col] = df[col].fillna(_get_default(col))
    return df


def build_features_for_training(
    ticker: str,
    years: int = 3,
    as_of_date: date | None = None,
    include_fundamentals: bool = True,
    include_macro: bool = True,
) -> pd.DataFrame:
    """Build complete feature matrix for model training.

    Returns DataFrame where each row is a trading day with:
    - All pillar features as columns (point-in-time, leakage-free)
    - 'target' column = next-day close % change
    - DatetimeIndex

    Args:
        ticker: NSE ticker symbol.
        years: Years of history to fetch.
        as_of_date: Cutoff date for look-ahead protection.
        include_fundamentals: Whether to include Pillar 2 (slower, needs Screener.in).
        include_macro: Whether to include Pillar 3 macro features.

    Returns:
        DataFrame with features + target. Rows with NaN target (last row) are dropped.
    """
    logger.info(f"Building training features for {ticker}, {years}Y history")

    df = _assemble_feature_frame(
        ticker,
        years=years,
        as_of_date=as_of_date,
        include_fundamentals=include_fundamentals,
        include_macro=include_macro,
    )
    if df.empty:
        return pd.DataFrame()

    # Target — next-day close % change (shift(-1) = tomorrow's return
    # observed from today's row)
    df["target"] = df["Close"].pct_change(1).shift(-1) * 100

    # Drop rows where target is NaN (last row — no next day yet)
    df = df.dropna(subset=["target"])

    # Drop initial rows where indicators haven't warmed up (EMA 200 needs 200 days)
    warmup = 200
    if len(df) > warmup:
        df = df.iloc[warmup:]

    df = _fill_feature_gaps(df)

    feature_cols = [c for c in ALL_FEATURES if c in df.columns]
    logger.info(f"Feature matrix for {ticker}: {len(df)} rows × {len(feature_cols)} features")
    return df


def build_features_for_prediction(
    ticker: str,
    include_fundamentals: bool = True,
    include_macro: bool = True,
) -> dict[str, float]:
    """Build feature vector for the LATEST trading day (for live prediction).

    Uses the exact same assembly as training, then overlays the live macro
    snapshot and freshly scored news sentiment on the last row (the live
    snapshot carries the same information set as the lagged history: at
    prediction time the most recent completed US session is yesterday's).

    Returns a dict of {feature_name: value} for the most recent trading day.
    """
    logger.info(f"Building prediction features for {ticker}")

    df = _assemble_feature_frame(
        ticker,
        years=1,
        include_fundamentals=include_fundamentals,
        include_macro=include_macro,
    )
    if df.empty or len(df) < 200:
        logger.error(f"Insufficient data for {ticker} prediction features")
        return {}

    df = _fill_feature_gaps(df)
    latest = df.iloc[-1]

    features: dict[str, float] = {}
    for col in ALL_FEATURES:
        val = latest.get(col)
        features[col] = float(val) if pd.notna(val) else _get_default(col)

    # Overlay live macro + sentiment on the latest row
    if include_macro:
        try:
            sector = get_sector(ticker)
            clean_ticker = ticker.replace(".NS", "").replace(".BO", "")
            live = compute_macro_sentiment_features(stock=clean_ticker, sector=sector)
            for col, val in live.to_dict().items():
                if val is not None:
                    features[col] = float(val)
        except Exception as e:
            logger.warning(f"Live macro overlay failed for {ticker}, keeping history values: {e}")

    # Also include latest close and ATR for price range calculation
    features["_latest_close"] = float(latest["Close"])
    features["_latest_atr_pct"] = float(latest.get("atr_pct", 0))

    logger.info(f"Prediction features for {ticker}: {len(features)} values")
    return features
