"""Feature Builder — combines all three pillars into a single feature matrix.

This is the central module that:
1. Fetches OHLCV data and computes Pillar 1 (technical) features
2. Fetches and computes Pillar 2 (fundamental) features
3. Fetches and computes Pillar 3 (macro + sentiment) features
4. Merges everything into one DataFrame ready for ML training/prediction
5. Enforces as_of_date to prevent look-ahead bias
6. Handles missing values with appropriate strategies

The target variable (next-day close % change) is also computed here for training.
"""

from datetime import date

import pandas as pd

from data.fetch_ohlc import fetch_ohlc
from features.fundamentals import (
    FUNDAMENTAL_FEATURES,
    compute_fundamental_features,
)
from features.macro_sentiment import (
    MACRO_SENTIMENT_FEATURES,
    compute_macro_sentiment_features,
)
from features.technicals import TECHNICAL_FEATURES, compute_all_technicals
from utils.logger import logger
from utils.sectors import get_sector

# ── All feature column names ─────────────────────────────────────

ALL_FEATURES = TECHNICAL_FEATURES + FUNDAMENTAL_FEATURES + MACRO_SENTIMENT_FEATURES

# Neutral defaults per feature — used when a pillar fails.
# 0.0 is the neutral value for z-scores, % changes, and sentiment.
# Specific features get domain-appropriate defaults.
_FEATURE_DEFAULTS: dict[str, float] = {
    "pe_zscore": 0.0,
    "roe_percentile": 0.5,        # Median
    "de_percentile": 0.5,         # Median
    "eps_cagr_3y": 0.0,           # No growth
    "promoter_holding_pct": 50.0, # Neutral holding level
    "macro_mood": 0.0,            # Neutral
    "news_sentiment": 0.0,        # Neutral
    "sp500_change": 0.0,
    "nasdaq_change": 0.0,
    "crude_change": 0.0,
    "usdinr_change": 0.0,
    "vix_value": 15.0,            # Normal VIX level
    "nifty_change": 0.0,
}


def _get_default(feature: str) -> float:
    """Return a domain-appropriate default value for a feature."""
    return _FEATURE_DEFAULTS.get(feature, 0.0)


def build_features_for_training(
    ticker: str,
    years: int = 3,
    as_of_date: date | None = None,
    include_fundamentals: bool = True,
    include_macro: bool = True,
) -> pd.DataFrame:
    """Build complete feature matrix for model training.

    Returns DataFrame where each row is a trading day with:
    - All pillar features as columns
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

    # ── Step 1: OHLCV + Pillar 1 (Technical) ────────────────
    df = fetch_ohlc(ticker, years=years, as_of_date=as_of_date)
    if df.empty:
        logger.error(f"No OHLCV data for {ticker}")
        return pd.DataFrame()

    df = compute_all_technicals(df)

    # ── Step 2: Target variable — next-day close % change ────
    # This is what we're predicting
    df["target"] = df["Close"].pct_change(1).shift(-1) * 100
    # shift(-1) = tomorrow's return computed from today's perspective

    # ── Step 3: Pillar 2 (Fundamental) — static per stock ────
    if include_fundamentals:
        try:
            fund_features = compute_fundamental_features(ticker)
            fund_dict = fund_features.to_dict()
            for col, val in fund_dict.items():
                df[col] = val if val is not None else _get_default(col)
        except Exception as e:
            logger.warning(f"Fundamental features failed for {ticker}, using defaults: {e}")
            for col in FUNDAMENTAL_FEATURES:
                df[col] = _get_default(col)
    else:
        for col in FUNDAMENTAL_FEATURES:
            df[col] = _get_default(col)

    # ── Step 4: Pillar 3 (Macro + Sentiment) ─────────────────
    if include_macro:
        try:
            sector = get_sector(ticker)
            clean_ticker = ticker.replace(".NS", "").replace(".BO", "")
            macro_features = compute_macro_sentiment_features(
                stock=clean_ticker, sector=sector
            )
            macro_dict = macro_features.to_dict()
            for col, val in macro_dict.items():
                df[col] = val if val is not None else _get_default(col)
        except Exception as e:
            logger.warning(f"Macro features failed for {ticker}, using defaults: {e}")
            for col in MACRO_SENTIMENT_FEATURES:
                df[col] = _get_default(col)
    else:
        for col in MACRO_SENTIMENT_FEATURES:
            df[col] = _get_default(col)

    # ── Step 5: Clean up ─────────────────────────────────────
    # Drop rows where target is NaN (last row — no next day yet)
    df = df.dropna(subset=["target"])

    # Drop initial rows where indicators haven't warmed up (EMA 200 needs 200 days)
    warmup = 200
    if len(df) > warmup:
        df = df.iloc[warmup:]

    # Handle remaining NaN in features — forward fill then zero
    feature_cols = [c for c in ALL_FEATURES if c in df.columns]
    df[feature_cols] = df[feature_cols].ffill().fillna(0)

    logger.info(
        f"Feature matrix for {ticker}: {len(df)} rows × {len(feature_cols)} features"
    )
    return df


def build_features_for_prediction(
    ticker: str,
    include_fundamentals: bool = True,
    include_macro: bool = True,
) -> dict[str, float]:
    """Build feature vector for the LATEST trading day (for live prediction).

    Returns a dict of {feature_name: value} for the most recent trading day.
    This is what gets fed into the trained model for inference.
    """
    logger.info(f"Building prediction features for {ticker}")

    # Fetch recent data (need enough for indicator warmup)
    df = fetch_ohlc(ticker, years=1)
    if df.empty or len(df) < 200:
        logger.error(f"Insufficient data for {ticker} prediction features")
        return {}

    df = compute_all_technicals(df)

    # Get latest row's technical features
    latest = df.iloc[-1]
    features: dict[str, float] = {}
    for col in TECHNICAL_FEATURES:
        val = latest.get(col)
        features[col] = float(val) if pd.notna(val) else 0.0

    # Fundamentals
    if include_fundamentals:
        try:
            fund = compute_fundamental_features(ticker)
            fund_dict = fund.to_dict()
            features.update({
                k: v if v is not None else _get_default(k)
                for k, v in fund_dict.items()
            })
        except Exception as e:
            logger.warning(f"Fundamental features failed, using defaults: {e}")
            features.update({col: _get_default(col) for col in FUNDAMENTAL_FEATURES})
    else:
        features.update({col: _get_default(col) for col in FUNDAMENTAL_FEATURES})

    # Macro + Sentiment
    if include_macro:
        try:
            sector = get_sector(ticker)
            clean_ticker = ticker.replace(".NS", "").replace(".BO", "")
            macro = compute_macro_sentiment_features(stock=clean_ticker, sector=sector)
            macro_dict = macro.to_dict()
            features.update({
                k: v if v is not None else _get_default(k)
                for k, v in macro_dict.items()
            })
        except Exception as e:
            logger.warning(f"Macro features failed, using defaults: {e}")
            features.update({col: _get_default(col) for col in MACRO_SENTIMENT_FEATURES})
    else:
        features.update({col: _get_default(col) for col in MACRO_SENTIMENT_FEATURES})

    # Also include latest close and ATR for price range calculation
    features["_latest_close"] = float(latest["Close"])
    features["_latest_atr_pct"] = float(latest.get("atr_pct", 0))

    logger.info(f"Prediction features for {ticker}: {len(features)} values")
    return features
