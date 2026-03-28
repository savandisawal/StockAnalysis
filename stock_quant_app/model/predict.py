"""Prediction module — inference using trained quantile models.

Takes the latest feature vector, runs it through P10/P50/P90 models,
and returns a predicted price range with confidence score.
"""

from dataclasses import dataclass

import numpy as np

from features.feature_builder import build_features_for_prediction
from model.model_registry import load_model_bundle
from utils.logger import logger


@dataclass
class PredictionResult:
    """Next-day price range prediction with confidence."""
    ticker: str
    prediction_date: str        # Date the prediction is for (next trading day)
    current_close: float        # Today's closing price

    predicted_low: float        # P10 — lower bound of range
    predicted_mid: float        # P50 — median prediction
    predicted_high: float       # P90 — upper bound of range

    predicted_change_pct: float # P50 % change from current close
    range_width_pct: float      # (P90 - P10) / close as %

    confidence: float           # 0-100 scale, derived from range width
    direction: str              # "Bullish", "Bearish", or "Neutral"

    model_version: str          # Which model made this prediction

    def to_dict(self) -> dict:
        return {
            "ticker": self.ticker,
            "prediction_date": self.prediction_date,
            "current_close": self.current_close,
            "predicted_low": self.predicted_low,
            "predicted_mid": self.predicted_mid,
            "predicted_high": self.predicted_high,
            "predicted_change_pct": self.predicted_change_pct,
            "range_width_pct": self.range_width_pct,
            "confidence": self.confidence,
            "direction": self.direction,
            "model_version": self.model_version,
        }


def _compute_confidence(range_width_pct: float, atr_pct: float) -> float:
    """Derive confidence from prediction interval width.

    Narrow range relative to typical volatility (ATR) = high confidence.
    Returns 0-100 scale.

    Logic:
    - If predicted range < ATR → very confident (model sees clear direction)
    - If predicted range ~ ATR → moderate confidence
    - If predicted range >> ATR → low confidence (uncertain)
    """
    if atr_pct <= 0:
        return 50.0

    ratio = range_width_pct / atr_pct

    # Map ratio to confidence: ratio=0.5 → 90%, ratio=1.0 → 65%, ratio=2.0 → 30%
    confidence = 100 * np.exp(-0.8 * ratio)
    return round(float(np.clip(confidence, 5, 95)), 1)


def _classify_direction(change_pct: float) -> str:
    """Classify predicted direction based on median change."""
    if change_pct > 0.3:
        return "Bullish"
    elif change_pct < -0.3:
        return "Bearish"
    else:
        return "Neutral"


def predict_next_day(
    ticker: str,
    model_version: str | None = None,
    include_fundamentals: bool = True,
    include_macro: bool = True,
) -> PredictionResult | None:
    """Generate next-day price range prediction for a stock.

    Args:
        ticker: NSE ticker symbol.
        model_version: Specific model version. None = latest.
        include_fundamentals: Include Pillar 2 features.
        include_macro: Include Pillar 3 features.

    Returns:
        PredictionResult or None if model not found / prediction fails.
    """
    # Load model
    bundle = load_model_bundle(ticker, version=model_version)
    if bundle is None:
        logger.error(f"No trained model for {ticker}. Train first.")
        return None

    models, feature_names, metadata = bundle
    version = metadata.get("version", "unknown")

    # Build feature vector
    features = build_features_for_prediction(
        ticker,
        include_fundamentals=include_fundamentals,
        include_macro=include_macro,
    )

    if not features:
        logger.error(f"Failed to build features for {ticker}")
        return None

    current_close = features.pop("_latest_close")
    atr_pct = features.pop("_latest_atr_pct", 2.0)

    # Align features to model's expected order
    X = np.array([[features.get(f, 0.0) for f in feature_names]])

    # Predict with each quantile model
    p10_change = float(models["p10"].predict(X)[0])
    p50_change = float(models["p50"].predict(X)[0])
    p90_change = float(models["p90"].predict(X)[0])

    # Ensure ordering: P10 <= P50 <= P90
    p10_change, p50_change, p90_change = sorted([p10_change, p50_change, p90_change])

    # Convert % changes to price levels
    predicted_low = round(current_close * (1 + p10_change / 100), 2)
    predicted_mid = round(current_close * (1 + p50_change / 100), 2)
    predicted_high = round(current_close * (1 + p90_change / 100), 2)

    range_width_pct = round(p90_change - p10_change, 3)
    confidence = _compute_confidence(range_width_pct, atr_pct)

    from utils.holidays import next_trading_day
    pred_date = next_trading_day().isoformat()

    result = PredictionResult(
        ticker=ticker.upper().replace(".NS", "").replace(".BO", ""),
        prediction_date=pred_date,
        current_close=round(current_close, 2),
        predicted_low=predicted_low,
        predicted_mid=predicted_mid,
        predicted_high=predicted_high,
        predicted_change_pct=round(p50_change, 3),
        range_width_pct=range_width_pct,
        confidence=confidence,
        direction=_classify_direction(p50_change),
        model_version=version,
    )

    logger.info(
        f"Prediction for {result.ticker}: "
        f"Rs.{result.predicted_low} - Rs.{result.predicted_mid} - Rs.{result.predicted_high} "
        f"({result.direction}, {result.confidence}% confidence)"
    )

    return result


def _sig(name: str, val: str, interp: str, sent: str) -> dict:
    return {
        "signal": name,
        "value": val,
        "interpretation": interp,
        "sentiment": sent,
    }


def get_signal_summary(ticker: str) -> list[dict]:
    """Generate human-readable signal summary from latest indicators.

    Returns list of dicts: signal, value, interpretation, sentiment.
    sentiment is one of "Bullish", "Bearish", "Neutral".
    """
    import pandas as pd

    from data.fetch_ohlc import fetch_ohlc
    from features.technicals import compute_all_technicals

    df = fetch_ohlc(ticker, years=1)
    if df.empty or len(df) < 50:
        return []

    df = compute_all_technicals(df)
    latest = df.iloc[-1]
    signals: list[dict] = []

    # RSI
    rsi = latest.get("rsi_14")
    if pd.notna(rsi):
        if rsi > 70:
            signals.append(_sig(
                "RSI (14)", f"{rsi:.1f}",
                "Overbought — potential reversal down", "Bearish",
            ))
        elif rsi < 30:
            signals.append(_sig(
                "RSI (14)", f"{rsi:.1f}",
                "Oversold — potential reversal up", "Bullish",
            ))
        else:
            signals.append(_sig(
                "RSI (14)", f"{rsi:.1f}",
                "Neutral zone", "Neutral",
            ))

    # MACD
    macd = latest.get("macd_hist_pct")
    if pd.notna(macd):
        sent = "Bullish" if macd > 0 else "Bearish"
        label = "Bullish" if macd > 0 else "Bearish"
        signals.append(_sig(
            "MACD Histogram", f"{macd:+.3f}%",
            f"{label} momentum", sent,
        ))

    # Bollinger %B
    bb = latest.get("bb_pct_b")
    if pd.notna(bb):
        if bb > 0.8:
            signals.append(_sig(
                "Bollinger %B", f"{bb:.3f}",
                "Near upper band — resistance", "Bearish",
            ))
        elif bb < 0.2:
            signals.append(_sig(
                "Bollinger %B", f"{bb:.3f}",
                "Near lower band — support", "Bullish",
            ))
        else:
            signals.append(_sig(
                "Bollinger %B", f"{bb:.3f}",
                "Mid-range", "Neutral",
            ))

    # EMA positioning
    ema_cols = [
        ("EMA 20", "ema_20_dist_pct"),
        ("EMA 50", "ema_50_dist_pct"),
        ("EMA 200", "ema_200_dist_pct"),
    ]
    for ema_name, col in ema_cols:
        val = latest.get(col)
        if pd.notna(val):
            sent = "Bullish" if val > 0 else "Bearish"
            pos = "Above" if val > 0 else "Below"
            signals.append(_sig(
                ema_name, f"{val:+.2f}%",
                f"{pos} {ema_name}", sent,
            ))

    # ADX + Regime
    adx = latest.get("adx_14")
    regime = latest.get("regime")
    if pd.notna(adx) and pd.notna(regime):
        regime_int = int(regime)
        label_map = {
            1: "Trending Up", -1: "Trending Down", 0: "Sideways",
        }
        sent_map = {
            1: "Bullish", -1: "Bearish", 0: "Neutral",
        }
        signals.append(_sig(
            "ADX / Regime", f"{adx:.1f}",
            f"Regime: {label_map.get(regime_int, 'Unknown')}",
            sent_map.get(regime_int, "Neutral"),
        ))

    # Volume Z-score
    vol_z = latest.get("volume_zscore")
    if pd.notna(vol_z):
        if abs(vol_z) > 2:
            interp = "Unusual volume — breakout signal"
        else:
            interp = "Normal volume"
        signals.append(_sig(
            "Volume Z-Score", f"{vol_z:.2f}", interp, "Neutral",
        ))

    # ATR volatility
    atr = latest.get("atr_pct")
    if pd.notna(atr):
        if atr > 3:
            vol_label = "High"
        elif atr < 1:
            vol_label = "Low"
        else:
            vol_label = "Normal"
        signals.append(_sig(
            "ATR %", f"{atr:.2f}%",
            f"{vol_label} volatility", "Neutral",
        ))

    return signals


def predict_batch(
    tickers: list[str],
    include_fundamentals: bool = True,
    include_macro: bool = True,
) -> list[PredictionResult]:
    """Predict next-day range for multiple stocks."""
    results = []
    for ticker in tickers:
        pred = predict_next_day(
            ticker,
            include_fundamentals=include_fundamentals,
            include_macro=include_macro,
        )
        if pred:
            results.append(pred)
    return results
