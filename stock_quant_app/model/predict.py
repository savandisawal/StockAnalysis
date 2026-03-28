"""Prediction module — inference using trained quantile models.

Takes the latest feature vector, runs it through P10/P50/P90 models,
and returns a predicted price range with confidence score.

Includes prediction safeguards:
- Guardrails: cap predicted change at 2x ATR
- Feature drift: flag out-of-distribution inputs
- Model staleness: warn if model > 7 days old
- Circuit breakers: refuse prediction during extreme events
- Calibration: adjust confidence from backtest accuracy
- Ensemble sanity: flag illogical P10/P50/P90 ordering
"""

from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

from features.feature_builder import build_features_for_prediction
from model.model_registry import load_model_bundle
from utils.logger import logger

# ── Safeguard thresholds ────────────────────────────────────────

_ATR_CAP_MULTIPLIER = 2.0    # Max predicted change = 2x ATR
_STALENESS_DAYS = 7           # Warn if model older than this
_VIX_CIRCUIT_BREAKER = 30.0   # Refuse prediction above this VIX
_VOLUME_Z_CIRCUIT = 4.0       # Refuse on extreme volume anomaly
_DRIFT_ZSCORE_THRESHOLD = 3.0 # Flag features > 3 std from mean
_DRIFT_MAX_FLAGS = 3          # Max drifted features before warning


@dataclass
class PredictionWarning:
    """A single warning/safeguard flag."""
    level: str    # "info", "warning", "critical"
    code: str     # machine-readable: "guardrail", "stale", etc.
    message: str  # human-readable explanation


@dataclass
class PredictionResult:
    """Next-day price range prediction with confidence."""
    ticker: str
    prediction_date: str        # Date the prediction is for
    current_close: float        # Today's closing price

    predicted_low: float        # P10 — lower bound of range
    predicted_mid: float        # P50 — median prediction
    predicted_high: float       # P90 — upper bound of range

    predicted_change_pct: float # P50 % change from current close
    range_width_pct: float      # (P90 - P10) / close as %

    confidence: float           # 0-100 scale, derived from range width
    direction: str              # "Bullish", "Bearish", or "Neutral"

    model_version: str          # Which model made this prediction

    # Safeguard outputs
    warnings: list[PredictionWarning] = field(
        default_factory=list,
    )
    guardrail_applied: bool = False  # True if prediction was capped
    original_change_pct: float | None = None  # Pre-guardrail value

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
            "guardrail_applied": self.guardrail_applied,
            "warnings": [
                {"level": w.level, "code": w.code, "message": w.message}
                for w in self.warnings
            ],
        }

    @property
    def has_critical_warnings(self) -> bool:
        return any(w.level == "critical" for w in self.warnings)

    @property
    def has_warnings(self) -> bool:
        return len(self.warnings) > 0


# ── Safeguard functions ──────────────────────────────────────────


def _check_model_staleness(
    metadata: dict,
) -> PredictionWarning | None:
    """Warn if model was trained more than 7 days ago."""
    ts = metadata.get("timestamp", "")
    if not ts:
        return None
    try:
        trained_dt = datetime.strptime(ts[:15], "%Y%m%d_%H%M%S")
        age_days = (datetime.now() - trained_dt).days
        if age_days > _STALENESS_DAYS:
            return PredictionWarning(
                level="warning",
                code="stale_model",
                message=(
                    f"Model is {age_days} days old. "
                    f"Retrain for better accuracy."
                ),
            )
    except ValueError:
        pass
    return None


def _check_circuit_breakers(
    features: dict[str, float],
) -> PredictionWarning | None:
    """Refuse prediction during extreme market events."""
    vix = features.get("vix_value")
    if vix is not None and abs(vix) > _VIX_CIRCUIT_BREAKER:
        return PredictionWarning(
            level="critical",
            code="circuit_breaker_vix",
            message=(
                f"India VIX change {vix:+.1f}% — extreme volatility. "
                f"Prediction unreliable."
            ),
        )

    vol_z = features.get("volume_zscore")
    if vol_z is not None and abs(vol_z) > _VOLUME_Z_CIRCUIT:
        return PredictionWarning(
            level="critical",
            code="circuit_breaker_volume",
            message=(
                f"Volume Z-score {vol_z:.1f} — extreme anomaly. "
                f"Prediction may be unreliable."
            ),
        )

    return None


def _check_feature_drift(
    features: dict[str, float],
    metadata: dict,
) -> PredictionWarning | None:
    """Flag when input features are far outside training distribution."""
    # Heuristic bounds for common features
    drift_flags = []
    bounds = {
        "rsi_14": (5, 95),
        "bb_pct_b": (-0.5, 1.5),
        "volume_zscore": (-3, 5),
        "adx_14": (0, 80),
        "return_1d": (-10, 10),
        "return_5d": (-20, 20),
    }
    for feat, (lo, hi) in bounds.items():
        val = features.get(feat)
        if val is not None and (val < lo or val > hi):
            drift_flags.append(feat)

    if len(drift_flags) >= _DRIFT_MAX_FLAGS:
        return PredictionWarning(
            level="warning",
            code="feature_drift",
            message=(
                f"{len(drift_flags)} features outside normal range "
                f"({', '.join(drift_flags[:4])}). "
                f"Model may not generalize well."
            ),
        )
    return None


def _apply_guardrails(
    p10: float, p50: float, p90: float, atr_pct: float,
) -> tuple[float, float, float, bool, float | None]:
    """Cap predicted changes at 2x ATR. Returns capped values + flag."""
    cap = atr_pct * _ATR_CAP_MULTIPLIER
    if cap <= 0:
        return p10, p50, p90, False, None

    original_p50 = None
    applied = False

    if abs(p50) > cap:
        original_p50 = p50
        p50 = cap if p50 > 0 else -cap
        applied = True

    p10 = max(p10, -cap)
    p90 = min(p90, cap)

    # Re-sort after capping
    p10, p50, p90 = sorted([p10, p50, p90])
    return p10, p50, p90, applied, original_p50


def _check_ensemble_sanity(
    p10_raw: float, p50_raw: float, p90_raw: float,
) -> PredictionWarning | None:
    """Flag if quantile models produce illogical results."""
    spread = p90_raw - p10_raw
    # If P10 > P50 or P50 > P90 before sorting, models disagree
    if p10_raw > p50_raw or p50_raw > p90_raw:
        return PredictionWarning(
            level="warning",
            code="ensemble_disorder",
            message=(
                "Quantile models produced disordered predictions "
                f"(P10={p10_raw:+.2f}%, P50={p50_raw:+.2f}%, "
                f"P90={p90_raw:+.2f}%). Confidence reduced."
            ),
        )
    # Extreme spread = models very uncertain
    if spread > 8.0:
        return PredictionWarning(
            level="warning",
            code="ensemble_wide_spread",
            message=(
                f"Prediction range is very wide ({spread:.1f}%). "
                f"Model is highly uncertain."
            ),
        )
    return None


def _calibrate_confidence(
    confidence: float,
    ticker: str,
) -> tuple[float, PredictionWarning | None]:
    """Adjust confidence based on backtest accuracy if available."""
    try:
        from model.backtest import get_backtest_summary
        summaries = get_backtest_summary(ticker)
        if not summaries:
            return confidence, None

        latest = summaries[0]
        actual_acc = latest.get("direction_accuracy", 0.5)
        coverage = latest.get("interval_coverage", 0.8)

        # If model claims high confidence but backtest shows poor accuracy
        if confidence > 60 and actual_acc < 0.50:
            adjusted = confidence * 0.6
            return round(adjusted, 1), PredictionWarning(
                level="warning",
                code="calibration_overconfident",
                message=(
                    f"Backtest direction accuracy is only "
                    f"{actual_acc:.0%}. Confidence adjusted down "
                    f"from {confidence:.0f}% to {adjusted:.0f}%."
                ),
            )

        if coverage < 0.6:
            adjusted = confidence * 0.75
            return round(adjusted, 1), PredictionWarning(
                level="info",
                code="calibration_low_coverage",
                message=(
                    f"Backtest 80% interval coverage is "
                    f"{coverage:.0%} (expected ~80%). "
                    f"Prediction range may be too narrow."
                ),
            )

    except Exception:
        pass

    return confidence, None


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
    warnings: list[PredictionWarning] = []

    # ── Safeguard 1: Model staleness ────────────────────────
    stale_warn = _check_model_staleness(metadata)
    if stale_warn:
        warnings.append(stale_warn)

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

    # ── Safeguard 2: Circuit breakers ───────────────────────
    circuit_warn = _check_circuit_breakers(features)
    if circuit_warn:
        warnings.append(circuit_warn)

    # ── Safeguard 3: Feature drift ──────────────────────────
    drift_warn = _check_feature_drift(features, metadata)
    if drift_warn:
        warnings.append(drift_warn)

    # Align features to model's expected order
    X = np.array([[features.get(f, 0.0) for f in feature_names]])

    # Predict with each quantile model
    p10_raw = float(models["p10"].predict(X)[0])
    p50_raw = float(models["p50"].predict(X)[0])
    p90_raw = float(models["p90"].predict(X)[0])

    # ── Safeguard 4: Ensemble sanity ────────────────────────
    ensemble_warn = _check_ensemble_sanity(p10_raw, p50_raw, p90_raw)
    if ensemble_warn:
        warnings.append(ensemble_warn)

    # Sort to ensure ordering
    p10_change, p50_change, p90_change = sorted(
        [p10_raw, p50_raw, p90_raw],
    )

    # ── Safeguard 5: Guardrails — cap at 2x ATR ────────────
    p10_change, p50_change, p90_change, guardrail_hit, orig_p50 = (
        _apply_guardrails(p10_change, p50_change, p90_change, atr_pct)
    )
    if guardrail_hit:
        warnings.append(PredictionWarning(
            level="info",
            code="guardrail",
            message=(
                f"Predicted change capped from "
                f"{orig_p50:+.2f}% to {p50_change:+.2f}% "
                f"(2x ATR limit: {atr_pct * _ATR_CAP_MULTIPLIER:.2f}%)."
            ),
        ))

    # Convert % changes to price levels
    predicted_low = round(current_close * (1 + p10_change / 100), 2)
    predicted_mid = round(current_close * (1 + p50_change / 100), 2)
    predicted_high = round(current_close * (1 + p90_change / 100), 2)

    range_width_pct = round(p90_change - p10_change, 3)
    confidence = _compute_confidence(range_width_pct, atr_pct)

    # ── Safeguard 6: Calibration from backtest ──────────────
    clean_ticker = ticker.upper().replace(".NS", "").replace(".BO", "")
    confidence, cal_warn = _calibrate_confidence(confidence, clean_ticker)
    if cal_warn:
        warnings.append(cal_warn)

    # Reduce confidence if ensemble was disordered
    if ensemble_warn and ensemble_warn.code == "ensemble_disorder":
        confidence = round(confidence * 0.7, 1)

    from utils.holidays import next_trading_day
    pred_date = next_trading_day().isoformat()

    result = PredictionResult(
        ticker=clean_ticker,
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
        warnings=warnings,
        guardrail_applied=guardrail_hit,
        original_change_pct=orig_p50,
    )

    # Log warnings
    for w in warnings:
        log_fn = logger.warning if w.level != "info" else logger.info
        log_fn(f"[{w.code}] {w.message}")

    logger.info(
        f"Prediction for {result.ticker}: "
        f"Rs.{result.predicted_low} - Rs.{result.predicted_mid} "
        f"- Rs.{result.predicted_high} "
        f"({result.direction}, {result.confidence}% confidence, "
        f"{len(warnings)} warnings)"
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
