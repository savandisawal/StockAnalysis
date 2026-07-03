"""Shared quantile-forecast metrics.

Single implementation used by training, cross-validation, backtesting,
and live-accuracy monitoring so every surface reports the same numbers.
"""

import numpy as np


def pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, alpha: float) -> float:
    """Mean pinball (quantile) loss for quantile level alpha.

    The proper scoring rule for quantile forecasts: penalizes under-
    prediction by alpha and over-prediction by (1 - alpha).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    diff = y_true - y_pred
    return float(np.mean(np.maximum(alpha * diff, (alpha - 1) * diff)))


def quantile_coverage(y_true: np.ndarray, y_pred: np.ndarray, alpha: float) -> float:
    """Empirical P(y <= prediction). Well calibrated when ≈ alpha."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(y_true <= y_pred))


def interval_metrics(y_true: np.ndarray, p10: np.ndarray, p90: np.ndarray) -> dict:
    """Coverage and width of the central 80% interval."""
    y_true = np.asarray(y_true, dtype=float)
    p10 = np.asarray(p10, dtype=float)
    p90 = np.asarray(p90, dtype=float)
    return {
        "coverage_80": round(float(np.mean((y_true >= p10) & (y_true <= p90))), 4),
        "mean_width": round(float(np.mean(p90 - p10)), 4),
    }


def crossing_rate(p10: np.ndarray, p50: np.ndarray, p90: np.ndarray) -> float:
    """Fraction of rows where the raw quantile predictions are disordered."""
    p10 = np.asarray(p10, dtype=float)
    p50 = np.asarray(p50, dtype=float)
    p90 = np.asarray(p90, dtype=float)
    return float(np.mean((p10 > p50) | (p50 > p90)))


def quantile_metrics_summary(
    y_true: np.ndarray,
    p10: np.ndarray,
    p50: np.ndarray,
    p90: np.ndarray,
) -> dict:
    """All standard metrics for a set of quantile predictions."""
    y_true = np.asarray(y_true, dtype=float)
    errors = y_true - np.asarray(p50, dtype=float)
    out = {
        "pinball_p10": round(pinball_loss(y_true, p10, 0.1), 4),
        "pinball_p50": round(pinball_loss(y_true, p50, 0.5), 4),
        "pinball_p90": round(pinball_loss(y_true, p90, 0.9), 4),
        "coverage_p10": round(quantile_coverage(y_true, p10, 0.1), 4),
        "coverage_p90": round(quantile_coverage(y_true, p90, 0.9), 4),
        "mae": round(float(np.mean(np.abs(errors))), 4),
        "rmse": round(float(np.sqrt(np.mean(errors**2))), 4),
        "direction_accuracy": round(float(np.mean(np.sign(y_true) == np.sign(p50))), 4),
        "crossing_rate": round(crossing_rate(p10, p50, p90), 4),
    }
    out.update(interval_metrics(y_true, p10, p90))
    return out
