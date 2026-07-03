"""Conformalized Quantile Regression (CQR) — split-conformal calibration.

LightGBM quantile models are often miscalibrated on financial returns
(intervals too narrow in volatile regimes). CQR fixes this with a simple,
distribution-free correction: measure how far realized values fall outside
the predicted quantiles on a held-out calibration window, and widen (or
tighten) the predicted interval by the empirical quantile of those errors.

We use per-quantile *asymmetric* offsets rather than the classic symmetric
two-sided score because equity returns are skewed — this targets valid
marginal coverage of P10 and P90 separately, not just the 80% interval.

Reference: Romano, Patterson, Candès — "Conformalized Quantile Regression"
(NeurIPS 2019).
"""

import numpy as np

from model.metrics import interval_metrics

CONFORMAL_METHOD = "cqr_asymmetric_v1"


def _finite_sample_quantile(scores: np.ndarray, level: float) -> float:
    """Empirical quantile with the (n+1) finite-sample correction."""
    n = len(scores)
    if n == 0:
        return 0.0
    q = min(1.0, np.ceil((n + 1) * level) / n)
    return float(np.quantile(scores, q, method="higher"))


def compute_conformal_offsets(
    y_cal: np.ndarray,
    q10: np.ndarray,
    q50: np.ndarray,
    q90: np.ndarray,
) -> dict:
    """Compute per-quantile conformal offsets from a calibration window.

    Offsets (all in target units, i.e. % change):
        d10: subtracted from raw P10 (positive = widen downward)
        d50: added to raw P50 (median bias correction)
        d90: added to raw P90 (positive = widen upward)

    Both tail offsets use the 90% finite-sample empirical quantile of the
    one-sided conformity scores, giving ~90% marginal coverage per tail
    (= 80% central interval when combined).

    Args:
        y_cal: Realized targets on the calibration window (never fitted).
        q10/q50/q90: Raw model predictions on the same window.

    Returns:
        Dict with offsets, method tag, calibration size, and pre/post
        coverage of the 80% interval on the calibration window.
    """
    y_cal = np.asarray(y_cal, dtype=float)
    q10 = np.asarray(q10, dtype=float)
    q50 = np.asarray(q50, dtype=float)
    q90 = np.asarray(q90, dtype=float)

    d10 = _finite_sample_quantile(q10 - y_cal, 0.9)
    d90 = _finite_sample_quantile(y_cal - q90, 0.9)
    d50 = float(np.median(y_cal - q50))

    pre = interval_metrics(y_cal, q10, q90)
    post = interval_metrics(y_cal, q10 - d10, q90 + d90)

    return {
        "method": CONFORMAL_METHOD,
        "d10": round(d10, 4),
        "d50": round(d50, 4),
        "d90": round(d90, 4),
        "cal_size": int(len(y_cal)),
        "coverage_pre": pre["coverage_80"],
        "coverage_post": post["coverage_80"],
        "mean_width_pre": pre["mean_width"],
        "mean_width_post": post["mean_width"],
    }


def apply_conformal(
    p10: float,
    p50: float,
    p90: float,
    offsets: dict | None,
) -> tuple[float, float, float]:
    """Apply conformal offsets to raw quantile predictions.

    Old model bundles without offsets pass offsets=None → identity.
    """
    if not offsets:
        return p10, p50, p90
    return (
        p10 - float(offsets.get("d10", 0.0)),
        p50 + float(offsets.get("d50", 0.0)),
        p90 + float(offsets.get("d90", 0.0)),
    )
