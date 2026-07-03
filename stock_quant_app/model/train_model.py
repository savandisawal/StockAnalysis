"""LightGBM Quantile Regression — trains P10, P50, P90 models.

Three separate models predict the 10th, 50th, and 90th percentile of
next-day close % change.

Training flow (all chronological, no shuffling):
1. Reserve the last `cal_days` rows as a conformal calibration window —
   never seen by any fit.
2. Run purged walk-forward CV on the remaining rows to get honest
   out-of-sample pinball/coverage metrics and a robust n_estimators
   (median of per-fold early-stopped best iterations).
3. Fit the final three models on all pre-calibration rows.
4. Evaluate on the calibration window and compute CQR offsets that are
   applied at inference for statistically valid interval coverage.

All estimators are seeded and deterministic — retraining on the same data
produces byte-identical models and metrics.
"""

import lightgbm as lgb
import numpy as np
import pandas as pd

from features.feature_builder import ALL_FEATURES, build_features_for_training
from model.conformal import compute_conformal_offsets
from model.cv import purged_walk_forward_splits
from model.metrics import pinball_loss, quantile_coverage, quantile_metrics_summary
from model.model_registry import save_model_bundle
from utils.logger import logger

# Quantile levels for the three models
QUANTILES = {"p10": 0.1, "p50": 0.5, "p90": 0.9}

# Default LightGBM parameters — tuned for financial time series.
# deterministic + force_row_wise + random_state make every fit reproducible
# (deterministic requires force_row_wise; n_jobs=-1 stays safe with these).
DEFAULT_PARAMS = {
    "n_estimators": 500,
    "max_depth": 6,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_child_samples": 20,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "verbose": -1,
    "n_jobs": -1,
    "random_state": 42,
    "deterministic": True,
    "force_row_wise": True,
}

# Purge/embargo rows between any train window and the data that follows it
# (label at row t is realized on day t+1 = feature day of row t+1).
_GAP = 2


def _compute_feature_stats(X: np.ndarray, feature_names: list[str]) -> dict:
    """Training-distribution stats per feature, stored in the bundle for
    inference-time drift detection."""
    stats = {}
    for i, name in enumerate(feature_names):
        col = X[:, i]
        stats[name] = {
            "mean": round(float(np.mean(col)), 6),
            "std": round(float(np.std(col)), 6),
            "p01": round(float(np.percentile(col, 1)), 6),
            "p99": round(float(np.percentile(col, 99)), 6),
        }
    return stats


def _run_cv(
    X: np.ndarray,
    y: np.ndarray,
    model_params: dict,
    n_splits: int,
) -> tuple[dict, dict[str, int]]:
    """Purged walk-forward CV. Returns (cv_metrics, best_n_estimators per quantile)."""
    min_train = min(252, max(100, int(len(X) * 0.5)))
    splits = list(
        purged_walk_forward_splits(len(X), n_splits=n_splits, min_train=min_train, gap=_GAP)
    )

    fold_preds: dict[str, list[np.ndarray]] = {q: [] for q in QUANTILES}
    fold_actuals: list[np.ndarray] = []
    best_iters: dict[str, list[int]] = {q: [] for q in QUANTILES}

    for fold_no, (train_idx, val_idx) in enumerate(splits, 1):
        fold_actuals.append(y[val_idx])
        for name, alpha in QUANTILES.items():
            model = lgb.LGBMRegressor(objective="quantile", alpha=alpha, **model_params)
            model.fit(
                X[train_idx],
                y[train_idx],
                eval_set=[(X[val_idx], y[val_idx])],
                eval_metric="quantile",
                callbacks=[
                    lgb.early_stopping(stopping_rounds=50, verbose=False),
                    lgb.log_evaluation(period=0),
                ],
            )
            fold_preds[name].append(model.predict(X[val_idx]))
            best_iters[name].append(int(model.best_iteration_ or model_params["n_estimators"]))
        logger.debug(f"CV fold {fold_no}/{len(splits)}: train={len(train_idx)} val={len(val_idx)}")

    y_all = np.concatenate(fold_actuals)
    preds = {q: np.concatenate(p) for q, p in fold_preds.items()}

    cv_metrics = {
        "n_folds": len(splits),
        "val_samples": int(len(y_all)),
        "per_quantile": {
            name: {
                "pinball": round(pinball_loss(y_all, preds[name], alpha), 4),
                "coverage": round(quantile_coverage(y_all, preds[name], alpha), 4),
                "best_iterations": best_iters[name],
            }
            for name, alpha in QUANTILES.items()
        },
    }
    cv_metrics.update(
        {
            k: v
            for k, v in quantile_metrics_summary(
                y_all, preds["p10"], preds["p50"], preds["p90"]
            ).items()
            if k in ("coverage_80", "mean_width", "direction_accuracy", "crossing_rate")
        }
    )

    # Robust final n_estimators: median of early-stopped fold iterations
    best_n = {q: max(50, int(np.median(iters))) for q, iters in best_iters.items()}
    return cv_metrics, best_n


def train_quantile_models(
    ticker: str,
    years: int = 3,
    include_fundamentals: bool = True,
    include_macro: bool = True,
    params: dict | None = None,
    save: bool = True,
    cal_days: int = 120,
    cv_folds: int = 4,
) -> tuple[dict[str, lgb.LGBMRegressor], list[str], dict]:
    """Train P10/P50/P90 quantile regression models for a stock.

    Args:
        ticker: NSE ticker symbol.
        years: Years of training data.
        include_fundamentals: Include Pillar 2 features.
        include_macro: Include Pillar 3 features.
        params: Override LightGBM hyperparameters.
        save: Save trained models to disk.
        cal_days: Rows reserved (from the end) for conformal calibration.
        cv_folds: Purged walk-forward CV folds.

    Returns:
        Tuple of (models_dict, feature_names, train_metrics).
    """
    logger.info(f"Training quantile models for {ticker}")

    df = build_features_for_training(
        ticker,
        years=years,
        include_fundamentals=include_fundamentals,
        include_macro=include_macro,
    )

    if df.empty or len(df) < 100:
        raise ValueError(f"Insufficient training data for {ticker}: {len(df)} rows (need 100+)")

    feature_cols = [c for c in ALL_FEATURES if c in df.columns]
    X = df[feature_cols].values.astype(float)
    y = df["target"].values.astype(float)

    logger.info(f"Training data: {X.shape[0]} samples × {X.shape[1]} features")

    # ── Reserve calibration window (never fitted) ────────────
    cal_n = max(60, min(cal_days, len(X) // 3))
    if len(X) - cal_n - _GAP < 80:
        raise ValueError(f"Insufficient data for {ticker} after reserving {cal_n} calibration rows")
    X_fit, y_fit = X[: -(cal_n + _GAP)], y[: -(cal_n + _GAP)]
    X_cal, y_cal = X[-cal_n:], y[-cal_n:]

    model_params = {**DEFAULT_PARAMS, **(params or {})}

    # ── Purged walk-forward CV on the fit region ─────────────
    cv_metrics, best_n = _run_cv(X_fit, y_fit, model_params, n_splits=cv_folds)
    logger.info(
        f"CV: pinball p50={cv_metrics['per_quantile']['p50']['pinball']}, "
        f"coverage_80={cv_metrics['coverage_80']}, "
        f"direction={cv_metrics['direction_accuracy']}"
    )

    # ── Final fit on all pre-calibration rows ────────────────
    models: dict[str, lgb.LGBMRegressor] = {}
    for name, alpha in QUANTILES.items():
        final_params = {**model_params, "n_estimators": best_n[name]}
        model = lgb.LGBMRegressor(objective="quantile", alpha=alpha, **final_params)
        model.fit(X_fit, y_fit)
        models[name] = model

    # ── Calibration-window evaluation + CQR offsets ──────────
    cal_preds = {name: models[name].predict(X_cal) for name in QUANTILES}
    cal_metrics = quantile_metrics_summary(
        y_cal, cal_preds["p10"], cal_preds["p50"], cal_preds["p90"]
    )
    conformal = compute_conformal_offsets(
        y_cal, cal_preds["p10"], cal_preds["p50"], cal_preds["p90"]
    )
    logger.info(
        f"Calibration: coverage {conformal['coverage_pre']:.3f} → "
        f"{conformal['coverage_post']:.3f} (offsets d10={conformal['d10']}, "
        f"d90={conformal['d90']})"
    )

    feature_stats = _compute_feature_stats(X_fit, feature_cols)

    train_metrics = {
        "cv_metrics": cv_metrics,
        "cal_metrics": cal_metrics,
        "conformal": conformal,
        # Headline numbers (calibration window = most recent unseen data)
        "interval_coverage_80": cal_metrics["coverage_80"],
        "direction_accuracy": cal_metrics["direction_accuracy"],
        "train_samples": int(len(X_fit)),
        "val_samples": int(cal_n),
        "n_features": int(len(feature_cols)),
        "fundamentals_point_in_time": df.attrs.get("fundamentals_point_in_time"),
        "seed": model_params.get("random_state"),
    }

    if save:
        version = save_model_bundle(
            ticker=ticker,
            models=models,
            feature_names=feature_cols,
            metrics=train_metrics,
            conformal=conformal,
            feature_stats=feature_stats,
        )
        train_metrics["version"] = version

    return models, feature_cols, train_metrics


def get_feature_importance(
    models: dict[str, lgb.LGBMRegressor],
    feature_names: list[str],
    top_n: int = 15,
) -> pd.DataFrame:
    """Extract feature importance from the P50 (median) model.

    Returns DataFrame with columns [feature, importance] sorted descending.
    """
    model = models.get("p50")
    if model is None:
        return pd.DataFrame(columns=["feature", "importance"])

    importance = model.feature_importances_
    df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": importance,
        }
    ).sort_values("importance", ascending=False)

    return df.head(top_n).reset_index(drop=True)
