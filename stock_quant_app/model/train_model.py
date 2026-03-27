"""LightGBM Quantile Regression — trains P10, P50, P90 models.

Three separate models predict the 10th, 50th, and 90th percentile
of next-day close % change. This gives a calibrated prediction range
and natural confidence score (narrow range = high confidence).

Training uses walk-forward expanding window by default.
"""

import lightgbm as lgb
import numpy as np
import pandas as pd

from features.feature_builder import ALL_FEATURES, build_features_for_training
from model.model_registry import save_model_bundle
from utils.logger import logger

# Quantile levels for the three models
QUANTILES = {"p10": 0.1, "p50": 0.5, "p90": 0.9}

# Default LightGBM parameters — tuned for financial time series
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
}


def train_quantile_models(
    ticker: str,
    years: int = 3,
    include_fundamentals: bool = True,
    include_macro: bool = True,
    params: dict | None = None,
    save: bool = True,
) -> tuple[dict[str, lgb.LGBMRegressor], list[str], dict]:
    """Train P10/P50/P90 quantile regression models for a stock.

    Args:
        ticker: NSE ticker symbol.
        years: Years of training data.
        include_fundamentals: Include Pillar 2 features.
        include_macro: Include Pillar 3 features.
        params: Override LightGBM hyperparameters.
        save: Save trained models to disk.

    Returns:
        Tuple of (models_dict, feature_names, train_metrics).
    """
    logger.info(f"Training quantile models for {ticker}")

    # Build feature matrix
    df = build_features_for_training(
        ticker, years=years,
        include_fundamentals=include_fundamentals,
        include_macro=include_macro,
    )

    if df.empty or len(df) < 100:
        raise ValueError(f"Insufficient training data for {ticker}: {len(df)} rows (need 100+)")

    # Identify available feature columns
    feature_cols = [c for c in ALL_FEATURES if c in df.columns]
    X = df[feature_cols].values
    y = df["target"].values

    logger.info(f"Training data: {X.shape[0]} samples × {X.shape[1]} features")

    # Train/validation split — last 20% for validation
    split_idx = int(len(X) * 0.8)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    model_params = {**DEFAULT_PARAMS, **(params or {})}
    models: dict[str, lgb.LGBMRegressor] = {}
    val_metrics: dict[str, dict] = {}

    for name, alpha in QUANTILES.items():
        logger.info(f"Training {name} model (alpha={alpha})")

        model = lgb.LGBMRegressor(
            objective="quantile",
            alpha=alpha,
            **model_params,
        )

        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric="quantile",
            callbacks=[
                lgb.early_stopping(stopping_rounds=50, verbose=False),
                lgb.log_evaluation(period=0),
            ],
        )

        models[name] = model

        # Validation metrics
        y_pred = model.predict(X_val)
        mae = np.mean(np.abs(y_val - y_pred))
        coverage = np.mean(y_val <= y_pred) if alpha < 0.5 else np.mean(y_val >= y_pred)

        val_metrics[name] = {
            "mae": round(float(mae), 4),
            "coverage": round(float(coverage), 4),
            "best_iteration": model.best_iteration_,
        }
        logger.info(f"  {name}: MAE={mae:.4f}, coverage={coverage:.4f}")

    # Combined metrics
    p10_pred = models["p10"].predict(X_val)
    p90_pred = models["p90"].predict(X_val)
    interval_coverage = np.mean((y_val >= p10_pred) & (y_val <= p90_pred))

    # Direction accuracy using P50
    p50_pred = models["p50"].predict(X_val)
    direction_acc = np.mean(np.sign(y_val) == np.sign(p50_pred))

    train_metrics = {
        "quantile_metrics": val_metrics,
        "interval_coverage_80": round(float(interval_coverage), 4),
        "direction_accuracy": round(float(direction_acc), 4),
        "train_samples": int(len(X_train)),
        "val_samples": int(len(X_val)),
        "n_features": int(len(feature_cols)),
    }

    logger.info(
        f"Training complete: 80% interval coverage={interval_coverage:.3f}, "
        f"direction accuracy={direction_acc:.3f}"
    )

    # Save model bundle
    if save:
        version = save_model_bundle(
            ticker=ticker,
            models=models,
            feature_names=feature_cols,
            metrics=train_metrics,
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
    df = pd.DataFrame({
        "feature": feature_names,
        "importance": importance,
    }).sort_values("importance", ascending=False)

    return df.head(top_n).reset_index(drop=True)
