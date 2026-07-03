"""Per-prediction SHAP explanations.

TreeExplainer on LightGBM is exact and fast (milliseconds for one row),
so every served prediction can carry its own explanation. The P50 (median)
model is explained — it drives the directional call.

shap is imported lazily so environments without it can still predict;
explanation generation must never block a prediction.
"""

import numpy as np

from utils.logger import logger


def compute_shap_explanation(
    model,
    X_row: np.ndarray,
    feature_names: list[str],
    top_k: int = 8,
) -> dict | None:
    """Explain a single prediction with SHAP values.

    Args:
        model: Trained LightGBM model (the P50 quantile model).
        X_row: Feature vector, shape (1, n_features).
        feature_names: Names aligned to X_row columns.
        top_k: Number of top contributors to return.

    Returns:
        {"quantile": "p50", "base_value": float,
         "top_features": [{"feature", "value", "shap"}, ...]}  sorted by
        |shap| descending, or None on any failure.
    """
    try:
        import shap

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_row)
        row_shap = np.asarray(shap_values).reshape(-1)
        base_value = float(np.asarray(explainer.expected_value).reshape(-1)[0])

        order = np.argsort(np.abs(row_shap))[::-1][:top_k]
        top_features = [
            {
                "feature": feature_names[i],
                "value": round(float(X_row[0, i]), 4),
                "shap": round(float(row_shap[i]), 4),
            }
            for i in order
        ]
        return {
            "quantile": "p50",
            "base_value": round(base_value, 4),
            "top_features": top_features,
        }
    except Exception as e:
        logger.warning(f"SHAP explanation failed: {e}")
        return None
