"""Model registry — versioning, saving, and loading trained models.

Each trained model set (P10 + P50 + P90) is saved as a versioned bundle
with metadata: ticker, train date, feature list, and a content hash for
tracking which model made which prediction.
"""

import hashlib
import json
from datetime import datetime

import joblib

from app.config import settings
from utils.logger import logger

# Ensure model directory exists
MODEL_DIR = settings.model_dir
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def _compute_hash(data: bytes) -> str:
    """SHA256 hash truncated to 8 chars."""
    return hashlib.sha256(data).hexdigest()[:8]


def save_model_bundle(
    ticker: str,
    models: dict[str, object],
    feature_names: list[str],
    metrics: dict | None = None,
    conformal: dict | None = None,
    feature_stats: dict | None = None,
) -> str:
    """Save a set of quantile models (P10/P50/P90) as a versioned bundle.

    Args:
        ticker: Stock ticker this model was trained for.
        models: {"p10": model, "p50": model, "p90": model}
        feature_names: Ordered list of feature column names.
        metrics: Optional training/backtest metrics to store with the model.
        conformal: CQR offsets computed on the calibration window.
        feature_stats: Per-feature training distribution stats (drift checks).

    Returns:
        Model version string (e.g. "RELIANCE_20260326_a1b2c3d4")
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    clean_ticker = ticker.upper().replace(".NS", "").replace(".BO", "")

    # Create bundle directory
    bundle_name = f"{clean_ticker}_{timestamp}"
    bundle_dir = MODEL_DIR / bundle_name
    bundle_dir.mkdir(parents=True, exist_ok=True)

    # Save each quantile model
    for quantile_name, model in models.items():
        model_path = bundle_dir / f"model_{quantile_name}.joblib"
        joblib.dump(model, model_path)

    # Save metadata (schema_version 2 adds conformal + feature_stats)
    metadata = {
        "schema_version": 2,
        "ticker": clean_ticker,
        "timestamp": timestamp,
        "feature_names": feature_names,
        "quantiles": list(models.keys()),
        "metrics": metrics or {},
        "conformal": conformal,
        "feature_stats": feature_stats,
    }

    meta_path = bundle_dir / "metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2))

    # Compute version hash from model bytes
    hash_input = meta_path.read_bytes()
    for qname in sorted(models.keys()):
        hash_input += (bundle_dir / f"model_{qname}.joblib").read_bytes()
    version_hash = _compute_hash(hash_input)

    version = f"{clean_ticker}_{timestamp}_{version_hash}"
    metadata["version"] = version
    meta_path.write_text(json.dumps(metadata, indent=2))

    # Update latest symlink file
    latest_file = MODEL_DIR / f"{clean_ticker}_latest.txt"
    latest_file.write_text(bundle_name)

    logger.info(f"Saved model bundle: {version}")
    return version


def load_model_bundle(
    ticker: str,
    version: str | None = None,
) -> tuple[dict[str, object], list[str], dict] | None:
    """Load a model bundle for a ticker.

    Args:
        ticker: Stock ticker.
        version: Specific version to load. If None, loads latest.

    Returns:
        Tuple of (models_dict, feature_names, metadata) or None if not found.
    """
    clean_ticker = ticker.upper().replace(".NS", "").replace(".BO", "")

    if version:
        # Find bundle by version prefix
        candidates = list(MODEL_DIR.glob(f"{clean_ticker}_*"))
        bundle_dir = None
        for c in candidates:
            meta_path = c / "metadata.json"
            if meta_path.exists():
                meta = json.loads(meta_path.read_text())
                if meta.get("version", "").startswith(version) or c.name == version:
                    bundle_dir = c
                    break
    else:
        # Load latest
        latest_file = MODEL_DIR / f"{clean_ticker}_latest.txt"
        if not latest_file.exists():
            logger.warning(f"No trained model found for {clean_ticker}")
            return None
        bundle_name = latest_file.read_text().strip()
        bundle_dir = MODEL_DIR / bundle_name

    if bundle_dir is None or not bundle_dir.exists():
        logger.warning(f"Model bundle not found for {clean_ticker}")
        return None

    # Load metadata
    meta_path = bundle_dir / "metadata.json"
    metadata = json.loads(meta_path.read_text())

    # Load models
    models = {}
    for qname in metadata["quantiles"]:
        model_path = bundle_dir / f"model_{qname}.joblib"
        models[qname] = joblib.load(model_path)

    feature_names = metadata["feature_names"]

    logger.info(f"Loaded model: {metadata.get('version', bundle_dir.name)}")
    return models, feature_names, metadata


def list_models(ticker: str | None = None) -> list[dict]:
    """List all saved model bundles, optionally filtered by ticker."""
    results = []
    for meta_path in MODEL_DIR.glob("*/metadata.json"):
        metadata = json.loads(meta_path.read_text())
        if ticker:
            clean = ticker.upper().replace(".NS", "").replace(".BO", "")
            if metadata.get("ticker") != clean:
                continue
        results.append(metadata)

    results.sort(key=lambda m: m.get("timestamp", ""), reverse=True)
    return results
