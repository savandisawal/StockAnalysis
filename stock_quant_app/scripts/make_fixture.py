"""Regenerate the deterministic OHLCV test fixture.

Usage: python -m scripts.make_fixture

Produces tests/fixtures/sample_ohlcv.csv — 520 trading days of synthetic
GBM prices with two volatility regimes (calm → stressed), lognormal
volume. Seeded, so the file is byte-identical across runs.

520 rows = 200 indicator warmup + ≥250 train + 60 calibration + margin.
"""

from pathlib import Path

import numpy as np
import pandas as pd

FIXTURE_PATH = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "sample_ohlcv.csv"

N_DAYS = 520
SEED = 42


def make_ohlcv(n_days: int = N_DAYS, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # Two volatility regimes: calm first 60%, stressed last 40%
    n_calm = int(n_days * 0.6)
    daily_vol = np.concatenate(
        [
            np.full(n_calm, 0.012),
            np.full(n_days - n_calm, 0.022),
        ]
    )
    drift = 0.0004

    log_returns = rng.normal(drift, daily_vol)
    close = 1000.0 * np.exp(np.cumsum(log_returns))

    # Intraday structure around the close path
    intraday = np.abs(rng.normal(0, daily_vol)) + 0.003
    open_ = close * (1 + rng.normal(0, daily_vol * 0.5))
    high = np.maximum(open_, close) * (1 + intraday)
    low = np.minimum(open_, close) * (1 - intraday)

    volume = np.round(np.exp(rng.normal(13.0, 0.4, n_days))).astype(int)

    dates = pd.bdate_range(end="2025-12-31", periods=n_days)
    return pd.DataFrame(
        {
            "Open": np.round(open_, 2),
            "High": np.round(high, 2),
            "Low": np.round(low, 2),
            "Close": np.round(close, 2),
            "Volume": volume,
        },
        index=pd.DatetimeIndex(dates, name="Date"),
    )


def main() -> None:
    df = make_ohlcv()
    FIXTURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(FIXTURE_PATH)
    print(f"Wrote {len(df)} rows to {FIXTURE_PATH}")


if __name__ == "__main__":
    main()
