"""Purged walk-forward cross-validation splits for time-series models.

Standard K-fold shuffles time away and leaks future information into
training. These splits are strictly forward: each fold trains on an
expanding window and validates on the next contiguous block, with a
purge/embargo gap between them.

Why gap=2 for daily data with a 1-day-ahead target: the label at row t
is realized at close(t+1), which is the feature day of row t+1 — one
purge row removes that direct overlap; one extra embargo row guards
against serially correlated residuals. More is wasteful at this horizon.
"""

from collections.abc import Iterator

import numpy as np


def purged_walk_forward_splits(
    n_samples: int,
    n_splits: int = 4,
    min_train: int = 252,
    gap: int = 2,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield (train_idx, val_idx) pairs for expanding-window walk-forward CV.

    The validation region (everything after min_train) is divided into
    n_splits contiguous blocks. For block k, training uses all rows up to
    the block start minus the purge gap.

    Args:
        n_samples: Total number of chronologically ordered rows.
        n_splits: Number of validation folds.
        min_train: Minimum rows in the first training window.
        gap: Rows dropped between train end and validation start.

    Yields:
        (train_indices, val_indices) as int arrays. Skips folds whose
        training window would fall below min_train or whose validation
        block would be empty.
    """
    if n_samples <= min_train + gap + n_splits:
        raise ValueError(
            f"Not enough samples for CV: {n_samples} rows, need > {min_train + gap + n_splits}"
        )

    val_region_start = min_train + gap
    val_region = n_samples - val_region_start
    fold_size = val_region // n_splits

    for k in range(n_splits):
        val_start = val_region_start + k * fold_size
        val_end = val_start + fold_size if k < n_splits - 1 else n_samples
        train_end = val_start - gap
        if train_end < min_train or val_end <= val_start:
            continue
        yield np.arange(0, train_end), np.arange(val_start, val_end)
