"""Unit tests for purged walk-forward CV splits."""

import numpy as np
import pytest

from model.cv import purged_walk_forward_splits


class TestPurgedWalkForward:
    def test_no_overlap_and_gap_respected(self):
        for train_idx, val_idx in purged_walk_forward_splits(500, n_splits=4, min_train=252, gap=2):
            assert len(np.intersect1d(train_idx, val_idx)) == 0
            # Purge gap: last train row at least `gap` before first val row
            assert val_idx[0] - train_idx[-1] >= 2 + 1  # gap rows fully excluded

    def test_strictly_forward(self):
        prev_val_end = -1
        for train_idx, val_idx in purged_walk_forward_splits(500, n_splits=4):
            assert train_idx[-1] < val_idx[0], "training must precede validation"
            assert val_idx[0] > prev_val_end, "folds must move forward"
            prev_val_end = val_idx[-1]

    def test_expanding_window(self):
        sizes = [len(tr) for tr, _ in purged_walk_forward_splits(500, n_splits=4)]
        assert sizes == sorted(sizes)
        assert sizes[0] >= 252

    def test_covers_tail(self):
        splits = list(purged_walk_forward_splits(500, n_splits=4))
        assert splits[-1][1][-1] == 499

    def test_insufficient_data_raises(self):
        with pytest.raises(ValueError):
            list(purged_walk_forward_splits(100, n_splits=4, min_train=252))

    def test_min_train_enforced(self):
        for train_idx, _ in purged_walk_forward_splits(400, n_splits=3, min_train=200, gap=2):
            assert len(train_idx) >= 200
