"""Synthetic tests for the Exp11 error-budget diagnostics."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from scripts.exp11_error_budget import aggregate_miou, _partition


def test_perfect_prediction_has_zero_miou_error_ceiling():
    confusion = np.diag([4, 3, 2])
    assert aggregate_miou(confusion) == 100.0


def test_aggregate_miou_matches_hand_calculation():
    confusion = np.array([[8, 2, 0], [1, 6, 3], [0, 2, 8]])
    expected = np.mean([80.0 / 91.0, 6.0 / 13.0, 8.0 / 13.0]) * 100.0
    assert abs(aggregate_miou(confusion) - expected) < 1e-10


def test_partition_is_disjoint_and_covers_all_errors():
    label = np.array([[0, 0, 1, 1, 2, 2],
                      [0, 1, 1, 2, 2, 0],
                      [0, 1, 1, 2, 2, 0]], dtype=np.int64)
    pred = np.array([[0, 1, 1, 1, 2, 0],
                     [0, 1, 2, 2, 0, 0],
                     [0, 1, 1, 2, 0, 0]], dtype=np.int64)
    categories, error = _partition(label, pred, primary_radius=1,
                                   small_area_px=2, instance_map=None)
    masks = list(categories.values())
    assert np.all(sum(mask.astype(np.int8) for mask in masks) <= 1)
    assert np.array_equal(sum(mask.astype(np.int8) for mask in masks) > 0, error)


def test_partition_does_not_assign_correct_pixels():
    label = np.zeros((8, 8), dtype=np.int64)
    pred = label.copy()
    categories, error = _partition(label, pred, primary_radius=1,
                                   small_area_px=2, instance_map=None)
    assert not error.any()
    assert all(not mask.any() for mask in categories.values())
