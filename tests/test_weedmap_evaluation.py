import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

from weedmap_evaluation import (IGNORE_TARGET, confusion_matrix, metrics_from_confusion,
                                prepare_target)


def test_nonclass_code_is_ignored_in_confusion_matrix():
    index_map = np.array([[0, 2, 10000], [1, 2, 0]], dtype=np.uint16)
    valid_pixels = np.ones(index_map.shape, dtype=bool)
    target = prepare_target(index_map, valid_pixels)
    prediction = np.array([[0, 2, 1], [1, 0, 2]])

    matrix = confusion_matrix(target, prediction)

    assert target[0, 2] == IGNORE_TARGET
    assert matrix.sum() == 5
    assert matrix[0, 0] == 1
    assert matrix[1, 1] == 1
    assert matrix[2, 2] == 1
    assert matrix[2, 0] == 1
    assert matrix[0, 2] == 1


def test_explicit_valid_mask_excludes_pixels_without_assuming_polarity():
    index_map = np.array([[0, 2], [1, 2]], dtype=np.uint8)
    valid_pixels = np.array([[True, False], [False, True]], dtype=bool)

    target = prepare_target(index_map, valid_pixels)

    assert target.tolist() == [[0, IGNORE_TARGET], [IGNORE_TARGET, 2]]


def test_crop_absent_omits_three_class_metrics():
    index_map = np.array([[0, 2, 10000]], dtype=np.uint16)
    target = prepare_target(index_map, np.ones(index_map.shape, dtype=bool))
    matrix = confusion_matrix(target, np.array([[0, 2, 1]]))

    metrics = metrics_from_confusion(matrix)

    assert metrics['three_class_metrics_available'] is False
    assert metrics['n_crop_gt_pixels'] == 0
    assert metrics['WeedRecall'] == 100.0
    assert metrics['WeedPrecision'] == 100.0
    assert metrics['IoU_weed'] == 100.0
    assert 'mIoU' not in metrics
    assert 'IoU_crop' not in metrics


def test_crop_present_enables_three_class_metrics():
    index_map = np.array([[0, 1, 2]], dtype=np.uint8)
    target = prepare_target(index_map, np.ones(index_map.shape, dtype=bool))
    matrix = confusion_matrix(target, np.array([[0, 1, 2]]))

    metrics = metrics_from_confusion(matrix)

    assert metrics['three_class_metrics_available'] is True
    assert metrics['IoU_bg'] == 100.0
    assert metrics['IoU_crop'] == 100.0
    assert metrics['IoU_weed'] == 100.0
    assert metrics['mIoU'] == 100.0


def test_unknown_index_codes_are_rejected():
    index_map = np.array([[0, 7]], dtype=np.uint16)
    with pytest.raises(ValueError, match='Unexpected WeedMap iMap codes'):
        prepare_target(index_map, np.ones(index_map.shape, dtype=bool))


def test_validity_mask_must_be_boolean_and_shape_matched():
    index_map = np.array([[0, 2]], dtype=np.uint8)
    with pytest.raises(ValueError, match='valid_pixels must be a boolean array'):
        prepare_target(index_map, np.array([[0, 255]], dtype=np.uint8))
    with pytest.raises(ValueError, match='valid_pixels must be a boolean array'):
        prepare_target(index_map, np.ones((1, 1), dtype=bool))


def test_metrics_handle_no_ground_truth_weed():
    matrix = np.zeros((3, 3), dtype=np.int64)
    matrix[0, 2] = 4

    metrics = metrics_from_confusion(matrix)

    assert metrics['WeedRecall'] is None
    assert metrics['WeedPrecision'] == 0.0
    assert metrics['IoU_weed'] == 0.0
from PIL import Image
from weedmap_evaluation import (BANDS, discover_frames, load_tile,
                                resolve_valid_pixels, aggregate_predictions)


def _write_synthetic_tile(root, frame='frame0020', mask=None, idx=None):
    if mask is None:
        mask = np.array([[0, 255], [0, 0]], dtype=np.uint8)
    if idx is None:
        idx = np.array([[0, 10000], [2, 2]], dtype=np.uint16)
    for j, band in enumerate(BANDS):
        path = root / 'tile' / band / f'{frame}.png'
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.full((2, 2), j + 11, dtype=np.uint8)).save(path)
    (root / 'mask').mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask).save(root / 'mask' / f'{frame}.png')
    (root / 'groundtruth').mkdir(parents=True, exist_ok=True)
    Image.fromarray(idx).save(root / 'groundtruth' / f'003_{frame}_GroundTruth_iMap.png')


def test_loader_preserves_band_order_and_uint16_labels(tmp_path):
    _write_synthetic_tile(tmp_path)
    assert discover_frames(tmp_path) == ['frame0020']
    tile = load_tile(tmp_path, 'frame0020')
    assert tile.image.shape == (5, 2, 2)
    assert tile.image.dtype == np.float32
    assert [int(round(float(v) * 255)) for v in tile.image[:, 0, 0]] == list(range(11, 16))
    assert tile.raw_index_map.dtype == np.uint16
    assert tile.raw_index_map[0, 1] == 10000


def test_mask_polarity_is_mandatory(tmp_path):
    _write_synthetic_tile(tmp_path)
    tile = load_tile(tmp_path, 'frame0020')
    with pytest.raises(ValueError, match='explicitly verified'):
        resolve_valid_pixels(tile.raw_mask)
    valid = resolve_valid_pixels(tile.raw_mask, valid_mask_code=0)
    gt = prepare_target(tile.raw_index_map, valid)
    assert gt.tolist() == [[0, -1], [2, 2]]
    assert resolve_valid_pixels(tile.raw_mask, valid_mask_code=255).sum() == 1


def test_frame_alignment_rejected(tmp_path):
    _write_synthetic_tile(tmp_path)
    (tmp_path / 'tile' / 'RE' / 'frame0020.png').unlink()
    with pytest.raises(ValueError, match='frame ID'):
        discover_frames(tmp_path)


def test_spatial_mismatch_rejected(tmp_path):
    _write_synthetic_tile(tmp_path)
    Image.fromarray(np.zeros((3, 2), dtype=np.uint8)).save(
        tmp_path / 'tile' / 'NIR' / 'frame0020.png')
    with pytest.raises(ValueError, match='spatial shapes'):
        load_tile(tmp_path, 'frame0020')


def test_aggregation_micro_counts_not_average():
    targets = {'a': np.array([[2, 2]]), 'b': np.array([[2]])}
    preds = {'a': np.array([[2, 0]]), 'b': np.array([[2]])}
    report = aggregate_predictions(targets, preds)
    assert report['confusion_matrix'][2][2] == 2
    assert report['confusion_matrix'][2][0] == 1
    assert report['metrics']['WeedRecall'] == pytest.approx(200 / 3)
    assert report['metrics']['three_class_metrics_available'] is False


def test_predictions_must_match_frames():
    with pytest.raises(ValueError, match='keys'):
        aggregate_predictions({'a': np.array([[2]])}, {'b': np.array([[2]])})


def test_float_prediction_not_accepted():
    with pytest.raises(ValueError, match='integer map'):
        confusion_matrix(np.array([[2]], dtype=np.int64),
                         np.array([[2.0]], dtype=np.float32))
