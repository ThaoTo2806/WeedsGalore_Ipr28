"""Fail-closed WeedMap RedEdge-M evaluation primitives (no model inference).

This module does NOT establish camera calibration, mask polarity, official split or
spatial independence. It never chooses them from target data. Use the loader ONLY
for data-integrity/synthetic testing until an independently reviewed, frozen
protocol documents those choices; offline prediction scoring must follow that
protocol. A successful unit test is not authorization to inspect test results.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
from PIL import Image

NUM_CLASSES = 3
IGNORE_CODE = 10000
IGNORE_TARGET = -1
CLASS_CODES = (0, 1, 2)
BANDS = ('R', 'G', 'B', 'NIR', 'RE')


@dataclass(frozen=True)
class Tile:
    frame_id: str
    image: np.ndarray  # float32 [5,H,W], numerical PNG normalization only
    raw_index_map: np.ndarray  # uint8 or uint16 [H,W]
    raw_mask: np.ndarray  # uint8 [H,W]; polarity deliberately unresolved


def _read_png(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(str(path))
    with Image.open(path) as img:
        if img.format != 'PNG':
            raise ValueError(f'Expected PNG: {path}')
        return np.array(img)


def discover_frames(root: str | Path) -> list[str]:
    """Validate identical frame IDs across all five bands and annotations.

    `root` is the exact index folder: .../Tiles/RedEdge/003/003.
    Empty/padded frames are retained; do not filter by target-label prevalence.
    """
    root = Path(root)
    band_sets = []
    for band in BANDS:
        folder = root / 'tile' / band
        if not folder.is_dir():
            raise FileNotFoundError(f'Missing band folder: {folder}')
        band_sets.append({p.stem for p in folder.glob('frame*.png')})
    if not band_sets[0]:
        raise ValueError('No frame PNG files')
    if any(ids != band_sets[0] for ids in band_sets[1:]):
        raise ValueError('Five-band frame ID sets differ')
    frames = sorted(band_sets[0])
    for frame in frames:
        for path in (root / 'mask' / f'{frame}.png',
                     root / 'groundtruth' / f'003_{frame}_GroundTruth_iMap.png'):
            if not path.is_file():
                raise FileNotFoundError(str(path))
    return frames


def load_tile(root: str | Path, frame_id: str) -> Tile:
    """Load image/annotation without determining valid pixels or calibration.

    PNG numerical mapping matches matplotlib PNG imread for unsigned uint8/16:
    division by 255/65535. This is NOT a physical radiometric conversion.
    """
    if not frame_id.startswith('frame') or not frame_id[5:].isdigit():
        raise ValueError('frame_id must have form frameNNNN')
    root = Path(root)
    channels = []
    for band in BANDS:
        arr = _read_png(root / 'tile' / band / f'{frame_id}.png')
        if arr.ndim != 2 or arr.dtype not in (np.dtype('uint8'), np.dtype('uint16')):
            raise ValueError(f'{band}: expected 2D uint8/uint16 grayscale PNG')
        divisor = 255.0 if arr.dtype == np.uint8 else 65535.0
        channels.append(arr.astype(np.float32) / divisor)
    shape = channels[0].shape
    if any(c.shape != shape for c in channels):
        raise ValueError(f'Band spatial shapes do not agree: {frame_id}')
    index_map = _read_png(root / 'groundtruth' / f'003_{frame_id}_GroundTruth_iMap.png')
    mask = _read_png(root / 'mask' / f'{frame_id}.png')
    if index_map.shape != shape or mask.shape != shape:
        raise ValueError(f'Label/mask spatial shapes do not agree: {frame_id}')
    if index_map.dtype not in (np.dtype('uint8'), np.dtype('uint16')):
        raise ValueError('iMap must be grayscale uint8 or uint16')
    if mask.dtype != np.uint8 or not np.isin(mask, (0, 255)).all():
        raise ValueError('Expected binary uint8 mask codes 0/255')
    if not np.isin(index_map, (*CLASS_CODES, IGNORE_CODE)).all():
        raise ValueError(f'Unexpected WeedMap iMap codes: {np.unique(index_map).tolist()}')
    return Tile(frame_id, np.stack(channels), index_map, mask)


def resolve_valid_pixels(mask: np.ndarray, *, valid_mask_code: int | None = None) -> np.ndarray:
    """Apply an EXTERNALLY VERIFIED mask code. Never infer it from signal."""
    if valid_mask_code not in (0, 255):
        raise ValueError('valid_mask_code must be explicitly verified as 0 or 255')
    arr = np.asarray(mask)
    if arr.ndim != 2 or arr.dtype != np.uint8 or not np.isin(arr, (0, 255)).all():
        raise ValueError('mask must be a 2D uint8 array containing only 0 and 255')
    return arr == valid_mask_code


def prepare_target(index_map: np.ndarray, valid_pixels: np.ndarray) -> np.ndarray:
    """Convert official iMap into [0,1,2,-1] with explicit valid-mask input."""
    labels = np.asarray(index_map)
    valid = np.asarray(valid_pixels)
    if labels.ndim != 2:
        raise ValueError(f'Expected a 2D iMap, got shape {labels.shape}')
    if valid.shape != labels.shape or valid.dtype != np.bool_:
        raise ValueError('valid_pixels must be a boolean array with the same shape as index_map')
    known_codes = np.isin(labels, (*CLASS_CODES, IGNORE_CODE))
    if not known_codes.all():
        raise ValueError(f'Unexpected WeedMap iMap codes: {np.unique(labels[~known_codes]).tolist()}')
    target = labels.astype(np.int64, copy=True)
    target[target == IGNORE_CODE] = IGNORE_TARGET
    target[~valid] = IGNORE_TARGET
    return target


def confusion_matrix(target: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    """3x3 additive counts with GT on rows, prediction on columns."""
    target_array = np.asarray(target)
    prediction_array = np.asarray(prediction)
    if target_array.ndim != 2 or prediction_array.shape != target_array.shape:
        raise ValueError('target and prediction must be same-sized 2D arrays')
    valid = target_array != IGNORE_TARGET
    if not np.isin(target_array[valid], CLASS_CODES).all():
        raise ValueError('target must contain only classes 0, 1, 2, or ignore=-1')
    if not np.issubdtype(prediction_array.dtype, np.integer) or not np.isin(prediction_array, CLASS_CODES).all():
        raise ValueError('prediction must be an integer map containing only classes 0, 1, 2')
    codes = target_array[valid].astype(np.int64) * NUM_CLASSES + prediction_array[valid].astype(np.int64)
    return np.bincount(codes, minlength=NUM_CLASSES ** 2).reshape(NUM_CLASSES, NUM_CLASSES)


def metrics_from_confusion(matrix: np.ndarray) -> dict:
    """Dataset-level micro/macro IoUs. Undefined ratios are None, never zero."""
    counts = np.asarray(matrix)
    if (counts.shape != (3, 3) or not np.issubdtype(counts.dtype, np.integer)
            or np.any(counts < 0)):
        raise ValueError('confusion matrix must be a nonnegative integer 3x3 array')
    def safe_ratio(num, den):
        return float(100.0 * num / den) if den else None
    tp = int(counts[2, 2]); gt = int(counts[2, :].sum()); pred = int(counts[:, 2].sum())
    union = gt + pred - tp
    result = {
        'WeedRecall': safe_ratio(tp, gt),
        'WeedPrecision': safe_ratio(tp, pred),
        'IoU_weed': safe_ratio(tp, union),
        'n_valid_pixels': int(counts.sum()),
        'n_crop_gt_pixels': int(counts[1].sum()),
        'three_class_metrics_available': bool(counts[1].sum() > 0),
    }
    if result['three_class_metrics_available']:
        unions = counts.sum(axis=0) + counts.sum(axis=1) - np.diag(counts)
        ious = [safe_ratio(int(counts[i, i]), int(unions[i])) for i in range(3)]
        result.update(IoU_bg=ious[0], IoU_crop=ious[1],
                      mIoU=float(np.mean(ious)) if all(v is not None for v in ious) else None)
    return result


def aggregate_predictions(targets: Mapping[str, np.ndarray],
                          predictions: Mapping[str, np.ndarray]) -> dict:
    """Score precomputed integer prediction maps; never runs a model.

    Keys must match exactly. Sum confusion counts BEFORE computing metrics;
    do not average per-frame ratios or change the frame set by label frequency.
    """
    if not targets or set(targets) != set(predictions):
        raise ValueError('Target/prediction frame keys must match and be nonempty')
    total = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    per_frame = {}
    for frame in sorted(targets):
        cm = confusion_matrix(targets[frame], predictions[frame])
        per_frame[frame] = cm.tolist()
        total += cm
    return {'confusion_matrix': total.tolist(),
            'per_frame_confusion': per_frame,
            'metrics': metrics_from_confusion(total),
            'frame_count': len(targets)}
