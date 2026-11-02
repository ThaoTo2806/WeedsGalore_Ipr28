"""Metrics for the CCMC-SR study (plan section 5 / 8.2).

- Pixel IoU from per-image intersection/union counts (bootstrap-able).
- Boundary IoU (Cheng et al., CVPR 2021 style: mask minus its erosion, d = ratio * diag).
- Instance-level Weed recall by size bin: official instance labels if available, otherwise
  connected components of the semantic Weed mask (logged as source='cc', NOT equivalent).
  One-to-one greedy matching on IoU; thresholds 0.5 (primary) and 0.3 (ablation).
  Secondary: coverage recall (>= 50 % of the instance pixels predicted as Weed).
- Aggregation from column sums so that paired cluster bootstrap is a matrix product.
"""

import json
import os

import cv2
import numpy as np
from PIL import Image

WEED = 2
NUM_CLASSES = 3
BIN_NAMES = ('small', 'medium', 'large')
IOU_THRESHOLDS = (50, 30)


# ----------------------------------------------------------------------------------------
# Instances
# ----------------------------------------------------------------------------------------

def load_instance_map(dataset_path, name):
    """Return the raw instance id map (int64, HxW) or None if no instance file exists."""
    path = os.path.join(dataset_path, name[:10], 'instances', name + '.png')
    if not os.path.exists(path):
        return None
    arr = np.array(Image.open(path))
    if arr.ndim == 3:  # colour-coded ids
        arr = arr[..., :3].astype(np.int64)
        arr = arr[..., 0] * 65536 + arr[..., 1] * 256 + arr[..., 2]
    return arr.astype(np.int64)


def weed_instances(binary_label, instance_map=None):
    """Weed instance map (int32, 0 = none, 1..n) and its source ('official' | 'cc').

    Official ids are restricted to Weed pixels. Weed pixels without an instance id
    (if any) are added as connected components so that no Weed pixel is ignored.
    """
    weed = binary_label == WEED
    if instance_map is None:
        _, comp = cv2.connectedComponents(weed.astype(np.uint8), connectivity=8)
        return comp.astype(np.int32), 'cc'
    ids = np.where(weed, instance_map, 0)
    uniq, inverse = np.unique(ids, return_inverse=True)
    inst = inverse.reshape(ids.shape).astype(np.int32)
    if uniq[0] != 0:  # every pixel is Weed with a non-zero id -> make ids start at 1
        inst = inst + 1
    inst[~weed] = 0
    orphan = weed & (inst == 0)  # Weed pixels with instance id 0
    if orphan.any():
        _, comp = cv2.connectedComponents(orphan.astype(np.uint8), connectivity=8)
        offset = inst.max()
        inst = np.where(comp > 0, comp + offset, inst).astype(np.int32)
    return inst, 'official'


def instance_areas(inst):
    counts = np.bincount(inst.ravel())
    ids = np.nonzero(counts)[0]
    ids = ids[ids > 0]
    return ids, counts[ids]


def area_bin(areas, edges):
    """0 = small, 1 = medium, 2 = large, from train tertile edges [q1, q2]."""
    return np.digitize(areas, edges, right=True)


def match_instances(inst, pred_weed, thresholds=IOU_THRESHOLDS):
    """One-to-one greedy IoU matching between GT Weed instances and predicted Weed CCs.

    Returns dict: ids, areas, best_iou, matched_<t> (bool arrays), coverage.
    """
    ids, areas = instance_areas(inst)
    result = {'ids': ids, 'areas': areas, 'coverage': np.zeros(len(ids)),
              'best_iou': np.zeros(len(ids))}
    for t in thresholds:
        result[f'matched_{t}'] = np.zeros(len(ids), dtype=bool)
    if len(ids) == 0:
        return result
    index_of = np.zeros(inst.max() + 1, dtype=np.int64)
    index_of[ids] = np.arange(len(ids))
    covered = np.bincount(inst[pred_weed & (inst > 0)], minlength=inst.max() + 1)[ids]
    result['coverage'] = covered / areas

    n_pred, pred_cc = cv2.connectedComponents(pred_weed.astype(np.uint8), connectivity=8)
    if n_pred <= 1:
        return result
    pred_areas = np.bincount(pred_cc.ravel(), minlength=n_pred)
    sel = (inst > 0) & (pred_cc > 0)
    if not sel.any():
        return result
    pair_codes = index_of[inst[sel]] * n_pred + pred_cc[sel]
    codes, inter = np.unique(pair_codes, return_counts=True)
    g_idx, p_idx = codes // n_pred, codes % n_pred
    iou = inter / (areas[g_idx] + pred_areas[p_idx] - inter)
    np.maximum.at(result['best_iou'], g_idx, iou)
    order = np.argsort(-iou, kind='stable')
    for t in thresholds:
        used_g, used_p = set(), set()
        matched = result[f'matched_{t}']
        for k in order:
            if iou[k] < t / 100.0:
                break
            g, p = g_idx[k], p_idx[k]
            if g in used_g or p in used_p:
                continue
            used_g.add(g)
            used_p.add(p)
            matched[g] = True
    return result


# ----------------------------------------------------------------------------------------
# Boundary IoU
# ----------------------------------------------------------------------------------------

def mask_boundary(mask, dilation):
    mask = mask.astype(np.uint8)
    padded = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    kernel = np.ones((3, 3), dtype=np.uint8)
    eroded = cv2.erode(padded, kernel, iterations=dilation)[1:-1, 1:-1]
    return (mask - eroded) > 0


def boundary_dilation(shape, ratio=0.02):
    return max(1, int(round(ratio * np.sqrt(shape[0] ** 2 + shape[1] ** 2))))


# ----------------------------------------------------------------------------------------
# Per-image statistics
# ----------------------------------------------------------------------------------------

def image_stats(pred, label, inst, edges, boundary_ratio=0.02, boundary_px=0):
    """All additive statistics of one image as a flat dict (numbers only).

    boundary_px > 0 sets a fixed boundary width (pixels); otherwise ratio * diagonal.
    """
    row = {}
    for c in range(NUM_CLASSES):
        p, g = pred == c, label == c
        row[f'inter_{c}'] = int((p & g).sum())
        row[f'union_{c}'] = int((p | g).sum())
    d = boundary_px if boundary_px > 0 else boundary_dilation(label.shape, boundary_ratio)
    for c in (1, 2):
        pb, gb = mask_boundary(pred == c, d), mask_boundary(label == c, d)
        row[f'binter_{c}'] = int((pb & gb).sum())
        row[f'bunion_{c}'] = int((pb | gb).sum())
    for gc in range(NUM_CLASSES):
        for pc in range(NUM_CLASSES):
            row[f'cm_{gc}{pc}'] = int(((label == gc) & (pred == pc)).sum())
    match = match_instances(inst, pred == WEED)
    bins = area_bin(match['areas'], edges) if len(match['areas']) else np.array([], dtype=int)
    for b, bname in enumerate(BIN_NAMES):
        in_bin = bins == b
        row[f'n_{bname}'] = int(in_bin.sum())
        for t in IOU_THRESHOLDS:
            row[f'tp{t}_{bname}'] = int(match[f'matched_{t}'][in_bin].sum())
        row[f'cov_{bname}'] = int((match['coverage'][in_bin] >= 0.5).sum())
    row['n_all'] = int(len(bins))
    for t in IOU_THRESHOLDS:
        row[f'tp{t}_all'] = int(match[f'matched_{t}'].sum())
    row['cov_all'] = int((match['coverage'] >= 0.5).sum())
    return row


def metrics_from_sums(s):
    """Metrics (percent) from a dict/Series of column sums. Safe divisions -> nan."""
    def div(a, b):
        return float(100.0 * a / b) if b > 0 else float('nan')
    m = {}
    ious = [div(s[f'inter_{c}'], s[f'union_{c}']) for c in range(NUM_CLASSES)]
    m['mIoU'] = float(np.nanmean(ious))
    m['IoU_bg'], m['IoU_crop'], m['IoU_weed'] = ious
    m['BIoU_crop'] = div(s['binter_1'], s['bunion_1'])
    m['BIoU_weed'] = div(s['binter_2'], s['bunion_2'])
    for bname in BIN_NAMES + ('all',):
        for t in IOU_THRESHOLDS:
            m[f'recall{t}_{bname}'] = div(s[f'tp{t}_{bname}'], s[f'n_{bname}'])
        m[f'cov_{bname}'] = div(s[f'cov_{bname}'], s[f'n_{bname}'])
        m[f'n_{bname}'] = int(s[f'n_{bname}'])
    m['SmallWeedRecall'] = m['recall50_small']
    return m


def stat_columns(row_keys):
    return [k for k in row_keys if k not in ('name', 'date', 'tile')]


def load_bins(path):
    with open(path) as f:
        data = json.load(f)
    return np.asarray(data['edges'], dtype=float), data
