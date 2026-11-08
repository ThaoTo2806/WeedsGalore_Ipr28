"""Exp11 validation error-budget and branch-activity diagnostic.

This script evaluates one checkpoint on validation only. It partitions semantic errors,
computes oracle repair ceilings, and optionally disables a model branch when the model
implements ``set_branch_scale``. It never reads test data and never trains.
"""

import json
from pathlib import Path
import sys

import cv2
import numpy as np
import torch
from absl import app, flags

# Allow direct execution as ``python src/scripts/exp11_error_budget.py``.
SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ccmc_common import build_model, define_model_flags, load_checkpoint
from ccmc_metrics import (NUM_CLASSES, WEED, area_bin, image_stats, load_bins,
                          load_instance_map, metrics_from_sums, weed_instances)
from datasets import WeedsGaloreDataset

FLAGS = flags.FLAGS
EXP11_ERROR_BUDGET_VERSION = '20261108-v2'
flags.DEFINE_string('dataset_path', None, 'WeedsGalore dataset root')
flags.DEFINE_enum('split', 'val', ['train', 'val', 'test'], 'evaluation split')
flags.DEFINE_string('ckpt', None, 'checkpoint path')
flags.DEFINE_string('bins_json', 'results/instance_bins.json', 'instance-bin metadata')
flags.DEFINE_string('run_name', '', 'diagnostic run name')
flags.DEFINE_string('out_prefix', None, 'output prefix')
flags.DEFINE_list('radii', ['1', '2', '3'], 'boundary radii for sensitivity')
flags.DEFINE_integer('primary_radius', 2, 'primary boundary radius')
flags.DEFINE_boolean('component_vote', True, 'compute component-vote diagnostic')
flags.DEFINE_boolean('activity_audit', True, 'disable branch when supported')
define_model_flags()


def aggregate_miou(confusion):
    """Return mIoU percent from a 3x3 ground-truth-by-prediction matrix."""
    matrix = np.asarray(confusion, dtype=np.float64)
    intersection = np.diag(matrix)
    union = matrix.sum(1) + matrix.sum(0) - intersection
    return float(np.nanmean(np.divide(100.0 * intersection, union,
                                       out=np.full(3, np.nan), where=union > 0)))


def _boundary_mask(label, radius):
    boundary = np.zeros(label.shape, dtype=bool)
    for cls in range(NUM_CLASSES):
        mask = (label == cls).astype(np.uint8)
        padded = cv2.copyMakeBorder(mask, radius, radius, radius, radius,
                                    cv2.BORDER_CONSTANT, value=0)
        kernel = np.ones((3, 3), dtype=np.uint8)
        eroded = cv2.erode(padded, kernel, iterations=radius)
        boundary |= mask.astype(bool) & ~eroded[radius:-radius, radius:-radius].astype(bool)
    return boundary


def _gt_components(label):
    components = np.zeros(label.shape, dtype=np.int32)
    next_id = 1
    for cls in (1, 2):
        count, connected = cv2.connectedComponents((label == cls).astype(np.uint8), 8)
        for component_id in range(1, count):
            components[connected == component_id] = next_id
            next_id += 1
    return components


def _component_flip_mask(label, pred):
    components = _gt_components(label)
    result = np.zeros(label.shape, dtype=bool)
    for component_id in range(1, components.max() + 1):
        selected = components == component_id
        gt_class = int(np.bincount(label[selected], minlength=NUM_CLASSES).argmax())
        prediction = pred[selected]
        if gt_class in (1, 2):
            vegetation = prediction[(prediction == 1) | (prediction == 2)]
            if vegetation.size and int(np.bincount(vegetation, minlength=3).argmax()) != gt_class:
                result |= selected
    return result


def _prediction_component_vote(pred):
    """Return a no-GT component probe based only on predicted vegetation components.

    A connected component already has one predicted class, so this majority vote is
    intentionally a conservative no-op probe. It is not an oracle correction.
    """
    repaired = pred.copy()
    for cls in (1, 2):
        count, components = cv2.connectedComponents((pred == cls).astype(np.uint8), 8)
        for component_id in range(1, count):
            selected = components == component_id
            majority = np.bincount(pred[selected], minlength=NUM_CLASSES).argmax()
            repaired[selected] = majority
    return repaired


def _partition(label, pred, primary_radius, small_area_px, instance_map):
    error = pred != label
    categories = {}
    assigned = np.zeros(label.shape, dtype=bool)

    boundary = _boundary_mask(label, primary_radius)
    categories['boundary_r2'] = error & boundary
    assigned |= categories['boundary_r2']

    instances, _ = weed_instances(label, instance_map)
    ids, areas = np.unique(instances[instances > 0], return_counts=True)
    small_ids = ids[areas <= small_area_px]
    categories['small_weed'] = error & np.isin(instances, small_ids)
    categories['small_weed'] &= ~assigned
    assigned |= categories['small_weed']

    categories['component_flip'] = error & _component_flip_mask(label, pred) & ~assigned
    assigned |= categories['component_flip']
    categories['background_fp'] = error & (label == 0) & (pred != 0) & ~assigned
    assigned |= categories['background_fp']
    categories['other'] = error & ~assigned
    return categories, error


def _confusion(label, pred):
    return np.bincount((label.ravel() * NUM_CLASSES + pred.ravel()),
                       minlength=NUM_CLASSES * NUM_CLASSES).reshape(NUM_CLASSES, NUM_CLASSES)


def _repair_confusion(label, pred, mask):
    repaired = pred.copy()
    repaired[mask] = label[mask]
    return _confusion(label, repaired)


def _category_record(label, pred, mask, error_count):
    base = _confusion(label, pred)
    repaired = _repair_confusion(label, pred, mask)
    return {
        'error_pixel_share': float(mask.sum() / max(error_count, 1)),
        'ceiling_delta_mIoU': aggregate_miou(repaired) - aggregate_miou(base),
    }


def _evaluate(net, dataset, device, edges, return_predictions=False):
    rows = []
    predictions = []
    net.eval()
    with torch.no_grad():
        for index in range(len(dataset)):
            name = str(dataset.img_list[index])
            image, _, binary_label = dataset[index]
            x = torch.from_numpy(np.ascontiguousarray(image)).float().unsqueeze(0).to(device)
            output = net(x)
            if isinstance(output, tuple):
                output = output[0]
            pred = output.argmax(1)[0].cpu().numpy().astype(np.int64)
            label = np.asarray(binary_label).astype(np.int64)
            instance_map = load_instance_map(FLAGS.dataset_path, name)
            row = image_stats( pred, label,
                               weed_instances(label, instance_map)[0], edges,
                               boundary_ratio=0.02, boundary_px=3)
            row['name'] = name
            rows.append(row)
            predictions.append((name, label, pred, instance_map))
    return rows, predictions


def _summary_from_rows(rows):
    keys = [key for key in rows[0] if key != 'name']
    sums = {key: sum(row[key] for row in rows) for key in keys}
    return metrics_from_sums(sums)


def main(_):
    if FLAGS.split == 'test':
        raise SystemExit('Exp11 error budget refuses split=test.')
    if not FLAGS.dataset_path or not FLAGS.ckpt or not FLAGS.out_prefix:
        raise SystemExit('--dataset_path, --ckpt and --out_prefix are required')
    edges, _ = load_bins(FLAGS.bins_json)
    dataset = WeedsGaloreDataset(FLAGS.dataset_path, None, FLAGS.in_channels,
                                  FLAGS.num_classes, False, FLAGS.split, False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = load_checkpoint(build_model(FLAGS), FLAGS.ckpt, device).to(device)
    rows, predictions = _evaluate(net, dataset, device, edges)
    base_confusion = sum((_confusion(label, pred) for _, label, pred, _ in predictions),
                         np.zeros((3, 3), dtype=np.int64))
    def class_iou(cls):
        union = base_confusion[cls].sum() + base_confusion[:, cls].sum() - base_confusion[cls, cls]
        return float(100.0 * base_confusion[cls, cls] / max(union, 1))

    baseline = {'mIoU': aggregate_miou(base_confusion),
                'IoU_bg': class_iou(0), 'IoU_crop': class_iou(1),
                'IoU_weed': class_iou(2)}
    partition_records = {}
    sensitivity_records = {}
    per_image = []
    for name, label, pred, instance_map in predictions:
        categories, error = _partition(label, pred, FLAGS.primary_radius,
                                        int(edges[0]), instance_map)
        row = {'name': name, 'base_confusion': _confusion(label, pred)}
        for category, mask in categories.items():
            row[category] = _repair_confusion(label, pred, mask)
        per_image.append(row)
        for category, mask in categories.items():
            partition_records.setdefault(category, {'pixels': 0, 'ceiling_deltas': []})
            partition_records[category]['pixels'] += int(mask.sum())
            partition_records[category]['ceiling_deltas'].append(
                aggregate_miou(_repair_confusion(label, pred, mask)) - aggregate_miou(_confusion(label, pred)))

    for category, record in partition_records.items():
        record['error_pixel_share'] = record['pixels'] / max(sum(np.count_nonzero(pred != label)
                                                               for _, label, pred, _ in predictions), 1)
        record['ceiling_delta_mIoU'] = float(np.mean(record.pop('ceiling_deltas')))
    for radius in (int(value) for value in FLAGS.radii):
        sensitivity_records[f'boundary_r{radius}'] = {'confusions': [], 'deltas': []}
        for _, label, pred, _ in predictions:
            mask = (pred != label) & _boundary_mask(label, radius)
            repaired = _repair_confusion(label, pred, mask)
            sensitivity_records[f'boundary_r{radius}']['confusions'].append(repaired)
            sensitivity_records[f'boundary_r{radius}']['deltas'].append(
                aggregate_miou(repaired) - aggregate_miou(_confusion(label, pred)))
        sensitivity_records[f'boundary_r{radius}']['ceiling_delta_mIoU'] = float(
            np.mean(sensitivity_records[f'boundary_r{radius}'].pop('deltas')))
        sensitivity_records[f'boundary_r{radius}']['error_pixel_share'] = None

    prediction_component_delta = None
    gt_component_vote_delta = None
    if FLAGS.component_vote:
        prediction_deltas = []
        oracle_deltas = []
        for _, label, pred, _ in predictions:
            repaired = _prediction_component_vote(pred)
            prediction_deltas.append(aggregate_miou(_confusion(label, repaired)) - aggregate_miou(_confusion(label, pred)))
            repaired_oracle = pred.copy()
            components = _gt_components(label)
            for component_id in range(1, components.max() + 1):
                selected = components == component_id
                majority = np.bincount(pred[selected], minlength=3).argmax()
                repaired_oracle[selected] = majority
            oracle_deltas.append(aggregate_miou(_confusion(label, repaired_oracle)) - aggregate_miou(_confusion(label, pred)))
        prediction_component_delta = float(np.mean(prediction_deltas))
        gt_component_vote_delta = float(np.mean(oracle_deltas))

    activity = {'branch_scale_supported': hasattr(net, 'set_branch_scale'),
                'delta_mIoU_when_branch_zero': None}
    if FLAGS.activity_audit and hasattr(net, 'set_branch_scale'):
        net.set_branch_scale(0.0)
        _, disabled = _evaluate(net, dataset, device, edges)
        disabled_confusion = sum((_confusion(label, pred) for _, label, pred, _ in disabled),
                                 np.zeros((3, 3), dtype=np.int64))
        activity['delta_mIoU_when_branch_zero'] = aggregate_miou(disabled_confusion) - aggregate_miou(base_confusion)
        net.set_branch_scale(1.0)

    prefix = Path(FLAGS.out_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    sensitivity_npz = {
        f'boundary_sensitivity_{radius}': np.stack(record['confusions'])
        for radius, record in ((key.rsplit('boundary_r', 1)[-1], value)
                               for key, value in sensitivity_records.items())
    }
    sensitivity_json = {
        key: {'error_pixel_share': record['error_pixel_share'],
              'ceiling_delta_mIoU': record['ceiling_delta_mIoU']}
        for key, record in sensitivity_records.items()
    }
    report = {
        'run_name': FLAGS.run_name, 'split': FLAGS.split, 'n_images': len(predictions),
        'aggregator': 'global_confusion_sum', 'baseline': baseline,
        'partition_categories': partition_records,
        'boundary_sensitivity': sensitivity_json,
        'partition_order': ['boundary_r2', 'small_weed', 'component_flip',
                            'background_fp', 'other'],
        'component_vote': {
            'prediction_component_probe_delta_mIoU': prediction_component_delta,
            'gt_component_vote_oracle_delta_mIoU': gt_component_vote_delta,
            'uses_ground_truth': False,
            'uses_ground_truth_for_oracle_only': True,
        },
        'activity': activity,
        'per_image_npz': str(prefix.with_name(prefix.name + '_budget_per_image.npz')),
        'test_accessed': False,
    }
    prefix.with_name(prefix.name + '_budget.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    npz_payload = {
        category: np.stack([row[category] for row in per_image])
        for category in ['base_confusion', *partition_records]
        if category in per_image[0]
    }
    npz_payload.update(sensitivity_npz)
    npz_payload['image_names'] = np.asarray([row['name'] for row in per_image])
    np.savez_compressed(prefix.with_name(prefix.name + '_budget_per_image.npz'),
                        **npz_payload)
    print(json.dumps({'run_name': FLAGS.run_name, 'mIoU': baseline['mIoU'],
                      'partition_categories': list(partition_records),
                      'boundary_sensitivity': list(sensitivity_records),
                      'activity': activity}, indent=2))


if __name__ == '__main__':
    app.run(main)
