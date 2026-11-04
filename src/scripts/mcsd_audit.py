"""Audit MCSD output marginals and occupancy-head error by GT/predicted occupancy."""

import csv
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from absl import app, flags

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ccmc_common import build_model, define_model_flags, load_checkpoint, name_parts  # noqa: E402
from datasets import WeedsGaloreDataset  # noqa: E402

FLAGS = flags.FLAGS
flags.DEFINE_string('dataset_path', 'weedsgalore-dataset', 'dataset directory')
flags.DEFINE_string('ckpt', None, 'trained MCSD-g checkpoint')
flags.DEFINE_string('out_prefix', 'results/mcsd_audit', 'output file prefix')
flags.DEFINE_integer('bootstrap_samples', 2000, 'image-cluster bootstrap draws')
define_model_flags()

CLASS_NAMES = ('background', 'crop', 'weed')
OCCUPANCY_BINS = ((0, 0, '0/16'), (1, 3, '1-3/16'), (4, 7, '4-7/16'), (8, 16, '8-16/16'))
PRED_WEED_BINS = ((0.0, 0.01, '[0,.01)'), (0.01, 0.05, '[.01,.05)'),
                  (0.05, 0.15, '[.05,.15)'), (0.15, 1.0, '[.15,1]'))


def bootstrap_ci(image_rows, n_boot, rng):
    n_images = len(image_rows)
    if n_images < 2 or n_boot <= 0:
        return None
    estimates = np.empty((n_boot, 3), dtype=np.float64)
    for draw in range(n_boot):
        selected = rng.integers(0, n_images, size=n_images)
        rows = [image_rows[index] for index in selected]
        n_cells = sum(row['n_cells'] for row in rows)
        if n_cells == 0:
            estimates[draw] = np.nan
            continue
        estimates[draw, 0] = sum(row['residual_sum'] for row in rows) / n_cells
        estimates[draw, 1] = sum(row['abs_residual_sum'] for row in rows) / n_cells
        estimates[draw, 2] = sum(row['occ_abs_error_sum'] for row in rows) / n_cells
    return {
        name: np.nanpercentile(estimates[:, index], [2.5, 97.5]).tolist()
        for index, name in enumerate(('residual_mean', 'marginal_mae', 'occupancy_mae'))
    }


def main(_):
    if FLAGS.head not in ('mcsd', 'occ_conditioned'):
        raise SystemExit('mcsd_audit requires --head=mcsd or --head=occ_conditioned')
    if FLAGS.sinkhorn_iters < 1:
        raise SystemExit('--sinkhorn_iters must be at least 1')
    if FLAGS.bootstrap_samples < 0:
        raise SystemExit('--bootstrap_samples cannot be negative')
    FLAGS.ccmc = False

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = load_checkpoint(build_model(FLAGS), FLAGS.ckpt, device).to(device).eval()
    dataset = WeedsGaloreDataset(FLAGS.dataset_path, None, FLAGS.in_channels, 3,
                                 False, 'val', False)
    image_rows = {}
    cell_values = {}
    pred_weed_rows = {}
    pred_weed_values = {}
    total_weed_probability = 0.0
    total_gt_weed = 0
    cell_area = net.cell ** 2

    with torch.no_grad():
        for image_index in range(len(dataset)):
            name = str(dataset.img_list[image_index])
            image, _, label = dataset[image_index]
            input_tensor = torch.from_numpy(np.ascontiguousarray(image)).float().unsqueeze(0).to(device)
            label_tensor = torch.from_numpy(np.ascontiguousarray(label)).long().unsqueeze(0).to(device)
            logp, extras = net(input_tensor, return_aux=True)
            probability = logp.float().exp()
            q_pred = torch.softmax(extras['occ_logits'].float(), dim=1)
            q_out = F.avg_pool2d(probability, net.cell)
            one_hot = F.one_hot(label_tensor.clamp(min=0), len(CLASS_NAMES)).permute(0, 3, 1, 2).float()
            q_gt = F.avg_pool2d(one_hot, net.cell)
            gt_counts = torch.round(q_gt * cell_area).long()
            residual = q_out - q_pred
            occupancy_error = q_pred - q_gt

            q_pred_weed = q_pred[0, 2]
            weed_residual = residual[0, 2]
            weed_occ_error = occupancy_error[0, 2]
            for bin_index, (lower, upper, bin_name) in enumerate(PRED_WEED_BINS):
                if bin_index == len(PRED_WEED_BINS) - 1:
                    selected = (q_pred_weed >= lower) & (q_pred_weed <= upper)
                else:
                    selected = (q_pred_weed >= lower) & (q_pred_weed < upper)
                residual_values = weed_residual[selected].cpu().numpy()
                occupancy_values = weed_occ_error[selected].cpu().numpy()
                key = bin_name
                pred_weed_values.setdefault(key, []).extend(residual_values.tolist())
                pred_weed_rows.setdefault(key, []).append({
                    'image': name,
                    'date': name[:10],
                    'tile': name[11:],
                    'n_cells': int(selected.sum().item()),
                    'residual_sum': float(residual_values.sum()),
                    'abs_residual_sum': float(np.abs(residual_values).sum()),
                    'occ_abs_error_sum': float(np.abs(occupancy_values).sum()),
                })

            total_weed_probability += probability[:, 2].sum().item()
            total_gt_weed += int((label_tensor == 2).sum().item())
            date, tile = name_parts(name)
            for class_id, class_name in enumerate(CLASS_NAMES):
                for lower, upper, bin_name in OCCUPANCY_BINS:
                    selected = ((gt_counts[0, class_id] >= lower)
                                & (gt_counts[0, class_id] <= upper))
                    marginal_values = residual[0, class_id][selected].cpu().numpy()
                    occupancy_values = occupancy_error[0, class_id][selected].cpu().numpy()
                    abs_values = np.abs(marginal_values)
                    key = (class_id, bin_name)
                    bucket = cell_values.setdefault(key, [])
                    bucket.extend(marginal_values.tolist())
                    row = {
                        'image': name,
                        'date': date,
                        'tile': tile,
                        'class_id': class_id,
                        'class_name': class_name,
                        'gt_occupancy_bin': bin_name,
                        'n_cells': int(selected.sum().item()),
                        'residual_sum': float(marginal_values.sum()),
                        'abs_residual_sum': float(abs_values.sum()),
                        'occ_abs_error_sum': float(np.abs(occupancy_values).sum()),
                    }
                    image_rows.setdefault(key, []).append(row)

    os.makedirs(os.path.dirname(FLAGS.out_prefix) or '.', exist_ok=True)
    csv_path = FLAGS.out_prefix + '_marginal_by_image.csv'
    pred_csv_path = FLAGS.out_prefix + '_predweed_by_image.csv'
    json_path = FLAGS.out_prefix + '_marginal_summary.json'
    fieldnames = ('image', 'date', 'tile', 'class_id', 'class_name', 'gt_occupancy_bin',
                  'n_cells', 'residual_mean', 'marginal_mae', 'occupancy_mae')
    summary = {
        'checkpoint': FLAGS.ckpt,
        'split': 'val',
        'head': FLAGS.head,
        'sinkhorn_iters': FLAGS.sinkhorn_iters,
        'n_images': len(dataset),
        'bootstrap_samples': FLAGS.bootstrap_samples,
        'pred_weed_mass_over_gt': total_weed_probability / max(total_gt_weed, 1),
        'groups': [],
        'pred_weed_occupancy_bins': [],
    }
    rng = np.random.default_rng(20261104)
    csv_rows = []
    for key, rows in image_rows.items():
        class_id, bin_name = key
        values = np.asarray(cell_values[key], dtype=np.float64)
        n_cells = int(sum(row['n_cells'] for row in rows))
        abs_values = np.abs(values)
        residual_mean = float(values.mean()) if n_cells else float('nan')
        marginal_mae = float(abs_values.mean()) if n_cells else float('nan')
        occupancy_mae = (sum(row['occ_abs_error_sum'] for row in rows) / n_cells
                         if n_cells else float('nan'))
        for row in rows:
            count = row['n_cells']
            csv_rows.append({
                'image': row['image'], 'date': row['date'], 'tile': row['tile'],
                'class_id': class_id, 'class_name': row['class_name'],
                'gt_occupancy_bin': bin_name, 'n_cells': count,
                'residual_mean': row['residual_sum'] / count if count else '',
                'marginal_mae': row['abs_residual_sum'] / count if count else '',
                'occupancy_mae': row['occ_abs_error_sum'] / count if count else '',
            })
        summary['groups'].append({
            'class_id': class_id,
            'class_name': CLASS_NAMES[class_id],
            'gt_occupancy_bin': bin_name,
            'n_cells': n_cells,
            'residual_mean': residual_mean,
            'marginal_mae': marginal_mae,
            'marginal_p95_abs': float(np.percentile(abs_values, 95)) if n_cells else None,
            'marginal_max_abs': float(abs_values.max()) if n_cells else None,
            'occupancy_mae': occupancy_mae,
            'image_bootstrap_ci95': bootstrap_ci(rows, FLAGS.bootstrap_samples, rng),
        })

    pred_csv_rows = []
    for bin_name, rows in pred_weed_rows.items():
        values = np.asarray(pred_weed_values[bin_name], dtype=np.float64)
        n_cells = int(sum(row['n_cells'] for row in rows))
        abs_values = np.abs(values)
        for row in rows:
            count = row['n_cells']
            pred_csv_rows.append({
                'image': row['image'], 'date': row['date'], 'tile': row['tile'],
                'pred_weed_occupancy_bin': bin_name, 'n_cells': count,
                'residual_mean': row['residual_sum'] / count if count else '',
                'marginal_mae': row['abs_residual_sum'] / count if count else '',
                'occupancy_mae': row['occ_abs_error_sum'] / count if count else '',
            })
        summary['pred_weed_occupancy_bins'].append({
            'pred_weed_occupancy_bin': bin_name,
            'n_cells': n_cells,
            'residual_mean': float(values.mean()) if n_cells else None,
            'marginal_mae': float(abs_values.mean()) if n_cells else None,
            'marginal_p95_abs': float(np.percentile(abs_values, 95)) if n_cells else None,
            'marginal_max_abs': float(abs_values.max()) if n_cells else None,
            'occupancy_mae': (sum(row['occ_abs_error_sum'] for row in rows) / n_cells
                              if n_cells else None),
            'image_bootstrap_ci95': bootstrap_ci(rows, FLAGS.bootstrap_samples, rng),
        })

    with open(csv_path, 'w', newline='') as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)
    with open(pred_csv_path, 'w', newline='') as output_file:
        writer = csv.DictWriter(output_file, fieldnames=(
            'image', 'date', 'tile', 'pred_weed_occupancy_bin', 'n_cells',
            'residual_mean', 'marginal_mae', 'occupancy_mae'))
        writer.writeheader()
        writer.writerows(pred_csv_rows)
    with open(json_path, 'w') as output_file:
        json.dump(summary, output_file, indent=2)
    print(json.dumps({'csv': csv_path, 'pred_weed_csv': pred_csv_path, 'summary': json_path,
                      'pred_weed_mass_over_gt': summary['pred_weed_mass_over_gt']}))


if __name__ == '__main__':
    app.run(main)