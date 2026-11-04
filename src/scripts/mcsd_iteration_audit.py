"""Compare MCSD predictions at two Sinkhorn iteration counts on validation images."""

import csv
import json
import os
import sys

import numpy as np
import torch
from absl import app, flags

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ccmc_common import build_model, define_model_flags, load_checkpoint, name_parts  # noqa: E402
from ccmc_metrics import area_bin, load_bins, load_instance_map, match_instances, weed_instances  # noqa: E402
from datasets import WeedsGaloreDataset  # noqa: E402

FLAGS = flags.FLAGS
flags.DEFINE_string('dataset_path', 'weedsgalore-dataset', 'dataset directory')
flags.DEFINE_string('ckpt', None, 'trained MCSD-g checkpoint')
flags.DEFINE_string('bins_json', 'results/instance_bins.json', 'locked train-derived instance bins')
flags.DEFINE_integer('iters_a', 1, 'iteration count for prediction A')
flags.DEFINE_integer('iters_b', 5, 'iteration count for prediction B')
flags.DEFINE_integer('bootstrap_samples', 2000, 'paired image-cluster bootstrap draws')
flags.DEFINE_string('out_prefix', 'results/mcsd_iters', 'output file prefix')
define_model_flags()


def safe_ratio(numerator, denominator):
    return float(numerator / denominator) if denominator else float('nan')


def aggregate_ci(rows, column_a, column_b, denominator_a, denominator_b, n_boot, rng):
    if len(rows) < 2 or n_boot <= 0:
        return None
    n_images = len(rows)
    draws = np.empty(n_boot, dtype=np.float64)
    for draw in range(n_boot):
        indices = rng.integers(0, n_images, size=n_images)
        sample = [rows[index] for index in indices]
        value_a = safe_ratio(sum(row[column_a] for row in sample),
                             sum(row[denominator_a] for row in sample))
        value_b = safe_ratio(sum(row[column_b] for row in sample),
                             sum(row[denominator_b] for row in sample))
        draws[draw] = value_a - value_b
    draws = draws[np.isfinite(draws)]
    return np.percentile(draws, [2.5, 97.5]).tolist() if draws.size else None


def ratio_ci(rows, numerator, denominator, n_boot, rng):
    if len(rows) < 2 or n_boot <= 0:
        return None
    n_images = len(rows)
    draws = np.empty(n_boot, dtype=np.float64)
    for draw in range(n_boot):
        indices = rng.integers(0, n_images, size=n_images)
        sample = [rows[index] for index in indices]
        draws[draw] = safe_ratio(sum(row[numerator] for row in sample),
                                 sum(row[denominator] for row in sample))
    draws = draws[np.isfinite(draws)]
    return np.percentile(draws, [2.5, 97.5]).tolist() if draws.size else None


def main(_):
    if FLAGS.head != 'mcsd':
        raise SystemExit('mcsd_iteration_audit requires --head=mcsd')
    if FLAGS.iters_a < 1 or FLAGS.iters_b < 1:
        raise SystemExit('iteration counts must be at least 1')
    if FLAGS.iters_a == FLAGS.iters_b:
        raise SystemExit('iters_a and iters_b must differ')
    if FLAGS.bootstrap_samples < 0:
        raise SystemExit('--bootstrap_samples cannot be negative')
    FLAGS.ccmc = False
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = load_checkpoint(build_model(FLAGS), FLAGS.ckpt, device).to(device).eval()
    dataset = WeedsGaloreDataset(FLAGS.dataset_path, None, FLAGS.in_channels, 3,
                                 False, 'val', False)
    edges, _ = load_bins(FLAGS.bins_json)
    per_image = []
    per_instance = []
    counts = {
        FLAGS.iters_a: {'confusion': np.zeros((3, 3), dtype=np.int64), 'small_tp50': 0,
                        'small_tp30': 0, 'small_covered': 0, 'small_n': 0},
        FLAGS.iters_b: {'confusion': np.zeros((3, 3), dtype=np.int64), 'small_tp50': 0,
                        'small_tp30': 0, 'small_covered': 0, 'small_n': 0},
    }

    with torch.no_grad():
        for image_index in range(len(dataset)):
            name = str(dataset.img_list[image_index])
            image, _, label = dataset[image_index]
            input_tensor = torch.from_numpy(np.ascontiguousarray(image)).float().unsqueeze(0).to(device)
            label = np.asarray(label)
            label_tensor = torch.from_numpy(np.ascontiguousarray(label)).long().unsqueeze(0).to(device)
            instance_map = load_instance_map(FLAGS.dataset_path, name)
            instances, instance_source = weed_instances(label, instance_map)
            small_matches = {}
            row = {'name': name, 'date': name_parts(name)[0], 'tile': name_parts(name)[1],
                   'instance_source': instance_source}

            for iters in (FLAGS.iters_a, FLAGS.iters_b):
                net.sinkhorn_iters = iters
                logp = net(input_tensor)
                prediction = logp.argmax(1)[0].cpu().numpy()
                confusion = np.bincount((label * 3 + prediction).ravel(), minlength=9).reshape(3, 3)
                matches = match_instances(instances, prediction == 2)
                bins = area_bin(matches['areas'], edges) if len(matches['areas']) else np.array([], dtype=int)
                small = bins == 0
                stats = counts[iters]
                stats['confusion'] += confusion
                stats['small_tp50'] += int(matches['matched_50'][small].sum())
                stats['small_tp30'] += int(matches['matched_30'][small].sum())
                stats['small_covered'] += int((matches['coverage'][small] >= 0.5).sum())
                stats['small_n'] += int(small.sum())

                weed_tp = int(confusion[2, 2])
                weed_fp = int(confusion[0, 2] + confusion[1, 2])
                crop_n = int((label == 1).sum())
                crop_to_weed = int(confusion[1, 2])
                row.update({
                    f'weed_tp_{iters}': weed_tp,
                    f'weed_fp_{iters}': weed_fp,
                    f'weed_pred_{iters}': weed_tp + weed_fp,
                    f'weed_precision_{iters}': safe_ratio(weed_tp, weed_tp + weed_fp),
                    f'crop_to_weed_pixels_{iters}': crop_to_weed,
                    f'crop_pixels_{iters}': crop_n,
                    f'crop_to_weed_rate_{iters}': safe_ratio(crop_to_weed, crop_n),
                    f'small_n_{iters}': int(small.sum()),
                    f'small_tp50_{iters}': int(matches['matched_50'][small].sum()),
                    f'small_tp30_{iters}': int(matches['matched_30'][small].sum()),
                    f'small_covered_{iters}': int((matches['coverage'][small] >= 0.5).sum()),
                })
                small_matches[iters] = (matches, small)

            per_image.append(row)
            match_a, small_a = small_matches[FLAGS.iters_a]
            match_b, small_b = small_matches[FLAGS.iters_b]
            if not np.array_equal(match_a['ids'], match_b['ids']):
                raise RuntimeError(f'GT instance ids changed across inference modes: {name}')
            for index in np.flatnonzero(small_a):
                detected_a = bool(match_a['matched_50'][index])
                detected_b = bool(match_b['matched_50'][index])
                per_instance.append({
                    'image': name,
                    'instance_source': instance_source,
                    'gt_instance_id': int(match_a['ids'][index]),
                    'area_px': int(match_a['areas'][index]),
                    f'matched50_it{FLAGS.iters_a}': detected_a,
                    f'matched50_it{FLAGS.iters_b}': detected_b,
                    'newly_detected_at_a': detected_a and not detected_b,
                    'lost_at_a': detected_b and not detected_a,
                    f'best_iou_it{FLAGS.iters_a}': float(match_a['best_iou'][index]),
                    f'best_iou_it{FLAGS.iters_b}': float(match_b['best_iou'][index]),
                    f'coverage_it{FLAGS.iters_a}': float(match_a['coverage'][index]),
                    f'coverage_it{FLAGS.iters_b}': float(match_b['coverage'][index]),
                })

    os.makedirs(os.path.dirname(FLAGS.out_prefix) or '.', exist_ok=True)
    image_path = FLAGS.out_prefix + '_per_image.csv'
    instance_path = FLAGS.out_prefix + '_small_instances.csv'
    summary_path = FLAGS.out_prefix + '_summary.json'
    with open(image_path, 'w', newline='') as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(per_image[0]))
        writer.writeheader()
        writer.writerows(per_image)
    with open(instance_path, 'w', newline='') as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(per_instance[0]) if per_instance else
                                ['image', 'gt_instance_id', 'area_px'])
        writer.writeheader()
        writer.writerows(per_instance)

    rng = np.random.default_rng(20261104)
    metric_specs = {
        'weed_precision': (f'weed_tp_{FLAGS.iters_a}', f'weed_tp_{FLAGS.iters_b}',
                           f'weed_pred_{FLAGS.iters_a}', f'weed_pred_{FLAGS.iters_b}'),
        'crop_to_weed_rate': (f'crop_to_weed_pixels_{FLAGS.iters_a}',
                              f'crop_to_weed_pixels_{FLAGS.iters_b}',
                              f'crop_pixels_{FLAGS.iters_a}', f'crop_pixels_{FLAGS.iters_b}'),
        'small_recall50': (f'small_tp50_{FLAGS.iters_a}', f'small_tp50_{FLAGS.iters_b}',
                           f'small_n_{FLAGS.iters_a}', f'small_n_{FLAGS.iters_b}'),
    }
    deltas = {}
    for metric, (num_a, num_b, den_a, den_b) in metric_specs.items():
        point_a = safe_ratio(sum(row[num_a] for row in per_image),
                             sum(row[den_a] for row in per_image))
        point_b = safe_ratio(sum(row[num_b] for row in per_image),
                             sum(row[den_b] for row in per_image))
        deltas[metric] = {
            f'it{FLAGS.iters_a}': point_a,
            f'it{FLAGS.iters_b}': point_b,
            'delta_a_minus_b': point_a - point_b,
            'paired_image_bootstrap_ci95': aggregate_ci(
                per_image, num_a, num_b, den_a, den_b, FLAGS.bootstrap_samples, rng),
        }

    for image_row in per_image:
        image_row['new_small_instances_at_a'] = sum(
            item['image'] == image_row['name'] and item['newly_detected_at_a']
            for item in per_instance)
    deltas['new_small_instance_rate_at_a'] = {
        'numerator': sum(row['new_small_instances_at_a'] for row in per_image),
        'denominator': sum(row[f'small_n_{FLAGS.iters_a}'] for row in per_image),
        'rate': safe_ratio(sum(row['new_small_instances_at_a'] for row in per_image),
                           sum(row[f'small_n_{FLAGS.iters_a}'] for row in per_image)),
        'image_bootstrap_ci95': ratio_ci(
            per_image, 'new_small_instances_at_a', f'small_n_{FLAGS.iters_a}',
            FLAGS.bootstrap_samples, rng),
    }

    for iters, stats in counts.items():
        confusion = stats['confusion']
        tp = int(confusion[2, 2])
        fp = int(confusion[0, 2] + confusion[1, 2])
        summary_counts = {
            'weed_precision': safe_ratio(tp, tp + fp),
            'crop_to_weed_rate': safe_ratio(int(confusion[1, 2]), int(confusion[1].sum())),
            'crop_to_weed_fraction_of_weed_fp': safe_ratio(int(confusion[1, 2]), fp),
            'weed_false_positive_pixels': fp,
            'crop_to_weed_pixels': int(confusion[1, 2]),
            'confusion_matrix_gt_rows_pred_columns': confusion.tolist(),
            'small_recall50': safe_ratio(stats['small_tp50'], stats['small_n']),
            'small_recall30': safe_ratio(stats['small_tp30'], stats['small_n']),
            'small_coverage_rate': safe_ratio(stats['small_covered'], stats['small_n']),
            'small_instances': stats['small_n'],
        }
        deltas[f'aggregate_it{iters}'] = summary_counts

    with open(summary_path, 'w') as output_file:
        json.dump({
            'checkpoint': FLAGS.ckpt,
            'split': 'val',
            'iterations_a': FLAGS.iters_a,
            'iterations_b': FLAGS.iters_b,
            'rate_scale': 'fraction_0_to_1',
            'n_images': len(per_image),
            'n_small_instances_compared': len(per_instance),
            'newly_detected_at_a': sum(row['newly_detected_at_a'] for row in per_instance),
            'lost_at_a': sum(row['lost_at_a'] for row in per_instance),
            'metrics': deltas,
            'outputs': {'per_image_csv': image_path, 'small_instances_csv': instance_path},
        }, output_file, indent=2)
    print(json.dumps({'per_image_csv': image_path, 'small_instances_csv': instance_path,
                      'summary_json': summary_path}))


if __name__ == '__main__':
    app.run(main)