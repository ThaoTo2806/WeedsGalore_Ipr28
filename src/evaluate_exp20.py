"""Evaluation for the CCMC-SR study: pixel IoU, Boundary IoU, size-binned instance Weed
recall, per-date breakdown, prior quality and oracle intervention.

Writes <out_prefix>_per_image.csv (additive stats, used by scripts/compare_runs.py for
paired cluster bootstrap) and <out_prefix>_summary.json.

Examples
  # L3 checkpoint, validation, learned prior
  python src/evaluate_ccmc.py --dataset_path=... --ckpt=.../best_val.pth \
      --prior_mode=learned_weed --bins_json=results/instance_bins.json --out_prefix=results/L3_s0_val
  # Oracle intervention on the SAME checkpoint (GT leakage, diagnostic only)
  ... --prior_override=gt_weed --out_prefix=results/L3_s0_val_ovr-gtweed
  ... --prior_override=gt_weed --prior_noise=dilate:5
"""

import csv
import json
import os

import numpy as np
import torch
from absl import app, flags

from ccmc_common import build_model, define_model_flags, load_checkpoint, name_parts
from nets.exp20_icr import ICRSegFormer
from ccmc_metrics import (BIN_NAMES, image_stats, load_bins, load_instance_map,
                          metrics_from_sums, stat_columns, weed_instances)
from datasets import WeedsGaloreDataset
from nets.ccmc_sr import GT_MODES, corrupt_weed_labels, occupancy
from exp12_utils import predict_labels
import exp12_config

FLAGS = flags.FLAGS
flags.DEFINE_string('dataset_path', 'weedsgalore-dataset', 'dataset directory')
flags.DEFINE_enum('split', 'val', ['train', 'val', 'test'], 'evaluation split')
flags.DEFINE_boolean('confirm_test_locked', False, 'required for split=test: config is frozen '
                     'on validation and test is run ONCE (plan section 5 / 8.2)')
flags.DEFINE_string('ckpt', None, 'checkpoint (best_val.pth or last.pth)')
flags.DEFINE_string('bins_json', 'results/instance_bins.json', 'from scripts/make_instance_bins.py')
flags.DEFINE_string('out_prefix', 'results/eval', 'output prefix')
flags.DEFINE_string('run_name', '', 'label written to summary')
flags.DEFINE_enum('prior_override', 'none', ['none', 'uniform', 'gt_veg', 'gt_weed', 'ndvi'],
                  'swap prior at inference on the same checkpoint (gt_* = GT leakage)')
flags.DEFINE_string('prior_noise', 'none', "corruption of the GT-Weed prior: 'dilate:5', "
                    "'erode:3', 'shift:8', 'drop:0.3' (only with prior_override=gt_weed or "
                    "prior_mode=gt_weed)")
flags.DEFINE_float('boundary_ratio', 0.02, 'Boundary IoU dilation ratio of image diagonal '
                   '(only used if boundary_px <= 0)')
flags.DEFINE_integer('boundary_px', 3, 'fixed Boundary IoU width in pixels. Ratio 0.02 gives '
                     '~17 px on 600x600, wider than most weeds (BIoU == IoU), so 3 px is used')
flags.DEFINE_integer('min_bin_n', 20, 'bins with fewer instances are flagged exploratory')
# EXP16_POINT_EVAL_FLAG
flags.DEFINE_bool('exp17_safety_gate', False, 'Exp17 gated refinement')
flags.DEFINE_bool('exp16_point_refine', False, 'Point refine architecture for Exp16 only')
flags.DEFINE_bool('exp20_icr', False, 'Exp20 single-network crop reference')
define_model_flags()
flags.DEFINE_string('arm_id', '', 'Exp12 registered arm (empty for legacy evaluation)')
flags.DEFINE_string('exp12_arms', 'configs/exp12/exp12_arms.json', 'Exp12 arm registry')
flags.DEFINE_float('input_scale', 1.0, 'Input inference resolution multiplier')
flags.DEFINE_enum('tta', 'none', ['none', 'hflip', 'flips', 'd4'], 'Single-network test-time transforms')
flags.DEFINE_float('weed_bias', 0.0, 'Registered Weed log-probability bias control')


def main(_):
    if FLAGS.split == 'test' and not FLAGS.confirm_test_locked:
        raise SystemExit('Refusing to evaluate on TEST: set --confirm_test_locked=True only after '
                         'the configuration is frozen on validation (single final run).')
    if FLAGS.num_classes != 3:
        raise SystemExit('CCMC-SR evaluation expects num_classes=3')
    if FLAGS.arm_id:
        arms = exp12_config.load_json(FLAGS.exp12_arms)
        problems = exp12_config.verify_eval(FLAGS.flag_values_dict(), arms, FLAGS.arm_id)
        if problems:
            raise SystemExit('Exp12 EVAL config mismatch: ' + '; '.join(problems))
        if FLAGS.split == 'test' and FLAGS.weed_bias != 0.0:
            raise SystemExit('Weed-bias probe is validation-only, never evaluate it on test')
    elif FLAGS.tta != 'none' or FLAGS.input_scale != 1.0 or FLAGS.weed_bias != 0.0:
        raise SystemExit('Non-default Exp12 TTA/scale/bias requires a registered --arm_id')
    if FLAGS.input_scale <= 0:
        raise SystemExit('input_scale must be positive')
    if FLAGS.ccmc and (FLAGS.tta != 'none' or FLAGS.input_scale != 1.0):
        raise SystemExit('Nontrivial Exp12 TTA/scale is not supported for CCMC prior models')
    if FLAGS.head != 'bilinear':
        FLAGS.ccmc = False  # MCSD / pixelshuffle heads are SegFormer5Band-based, no prior
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if FLAGS.exp20_icr:
        if FLAGS.ccmc or FLAGS.backbone != 'b2' or FLAGS.head != 'bilinear' or not FLAGS.use_low_level_refine or FLAGS.llr_source != 'raw' or FLAGS.exp16_point_refine or FLAGS.exp17_safety_gate:
            raise SystemExit('Exp20 eval architecture mismatch')
        model = ICRSegFormer(num_classes=3, in_channels=5, pretrained_backbone=True, backbone='b2', use_low_level_refine=True, use_multiscale_fusion=True, llr_source='raw')
    else:
        model = build_model(FLAGS)
    net = load_checkpoint(model, FLAGS.ckpt, device).to(device).eval()
    edges, bins_meta = load_bins(FLAGS.bins_json)

    override = None if FLAGS.prior_override == 'none' else FLAGS.prior_override
    uses_gt = FLAGS.ccmc and (FLAGS.prior_mode in GT_MODES or override in GT_MODES)
    if FLAGS.prior_noise != 'none' and not uses_gt:
        raise SystemExit('prior_noise only applies to GT priors')
    if override is not None and not FLAGS.ccmc:
        raise SystemExit('prior_override needs a CCMC checkpoint')

    dataset = WeedsGaloreDataset(dataset_path=FLAGS.dataset_path, dataset_size=None,
                                 in_bands=FLAGS.in_channels, num_classes=FLAGS.num_classes,
                                 is_training=False, split=FLAGS.split, augmentation=False)
    rows, sources = [], {}
    with torch.no_grad():
        for idx in range(len(dataset)):
            name = str(dataset.img_list[idx])
            img, _, binary_label = dataset[idx]
            x = torch.from_numpy(np.ascontiguousarray(img)).float().unsqueeze(0).to(device)
            label_t = torch.from_numpy(np.ascontiguousarray(binary_label)).long().unsqueeze(0).to(device)
            extras = {}
            if FLAGS.ccmc:
                prior_mask = None
                if uses_gt:
                    prior_mask = corrupt_weed_labels(label_t, FLAGS.prior_noise, seed=idx)
                logits, extras = net(x, prior_mask=prior_mask, prior_override=override,
                                     return_aux=True)
            else:
                if FLAGS.arm_id:
                    pred = predict_labels(net, x, scale=FLAGS.input_scale, tta=FLAGS.tta,
                                          weed_bias=FLAGS.weed_bias)[0].cpu().numpy()
                else:
                    logits = net(x)
                    pred = logits.argmax(1)[0].cpu().numpy()
            if FLAGS.ccmc:
                pred = logits.argmax(1)[0].cpu().numpy()
            label = np.asarray(binary_label)
            inst, src = weed_instances(label, load_instance_map(FLAGS.dataset_path, name))
            sources[src] = sources.get(src, 0) + 1
            row = {'name': name}
            row['date'], row['tile'] = name_parts(name)
            row.update(image_stats(pred, label, inst, edges, FLAGS.boundary_ratio, FLAGS.boundary_px))
            # Prior quality (head prediction if a head exists, otherwise the used prior)
            prior = None
            if 'prior_logits' in extras:
                prior = torch.sigmoid(extras['prior_logits'].float())
            elif 'prior' in extras:
                prior = extras['prior'].float()
            if prior is not None:
                for kind in ('weed', 'veg'):
                    target = occupancy(label_t, kind, prior.shape[-2:])
                    row[f'pq_abs_{kind}'] = float((prior - target).abs().sum())
                    row[f'pq_min_{kind}'] = float(torch.minimum(prior, target).sum())
                    row[f'pq_max_{kind}'] = float(torch.maximum(prior, target).sum())
                row['pq_n'] = int(prior.numel())
            rows.append(row)

    os.makedirs(os.path.dirname(FLAGS.out_prefix) or '.', exist_ok=True)
    keys = list(rows[0].keys())
    with open(FLAGS.out_prefix + '_per_image.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)

    cols = stat_columns(keys)
    sums = {k: sum(r[k] for r in rows) for k in cols}
    summary = {'run_name': FLAGS.run_name, 'ckpt': FLAGS.ckpt, 'split': FLAGS.split,
               'n_images': len(rows), 'prior_mode': FLAGS.prior_mode if FLAGS.ccmc else 'n/a',
               'injection': FLAGS.injection if FLAGS.ccmc else 'n/a',
               'prior_perturb': FLAGS.prior_perturb if FLAGS.ccmc else 'n/a',
               'prior_override': FLAGS.prior_override, 'prior_noise': FLAGS.prior_noise,
               'GT_LEAKAGE': bool(uses_gt), 'instance_source': sources,
               'bin_edges_px': edges.tolist(), 'bins_source': bins_meta.get('source'),
               'boundary_px': FLAGS.boundary_px, 'head': FLAGS.head,
               'arm_id': FLAGS.arm_id, 'input_scale': FLAGS.input_scale,
               'tta': FLAGS.tta, 'weed_bias': FLAGS.weed_bias,
               'lambdas': net.encoder.lambdas() if FLAGS.ccmc else [],
               'metrics': metrics_from_sums(sums)}
    cm = np.array([[sums[f'cm_{g}{p}'] for p in range(3)] for g in range(3)], dtype=float)
    summary['confusion_row_norm'] = (cm / cm.sum(1, keepdims=True).clip(min=1)).round(4).tolist()
    weed_tp = float(cm[2, 2]); weed_pred = float(cm[:, 2].sum()); weed_gt = float(cm[2].sum())
    summary['weed_precision_percent'] = 100.0 * weed_tp / weed_pred if weed_pred else None
    summary['weed_pred_gt_mass_ratio'] = weed_pred / weed_gt if weed_gt else None
    if 'pq_n' in sums:
        summary['prior_quality'] = {
            f'mae_{k}': sums[f'pq_abs_{k}'] / sums['pq_n'] for k in ('weed', 'veg')}
        summary['prior_quality'].update({
            f'soft_iou_{k}': sums[f'pq_min_{k}'] / max(sums[f'pq_max_{k}'], 1e-9)
            for k in ('weed', 'veg')})
    summary['exploratory_bins'] = [b for b in BIN_NAMES if sums[f'n_{b}'] < FLAGS.min_bin_n]
    per_date = {}
    for date in sorted({r['date'] for r in rows}):
        sel = [r for r in rows if r['date'] == date]
        per_date[date] = metrics_from_sums({k: sum(r[k] for r in sel) for k in cols})
        per_date[date]['n_images'] = len(sel)
    summary['per_date'] = per_date
    with open(FLAGS.out_prefix + '_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    m = summary['metrics']
    print(json.dumps({k: summary[k] for k in ('run_name', 'split', 'prior_override',
                                              'prior_noise', 'GT_LEAKAGE', 'instance_source',
                                              'lambdas')}, default=str))
    print(f"mIoU {m['mIoU']:.2f} | bg {m['IoU_bg']:.2f} crop {m['IoU_crop']:.2f} "
          f"weed {m['IoU_weed']:.2f} | BIoU crop {m['BIoU_crop']:.2f} weed {m['BIoU_weed']:.2f}")
    for b in BIN_NAMES + ('all',):
        print(f"  {b:6s} n={m[f'n_{b}']:4d} recall@0.5={m[f'recall50_{b}']:.2f} "
              f"recall@0.3={m[f'recall30_{b}']:.2f} cov>=0.5={m[f'cov_{b}']:.2f}")
    if summary['exploratory_bins']:
        print(f"  WARNING exploratory bins (n<{FLAGS.min_bin_n}): {summary['exploratory_bins']}")
    if uses_gt:
        print('  NOTE: GT prior used -> label leakage; diagnostic only, not a deployable result.')


if __name__ == '__main__':
    app.run(main)
