"""Paired cluster bootstrap between two models (optionally several paired seeds) + gate rule.

Delta = mean_over_seeds(metric(A_seed)) - mean_over_seeds(metric(B_seed)), with the SAME
resampled clusters for every seed and both models (paired by image name). Clusters are
images (default) or tiles (same tile id across flight dates, e.g. '0956').

python src/scripts/compare_runs.py \
    --a=results/L3_s0_val_per_image.csv,results/L3_s1_val_per_image.csv \
    --b=results/L2_s0_val_per_image.csv,results/L2_s1_val_per_image.csv \
    --name_a=L3 --name_b=L2 --gate=h2 --out=results/cmp_L3_vs_L2.json

Gates (plan section 7 / 8.2, thresholds locked BEFORE looking at validation):
  h2 / g3 : SmallWeedRecall delta >= +3.0 pp, 95% CI low > 0, Crop IoU drop <= 1.0 pp
  g4      : SmallWeedRecall delta > 0, 95% CI low > 0, Crop IoU drop <= 1.0 pp
  abs     : L3 - L0, same as h2
  screen  : C3 vs C2 one-way screen (SCREEN-PASS / NO-GO / INCONCLUSIVE), never 'H2 PASS'
"""

import csv
import json
import os

import numpy as np
from absl import app, flags

FLAGS = flags.FLAGS
flags.DEFINE_list('a', None, 'per-image CSVs of model A (one per seed)')
flags.DEFINE_list('b', None, 'per-image CSVs of model B (same seed order)')
flags.DEFINE_string('name_a', 'A', 'label A')
flags.DEFINE_string('name_b', 'B', 'label B')
flags.DEFINE_enum('cluster', 'image', ['image', 'tile'], 'bootstrap cluster unit')
flags.DEFINE_integer('n_boot', 10000, 'bootstrap replicates')
flags.DEFINE_integer('seed', 0, 'bootstrap seed')
flags.DEFINE_list('metrics', ['SmallWeedRecall', 'recall30_small', 'cov_small', 'recall50_medium',
                              'recall50_large', 'recall50_all', 'IoU_weed', 'IoU_crop', 'IoU_bg',
                              'mIoU', 'BIoU_weed', 'BIoU_crop'], 'metrics to compare')
flags.DEFINE_enum('gate', 'none', ['none', 'h2', 'g3', 'g4', 'abs', 'screen'], 'decision rule')
flags.DEFINE_float('min_delta', None, 'override primary-metric threshold (pp)')
flags.DEFINE_float('max_crop_drop', 1.0, 'max allowed Crop IoU drop (pp)')
flags.DEFINE_string('primary', 'SmallWeedRecall', 'primary metric of the gate')
flags.DEFINE_string('co_primary', 'cov_small', 'PASS also requires this metric delta > 0 '
                    '(added after Exp01 review; empty string disables)')
flags.DEFINE_string('out', None, 'output json')


def load_csv(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    names = [r['name'] for r in rows]
    cols = [k for k in rows[0] if k not in ('name', 'date', 'tile')]
    data = np.array([[float(r[k]) if r[k] != '' else 0.0 for k in cols] for r in rows])
    return names, cols, data


def metric_values(sums, col, metric):
    """Vectorised metric (percent) from column sums, sums: (..., K)."""
    def get(k):
        return sums[..., col[k]]

    def ratio(a, b):
        with np.errstate(invalid='ignore', divide='ignore'):
            return np.where(get(b) > 0, 100.0 * get(a) / np.maximum(get(b), 1e-12), np.nan)

    if metric == 'mIoU':
        return np.nanmean(np.stack([ratio(f'inter_{c}', f'union_{c}') for c in range(3)]), axis=0)
    if metric.startswith('IoU_'):
        c = {'bg': 0, 'crop': 1, 'weed': 2}[metric[4:]]
        return ratio(f'inter_{c}', f'union_{c}')
    if metric.startswith('BIoU_'):
        c = {'crop': 1, 'weed': 2}[metric[5:]]
        return ratio(f'binter_{c}', f'bunion_{c}')
    if metric == 'SmallWeedRecall':
        return ratio('tp50_small', 'n_small')
    if metric.startswith('recall'):
        t, b = metric[6:].split('_')
        return ratio(f'tp{t}_{b}', f'n_{b}')
    if metric.startswith('cov_'):
        b = metric[4:]
        return ratio(f'cov_{b}', f'n_{b}')
    if metric.startswith('prior_soft_iou_'):
        k = metric[len('prior_soft_iou_'):]
        return ratio(f'pq_min_{k}', f'pq_max_{k}')
    raise ValueError(f'unknown metric {metric}')


def cluster_sums(data, keys):
    uniq = sorted(set(keys))
    index = {k: i for i, k in enumerate(uniq)}
    out = np.zeros((len(uniq), data.shape[1]))
    for row, k in zip(data, keys):
        out[index[k]] += row
    return out


def decide(gate, delta, lo, crop_delta, min_delta, max_crop_drop):
    if gate == 'screen':
        if delta <= 0:
            return 'NO-GO (oracle does not support a Weed-specific prior)'
        if lo > 0:
            return 'SCREEN-PASS (not H2 evidence; needs L3 vs L2 and K5/K6)'
        return 'INCONCLUSIVE'
    if crop_delta < -max_crop_drop:
        return f'FAIL (Crop IoU drop {-crop_delta:.2f} > {max_crop_drop})'
    if delta <= 0:
        return 'FAIL'
    if delta >= min_delta and lo > 0:
        return 'PASS'
    return 'INCONCLUSIVE'


def main(_):
    if not FLAGS.a or not FLAGS.b or len(FLAGS.a) != len(FLAGS.b):
        raise SystemExit('--a and --b need the same number of per-image CSVs (paired seeds)')
    runs_a = [load_csv(p) for p in FLAGS.a]
    runs_b = [load_csv(p) for p in FLAGS.b]
    names = runs_a[0][0]
    cols = runs_a[0][1]
    for n, c, _ in runs_a + runs_b:
        if sorted(n) != sorted(names):
            raise SystemExit('Runs are not evaluated on the same images/split')
        missing = [k for k in cols if k not in c]
        if missing:
            raise SystemExit(f'Missing columns {missing[:5]}')
    col = {k: i for i, k in enumerate(cols)}

    def aligned(run):
        n, c, d = run
        pos = {k: i for i, k in enumerate(n)}
        cidx = [c.index(k) for k in cols]
        return d[[pos[k] for k in names]][:, cidx]

    keys = names if FLAGS.cluster == 'image' else [nm[11:] for nm in names]
    sums_a = [cluster_sums(aligned(r), keys) for r in runs_a]
    sums_b = [cluster_sums(aligned(r), keys) for r in runs_b]
    n_clusters = sums_a[0].shape[0]
    rng = np.random.default_rng(FLAGS.seed)
    weights = rng.multinomial(n_clusters, np.full(n_clusters, 1.0 / n_clusters), size=FLAGS.n_boot)

    result = {'A': FLAGS.name_a, 'B': FLAGS.name_b, 'files_A': FLAGS.a, 'files_B': FLAGS.b,
              'n_seeds': len(FLAGS.a), 'cluster': FLAGS.cluster, 'n_clusters': n_clusters,
              'n_images': len(names), 'n_boot': FLAGS.n_boot, 'metrics': {}}
    metrics = list(dict.fromkeys(FLAGS.metrics + [FLAGS.primary, 'IoU_crop'] +
                                 ([FLAGS.co_primary] if FLAGS.co_primary else [])))
    for metric in metrics:
        point_a = [float(metric_values(s.sum(0), col, metric)) for s in sums_a]
        point_b = [float(metric_values(s.sum(0), col, metric)) for s in sums_b]
        boot_a = np.mean([metric_values(weights @ s, col, metric) for s in sums_a], axis=0)
        boot_b = np.mean([metric_values(weights @ s, col, metric) for s in sums_b], axis=0)
        boot_delta = boot_a - boot_b
        boot_delta = boot_delta[~np.isnan(boot_delta)]
        delta = float(np.mean(point_a) - np.mean(point_b))
        lo, hi = (np.percentile(boot_delta, [2.5, 97.5]).tolist() if len(boot_delta)
                  else [float('nan'), float('nan')])
        result['metrics'][metric] = {
            'A_per_seed': point_a, 'B_per_seed': point_b,
            'A_mean': float(np.mean(point_a)), 'B_mean': float(np.mean(point_b)),
            'A_seed_std': float(np.std(point_a, ddof=1)) if len(point_a) > 1 else None,
            'B_seed_std': float(np.std(point_b, ddof=1)) if len(point_b) > 1 else None,
            'delta': delta, 'ci95': [lo, hi],
            'p_delta_le_0': float((boot_delta <= 0).mean()) if len(boot_delta) else None,
            'per_seed_delta': [a - b for a, b in zip(point_a, point_b)]}

    if FLAGS.gate != 'none':
        min_delta = FLAGS.min_delta
        if min_delta is None:
            min_delta = 3.0 if FLAGS.gate in ('h2', 'g3', 'abs') else 0.0
        p = result['metrics'][FLAGS.primary]
        crop = result['metrics']['IoU_crop']
        result['gate'] = {'rule': FLAGS.gate, 'primary': FLAGS.primary, 'min_delta': min_delta,
                          'max_crop_drop': FLAGS.max_crop_drop,
                          'decision': decide(FLAGS.gate, p['delta'], p['ci95'][0], crop['delta'],
                                             min_delta, FLAGS.max_crop_drop),
                          'crop_ci95': crop['ci95']}
        if FLAGS.co_primary and FLAGS.gate != 'screen':
            co = result['metrics'][FLAGS.co_primary]
            result['gate']['co_primary'] = {'metric': FLAGS.co_primary, 'delta': co['delta'],
                                            'ci95': co['ci95']}
            if result['gate']['decision'] == 'PASS' and not co['delta'] > 0:
                result['gate']['decision'] = (f'INCONCLUSIVE ({FLAGS.co_primary} delta '
                                              f"{co['delta']:+.2f} disagrees in sign)")
        if FLAGS.gate == 'screen':
            result['gate']['note'] = ('Oracle priors use GT of the evaluated image (label '
                                      'leakage). One-way screen only; NOT evidence for H2.')

    print(f"{FLAGS.name_a} vs {FLAGS.name_b} | seeds={len(FLAGS.a)} | cluster={FLAGS.cluster} "
          f"(n={n_clusters})")
    print(f"{'metric':18s} {'A':>8s} {'B':>8s} {'delta':>8s}   95% CI           P(d<=0)")
    for metric, r in result['metrics'].items():
        print(f"{metric:18s} {r['A_mean']:8.2f} {r['B_mean']:8.2f} {r['delta']:+8.2f}   "
              f"[{r['ci95'][0]:+6.2f}, {r['ci95'][1]:+6.2f}]   {r['p_delta_le_0']}")
    if 'gate' in result:
        print('GATE', result['gate']['rule'], '->', result['gate']['decision'])
    if FLAGS.out:
        os.makedirs(os.path.dirname(FLAGS.out) or '.', exist_ok=True)
        with open(FLAGS.out, 'w') as f:
            json.dump(result, f, indent=2)


if __name__ == '__main__':
    app.run(main)
