"""Collect *_summary.json (evaluate_ccmc) and run_summary.json (train) into one CSV table.

python src/scripts/collect_results.py --results_dir=results --runs_dir=runs --out=results/metrics.csv
"""

import csv
import glob
import json
import os

from absl import app, flags

FLAGS = flags.FLAGS
flags.DEFINE_string('results_dir', 'results', 'directory with *_summary.json')
flags.DEFINE_string('runs_dir', 'runs', 'directory with <run>/run_summary.json')
flags.DEFINE_string('out', 'results/metrics.csv', 'output csv')

KEYS = ('mIoU', 'IoU_bg', 'IoU_crop', 'IoU_weed', 'BIoU_crop', 'BIoU_weed', 'SmallWeedRecall',
        'recall30_small', 'cov_small', 'recall50_medium', 'recall50_large', 'recall50_all',
        'n_small', 'n_medium', 'n_large')


def main(_):
    rows = []
    for path in sorted(glob.glob(os.path.join(FLAGS.results_dir, '**', '*_summary.json'),
                                 recursive=True)):
        with open(path) as f:
            s = json.load(f)
        if 'metrics' not in s:
            continue
        row = {'file': os.path.relpath(path, FLAGS.results_dir), 'run_name': s.get('run_name'),
               'split': s.get('split'), 'prior_mode': s.get('prior_mode'),
               'injection': s.get('injection'), 'prior_perturb': s.get('prior_perturb'),
               'override': s.get('prior_override'), 'noise': s.get('prior_noise'),
               'GT_LEAKAGE': s.get('GT_LEAKAGE'),
               'lambdas': ';'.join(f'{v:.4f}' for v in s.get('lambdas', []))}
        row.update({k: round(s['metrics'][k], 3) if isinstance(s['metrics'][k], float)
                    else s['metrics'][k] for k in KEYS})
        pq = s.get('prior_quality', {})
        row['prior_soft_iou_weed'] = round(pq['soft_iou_weed'], 4) if pq else ''
        rows.append(row)
    os.makedirs(os.path.dirname(FLAGS.out) or '.', exist_ok=True)
    if rows:
        with open(FLAGS.out, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    train_rows = []
    for path in sorted(glob.glob(os.path.join(FLAGS.runs_dir, '*', 'run_summary.json'))):
        with open(path) as f:
            train_rows.append(json.load(f))
    print(f'{len(rows)} eval summaries -> {FLAGS.out}')
    for r in rows:
        print(f"{r['run_name']:28s} {r['split']:5s} ovr={r['override']:8s} mIoU={r['mIoU']:.2f} "
              f"crop={r['IoU_crop']:.2f} weed={r['IoU_weed']:.2f} SWR={r['SmallWeedRecall']:.2f} "
              f"lam={r['lambdas']}")
    for t in train_rows:
        print('train:', json.dumps(t))


if __name__ == '__main__':
    app.run(main)
