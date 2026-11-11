"""E13 validation gate -> single-shot TEST.

  1. reads per-seed VALIDATION summaries (val_<tta>_summary.json) of the CONFIRMATION seeds
  2. gate passes only if mean val mIoU > threshold (default 86.0)
  3. with --run_test and a passing gate: creates an exclusive marker (fails if it exists, so the
     test can never be run twice) and evaluates split=test ONCE per seed with final_ema.pth.

Weed-bias is always 0 for the test. Never edit the arm/TTA after seeing validation numbers.
"""
import argparse
import json
import math
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from exp12_config import eval_cli, load_json  # noqa: E402


def ci95(xs):
    n = len(xs)
    m = statistics.mean(xs)
    if n < 2:
        return m, float('nan'), float('nan')
    sem = statistics.stdev(xs) / math.sqrt(n)
    try:
        from scipy.stats import t
        crit = float(t.ppf(0.975, n - 1))
    except ImportError:
        crit = {4: 2.776, 5: 2.571}.get(n - 1, 2.0)
    return m, m - crit * sem, m + crit * sem


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', required=True)
    ap.add_argument('--tta', default='d4')
    ap.add_argument('--seeds', default='5,6,7,8,9')
    ap.add_argument('--threshold', type=float, default=86.0)
    ap.add_argument('--results_root', default='results/exp13')
    ap.add_argument('--arms', default='configs/exp13/exp13_arms.json')
    ap.add_argument('--dataset_path', required=True)
    ap.add_argument('--bins_json', required=True)
    ap.add_argument('--run_test', action='store_true')
    a = ap.parse_args()

    seeds = [int(s) for s in a.seeds.split(',')]
    root = ROOT / a.results_root / a.arm
    vals = []
    for s in seeds:
        p = root / f'seed{s}' / f'val_{a.tta}_summary.json'
        if not p.is_file():
            raise SystemExit(f'Missing validation summary: {p}')
        vals.append(float(json.loads(p.read_text())['metrics']['mIoU']))
    mean, lo, hi = ci95(vals)
    passed = mean > a.threshold
    report = {'arm': a.arm, 'tta': a.tta, 'seeds': seeds, 'val_miou': vals, 'mean': mean,
              'ci95': [lo, hi], 'threshold': a.threshold, 'gate': 'PASS' if passed else 'FAIL'}
    print(json.dumps(report, indent=2))
    (root / f'gate_{a.tta}.json').write_text(json.dumps(report, indent=2))
    if not passed:
        print('GATE FAIL: TEST NOT RUN. Report the validation number as is; do not tune and re-gate.')
        raise SystemExit(2)
    if not a.run_test:
        print('GATE PASS. Re-run with --run_test to open TEST once.')
        return

    marker = root / 'TEST_RUN_MARKER.json'
    with open(marker, 'x') as f:  # raises FileExistsError if test was ever started
        json.dump({'gate': report}, f, indent=2)
    arms = load_json(a.arms)
    tests = []
    for s in seeds:
        out = root / f'seed{s}'
        cmd = [sys.executable, 'src/evaluate_ccmc.py'] + eval_cli(
            arms, a.arm, str(out / 'final_ema.pth'), a.dataset_path, 'test', a.bins_json,
            str(out / f'test_{a.tta}'), f'{a.arm}_s{s}_TEST', a.arms, tta=a.tta)
        cmd.append('--confirm_test_locked=True')
        subprocess.run(cmd, cwd=ROOT, check=True)
        tests.append(float(json.loads((out / f'test_{a.tta}_summary.json').read_text())['metrics']['mIoU']))
    tm, tlo, thi = ci95(tests)
    final = {'test_miou': tests, 'mean': tm, 'ci95_over_seeds': [tlo, thi]}
    (root / f'test_result_{a.tta}.json').write_text(json.dumps(final, indent=2))
    print(json.dumps(final, indent=2))
    print('Report this number as is (including if < 86). Do not run test again.')


if __name__ == '__main__':
    main()
