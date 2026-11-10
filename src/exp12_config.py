"""Exp12 configuration registry: a COMMON frozen config plus REGISTERED per-arm configs.

  configs/exp12/exp12_common_config.json : flags that must be identical in every run
  configs/exp12/exp12_arms.json          : flags that may differ, locked per arm BEFORE running

A run is valid only if (i) every common flag equals the frozen value, (ii) every flag registered
for its arm (backbone, head, resolution, schedule, EMA...) equals the registered value, (iii) its seed
is allowed for that arm, (iv) every other training flag equals its default. Torch-free on purpose.

CLI:  python src/exp12_config.py --check_runs='runs/*/run_config.json'
"""
import argparse
import glob
import hashlib
import json
import math
import sys

IGNORED = {'dataset_path', 'num_workers', 'out_dir', 'run_name', 'resume', 'log_interval',
           'ckpt_interval', 'measure_stats', 'detect_anomaly', 'ckpt_resnet', 'arm_id',
           'exp12_common_config', 'exp12_arms', 'seed'}
EVAL_KEYS = ('backbone', 'head', 'use_low_level_refine', 'llr_source', 'spectral_guide',
             'input_scale', 'mcsd_tau', 'sinkhorn_iters', 'use_multiscale_fusion',
             'in_channels', 'num_classes')


def load_json(path):
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def sha256_file(path):
    with open(path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()


def same(a, b):
    if isinstance(a, bool) or isinstance(b, bool):
        return isinstance(a, bool) and isinstance(b, bool) and a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=1e-12)
    return a == b


def expected_flags(common, arms, arm_id):
    if arm_id not in arms['arms']:
        raise KeyError(f'unknown arm {arm_id!r}; registered: {sorted(arms["arms"])}')
    merged = dict(common['common'])
    overlap = set(merged) & set(arms['arms'][arm_id]['flags'])
    if overlap:
        raise ValueError(f'arm {arm_id} redefines common flags: {sorted(overlap)}')
    merged.update(arms['arms'][arm_id]['flags'])
    return merged


def verify_flags(flag_values, common, arms, arm_id, defaults=None, train_flags=None):
    """Return a list of error strings (empty = valid)."""
    errors = []
    try:
        expected = expected_flags(common, arms, arm_id)
    except (KeyError, ValueError) as e:
        return [str(e)]
    for k, v in expected.items():
        if k not in flag_values:
            errors.append(f'flag {k} missing from run')
        elif not same(flag_values[k], v):
            errors.append(f'{k}={flag_values[k]!r} but registered {v!r}')
    seeds = arms['arms'][arm_id].get('seeds')
    if seeds is not None and flag_values.get('seed') not in seeds:
        errors.append(f"seed {flag_values.get('seed')!r} not allowed for arm {arm_id} {seeds}")
    if defaults is not None and train_flags is not None:
        for k in train_flags:
            if k in expected or k in IGNORED:
                continue
            if k in defaults and k in flag_values and not same(flag_values[k], defaults[k]):
                errors.append(f'unregistered flag {k}={flag_values[k]!r} differs from default {defaults[k]!r}')
    return errors


def verify_eval(flag_values, arms, arm_id):
    """Evaluation flags (architecture, scale, TTA, weed bias) must be registered for the arm."""
    errors = []
    if arm_id not in arms['arms']:
        return [f'unknown arm {arm_id!r}']
    spec = arms['arms'][arm_id]
    for k in EVAL_KEYS:
        if k in spec['flags'] and (k not in flag_values or not same(flag_values[k], spec['flags'][k])):
            errors.append(f'{k}={flag_values.get(k)!r} but registered {spec["flags"][k]!r}')
    if flag_values.get('tta', 'none') not in spec.get('allowed_tta', ['none']):
        errors.append(f"tta={flag_values.get('tta')!r} not registered for {arm_id}")
    bias = float(flag_values.get('weed_bias', 0.0))
    if bias != 0.0 and not any(math.isclose(bias, g) for g in arms.get('rp_bias_grid', [])):
        errors.append(f'weed_bias={bias} not in the registered grid {arms.get("rp_bias_grid")}')
    return errors


def _fmt(v):
    return str(v)


def build_cli(common, arms, arm_id, seed, dataset_path, out_dir, run_name,
              common_path, arms_path):
    """Command-line flags for src/train.py. One source of truth: the JSON registry."""
    flags = expected_flags(common, arms, arm_id)
    args = [f'--dataset_path={dataset_path}', f'--seed={seed}', f'--run_name={run_name}',
            f'--out_dir={out_dir}', f'--arm_id={arm_id}', f'--exp12_common_config={common_path}',
            f'--exp12_arms={arms_path}', '--log_interval=50', '--ckpt_interval=1000000',
            '--num_workers=4', '--resume=True']
    args += [f'--{k}={_fmt(v)}' for k, v in flags.items()]
    return args


def eval_cli(arms, arm_id, ckpt, dataset_path, split, bins_json, out_prefix, run_name,
             arms_path, tta='none', weed_bias=0.0):
    spec = arms['arms'][arm_id]['flags']
    args = [f'--dataset_path={dataset_path}', f'--split={split}', f'--ckpt={ckpt}', '--ccmc=False',
            f'--bins_json={bins_json}', '--boundary_px=3', f'--run_name={run_name}',
            f'--out_prefix={out_prefix}', f'--arm_id={arm_id}', f'--exp12_arms={arms_path}',
            f'--tta={tta}', f'--weed_bias={weed_bias}']
    args += [f'--{k}={_fmt(spec[k])}' for k in EVAL_KEYS if k in spec]
    return args


def check_run_config(path, common, arms):
    cfg = load_json(path)
    flags = cfg.get('flags', {})
    arm_id = flags.get('arm_id', '')
    if not arm_id:
        return [f'{path}: no arm_id (not an Exp12 run)']
    return [f'{path}: {e}' for e in verify_flags(flags, common, arms, arm_id)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--check_runs', required=True, help="glob of run_config.json files")
    ap.add_argument('--common', default='configs/exp12/exp12_common_config.json')
    ap.add_argument('--arms', default='configs/exp12/exp12_arms.json')
    a = ap.parse_args()
    common, arms = load_json(a.common), load_json(a.arms)
    paths = sorted(glob.glob(a.check_runs))
    if not paths:
        raise SystemExit('No run_config.json matches --check_runs; refusing false PASS')
    bad = 0
    for p in paths:
        errs = check_run_config(p, common, arms)
        if errs:
            bad += 1
            print('\n'.join(errs))
    print(f'checked {len(paths)} runs, {bad} invalid | common sha256 {sha256_file(a.common)[:16]} '
          f'arms sha256 {sha256_file(a.arms)[:16]}')
    sys.exit(1 if bad else 0)


if __name__ == '__main__':
    main()
