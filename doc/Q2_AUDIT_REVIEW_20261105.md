- A4 compute inventory | `PASS_PROFILE_INVENTORY`; 6/6 profiles share 600x600, Tesla T4, FP32, batch 1 | Profiles are DF1, Dnone, LiteSent, MCSD, MCSD-g, and PS. It does not include L0 or OccCond-g, so do not claim matched compute/latency superiority over those methods. |

# Q2 Audit Review and Evidence Closeout

## Scope

Reviewed the Kaggle output run `20261105T161337Z`, copied under `doc/q2_audit_20261105T161337Z/results/q2_audit/20261105T161337Z/`. The copied output includes A0–A4 artifacts and the initial `q2_contribution_gate.json`. This review does not authorize or request test access, an Exp08 rerun, WeedMap evaluation, training, or new inference.

## Review Summary

| Gate                       | Recorded result                                                                           | Review                                                                                                                                                                                                                                                     |
| -------------------------- | ----------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| A0 source audit            | `SOURCE_CAPTURED_MANUAL_REVIEW_REQUIRED`; 15/30 configs found                             | Source excerpts support the implementation description below. The 15 missing configs were not absent from the repository: they are in earlier experiment `runs_meta` folders. Run the reconciliation cell below before treating A2 provenance as complete. |
| A1 synthetic operators     | `PASS`                                                                                    | Shapes, normalized class probabilities, finite outputs/gradients, and occupancy loss gradient passed. This is a primitive-level synthetic check, not a full-model integration test.                                                                        |
| A2 paired validation table | Six contrasts; five seeds; 26 validation images; 10,000 paired seed/image bootstrap draws | Statistics are present and internally labeled validation. The original status is `PARTIAL_CONFIG_PROVENANCE`; do not present provenance as complete until the continuation cell confirms configs and evaluation summaries.                                 |
| A3 occupancy summaries     | `PASS_DESCRIPTIVE_BINS`; 10/10 summaries hashed                                           | Descriptive marginal/occupancy bins are covered. The listed instance CSVs compare MCSD-g inference iterations 1 vs 5, not MCSD-g vs OccCond-g; the requested between-method paired instance-ID analysis remains unestablished.                             |
| A4 compute inventory       | `PASS_PROFILE_INVENTORY`; 6/6 profiles share 600x600, Tesla T4, FP32, batch 1             | Valid matched profile inventory for the six profiled methods. It does not include L0 or OccCond-g, so do not claim matched compute/latency superiority over those methods.                                                                                 |
| A5/A6                      | `NOT RUN`                                                                                 | Appropriate for the one-day closeout. No new baseline or factorial training is needed to finish this evidence pack.                                                                                                                                        |
| Test / WeedMap             | Blocked / not authorized                                                                  | Keep unchanged.                                                                                                                                                                                                                                            |

## Findings for the Paper

- The saved paired validation table reports OccCond-g vs L0: Small-Weed Recall `+10.55 pp` (95% CI `[+6.39, +14.84]`), `cov_small` `+7.55 pp` (`[+2.97, +12.47]`), and mIoU `+0.88 pp` (`[+0.42, +1.54]`). This is validation evidence, not held-out test evidence.
- MCSD-g vs L0 reports Small-Weed Recall `+7.84 pp` (`[+4.46, +11.66]`) and mIoU `+0.93 pp` (`[+0.35, +1.78]`). `cov_small` is `+3.10 pp` with CI spanning zero (`[-0.38, +7.20]`).
- MCSD-g minus OccCond-g reports Small-Weed Recall `-2.71 pp` (`[-6.02, +0.40]`), `cov_small` `-4.45 pp` (`[-7.56, -1.32]`), and mIoU `+0.06 pp` (`[-0.17, +0.36]`); Weed BIoU is `+1.10 pp` (`[+0.16, +2.17]`) and pixel Weed precision is `+0.98 pp` (`[+0.28, +2.19]`). Describe this as an observed sensitivity/boundary trade-off, not a causal effect of Sinkhorn.
- PS vs L0 increases Small-Weed Recall by `+6.06 pp` but reduces mIoU by `-1.28 pp` and Crop IoU by `-2.51 pp`, with the listed CIs excluding zero. This supports the importance of the occupancy/constrained variants relative to unconstrained pixel shuffle, but does not by itself establish novelty.
- Dnone and LiteSent also improve validation metrics in this table. Keep them as resolution/refinement comparators; do not attribute their gains to OccCond-g or MCSD-g.
- A1's extreme-occupancy case has finite but very small gradients (about `1e-12` at `q_min ~= 9.36e-14`). Report numerical finiteness only; do not claim strong learning signal at near-zero occupancy. In the synthetic stress test, 5 Sinkhorn iterations have row-marginal MAE `0.10384` in the recorded mass units, while 20 iterations reduce it to `4.14e-5`. This reinforces “approximate marginal matching,” not exact conservation; these synthetic values are not model-validation metrics.
- The reviewed source implements OccCond-g as `log_softmax(log_q + scores / tau)` over classes independently per sub-cell. It does not constrain the mean sub-cell prediction to equal `q`. MCSD-g uses finite log-domain Sinkhorn; columns are normalized at the end of each iteration and row marginals are approximate. The raw-band spectral guide is added to pixel-shuffled placement scores. `occupancy_loss` is called in the training loop; the shown validation loop does not add that loss. Do not claim exact mass conservation, first/SOTA, external validation, causal mechanism, or held-out gains.
- The saved matched profiles report PS: 13.544 GFLOPs, 14.522 ms median, 3.462M parameters; MCSD-g: 14.097 GFLOPs, 17.746 ms, 3.463M; Dnone: 40.069 GFLOPs, 21.160 ms, 3.605M. These are descriptive values from the same recorded T4/FP32/batch-1/600x600 profile group, not a complete L0/OccCond-g Pareto comparison.
- The reviewed source implements OccCond-g as `log_softmax(log_q + scores / tau)` over classes independently per sub-cell. It does not constrain the mean sub-cell prediction to equal `q`. MCSD-g uses finite log-domain Sinkhorn; columns are normalized at the end of each iteration and row marginals are approximate. The raw-band spectral guide is added to pixel-shuffled placement scores. `occupancy_loss` is called in the training loop; the shown validation loop does not add that loss. Do not claim exact mass conservation, first/SOTA, external validation, causal mechanism, or held-out gains.

## Remaining Provenance Note

The original A0 searched only Exp06/07 `runs_meta` plus Kaggle checkpoint roots. The missing configs can be found in the repository's earlier metadata:

- L0 seeds 0–1 and LiteSent seed 0: Exp01; L0 seed 2: Exp03.
- Dnone seeds 0–2 and LiteSent seeds 1–2: Exp04.
- MCSD-g and PS seeds 0–2: Exp05.
- The remaining seed 3–4 configs and OccCond-g configs are already in Exp06/07 metadata.

The next cell searches these read-only metadata files, checks run name, seed, split hash and method flags, checks each saved validation summary and per-image CSV, verifies the checkpoint hashes already recorded by A0, and validates all six existing comparison JSONs. It **does not rerun bootstrap, evaluate a model, open a dataset, or modify the original A0/A2 files**. It writes a separate reconciliation JSON and an A3 instance-artifact scope inventory under the existing `OUT` folder.

## Next Kaggle Cells

Run these in the same Kaggle kernel if `PROJECT`, `OUT`, and `RUN_ID` still exist. After a kernel restart, define them as shown at the start of the first cell. Do not rerun Q2-00, since that creates a new run directory.

### Q2-A2-PROVENANCE — Reconcile Configs and Existing Validation Outputs

```python
from collections import defaultdict
from pathlib import Path
import csv
import hashlib
import json

PROJECT = Path('/kaggle/working/WeedsGalore_Ipr28')
RUN_ID = '20261105T161337Z'
OUT = PROJECT / 'results' / 'q2_audit' / RUN_ID
assert PROJECT.is_dir() and OUT.is_dir(), (PROJECT, OUT)

def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()

methods = ('L0', 'PS', 'MCSDg', 'OccCondg', 'Dnone', 'LiteSent')
expected_runs = {f'{method}_s{seed}': (method, seed)
                 for method in methods for seed in range(5)}
expected_split_hash = '675667edf3f35f16'
expected_flags = {
    'L0': {'ccmc': True, 'prior_mode': 'none', 'use_low_level_refine': False},
     'PS': {'ccmc': False, 'head': 'pixelshuffle', 'spectral_guide': False,
              'use_low_level_refine': False},
    'MCSDg': {'ccmc': False, 'head': 'mcsd', 'spectral_guide': True,
                  'use_low_level_refine': False, 'sinkhorn_iters': 5,
                  'mcsd_tau': 1.0, 'occ_weight': 0.3},
    'OccCondg': {'ccmc': False, 'head': 'occ_conditioned', 'spectral_guide': True,
                      'use_low_level_refine': False, 'mcsd_tau': 1.0, 'occ_weight': 0.3},
    'Dnone': {'ccmc': False, 'use_low_level_refine': True, 'llr_source': 'none'},
    'LiteSent': {'ccmc': False, 'use_low_level_refine': True, 'llr_source': 'raw'},
}

config_paths = defaultdict(list)
doc_root = PROJECT / 'doc'
for path in doc_root.rglob('run_config.json'):
    if path.parent.name in expected_runs:
        config_paths[path.parent.name].append(path)
runs_root = PROJECT / 'runs'
if runs_root.is_dir():
    for path in runs_root.glob('*/run_config.json'):
        if path.parent.name in expected_runs:
            config_paths[path.parent.name].append(path)
input_root = Path('/kaggle/input')
if input_root.is_dir():
    for pattern in ('**/runs_meta/*/run_config.json', '**/runs/*/run_config.json'):
        for path in input_root.glob(pattern):
            if path.parent.name in expected_runs:
                config_paths[path.parent.name].append(path)

def validate_config(path, method, seed):
    config = json.loads(path.read_text(encoding='utf-8'))
    flags = config.get('flags', {})
    errors = []
    if config.get('split_hash') != expected_split_hash:
        errors.append(f'split_hash={config.get("split_hash")}')
    if int(flags.get('seed', -1)) != seed:
        errors.append(f'seed={flags.get("seed")}')
    if flags.get('in_channels') != 5 or flags.get('num_classes') != 3:
        errors.append('expected 5 input channels and 3 classes')
    if flags.get('use_multiscale_fusion') is not True:
        errors.append('use_multiscale_fusion is not true')
    common_flags = {
        'dataset_size_train': 104,
        'batch_size': 8,
        'lr': 0.0003,
        'weight_bg': 0.9,
        'weight_crop': 1.0,
        'weight_weed': 1.3,
        'use_dice_loss': True,
        'dice_weight': 0.3,
        'epochs': 100,
    }
    for key, expected_value in common_flags.items():
        if flags.get(key) != expected_value:
            errors.append(f'{key}={flags.get(key)!r}, expected {expected_value!r}')
    for key, expected_value in expected_flags[method].items():
        actual = flags.get(key)
        if actual != expected_value:
            errors.append(f'{key}={actual!r}, expected {expected_value!r}')
    expected_names = {f'{method}__s{seed}', f'{method}_s{seed}'}
    if config.get('run_name') not in expected_names:
        errors.append(f'run_name={config.get("run_name")!r}')
    return config, errors

config_records = {}
for run, (method, seed) in expected_runs.items():
    candidates = sorted(set(config_paths.get(run, [])))
    valid = []
    rejected = []
    for path in candidates:
        config, errors = validate_config(path, method, seed)
        record = {
            'path': str(path),
            'sha256': sha256_file(path),
            'git_commit': config.get('env', {}).get('git_commit'),
            'run_name': config.get('run_name'),
            'split_hash': config.get('split_hash'),
            'errors': errors,
        }
        (valid if not errors else rejected).append(record)
    distinct_valid_hashes = sorted({item['sha256'] for item in valid})
    config_records[run] = {
        'status': ('PASS' if len(distinct_valid_hashes) == 1 else
                   'MULTIPLE_VALID_CONFIG_HASHES' if len(distinct_valid_hashes) > 1 else 'MISSING_OR_MISMATCH'),
        'valid_candidates': valid,
        'rejected_candidates': rejected,
    }

a0_path = OUT / 'q2_decoder_implementation_audit.json'
a2_path = OUT / 'q2_evidence_consistency.json'
a0 = json.loads(a0_path.read_text(encoding='utf-8'))
a2 = json.loads(a2_path.read_text(encoding='utf-8'))
source_checks = {}
for relative_path, expected_hash in a0.get('source_sha256', {}).items():
    source_path = PROJECT / relative_path
    actual_hash = sha256_file(source_path) if source_path.is_file() else None
    source_checks[relative_path] = {
        'status': 'PASS' if actual_hash == expected_hash else 'MISMATCH',
        'expected_sha256': expected_hash,
        'actual_sha256': actual_hash,
    }

def find_experiment_dir(experiment):
    relative = Path(f'{experiment}_results') / 'results' / experiment
    candidates = [PROJECT / 'doc' / relative]
    if input_root.is_dir():
        candidates.extend(sorted(input_root.glob(f'**/{relative.as_posix()}')))
    return next((path for path in candidates if path.is_dir()), None)

exp06 = find_experiment_dir('exp06')
exp07 = find_experiment_dir('exp07')
assert exp06 is not None and exp07 is not None, 'Exp06/07 artifacts not found'
experiment_dir = {'OccCondg': exp07, 'L0': exp06, 'PS': exp06,
                  'MCSDg': exp06, 'Dnone': exp06, 'LiteSent': exp06}

run_records = {}
for run, (method, seed) in expected_runs.items():
    root = experiment_dir[method]
    csv_path = root / f'{run}_val_per_image.csv'
    summary_path = root / f'{run}_val_summary.json'
    a0_record = a0.get('run_configs', {}).get(run, {})
    checkpoint = a0_record.get('checkpoint', {})
    issues = []
    if not csv_path.is_file():
        issues.append('per-image CSV missing')
    if not summary_path.is_file():
        issues.append('evaluation summary missing')
    if checkpoint.get('status') != 'FOUND' or not Path(checkpoint.get('path', '')).is_file():
        issues.append('checkpoint missing from original A0 audit')
    else:
        current_checkpoint_hash = sha256_file(checkpoint['path'])
        if current_checkpoint_hash != checkpoint.get('sha256'):
            issues.append('checkpoint SHA-256 changed since A0')
    if csv_path.is_file():
        original_csv_hash = a2.get('source_csvs', {}).get(run, {}).get('sha256')
        if sha256_file(csv_path) != original_csv_hash:
            issues.append('per-image CSV SHA-256 differs from original A2 input')
        with csv_path.open(newline='', encoding='utf-8') as stream:
            csv_rows = list(csv.DictReader(stream))
        if len(csv_rows) != 26 or len({row['name'] for row in csv_rows}) != 26:
            issues.append('per-image CSV does not contain 26 unique image rows')
    summary_info = None
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding='utf-8'))
        expected_head = {'L0': 'bilinear', 'PS': 'pixelshuffle', 'MCSDg': 'mcsd',
                         'OccCondg': 'occ_conditioned', 'Dnone': 'bilinear', 'LiteSent': 'bilinear'}[method]
        ckpt_parts = Path(str(summary.get('ckpt', ''))).parts
        if summary.get('run_name') != run:
            issues.append(f'evaluation summary run_name={summary.get("run_name")!r}')
        if summary.get('split') != 'val' or summary.get('n_images') != 26:
            issues.append('evaluation summary split/image count mismatch')
        if summary.get('GT_LEAKAGE') is not False:
            issues.append('evaluation summary indicates GT leakage or lacks false flag')
        if summary.get('head') != expected_head:
            issues.append(f'evaluation summary head={summary.get("head")!r}')
        if len(ckpt_parts) < 2 or ckpt_parts[-2:] != (run, 'best_val.pth'):
            issues.append(f'evaluation summary checkpoint path={summary.get("ckpt")!r}')
        summary_info = {
            'path': str(summary_path),
            'sha256': sha256_file(summary_path),
            'run_name': summary.get('run_name'),
            'split': summary.get('split'),
            'n_images': summary.get('n_images'),
            'ckpt': summary.get('ckpt'),
            'head': summary.get('head'),
            'GT_LEAKAGE': summary.get('GT_LEAKAGE'),
        }
    config_info = config_records[run]
    if config_info['status'] != 'PASS':
        issues.append(f'configuration audit status={config_info["status"]}')
    run_records[run] = {
        'status': 'PASS' if not issues else 'PARTIAL',
        'issues': issues,
        'config': config_info,
        'per_image_csv': {
            'path': str(csv_path),
            'sha256': sha256_file(csv_path) if csv_path.is_file() else None,
        },
        'evaluation_summary': summary_info,
        'checkpoint': {
            'path': checkpoint.get('path'),
            'sha256': checkpoint.get('sha256'),
        },
    }

comparison_checks = {}
for relative in a2.get('comparison_jsons', []):
    path = PROJECT / relative
    if not path.is_file():
        comparison_checks[relative] = {'status': 'MISSING'}
        continue
    result = json.loads(path.read_text(encoding='utf-8'))
    ok = (result.get('n_seeds') == 5 and result.get('n_images') == 26
          and result.get('n_boot') == 10000 and result.get('seed_bootstrap') is True)
    comparison_checks[relative] = {
        'status': 'PASS' if ok else 'MISMATCH',
        'sha256': sha256_file(path),
        'n_seeds': result.get('n_seeds'),
        'n_images': result.get('n_images'),
        'n_boot': result.get('n_boot'),
        'seed_bootstrap': result.get('seed_bootstrap'),
    }

instance_files = []
for path in sorted(exp07.glob('iteration_audit_*_small_instances.csv')):
    instance_files.append({
        'path': str(path),
        'sha256': sha256_file(path),
        'scope': 'MCSDg inference iteration 1 vs 5; not a MCSDg vs OccCondg comparison',
    })

all_runs_pass = all(item['status'] == 'PASS' for item in run_records.values())
all_comparisons_pass = bool(comparison_checks) and all(
    item['status'] == 'PASS' for item in comparison_checks.values()
)
all_sources_pass = bool(source_checks) and all(
    item['status'] == 'PASS' for item in source_checks.values()
)
report = {
    'status': ('PASS_PROVENANCE_RECONCILED'
               if all_runs_pass and all_comparisons_pass and all_sources_pass else 'PARTIAL'),
    'run_id': RUN_ID,
    'evaluation_split': 'validation only; verified from per-run evaluation summaries',
    'split_hash_expected': expected_split_hash,
    'run_count': len(run_records),
    'run_records': run_records,
    'source_hash_checks': source_checks,
    'comparison_count': len(comparison_checks),
    'comparison_checks': comparison_checks,
    'a3_between_method_paired_instance_ids': 'NOT_ESTABLISHED',
    'a3_existing_instance_artifacts': instance_files,
    'test_data_accessed_by_this_cell': False,
    'model_inference_run': False,
    'bootstrap_recomputed': False,
    'note': 'This cell reconciles existing configs, summaries, CSV hashes, checkpoint hashes and comparison JSONs only; it does not rerun evaluation or bootstrap.',
}
output_path = OUT / 'q2_a2_provenance_reconciliation.json'
output_path.write_text(json.dumps(report, indent=2), encoding='utf-8')
print('Saved:', output_path)
print('Status:', report['status'], '| validated run records:', sum(
    item['status'] == 'PASS' for item in run_records.values()), '/', len(run_records))
print('Validated existing paired comparisons:', sum(
    item['status'] == 'PASS' for item in comparison_checks.values()), '/', len(comparison_checks))
print('A3 between-method instance IDs:', report['a3_between_method_paired_instance_ids'])
```

### Q2-PACK-REVIEWED — Final Gate and Updated Archive

```python
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile
import json
import subprocess

PROJECT = Path('/kaggle/working/WeedsGalore_Ipr28')
RUN_ID = '20261105T161337Z'
OUT = PROJECT / 'results' / 'q2_audit' / RUN_ID
assert OUT.is_dir(), OUT

def read_json(name, default):
    path = OUT / name
    return json.loads(path.read_text(encoding='utf-8')) if path.is_file() else default

a0 = read_json('q2_decoder_implementation_audit.json', {'status': 'NOT_RUN'})
a1 = read_json('q2_decoder_unit_tests.json', {'status': 'NOT_RUN'})
a2 = read_json('q2_evidence_consistency.json', {'status': 'NOT_RUN'})
a2_reconciled = read_json('q2_a2_provenance_reconciliation.json', {'status': 'NOT_RUN'})
a3 = read_json('q2_occupancy_audit_status.json', {'status': 'NOT_RUN'})
a4 = read_json('q2_compute_tradeoff_status.json', {'status': 'NOT_RUN'})
head = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=PROJECT, text=True,
                      stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

gate = {
    'run_id': RUN_ID,
    'git_head': head.stdout.strip() if head.returncode == 0 else None,
    'evaluation_split': 'validation artifacts and synthetic tensors only',
    'A0_source_audit': a0.get('status'),
    'A0_source_excerpt_review': 'REVIEWED; PI approval is not implied',
    'A1_synthetic_tests': a1.get('status'),
    'A2_original_recompute': a2.get('status'),
    'A2_provenance_reconciliation': a2_reconciled.get('status'),
    'A2_bootstrap_rerun': False,
    'A3_descriptive_occupancy_bins': a3.get('status'),
    'A3_MCSDg_vs_OccCondg_paired_instance_analysis': 'NOT_ESTABLISHED',
    'A4_matched_profile_inventory': a4.get('status'),
    'A4_profile_scope': '6 methods profiled; L0 and OccCondg profile absent',
    'A5_new_baseline': 'NOT_RUN',
    'A6_factorial_ablation': 'NOT_RUN',
    'K8.3_WeedsGalore_test': 'BLOCKED_ABORTED_RERUN_UNAUTHORIZED',
    'WeedMap': 'NOT_AUTHORIZED',
    'test_data_accessed_by_closeout_cells': False,
    'paper_claim_boundary': [
        'Report Exp00-07 performance only as validation evidence.',
        'Describe finite Sinkhorn as approximate marginal matching, not exact conservation.',
        'Describe the MCSDg/OccCondg outcome as an observed trade-off, not a causal effect.',
        'Do not claim first, SOTA, external-domain validation, or held-out gains.',
    ],
}
gate_path = OUT / 'q2_contribution_gate_reviewed.json'
gate_path.write_text(json.dumps(gate, indent=2), encoding='utf-8')

archive_path = Path('/kaggle/working') / f'q2_audit_{RUN_ID}_reviewed.zip'
with ZipFile(archive_path, 'w', compression=ZIP_DEFLATED) as archive:
    for path in sorted(OUT.rglob('*')):
        if path.is_file():
            archive.write(path, arcname=str(path.relative_to(PROJECT)))

print(json.dumps(gate, indent=2))
print('Reviewed evidence archive:', archive_path)
```

## Closeout Decision

Run `Q2-A2-PROVENANCE` first; it does not recompute the metrics. If it reports `PASS_PROVENANCE_RECONCILED`, run `Q2-PACK-REVIEWED` and send the reviewed ZIP back for final table/claim review. If it remains `PARTIAL`, preserve that status and send the report; do not substitute a one-seed result or reopen test. No further training or inference is recommended for this deadline closeout.
