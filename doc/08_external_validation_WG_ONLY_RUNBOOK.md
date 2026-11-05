# Exp08 Kaggle Runbook: K8.3-WG-only

## Phạm vi

Runbook này là đường chạy chính cho session Kaggle mới. Không dùng các cell cũ trộn WeedMap với WeedsGalore.

- WeedMap: `NOT_AUTHORIZED`.
- Không chạy WeedMap inference hoặc WeedMap evaluator end-to-end.
- Giữ nguyên toàn bộ WeedMap audit artifact cũ, không ghi đè.
- Chỉ mở WeedsGalore held-out test một lần sau khi manifest đã khóa và commit.
- Không chọn checkpoint, tune model hoặc fit transform từ test data.

Method set đã khóa từ validation: `L0`, `OccCondg`, `MCSDg`; seeds `0, 1, 2, 3, 4`.

## Thứ tự chạy trên Kaggle

1. Session setup.
2. Chạy **Cell A: K8.2b Restore**.
3. Chạy **Cell B: WeedMap amendment**.
4. Chạy **Cell C: K8.2-WG-only Preflight + Manifest**.
5. Commit manifest bằng Git.
6. Chạy **Cell D: K8.3-WG-only One-time Test**.

Không chạy Cell D nếu bất kỳ bước trước đó fail.

## Bằng chứng đã PASS

- Strict loading + synthetic forward: PASS.
- THOP buffers trong RAM: PASS.
- Checkpoint provenance: `15/15` checkpoint SHA-256 PASS.
- Config provenance: `15/15` `run_config.json` PASS.
- WeedMap mask validity: `PENDING`.
- WeedMap radiometric compatibility: `PENDING`.
- WeedMap evaluator end-to-end: chưa được chấp thuận.

## Session setup

Attach các Kaggle Input dataset chứa checkpoint roots:

```text
ccmc-ckpts
exp02-ckpts
exp03-ckpts
exp04-ckpts
exp05-ckpts
exp06-ckpts
exp07-occcond-ckpts
```

WeedsGalore dataset phải có một root chứa `2023-05-25/images`, `semantics`, và `instances`.

```python
from pathlib import Path
import os

PROJECT = Path('/kaggle/working/WeedsGalore_Ipr28')
assert PROJECT.is_dir(), PROJECT
os.chdir(PROJECT)
print('Project:', PROJECT)
```

Cell setup không mở test images/labels.

## Cell A — K8.2b Restore frozen checkpoint/config paths

Cell này chỉ đọc checkpoint/config sources, hash và tạo các path cần thiết trong `runs/`. Không inference, không đọc test images/labels.

```python
from pathlib import Path
import os, json, hashlib, shutil

PROJECT = Path('/kaggle/working/WeedsGalore_Ipr28')
assert PROJECT.is_dir(), PROJECT
os.chdir(PROJECT)
OUT = PROJECT / 'results/exp08'
EVIDENCE = PROJECT / 'doc/exp08_results/results/exp08'
assert not (OUT / 'test_started.json').exists(), 'Test marker exists. STOP.'
assert not (OUT / 'test_lock_manifest.json').exists(), 'Manifest already exists. STOP.'

provenance_path = EVIDENCE / 'checkpoint_provenance_verified.json'
assert provenance_path.is_file(), provenance_path
provenance = json.loads(provenance_path.read_text(encoding='utf-8'))
assert provenance['status'] == 'CHECKPOINT_PROVENANCE_PASS'
assert provenance['checkpoint_count'] == 15

def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

expected_runs = {f'{method}_s{seed}' for method in ('L0', 'OccCondg', 'MCSDg') for seed in range(5)}
assert set(provenance['runs']) == expected_runs
plans = []
for run, record in sorted(provenance['runs'].items()):
    source_ckpt = Path(record['checkpoint_source'])
    source_cfg = Path(record['config_source'])
    assert source_ckpt.is_file(), f'Missing source checkpoint: {source_ckpt}'
    assert source_cfg.is_file(), f'Missing source config: {source_cfg}'
    assert sha256_file(source_ckpt) == record['checkpoint_sha256'], f'Checkpoint SHA mismatch: {run}'
    assert sha256_file(source_cfg) == record['config_sha256'], f'Config SHA mismatch: {run}'
    dest_dir = PROJECT / 'runs' / run
    plans.append((run, source_ckpt, source_cfg, dest_dir / 'best_val.pth', dest_dir / 'run_config.json'))

assert len(plans) == 15
print('Original checkpoint/config sources: 15/15 VERIFIED')
for run, source_ckpt, source_cfg, dest_ckpt, dest_cfg in plans:
    dest_ckpt.parent.mkdir(parents=True, exist_ok=True)
    if dest_ckpt.exists() or dest_ckpt.is_symlink():
        assert sha256_file(dest_ckpt) == sha256_file(source_ckpt), f'Existing checkpoint differs: {run}'
    else:
        os.symlink(str(source_ckpt), str(dest_ckpt))
    if dest_cfg.exists():
        assert sha256_file(dest_cfg) == sha256_file(source_cfg), f'Existing config differs: {run}'
    else:
        shutil.copy2(source_cfg, dest_cfg)
    assert sha256_file(dest_ckpt) == sha256_file(source_ckpt)
    assert sha256_file(dest_cfg) == sha256_file(source_cfg)
    print(run, '| checkpoint OK | config OK')

print('\nRESTORE PASS: 15/15')
print('Checkpoint contents unchanged.')
print('No test inference performed.')
print('K8.3: NOT AUTHORIZED until WG-only manifest is committed.')
```

## Cell B — WeedMap protocol amendment

Cell này công khai thay đổi protocol. Không ghi đè các audit WeedMap cũ.

```python
from pathlib import Path
from datetime import datetime, timezone
import json, os

PROJECT = Path('/kaggle/working/WeedsGalore_Ipr28')
os.chdir(PROJECT)
OUT = PROJECT / 'results/exp08'
OUT.mkdir(parents=True, exist_ok=True)
amendment_path = OUT / 'weedmap_scope_amendment_not_authorized.json'
assert not amendment_path.exists(), 'Amendment already exists. Do not overwrite it.'

amendment = {
    'kind': 'PROTOCOL_AMENDMENT',
    'generated_at_utc': datetime.now(timezone.utc).isoformat(),
    'weedmap_status': 'NOT_AUTHORIZED',
    'scope': 'WeedsGalore-only one-time test; WeedMap excluded from current evaluation',
    'reasons': [
        'Mask validity/polarity is unresolved.',
        'Radiometric compatibility and a fixed source-compatible transform are unresolved.',
        'WeedMap evaluator end-to-end acceptance is not approved.',
    ],
    'preserved_audits': [
        'results/exp08/weedmap_mask_signal_contingency.json',
        'results/exp08/weedmap_radiometry_provenance_gap.json',
        'results/exp08/checkpoint_source_presence_audit.json',
    ],
    'retained_protocol': {
        'methods': ['L0', 'OccCondg', 'MCSDg'],
        'seeds': [0, 1, 2, 3, 4],
        'selection': 'validation-selected checkpoints only',
        'no_test_based_selection': True,
    },
    'test_data_opened': False,
    'inference_run': False,
    'authorization': 'WEEDSGALORE_ONLY; WEEDMAP_NOT_AUTHORIZED',
}
amendment_path.write_text(json.dumps(amendment, indent=2), encoding='utf-8')
print('Saved:', amendment_path)
print('WeedMap status: NOT_AUTHORIZED')
print('No test images/labels or WeedMap data opened.')
```

## Cell C — K8.2-WG-only Preflight + Manifest lock

Cell này chỉ đọc metadata, split files, CLI help và hashes. Không mở test images/labels, không inference, không tạo `test_started.json`.

```python
from pathlib import Path
from datetime import datetime, timezone
import glob, hashlib, json, os, subprocess, sys

PROJECT = Path('/kaggle/working/WeedsGalore_Ipr28')
os.chdir(PROJECT)
OUT = PROJECT / 'results/exp08'
assert not (OUT / 'test_started.json').exists(), 'Test marker exists. STOP.'
assert not (OUT / 'test_lock_manifest.json').exists(), 'Manifest exists. STOP.'
amendment_path = OUT / 'weedmap_scope_amendment_not_authorized.json'
assert amendment_path.is_file(), 'Run Cell B first.'
amendment = json.loads(amendment_path.read_text(encoding='utf-8'))
assert amendment['weedmap_status'] == 'NOT_AUTHORIZED'

def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

specs = {
    'L0': '--ccmc=True --prior_mode=none --use_low_level_refine=False',
    'OccCondg': '--ccmc=False --head=occ_conditioned --spectral_guide=True --mcsd_tau=1.0',
    'MCSDg': '--ccmc=False --head=mcsd --spectral_guide=True --sinkhorn_iters=5 --mcsd_tau=1.0',
}
checkpoints, run_configs = {}, {}
for method in specs:
    checkpoints[method], run_configs[method] = {}, {}
    for seed in range(5):
        run = f'{method}_s{seed}'
        ckpt, config_path = Path(f'runs/{run}/best_val.pth'), Path(f'runs/{run}/run_config.json')
        assert ckpt.is_file(), ckpt
        assert config_path.is_file(), config_path
        checkpoints[method][str(seed)] = {'path': str(ckpt), 'sha256': sha256_file(ckpt)}
        config = json.loads(config_path.read_text(encoding='utf-8'))
        assert config['split_hash'] == '675667edf3f35f16'
        flags = config['flags']
        assert int(flags['seed']) == seed
        assert int(flags['in_channels']) == 5 and int(flags['num_classes']) == 3
        assert bool(flags['use_multiscale_fusion']) is True
        expected_head = {'L0': 'bilinear', 'OccCondg': 'occ_conditioned', 'MCSDg': 'mcsd'}[method]
        observed_head = flags.get('head')
        if observed_head is None:
            assert method == 'L0', f'Missing serialized head for {method}: {config_path}'
            observed_head = 'bilinear'
        assert observed_head == expected_head, f'Method/head mismatch: {config_path}'
        assert bool(flags['ccmc']) is (method == 'L0')
        observed_spectral_guide = flags.get('spectral_guide')
        if observed_spectral_guide is None:
            assert method == 'L0', f'Missing serialized spectral_guide for {method}: {config_path}'
            observed_spectral_guide = False
        assert bool(observed_spectral_guide) is (method != 'L0')
        run_configs[method][str(seed)] = {'path': str(config_path), 'sha256': sha256_file(config_path)}

split_dir = Path('src/datasets/weedsgalore/splits')
split_digest = hashlib.sha256()
for split_name in ('train.txt', 'val.txt', 'test.txt'):
    split_path = split_dir / split_name
    assert split_path.is_file(), split_path
    split_digest.update(split_path.read_bytes())
split_hash = split_digest.hexdigest()[:16]
assert split_hash == '675667edf3f35f16', split_hash

dataset_candidates = sorted(
    [Path(path).parent.parent for path in glob.glob('/kaggle/input/**/2023-05-25/semantics', recursive=True)]
    + ([Path('/kaggle/working/weedsgalore-dataset')]
       if Path('/kaggle/working/weedsgalore-dataset/2023-05-25/semantics').is_dir() else [])
)
assert dataset_candidates, 'WeedsGalore dataset root not found'
dataset_root = dataset_candidates[0]
for subdirectory in ('images', 'semantics', 'instances'):
    assert (dataset_root / '2023-05-25' / subdirectory).is_dir()

def check_help(command, required_flags):
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=False)
    assert all(flag in result.stdout for flag in required_flags), (
        f'Help output missing flags {required_flags}; exit={result.returncode}\n{result.stdout}')

check_help([sys.executable, 'src/evaluate_ccmc.py', '--helpshort'],
           ('dataset_path', 'split', 'ckpt', 'confirm_test_locked'))
check_help([sys.executable, 'src/scripts/compare_runs.py', '--helpshort'],
           ('--a', '--b', '--metrics', '--out'))

source_files = (
    'src/nets/mcsd.py', 'src/nets/segformer.py', 'src/ccmc_common.py',
    'src/ccmc_metrics.py', 'src/evaluate_ccmc.py', 'src/scripts/compare_runs.py',
    'src/train.py', 'src/scripts/make_instance_bins.py', 'src/datasets/weedsgalore/weedsgalore.py',
)
source_hashes = {path: sha256_file(path) for path in source_files}
bins_path = Path('results/instance_bins.json')
if not bins_path.is_file():
    canonical_bins = Path('doc/exp01_results/results/instance_bins.json')
    assert canonical_bins.is_file(), (
        f'Missing instance-bin artifact and canonical source: {canonical_bins}')
    bins_path.parent.mkdir(parents=True, exist_ok=True)
    bins_path.write_bytes(canonical_bins.read_bytes())
    print('Restored train-derived instance bins from:', canonical_bins)
bins_data = json.loads(bins_path.read_text(encoding='utf-8'))
assert bins_data['edges'] == [45.0, 147.0]

manifest = {
    'protocol': 'one-time WeedsGalore test; WG-only frozen external-method comparison',
    'generated_at_utc': datetime.now(timezone.utc).isoformat(),
    'repository_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
    'dataset_root': str(dataset_root),
    'source_files': source_hashes,
    'protocol_amendment': {'path': str(amendment_path), 'sha256': sha256_file(amendment_path), 'weedmap_status': 'NOT_AUTHORIZED'},
    'split_hash': split_hash, 'methods': specs, 'checkpoints': checkpoints, 'run_configs': run_configs,
    'instance_bins': {'path': str(bins_path), 'sha256': sha256_file(bins_path), 'edges': [45.0, 147.0]},
    'evaluation': {
        'split': 'test', 'input_channels': ['R', 'G', 'B', 'NIR', 'RE'], 'boundary_px': 3,
        'primary': 'SmallWeedRecall', 'co_primary': 'cov_small',
        'metrics': ['mIoU', 'IoU_bg', 'IoU_crop', 'IoU_weed', 'BIoU_crop', 'BIoU_weed', 'SmallWeedRecall', 'recall30_small', 'cov_small', 'PrecisionWeed', 'CropToWeedRate'],
        'contrasts': [['OccCondg', 'L0'], ['MCSDg', 'L0'], ['MCSDg', 'OccCondg']],
        'bootstrap': {'replicates': 10000, 'cluster': 'image', 'paired_seed_resampling': True, 'seed': 0},
        'no_test_based_selection': True,
    },
    'weedmap_protocol': {'status': 'NOT_AUTHORIZED', 'excluded_from_current_test': True, 'reason': 'mask validity, radiometric compatibility, and evaluator acceptance unresolved'},
}
manifest_path = OUT / 'test_lock_manifest.json'
manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
print('WG-only PREFLIGHT PASS')
print('Manifest saved:', manifest_path)
print('No test images/labels opened; no test_started.json created.')
```

## Commit manifest trước Cell D

Sau khi Cell C PASS, manifest đúng phải nằm tại:

```text
/kaggle/working/WeedsGalore_Ipr28/results/exp08/test_lock_manifest.json
```

File dưới `doc/exp08_results/results/exp08/` là evidence/archive cũ, không dùng thay cho manifest hiện tại.

Trên Kaggle **không cần `git push`**. Chỉ cần tạo local commit trong Kaggle vì Cell D kiểm tra `git ls-files` và `git show HEAD`.

Chạy cell sau từ project root:

```python
from pathlib import Path
import subprocess

manifest = Path('results/exp08/test_lock_manifest.json')
assert manifest.is_file(), manifest

status = subprocess.run(
    ['git', 'status', '--short', '--', str(manifest)],
    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=True,
)
print('Manifest status before commit:', status.stdout.strip() or 'clean')

subprocess.run(['git', 'config', 'user.name', 'Kaggle Exp08'], check=True)
subprocess.run(['git', 'config', 'user.email', 'kaggle-exp08@localhost'], check=True)
subprocess.run(['git', 'add', str(manifest)], check=True)
subprocess.run(['git', 'commit', '-m', 'Lock WeedsGalore-only test manifest'], check=True)

subprocess.run(
    ['git', 'ls-files', '--error-unmatch', str(manifest)],
    check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
)
print('Manifest is committed and tracked. No git push is required.')
```

Không sửa source, checkpoint, config hoặc manifest sau commit.

```python
from pathlib import Path
import subprocess

manifest = Path('results/exp08/test_lock_manifest.json')
assert manifest.is_file()
subprocess.run(['git', 'ls-files', '--error-unmatch', str(manifest)], check=True,
               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
subprocess.run(['git', 'show', f'HEAD:{manifest.as_posix()}'], check=True,
               stdout=subprocess.PIPE, stderr=subprocess.PIPE)
print('Manifest is tracked and ready for K8.3-WG-only.')
```

### Nếu Cell D báo manifest hash mismatch

Lỗi ở dòng `hashlib.sha256(committed).hexdigest() == sha256_file(manifest_path)` nghĩa là file hiện tại trên disk khác byte-for-byte với file đã commit trong `HEAD`. Không chạy tiếp và không tạo lại manifest.

Chạy cell chẩn đoán này; cell chưa mở test data:

```python
from pathlib import Path
import hashlib, subprocess

manifest = Path('results/exp08/test_lock_manifest.json')

def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()

working = manifest.read_bytes()
committed = subprocess.check_output(
    ['git', 'show', f'HEAD:{manifest.as_posix()}'],
    stderr=subprocess.STDOUT,
)
print('Working SHA  :', sha256_bytes(working))
print('Committed SHA:', sha256_bytes(committed))
print('Git diff:')
subprocess.run(['git', 'diff', '--', str(manifest)], check=False)
```

Nếu bạn không cố ý sửa manifest, khôi phục đúng bản đã commit rồi xác minh lại:

```python
from pathlib import Path
import subprocess

manifest = Path('results/exp08/test_lock_manifest.json')
subprocess.run(['git', 'restore', '--source=HEAD', '--', str(manifest)], check=True)
print('Restored manifest from HEAD:', manifest)
```

Sau đó chạy lại Cell D từ đầu. Cell D chưa tạo `test_started.json` khi dừng ở hash check, nên vẫn có thể chạy lại an toàn.

Nếu manifest đã bị sửa có chủ ý, không restore. Hãy dừng, tạo lại manifest bằng Cell C, commit thành commit mới, rồi chỉ sau đó chạy Cell D.

## Cell D — K8.3-WG-only one-time test

Chỉ chạy sau khi manifest đã commit. Đây là cell duy nhất được phép đọc test images/labels và chạy inference.

```python
from pathlib import Path
import hashlib, json, os, subprocess, sys

PROJECT = Path('/kaggle/working/WeedsGalore_Ipr28')
os.chdir(PROJECT)
OUT, LOGS = Path('results/exp08'), Path('results/exp08/logs')
manifest_path, marker_path = OUT / 'test_lock_manifest.json', OUT / 'test_started.json'
LOGS.mkdir(parents=True, exist_ok=True)
assert manifest_path.is_file()
assert not marker_path.exists(), 'One-time marker exists. Do not rerun.'

def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
assert manifest['protocol_amendment']['weedmap_status'] == 'NOT_AUTHORIZED'
# K8.2-WG-only already locked the manifest. Cell D intentionally does not run Git.

def check_help(command, required_flags):
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, check=False)
    assert all(flag in result.stdout for flag in required_flags), (
        f'Evaluator precheck missing {required_flags}; exit={result.returncode}\n{result.stdout}')

# Metadata-only evaluator precheck; no images, labels, checkpoints, or inference are opened here.
check_help([sys.executable, 'src/evaluate_ccmc.py', '--helpshort'],
           ('dataset_path', 'split', 'ckpt', 'confirm_test_locked'))
check_help([sys.executable, 'src/scripts/compare_runs.py', '--helpshort'],
           ('--a', '--b', '--metrics', '--out'))

for path, expected_hash in manifest['source_files'].items():
    assert Path(path).is_file() and sha256_file(Path(path)) == expected_hash, path
for method, seeds in manifest['checkpoints'].items():
    for seed, item in seeds.items():
        assert Path(item['path']).is_file() and sha256_file(Path(item['path'])) == item['sha256']
        config = manifest['run_configs'][method][seed]
        assert Path(config['path']).is_file() and sha256_file(Path(config['path'])) == config['sha256']

DS = manifest['dataset_root']
assert Path(DS).is_dir()
split_dir = Path('src/datasets/weedsgalore/splits')
split_digest = hashlib.sha256()
for split_name in ('train.txt', 'val.txt', 'test.txt'):
    split_digest.update((split_dir / split_name).read_bytes())
assert split_digest.hexdigest()[:16] == manifest['split_hash']
assert not any(OUT.glob('*_test_per_image.csv'))

def run_live(args, log_path):
    with Path(log_path).open('w', encoding='utf-8', buffering=1) as log_file:
        process = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in process.stdout:
            print(line, end='', flush=True)
            log_file.write(line)
        code = process.wait()
    if code != 0:
        raise subprocess.CalledProcessError(code, args)

with marker_path.open('x', encoding='utf-8') as stream:
    json.dump({'started': True, 'manifest_sha256': sha256_file(manifest_path)}, stream, indent=2)

for method, seeds in manifest['checkpoints'].items():
    for seed, item in seeds.items():
        name = f'{method}_s{seed}_test'
        args = [sys.executable, 'src/evaluate_ccmc.py', f'--dataset_path={DS}', '--split=test', '--confirm_test_locked=True', f'--ckpt={item["path"]}', *manifest['methods'][method].split(), '--bins_json=results/instance_bins.json', '--boundary_px=3', f'--run_name={name}', f'--out_prefix={OUT}/{name}']
        run_live(args, LOGS / f'eval_{name}.log')
        assert (OUT / f'{name}_per_image.csv').is_file()
        assert (OUT / f'{name}_summary.json').is_file()

for method_a, method_b in manifest['evaluation']['contrasts']:
    files_a = ','.join(str(OUT / f'{method_a}_s{seed}_test_per_image.csv') for seed in range(5))
    files_b = ','.join(str(OUT / f'{method_b}_s{seed}_test_per_image.csv') for seed in range(5))
    output = OUT / f'cmp_TEST_{method_a}_vs_{method_b}.json'
    args = [sys.executable, 'src/scripts/compare_runs.py', f'--a={files_a}', f'--b={files_b}', f'--name_a={method_a}', f'--name_b={method_b}', '--gate=none', '--cluster=image', '--seed_bootstrap=True', '--n_boot=10000', '--metrics=' + ','.join(manifest['evaluation']['metrics']), f'--out={output}']
    run_live(args, LOGS / f'cmp_TEST_{method_a}_vs_{method_b}.log')
    assert output.is_file()

print('ONE-TIME WEEDSGALORE-ONLY TEST COMPLETE. Do not rerun.')
```

## Nếu Cell D báo marker đã tồn tại

Lỗi:

```text
AssertionError: One-time marker exists. Do not rerun.
```

nghĩa là một lần test đã được tiêu thụ ở mức protocol. Cell D tạo `test_started.json` **trước** evaluator đầu tiên, nên `git restore` manifest không làm marker biến mất.

Không làm các việc sau:

- Không xóa `results/exp08/test_started.json`.
- Không đổi tên marker để chạy lại.
- Không chạy lại Cell D như một test mới.

Giữ nguyên marker, log và các output đã tạo. Kiểm tra trạng thái chỉ đọc:

```python
from pathlib import Path
import json

OUT = Path('results/exp08')
marker = OUT / 'test_started.json'
assert marker.is_file()
print('Marker:')
print(marker.read_text(encoding='utf-8'))
print('\nEvaluation logs:')
for path in sorted((OUT / 'logs').glob('eval_*_test.log')):
    print(path, path.stat().st_size, 'bytes')
print('\nCompleted per-image outputs:', len(list(OUT.glob('*_test_per_image.csv'))))
print('Completed summaries:', len(list(OUT.glob('*_test_summary.json'))))
```

Nếu test bị lỗi sau marker, ghi nhận đây là một interrupted/failed one-time run và bảo toàn evidence. Chỉ reviewer có thẩm quyền mới quyết định protocol deviation tiếp theo; không tự ý reset marker.

### Trường hợp marker có nhưng mọi output đều bằng zero

Nếu audit cho kết quả:

```text
Completed per-image outputs: 0
Completed summaries: 0
```

thì trạng thái phải được ghi là:

```text
K8.3-WG-only: ABORTED_AFTER_MARKER_BEFORE_COMPLETED_EVALUATION
```

Marker chứng minh one-time release đã được mở, nhưng không chứng minh evaluator đã hoàn tất hoặc đã tạo prediction. Không xóa marker và không chạy lại Cell D trong cùng manifest. Dùng cell read-only sau để lưu audit trạng thái:

```python
from pathlib import Path
from datetime import datetime, timezone
import json

OUT = Path('results/exp08')
marker = OUT / 'test_started.json'
logs = OUT / 'logs'
assert marker.is_file()

audit = {
    'status': 'ABORTED_AFTER_MARKER_BEFORE_COMPLETED_EVALUATION',
    'audited_at_utc': datetime.now(timezone.utc).isoformat(),
    'marker': str(marker),
    'marker_content': json.loads(marker.read_text(encoding='utf-8')),
    'evaluation_log_count': len(list(logs.glob('eval_*_test.log'))),
    'per_image_count': len(list(OUT.glob('*_test_per_image.csv'))),
    'summary_count': len(list(OUT.glob('*_test_summary.json'))),
    'inference_completion_verified': False,
    'rerun_authorized': False,
}
audit_path = OUT / 'k8_3_wg_only_aborted_audit.json'
if audit_path.exists():
    existing = json.loads(audit_path.read_text(encoding='utf-8'))
    print('Abort audit already exists; preserved without overwrite.')
    print(json.dumps(existing, indent=2))
else:
    audit_path.write_text(json.dumps(audit, indent=2), encoding='utf-8')
    print(json.dumps(audit, indent=2))
print('Preserve marker and logs. Do not rerun Cell D.')
```

Nếu muốn chạy lại sau trạng thái này, đó phải là một **protocol deviation được phê duyệt**, với manifest/marker mới và giải thích rõ lần release trước đã abort; không được reset hoặc xóa artifact cũ để giả lập một lần chạy mới.

## Cell E — Closeout attempt K8.3 hiện tại

Cell này chỉ ghi nhận trạng thái đã abort; không chạy inference, không xóa marker và không tạo kết quả giả.

```python
from pathlib import Path
from datetime import datetime, timezone
import json

OUT = Path('results/exp08')
marker = OUT / 'test_started.json'
assert marker.is_file()
closeout_path = OUT / 'k8_3_wg_only_closeout.json'

closeout = {
    'status': 'ABORTED_AFTER_MARKER_BEFORE_COMPLETED_EVALUATION',
    'closed_at_utc': datetime.now(timezone.utc).isoformat(),
    'marker_path': str(marker),
    'marker': json.loads(marker.read_text(encoding='utf-8')),
    'inference_completed': False,
    'per_image_outputs': len(list(OUT.glob('*_test_per_image.csv'))),
    'summary_outputs': len(list(OUT.glob('*_test_summary.json'))),
    'comparison_outputs': len(list(OUT.glob('cmp_TEST_*.json'))),
    'weedmap_status': 'NOT_AUTHORIZED',
    'rerun_authorized': False,
    'next_action': 'Reviewer-approved protocol deviation required before any new one-time release',
}

if closeout_path.exists():
    print('Closeout already exists; preserved without overwrite.')
    print(closeout_path.read_text(encoding='utf-8'))
else:
    closeout_path.write_text(json.dumps(closeout, indent=2), encoding='utf-8')
    print(json.dumps(closeout, indent=2))
print('Current K8.3 attempt is closed as ABORTED, not COMPLETE.')
```

## Không chạy

- WeedMap inference hoặc WeedMap evaluator end-to-end.
- Fit normalization/transform từ WeedMap test tiles.
- Chọn checkpoint sau khi xem test result.
- Cell K8.2/K8.3 cũ yêu cầu WeedMap `GO` hoặc evaluator acceptance `PASS`.
- Bất kỳ test cell nào nếu `test_started.json` đã tồn tại.

## Artifact cần tải sau K8.3

```text
results/exp08/test_lock_manifest.json
results/exp08/weedmap_scope_amendment_not_authorized.json
results/exp08/test_started.json
results/exp08/logs/
results/exp08/*_test_per_image.csv
results/exp08/*_test_summary.json
results/exp08/cmp_TEST_*.json
```

## Checklist hoàn tất Exp08

- [ ] K8.2 WG-only preflight và manifest đã PASS, khóa và commit.
- [ ] Evaluator CLI precheck đã PASS trước marker.
- [ ] K8.3-WG-only đã chạy đúng một lần.
- [ ] Đã xuất 15 `*_test_summary.json`, 15 `*_test_per_image.csv` và 3 `cmp_TEST_*.json`.
- [ ] Đã tổng hợp bảng kết quả, các comparison và phân tích lỗi.
- [ ] Đã viết kết luận Exp08, nêu rõ WeedMap `NOT_AUTHORIZED` và các giới hạn nghiên cứu.

## Câu mô tả trong paper

> WeedsGalore held-out test results were obtained in a one-time, preregistered WG-only evaluation of three validation-selected methods across five seeds. WeedMap external validation was not authorized because mask polarity, radiometric compatibility, and end-to-end evaluator acceptance remained unresolved; all WeedMap audit artifacts were preserved and no WeedMap inference was performed.
