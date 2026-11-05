# Q2 Kaggle Evidence Cells (One-Day Closeout)

> Mục tiêu: trong phiên Kaggle ngày cuối, tạo evidence có thể truy vết cho paper từ code và validation artifacts đã có. **Không chạy test, không chạy lại K8.3/Cell D, không đọc hay đổi `results/exp08/test_started.json`, không chạy WeedMap, không train model mới.** K8.3 đã `ABORTED_AFTER_MARKER_BEFORE_COMPLETED_EVALUATION`, `rerun_authorized=false`; giữ nguyên toàn bộ trạng thái đó.

Tài liệu này thực thi phần khả thi của [README_Q2_CONTRIBUTION_EXPERIMENT_PLAN.md](README_Q2_CONTRIBUTION_EXPERIMENT_PLAN.md) mục 5 theo ưu tiên ngày cuối: A0 → A1 → A2; A3 → A4 chỉ tổng hợp artifacts có sẵn; A5/A6 không chạy. Chạy các code block theo thứ tự trong một Kaggle notebook. **Các cell trong file này chưa được chạy.**

## Quyết định và đầu vào

- Chạy trên Kaggle clone của repository, không chạy trên máy local. A0/A1 không cần dataset hay checkpoint. A2 cần các validation per-image CSV của Exp06–07. A3 cần summary JSON marginal audit Exp07. A4 cần profile JSON Exp04–05.
- Nếu artifacts không nằm trong clone, attach Kaggle Input dataset chứa `exp06_results/results/exp06/` và `exp07_results/results/exp07/` (có thể gồm `exp04_results/results/exp04/`, `exp05_results/results/exp05/`); các cell sẽ dò các thư mục này dưới `/kaggle/input/`. Không tải/khôi phục dataset test để chạy các cell này.
- Mọi output mới chỉ ghi vào `results/q2_audit/<run-id>/`. Không ghi đè artifact Exp00–08. Cell khởi tạo sẽ dừng nếu trùng run-id; bỏ session và tạo run mới nếu cần chạy lại.
- CSV kết quả cũ mang nhãn validation theo tên/split provenance. Cell chỉ phân tích các CSV đó, không load ảnh, label, checkpoint hoặc dataset.
- Không đánh dấu `test_accessed=false` cho toàn bộ lịch sử nghiên cứu. Metadata chỉ khẳng định các cell mới trong file này không mở test data; lịch sử trước đó được mô tả đúng theo audit hiện có.

## Kaggle Preflight (chạy trước Q2-00)

Đúng: các cell Q2 này **kế thừa outputs đã lưu từ E04–E07**, không chạy lại các experiment đó. Trước khi paste/chạy Q2 cells:

1. Mở Kaggle notebook mới và attach các result datasets đã xuất từ trước:
   - Exp06: `exp06_results/results/exp06/`, gồm per-image validation CSV 5 seed của `L0`, `MCSDg`, `PS`, `Dnone`, `LiteSent`, cùng `runs_meta/` nếu có.
   - Exp07: `exp07_results/results/exp07/`, gồm per-image validation CSV của `OccCondg`, marginal audit summary của `MCSDg` và `OccCondg` (seed 0–4, 5 iterations), cùng `runs_meta/`.
   - Exp04/05: `exp04_results/results/exp04/` và `exp05_results/results/exp05/` nếu muốn chạy A4 từ profile artifacts.
   - Attach các Kaggle Input checkpoint datasets hiện có nếu muốn ghi SHA-256 checkpoint trong A0/A2. Các file cần nhận diện được dưới `runs/<method>_s<seed>/best_val.pth`; cell chỉ hash, không load/inference. Thiếu checkpoint/config thì bảng A2 vẫn có thể được tính nhưng trạng thái provenance là `PARTIAL`.
2. Đảm bảo project source có ở `/kaggle/working/WeedsGalore_Ipr28` và chứa source của experiment (`src/nets/mcsd.py`, `src/train.py`, `src/evaluate_ccmc.py`, `src/scripts/compare_runs.py`). Làm theo phần clone của [00_setup_kaggle.md](00_setup_kaggle.md) **chỉ khi checkout chưa có**. Không chạy nguyên Cell S2 nếu nó sẽ `rm -rf` thư mục đang có; tuyệt đối không xóa hoặc thay checkout chứa marker/artifacts Exp08. Ghi nhận commit hiện tại và không sửa source giữa các cell.
3. Kiểm tra dependencies: `torch`, `numpy`, `transformers`, `absl-py`; `matplotlib` chỉ cần cho plot A4. Dùng phần cài package S3 trong setup runbook nếu thiếu, nhưng không cài lại/đổi PyTorch của Kaggle nếu đã hoạt động. Không cần GPU, WeedMap hay raw WeedsGalore dataset cho A0–A4.
4. **Không chạy lại E00–E07**, không chạy setup Cell S4 (dataset), S5 (khởi tạo pretrained model), S6 (full pytest/HF integration) cho phiên evidence-only này. A1 là synthetic operator check riêng; A2–A4 đọc artifacts E đã lưu. Không chạy các cell của Exp08.
5. Chạy cell Q2-PREFLIGHT dưới đây trước, xem trạng thái artifact/dependency. Sau đó chạy Q2-00, A0, A1, A2, A3, A4 và Q2-PACK theo thứ tự, trong cùng một kernel.

```python
from importlib.util import find_spec
from pathlib import Path

PROJECT = Path('/kaggle/working/WeedsGalore_Ipr28')
assert PROJECT.is_dir(), f'Project checkout not found: {PROJECT}'
required_sources = (
    'src/nets/mcsd.py',
    'src/train.py',
    'src/evaluate_ccmc.py',
    'src/scripts/compare_runs.py',
)
missing_sources = [name for name in required_sources if not (PROJECT / name).is_file()]
print('Project source:', 'READY' if not missing_sources else f'MISSING {missing_sources}')

def locate_artifact_dir(experiment):
    relative = Path(f'{experiment}_results') / 'results' / experiment
    candidates = [PROJECT / 'doc' / relative]
    input_root = Path('/kaggle/input')
    if input_root.is_dir():
        candidates.extend(sorted(input_root.glob(f'**/{relative.as_posix()}')))
    return next((path for path in candidates if path.is_dir()), None)

for experiment in ('exp04', 'exp05', 'exp06', 'exp07'):
    root = locate_artifact_dir(experiment)
    if root is None:
        print(experiment, 'NOT FOUND')
        continue
    profiles = len(list(root.glob('profile_*.json')))
    per_image = len(list(root.glob('*_val_per_image.csv')))
    audit_summaries = len(list(root.glob('audit_*_marginal_summary.json')))
    print(experiment, 'FOUND:', root, '| profiles:', profiles,
          '| val CSVs:', per_image, '| marginal summaries:', audit_summaries)

checkpoint_roots = [path for path in Path('/kaggle/input').glob('**/runs') if path.is_dir()]
expected_runs = [f'{method}_s{seed}'
                 for method in ('L0', 'PS', 'MCSDg', 'OccCondg', 'Dnone', 'LiteSent')
                 for seed in range(5)]
checkpoint_count = sum(
    any((root / run / 'best_val.pth').is_file() for root in checkpoint_roots)
    for run in expected_runs
)
print('Existing checkpoint files found:', checkpoint_count, '/', len(expected_runs))

modules = ('torch', 'numpy', 'transformers', 'absl', 'matplotlib')
print('Module availability:', {name: find_spec(name) is not None for name in modules})
print('This preflight does not open a dataset, checkpoint contents, test data, or Exp08 marker.')
```

## Run Order

| Cell         | Công việc                                                                       | Gate                                                           |
| ------------ | ------------------------------------------------------------------------------- | -------------------------------------------------------------- |
| Q2-PREFLIGHT | Kiểm tra project, artifacts và package trên Kaggle                              | Chuẩn bị/attach inputs trước khi chạy Q2-00                    |
| Q2-00        | Tạo output folder riêng, xác định project root                                  | Dừng nếu sai checkout hoặc output tồn tại                      |
| Q2-A0        | Hash và trích source/config làm implementation audit                            | Cần người đọc source xác nhận, cell không tự cấp PASS khoa học |
| Q2-A1        | Kiểm tra tensor/operator synthetic, gradient và residual Sinkhorn               | Mọi assertion pass, nếu fail thì không dùng model claim        |
| Q2-A2        | Recompute paired comparisons từ per-image validation CSV, 5 seed, 10k bootstrap | Chỉ PASS nếu đủ 5 seed, cùng 26 ảnh val, provenance có hash    |
| Q2-A3        | Tổng hợp marginal/occupancy audit JSON đã có, không inference                   | Nếu thiếu artifact/paired IDs thì PARTIAL hoặc NOT RUN         |
| Q2-A4        | Chuẩn hóa profiler JSON đã có và vẽ trade-off khi metadata khớp                 | Không so profile khác shape/device/precision/batch             |
| Q2-PACK      | Ghi gate summary và zip evidence                                                | Giữ nguyên trạng thái A5/A6/test là NOT RUN/BLOCKED            |

**Không chạy** các cell train/evaluate trong runbook Exp06/07, `evaluate_ccmc.py --split=test`, `src/scripts/mcsd_audit.py` hoặc `src/scripts/profile_model.py` trong phiên closeout này. Q2-A3/A4 ở đây chỉ đọc artifact đã lưu.

## Q2-00 — Project và output mới

```python
from datetime import datetime, timezone
from pathlib import Path
import os

PROJECT = Path('/kaggle/working/WeedsGalore_Ipr28')
assert PROJECT.is_dir(), f'Project checkout not found: {PROJECT}'
os.chdir(PROJECT)

RUN_ID = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
OUT = PROJECT / 'results' / 'q2_audit' / RUN_ID
OUT.mkdir(parents=True, exist_ok=False)
(OUT / 'logs').mkdir()

print('Project:', PROJECT)
print('Evidence output:', OUT)
print('No dataset, checkpoint, test image, or test label was opened by this cell.')
```

## Q2-A0 — Source and configuration audit

Cell này không import model, không mở dataset/checkpoint và không tự kết luận công thức đúng. Nó đóng dấu SHA-256, source excerpts có số dòng, cùng các `run_config.json` tìm thấy để người phụ trách đối chiếu trước khi chốt phương trình paper.

```python
import hashlib
import json
import subprocess
from pathlib import Path

def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()

def capture_between(path, start_marker, end_marker):
    lines = Path(path).read_text(encoding='utf-8').splitlines()
    start = next((i for i, line in enumerate(lines) if start_marker in line), None)
    if start is None:
        return None
    end = next((i for i in range(start + 1, len(lines)) if end_marker in lines[i]), None)
    if end is None:
        return None
    return {'start_line': start + 1, 'end_line': end + 1,
            'text': '\n'.join(lines[start:end + 1])}

source_specs = {
    'src/nets/mcsd.py': [
        ('sinkhorn_subcell', 'def sinkhorn_subcell', 'def occupancy_conditioned_subcell'),
        ('occupancy_conditioned_subcell', 'def occupancy_conditioned_subcell', 'class MCSDSegFormer'),
        ('forward_subcell_path', '        if self.guide is not None:', "        return (out, {'occ_logits': occ_logits})"),
        ('occupancy_loss', 'def occupancy_loss', None),
    ],
    'src/train.py': [
        ('auxiliary_loss_flags', '    use_aux =', '    # Train'),
        ('training_forward', '    def forward_train', '    # Train'),
        ('occupancy_auxiliary_loss', '            if use_occ:', '            epoch_steps += 1'),
        ('validation_loop', '        net.eval()', '        val_scores ='),
    ],
    'src/evaluate_ccmc.py': [
        ('test_confirmation_guard', "    if FLAGS.split == 'test'", '    device ='),
    ],
}

source_hashes = {}
snippets = {}
issues = []
for relative_path, specs in source_specs.items():
    path = PROJECT / relative_path
    if not path.is_file():
        issues.append(f'Missing source file: {relative_path}')
        continue
    source_hashes[relative_path] = sha256_file(path)
    snippets[relative_path] = {}
    for name, start_marker, end_marker in specs:
        if end_marker is None:
            lines = path.read_text(encoding='utf-8').splitlines()
            start = next((i for i, line in enumerate(lines) if start_marker in line), None)
            snippet = None if start is None else {
                'start_line': start + 1,
                'end_line': len(lines),
                'text': '\n'.join(lines[start:]),
            }
        else:
            snippet = capture_between(path, start_marker, end_marker)
        snippets[relative_path][name] = snippet
        if snippet is None:
            issues.append(f'Could not capture {relative_path}:{name}')

head = subprocess.run(
    ['git', 'rev-parse', 'HEAD'], cwd=PROJECT, text=True,
    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
)
git_head = head.stdout.strip() if head.returncode == 0 else None

config_roots = [
    PROJECT / 'runs',
    PROJECT / 'doc' / 'exp06_results' / 'results' / 'exp06' / 'runs_meta',
    PROJECT / 'doc' / 'exp07_results' / 'results' / 'exp07' / 'runs_meta',
]
for experiment in ('exp06', 'exp07'):
    relative = Path(f'{experiment}_results') / 'results' / experiment / 'runs_meta'
    kaggle_input = Path('/kaggle/input')
    if kaggle_input.is_dir():
        config_roots.extend(sorted(kaggle_input.glob(f'**/{relative.as_posix()}')))
checkpoint_roots = [PROJECT / 'runs']
kaggle_input = Path('/kaggle/input')
if kaggle_input.is_dir():
    checkpoint_roots.extend(path for path in kaggle_input.glob('**/runs') if path.is_dir())
configs = {}
for method in ('L0', 'PS', 'MCSDg', 'OccCondg', 'Dnone', 'LiteSent'):
    for seed in range(5):
        run = f'{method}_s{seed}'
        candidates = [root / run / 'run_config.json' for root in config_roots]
        config_path = next((path for path in candidates if path.is_file()), None)
        checkpoint_candidates = [root / run / 'best_val.pth' for root in checkpoint_roots]
        checkpoint_path = next((path for path in checkpoint_candidates if path.is_file()), None)
        checkpoint = {
            'status': 'FOUND' if checkpoint_path else 'MISSING',
            'sha256': sha256_file(checkpoint_path) if checkpoint_path else None,
            'path': str(checkpoint_path),
        }
        if config_path is None:
            configs[run] = {'status': 'MISSING', 'checkpoint': checkpoint}
            continue
        config = json.loads(config_path.read_text(encoding='utf-8'))
        flags = config.get('flags', {})
        try:
            config_location = str(config_path.relative_to(PROJECT))
        except ValueError:
            config_location = str(config_path)
        configs[run] = {
            'status': 'FOUND',
            'path': config_location,
            'sha256': sha256_file(config_path),
            'checkpoint': checkpoint,
            'split_hash': config.get('split_hash'),
            'flags': {key: flags.get(key) for key in (
                'seed', 'head', 'ccmc', 'in_channels', 'num_classes',
                'spectral_guide', 'sinkhorn_iters', 'mcsd_tau', 'occ_weight',
            )},
        }

audit = {
    'status': 'SOURCE_CAPTURED_MANUAL_REVIEW_REQUIRED' if not issues else 'PARTIAL',
    'git_head': git_head,
    'source_sha256': source_hashes,
    'source_excerpts': snippets,
    'run_configs': configs,
    'issues': issues,
    'interpretation_checklist': [
        'Confirm OccCond-g computes class-wise softmax(log_q + scores / tau) per sub-cell.',
        'Confirm this normalization does not enforce mean_subcell(P) == q.',
        'Confirm MCSD uses finite log-domain Sinkhorn; columns are normalized each iteration and row marginals are approximate.',
        'Confirm spectral guide is added after pixel_shuffle(scores) and before pixel_unshuffle.',
        'Confirm occupancy_loss(extras[occ_logits], labels, ...) is called inside the train loop only.',
        'Compare each frozen run_config head/guide/iteration/loss flags and split hash with the intended source/checkpoint record.',
        'Keep the paper wording aligned to implementation; do not describe finite Sinkhorn as exact conservation.',
    ],
    'test_data_accessed_by_this_cell': False,
}
(OUT / 'q2_decoder_implementation_audit.json').write_text(
    json.dumps(audit, indent=2), encoding='utf-8'
)
print('Saved:', OUT / 'q2_decoder_implementation_audit.json')
print('Status:', audit['status'])
print('Review source excerpts and each available config before assigning A0 PASS.')
print('Config records found:', sum(item['status'] == 'FOUND' for item in configs.values()), '/', len(configs))
```

**A0 đọc kết quả:** đây là source capture, không phải tự động `PASS`. Nếu source/config không khớp, hạ claim và ghi khác biệt; không sửa checkpoint hay diễn giải lại checkpoint cũ bằng semantics mới. Run config thiếu thì ghi rõ, không tự gán giá trị dự kiến.

## Q2-A1 — Synthetic decoder operators

Kiểm tra primitives trên tensor nhỏ, không instantiate SegFormer, không tải pretrained weights và không cần data. Cell kiểm tra shape output `[B,3,H,W]`, phân phối xác suất, extreme occupancy, gradient hữu hạn và residual theo 1/5/20 Sinkhorn iterations. Residual được ghi nhận; không gọi 5 vòng là exact mass conservation.

```python
import json
import sys
import torch
import torch.nn.functional as F

sys.path.insert(0, str(PROJECT / 'src'))
from nets.mcsd import occupancy_conditioned_subcell, occupancy_loss, sinkhorn_subcell

torch.manual_seed(20261105)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
checks = []
residuals = []

def run_check(name, function):
    try:
        details = function()
        checks.append({'name': name, 'status': 'PASS', 'details': details or {}})
    except Exception as error:
        checks.append({'name': name, 'status': 'FAIL', 'error': repr(error)})

def check_shapes_and_probabilities():
    batch, classes, subcells, height, width = 2, 3, 16, 3, 5
    occ_logits = torch.randn(batch, classes, height, width, device=device)
    log_q = F.log_softmax(occ_logits, dim=1)
    scores = torch.randn(batch, classes, subcells, height, width, device=device)
    outputs = {}
    for name, operator in (
        ('OccCond-g', lambda: occupancy_conditioned_subcell(log_q, scores)),
        ('MCSD-g', lambda: sinkhorn_subcell(log_q, scores, iters=5)),
    ):
        log_p = operator()
        p = log_p.exp()
        assert p.shape == (batch, classes, subcells, height, width), (name, p.shape)
        assert torch.isfinite(log_p).all(), name
        assert torch.all(p >= 0), name
        assert torch.allclose(p.sum(dim=1), torch.ones_like(p[:, 0]), atol=2e-5), name
        pixel_logp = F.pixel_shuffle(log_p.reshape(batch, classes * subcells, height, width), 4)
        pixel_p = pixel_logp.exp()
        assert pixel_logp.shape == (batch, classes, height * 4, width * 4), name
        assert torch.allclose(pixel_p.sum(dim=1), torch.ones_like(pixel_p[:, 0]), atol=2e-5), name
        outputs[name] = list(pixel_logp.shape)
    q = log_q.exp()
    assert torch.all(q >= 0)
    assert torch.allclose(q.sum(dim=1), torch.ones_like(q[:, 0]), atol=1e-6)
    return {'pixel_output_shapes': outputs, 'q_min': float(q.min()), 'q_max': float(q.max())}

def check_extreme_occupancy_and_gradients():
    logits = torch.tensor([0.0, -12.0, -30.0], device=device).view(1, 3, 1, 1).requires_grad_()
    log_q = F.log_softmax(logits, dim=1)
    scores = torch.randn(1, 3, 16, 1, 1, device=device, requires_grad=True)
    position_weights = torch.linspace(0.1, 1.0, 16, device=device).view(1, 1, 16, 1, 1)
    records = {}
    for name, operator in (
        ('OccCond-g', lambda: occupancy_conditioned_subcell(log_q, scores)),
        ('MCSD-g-5', lambda: sinkhorn_subcell(log_q, scores, iters=5)),
    ):
        p = operator().exp()
        assert torch.isfinite(p).all(), name
        objective = (p[:, 2:3] * position_weights).sum()
        grad_logits, grad_scores = torch.autograd.grad(
            objective, (logits, scores), retain_graph=True, allow_unused=False,
        )
        assert torch.isfinite(grad_logits).all(), name
        assert torch.isfinite(grad_scores).all(), name
        assert grad_scores.abs().sum() > 0, name
        records[name] = {
            'minimum_q': float(log_q.exp().min()),
            'gradient_norm_logits': float(grad_logits.norm()),
            'gradient_norm_scores': float(grad_scores.norm()),
        }
    return records

def check_sinkhorn_iteration_residuals():
    occ_logits = torch.randn(1, 3, 2, 4, device=device)
    log_q = F.log_softmax(occ_logits, dim=1)
    scores = torch.randn(1, 3, 16, 2, 4, device=device) * 2.0
    for iterations in (1, 5, 20):
        p = sinkhorn_subcell(log_q, scores, iters=iterations).exp()
        row_target = 16.0 * log_q.exp()
        row_residual = p.sum(dim=2) - row_target
        column_residual = p.sum(dim=1) - 1.0
        assert torch.isfinite(p).all(), f'iters={iterations}'
        assert torch.allclose(p.sum(dim=1), torch.ones_like(p[:, 0]), atol=2e-5), iterations
        residuals.append({
            'iterations': iterations,
            'row_marginal_mae': float(row_residual.abs().mean()),
            'row_marginal_max_abs': float(row_residual.abs().max()),
            'column_marginal_max_abs': float(column_residual.abs().max()),
        })
    return {'residuals': residuals}

def check_occupancy_loss_gradient():
    labels = torch.zeros(1, 8, 12, dtype=torch.long, device=device)
    labels[:, 1:3, 4:7] = 2
    logits = torch.randn(1, 3, 2, 3, device=device, requires_grad=True)
    loss = occupancy_loss(logits, labels, 3)
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(logits.grad).all()
    assert logits.grad.abs().sum() > 0
    return {'loss': float(loss.detach()), 'gradient_norm': float(logits.grad.norm())}

run_check('output_shapes_and_probabilities', check_shapes_and_probabilities)
run_check('extreme_occupancy_and_finite_gradients', check_extreme_occupancy_and_gradients)
run_check('sinkhorn_residuals_1_5_20', check_sinkhorn_iteration_residuals)
run_check('occupancy_loss_gradient', check_occupancy_loss_gradient)

report = {
    'status': 'PASS' if all(item['status'] == 'PASS' for item in checks) else 'FAIL',
    'scope': 'synthetic decoder operators only; no full model, dataset, or checkpoint',
    'device': str(device),
    'checks': checks,
    'test_data_accessed_by_this_cell': False,
}
(OUT / 'q2_decoder_unit_tests.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
print(json.dumps(report, indent=2))
assert report['status'] == 'PASS', 'A1 failed: stop and do not claim unit-test PASS.'
```

## Q2-A2 — Recompute paired validation evidence

Cell dùng đúng `src/scripts/compare_runs.py`: paired image-cluster bootstrap, resample paired seeds, 10,000 replicates. Nó không evaluate model. Bảng này ưu tiên contrast OccCond-g−L0, MCSD-g−L0, MCSD-g−OccCond-g, PS−L0, D½-none−L0 và LiteSent−L0; nguồn là per-image validation CSV của Exp06/07.

```python
import csv
import hashlib
import json
import subprocess
import sys

def find_artifact_dir(experiment):
    relative = Path(f'{experiment}_results') / 'results' / experiment
    candidates = [PROJECT / 'doc' / relative]
    kaggle_input = Path('/kaggle/input')
    if kaggle_input.is_dir():
        candidates.extend(sorted(kaggle_input.glob(f'**/{relative.as_posix()}')))
    result = next((path for path in candidates if path.is_dir()), None)
    if result is None:
        raise FileNotFoundError(f'Cannot locate {relative} under doc/ or /kaggle/input/')
    return result

def artifact_ref(path):
    try:
        return str(Path(path).relative_to(PROJECT))
    except ValueError:
        return str(path)

EXP06 = find_artifact_dir('exp06')
EXP07 = find_artifact_dir('exp07')

def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()

def per_image_files(method):
    roots = [EXP07, EXP06] if method == 'OccCondg' else [EXP06, EXP07]
    files = []
    for seed in range(5):
        filename = f'{method}_s{seed}_val_per_image.csv'
        path = next((root / filename for root in roots if (root / filename).is_file()), None)
        if path is None:
            raise FileNotFoundError(f'Missing validation CSV: {filename}')
        files.append(path)
    return files

contrasts = [
    ('OccCondg', 'L0'),
    ('MCSDg', 'L0'),
    ('MCSDg', 'OccCondg'),
    ('PS', 'L0'),
    ('Dnone', 'L0'),
    ('LiteSent', 'L0'),
]
methods = sorted({method for pair in contrasts for method in pair})
inputs = {method: per_image_files(method) for method in methods}
input_hashes = {}
reference_images = None
for method, paths in inputs.items():
    for seed, path in enumerate(paths):
        with path.open(newline='', encoding='utf-8') as stream:
            names = [row['name'] for row in csv.DictReader(stream)]
        if reference_images is None:
            reference_images = sorted(names)
        assert sorted(names) == reference_images, f'Image mismatch: {path}'
        assert len(names) == 26, f'Expected 26 validation images in {path}, got {len(names)}'
        input_hashes[f'{method}_s{seed}'] = {
            'path': artifact_ref(path),
            'sha256': sha256_file(path),
            'rows': len(names),
            'filename_split_label': 'val',
        }

metric_list = 'SmallWeedRecall,cov_small,mIoU,IoU_weed,IoU_crop,BIoU_weed,PrecisionWeed'
comparison_jsons = []
table_rows = []
for method_a, method_b in contrasts:
    name = f'{method_a}_vs_{method_b}'
    output_json = OUT / f'cmp_{name}.json'
    log_path = OUT / 'logs' / f'cmp_{name}.log'
    command = [
        sys.executable, str(PROJECT / 'src' / 'scripts' / 'compare_runs.py'),
        f"--a={','.join(str(path) for path in inputs[method_a])}",
        f"--b={','.join(str(path) for path in inputs[method_b])}",
        f'--name_a={method_a}', f'--name_b={method_b}',
        '--cluster=image', '--seed_bootstrap=True', '--min_seeds_pass=5',
        '--n_boot=10000', '--seed=0', '--gate=none', f'--metrics={metric_list}',
        f'--out={output_json}',
    ]
    with log_path.open('w', encoding='utf-8') as log:
        subprocess.run(command, cwd=PROJECT, stdout=log, stderr=subprocess.STDOUT,
                       text=True, check=True)
    result = json.loads(output_json.read_text(encoding='utf-8'))
    assert result['n_seeds'] == 5, (name, result['n_seeds'])
    assert result['n_images'] == 26, (name, result['n_images'])
    assert result['seed_bootstrap'] is True, (name, result['seed_bootstrap'])
    assert result['n_boot'] == 10000, (name, result['n_boot'])
    comparison_jsons.append(str(output_json.relative_to(PROJECT)))
    for metric, values in result['metrics'].items():
        table_rows.append({
            'method_A': method_a,
            'method_B': method_b,
            'metric': metric,
            'A_mean': values['A_mean'],
            'B_mean': values['B_mean'],
            'delta_A_minus_B': values['delta'],
            'ci95_low': values['ci95'][0],
            'ci95_high': values['ci95'][1],
            'n_seeds': result['n_seeds'],
            'n_images': result['n_images'],
            'cluster': result['cluster'],
            'n_boot': result['n_boot'],
            'seed_bootstrap': result['seed_bootstrap'],
        })

table_path = OUT / 'q2_validation_main_table.csv'
with table_path.open('w', newline='', encoding='utf-8') as stream:
    writer = csv.DictWriter(stream, fieldnames=list(table_rows[0]))
    writer.writeheader()
    writer.writerows(table_rows)

config_audit_path = OUT / 'q2_decoder_implementation_audit.json'
assert config_audit_path.is_file(), 'Run Q2-A0 before A2.'
config_audit = json.loads(config_audit_path.read_text(encoding='utf-8'))
config_gaps = []
for method in methods:
    for seed in range(5):
        run = f'{method}_s{seed}'
        record = config_audit.get('run_configs', {}).get(run, {})
        if record.get('status') != 'FOUND':
            config_gaps.append(f'{run}: config missing')
        elif record.get('split_hash') != '675667edf3f35f16':
            config_gaps.append(f'{run}: split hash mismatch/missing ({record.get("split_hash")})')
        elif record.get('flags', {}).get('seed') != seed:
            config_gaps.append(f'{run}: seed flag mismatch/missing')
        elif record.get('checkpoint', {}).get('status') != 'FOUND':
            config_gaps.append(f'{run}: checkpoint SHA-256 missing')

evidence = {
    'status': 'PASS' if not config_gaps else 'PARTIAL_CONFIG_PROVENANCE',
    'evaluation_split': 'validation, as identified by source artifact names and experiment records',
    'split_hash_expected': '675667edf3f35f16',
    'methods': methods,
    'contrasts': [list(pair) for pair in contrasts],
    'n_seeds_per_method': 5,
    'n_validation_images': len(reference_images),
    'bootstrap': {'replicates': 10000, 'cluster': 'image', 'paired_seed_resampling': True, 'seed': 0},
    'source_csvs': input_hashes,
    'run_config_crosscheck': 'PASS' if not config_gaps else 'PARTIAL',
    'run_config_gaps': config_gaps,
    'run_config_sha256': {
        run: config_audit['run_configs'].get(run, {}).get('sha256') for run in
        (f'{method}_s{seed}' for method in methods for seed in range(5))
    },
    'checkpoint_sha256': {
        run: config_audit['run_configs'].get(run, {}).get('checkpoint', {}).get('sha256') for run in
        (f'{method}_s{seed}' for method in methods for seed in range(5))
    },
    'comparison_jsons': comparison_jsons,
    'test_data_accessed_by_this_cell': False,
    'note': 'This cell recomputes statistics from saved validation per-image sums; it does not load data or run inference.',
}
(OUT / 'q2_evidence_consistency.json').write_text(json.dumps(evidence, indent=2), encoding='utf-8')
print('Saved:', table_path)
print('Saved:', OUT / 'q2_evidence_consistency.json')
print('Paired comparisons:', len(contrasts), '| seeds:', 5, '| val images:', len(reference_images))
```

**A2 gate:** đủ 5 seed cho từng method, cùng 26 image IDs, hashes được lưu và compare output ghi `seed_bootstrap=true`. Nếu thiếu input/ảnh/seed, cell dừng; không thay bằng seed đơn lẻ, không tự tạo CSV PASS. `split_hash_expected` cần đối chiếu với config metadata A0; nếu config thiếu hoặc lệch thì đổi gate thành `PARTIAL` và ghi mismatch trước khi dùng bảng trong paper.

## Q2-A3 — Reuse marginal/occupancy audit

Cell chỉ đọc các summary marginal audit đã có trong Exp07; không chạy `mcsd_audit.py`, không đọc labels/ảnh và không inference. Nó xuất bảng theo lớp/GT-occupancy bins và predicted-Weed-occupancy bins. Audit instance ID paired giữa MCSD-g và OccCond-g chỉ được báo có nếu đã có artifact ID-level tương ứng; không suy ra ID match từ các bảng aggregate.

```python
import csv
import hashlib
import json

summary_rows = []
source_hashes = {}
missing = []
for method in ('MCSDg', 'OccCondg'):
    for seed in range(5):
        iterations = 5
        source = EXP07 / f'audit_{method}_s{seed}_it{iterations}_marginal_summary.json'
        if not source.is_file():
            missing.append(str(source))
            continue
        source_hashes[artifact_ref(source)] = sha256_file(source)
        summary = json.loads(source.read_text(encoding='utf-8'))
        for group in summary.get('groups', []):
            summary_rows.append({
                'method': method,
                'seed': seed,
                'analysis_type': 'gt_occupancy_by_class',
                'class_id': group.get('class_id'),
                'class_name': group.get('class_name'),
                'occupancy_bin': group.get('gt_occupancy_bin'),
                'n_cells': group.get('n_cells'),
                'residual_mean': group.get('residual_mean'),
                'marginal_mae': group.get('marginal_mae'),
                'occupancy_mae': group.get('occupancy_mae'),
                'marginal_p95_abs': group.get('marginal_p95_abs'),
                'marginal_max_abs': group.get('marginal_max_abs'),
                'image_bootstrap_ci95': json.dumps(group.get('image_bootstrap_ci95')),
                'source_summary': artifact_ref(source),
            })
        for group in summary.get('pred_weed_occupancy_bins', []):
            summary_rows.append({
                'method': method,
                'seed': seed,
                'analysis_type': 'predicted_weed_occupancy',
                'class_id': 2,
                'class_name': 'weed',
                'occupancy_bin': group.get('pred_weed_occupancy_bin'),
                'n_cells': group.get('n_cells'),
                'residual_mean': group.get('residual_mean'),
                'marginal_mae': group.get('marginal_mae'),
                'occupancy_mae': group.get('occupancy_mae'),
                'marginal_p95_abs': group.get('marginal_p95_abs'),
                'marginal_max_abs': group.get('marginal_max_abs'),
                'image_bootstrap_ci95': json.dumps(group.get('image_bootstrap_ci95')),
                'source_summary': artifact_ref(source),
            })

output_csv = OUT / 'q2_occupancy_calibration_bins.csv'
fields = [
    'method', 'seed', 'analysis_type', 'class_id', 'class_name', 'occupancy_bin',
    'n_cells', 'residual_mean', 'marginal_mae', 'occupancy_mae',
    'marginal_p95_abs', 'marginal_max_abs', 'image_bootstrap_ci95', 'source_summary',
]
with output_csv.open('w', newline='', encoding='utf-8') as stream:
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    writer.writerows(summary_rows)

paired_id_candidates = [
    artifact_ref(path)
    for path in EXP07.glob('*')
    if path.is_file() and ('paired' in path.name.lower() or 'instance' in path.name.lower())
]
status = 'PASS_DESCRIPTIVE_BINS' if not missing and summary_rows else (
    'PARTIAL' if summary_rows else 'NOT_RUN_NO_AUDIT_ARTIFACTS'
)
report = {
    'status': status,
    'summary_count': len(source_hashes),
    'missing_summaries': missing,
    'source_sha256': source_hashes,
    'paired_instance_artifact_candidates': paired_id_candidates,
    'paired_instance_id_analysis': 'NOT_ESTABLISHED_BY_THIS_CELL; inspect candidate files before any claim',
    'test_data_accessed_by_this_cell': False,
    'interpretation': 'q_out-q_pred is marginal residual; q_pred-q_GT is occupancy-head error. Do not merge these measures or claim causality.',
}
(OUT / 'q2_occupancy_audit_status.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
print('Saved:', output_csv)
print(json.dumps(report, indent=2))
```

## Q2-A4 — Reuse compute profiles

Cell chuẩn hóa các profile JSON 600×600 đã lưu ở Exp04/05, giữ lại metadata thiết bị/precision/batch và chỉ vẽ trade-off từ nhóm profile có metadata khớp. Đây không phải phép đo latency mới. Không dùng con số THOP 512×512 trong bài Word cũ để so cùng trục với FlopCounter 600×600.

```python
import csv
import hashlib
import json
from collections import Counter, defaultdict

profile_roots = [
    PROJECT / 'doc' / 'exp04_results' / 'results' / 'exp04',
    PROJECT / 'doc' / 'exp05_results' / 'results' / 'exp05',
]
profiles = []
for root in profile_roots:
    if not root.is_dir():
        continue
    for path in sorted(root.glob('profile_*.json')):
        item = json.loads(path.read_text(encoding='utf-8'))
        input_shape = tuple(item.get('input', []))
        key = (input_shape, item.get('device'), item.get('precision'), item.get('batch'))
        profiles.append({
            'method': item.get('label', path.stem.replace('profile_', '')),
            'input_shape': 'x'.join(str(value) for value in input_shape),
            'device': item.get('device'),
            'torch': item.get('torch'),
            'precision': item.get('precision'),
            'batch': item.get('batch'),
            'params_M': item.get('params_M'),
            'GFLOPs_flopcounter': item.get('GFLOPs_flopcounter'),
            'GMACs': item.get('GMACs'),
            'latency_ms_median': item.get('latency_ms_median'),
            'latency_ms_p10': item.get('latency_ms_p10'),
            'latency_ms_p90': item.get('latency_ms_p90'),
            'peak_mem_MB': item.get('peak_mem_MB'),
            'runs': item.get('runs'),
            'warmup': item.get('warmup'),
            'profile_group': repr(key),
            'source_path': artifact_ref(path),
            'source_sha256': sha256_file(path),
            '_group_key': key,
        })

assert profiles, 'No saved profile JSON found; A4 is NOT RUN, do not launch a new profile in this closeout.'
group_counts = Counter(profile['_group_key'] for profile in profiles)
matched_key, matched_count = group_counts.most_common(1)[0]
matched_profiles = [profile for profile in profiles if profile['_group_key'] == matched_key]

profile_csv = OUT / 'q2_compute_tradeoff.csv'
profile_fields = [key for key in profiles[0] if key != '_group_key']
with profile_csv.open('w', newline='', encoding='utf-8') as stream:
    writer = csv.DictWriter(stream, fieldnames=profile_fields)
    writer.writeheader()
    writer.writerows([{key: value for key, value in profile.items() if key != '_group_key'}
                      for profile in profiles])

validation = defaultdict(dict)
main_table = OUT / 'q2_validation_main_table.csv'
if main_table.is_file():
    with main_table.open(newline='', encoding='utf-8') as stream:
        for row in csv.DictReader(stream):
            metric = row['metric']
            for method, field in ((row['method_A'], 'A_mean'), (row['method_B'], 'B_mean')):
                if metric in ('SmallWeedRecall', 'mIoU'):
                    validation[method].setdefault(metric, []).append(float(row[field]))

plot_status = 'NOT_CREATED_NO_MATCHED_VALIDATION_METRICS'
try:
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    plotted = 0
    for profile in matched_profiles:
        method = profile['method']
        recall_values = validation.get(method, {}).get('SmallWeedRecall', [])
        if not recall_values:
            continue
        recall = sum(recall_values) / len(recall_values)
        axes[0].scatter(profile['GFLOPs_flopcounter'], recall)
        axes[0].annotate(method, (profile['GFLOPs_flopcounter'], recall), xytext=(4, 4),
                         textcoords='offset points')
        axes[1].scatter(profile['latency_ms_median'], recall)
        axes[1].annotate(method, (profile['latency_ms_median'], recall), xytext=(4, 4),
                         textcoords='offset points')
        plotted += 1
    if plotted:
        axes[0].set_xlabel('GFLOPs (FlopCounter)')
        axes[0].set_ylabel('Small-Weed Recall (%)')
        axes[0].set_title('Validation recall vs compute')
        axes[1].set_xlabel('Median latency (ms)')
        axes[1].set_ylabel('Small-Weed Recall (%)')
        axes[1].set_title('Validation recall vs latency')
        for axis in axes:
            axis.grid(True, alpha=0.25)
        figure.tight_layout()
        figure.savefig(OUT / 'q2_compute_tradeoff.png', dpi=180)
        plot_status = 'CREATED_FROM_MATCHED_PROFILE_GROUP'
    plt.close(figure)
except ImportError:
    plot_status = 'NOT_CREATED_MATPLOTLIB_UNAVAILABLE'

report = {
    'status': 'PASS_PROFILE_INVENTORY' if matched_count == len(profiles) else 'PARTIAL_PROFILE_GROUPS',
    'profile_count': len(profiles),
    'matched_group_count': matched_count,
    'matched_group': repr(matched_key),
    'plot_status': plot_status,
    'source_note': 'Saved profile artifacts only; this cell does not run or re-time a model.',
    'test_data_accessed_by_this_cell': False,
}
(OUT / 'q2_compute_tradeoff_status.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
print('Saved:', profile_csv)
print(json.dumps(report, indent=2))
```

**A4 interpretation:** chỉ mô tả kết quả profile được lưu và metadata khớp. Nếu thiếu profile OccCond-g hoặc profile không cùng thiết bị/precision/input/batch, ghi limitation; không tự đo lại hoặc nội suy thành số liệu matched. FLOPs của FlopCounter có thể bỏ qua elementwise work theo profile source.

## Q2-PACK — Gate summary and download

Chạy sau các cell đã chọn. Cell tạo gate summary bảo thủ và zip evidence; không đụng `results/exp08/`. A0 luôn chờ manual source/config review; A3 partial không chặn paper nếu được mô tả là exploratory; A5/A6/test vẫn `NOT RUN/BLOCKED`.

```python
import json
import subprocess
from zipfile import ZIP_DEFLATED, ZipFile

def load_status(filename, default):
    path = OUT / filename
    return json.loads(path.read_text(encoding='utf-8')) if path.is_file() else default

a0 = load_status('q2_decoder_implementation_audit.json', {'status': 'NOT_RUN'})
a1 = load_status('q2_decoder_unit_tests.json', {'status': 'NOT_RUN'})
a2 = load_status('q2_evidence_consistency.json', {'status': 'NOT_RUN'})
a3 = load_status('q2_occupancy_audit_status.json', {'status': 'NOT_RUN'})
a4 = load_status('q2_compute_tradeoff_status.json', {'status': 'NOT_RUN'})
head = subprocess.run(
    ['git', 'rev-parse', 'HEAD'], cwd=PROJECT, text=True,
    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
)

gate = {
    'protocol': 'Q2 one-day validation/synthetic evidence closeout',
    'run_id': RUN_ID,
    'git_head': head.stdout.strip() if head.returncode == 0 else None,
    'evaluation_split': 'validation artifacts and synthetic tensors only',
    'test_data_accessed_by_cells_in_this_readme': False,
    'A0_source_audit': a0.get('status'),
    'A0_manual_review_required': True,
    'A1_synthetic_operator_tests': a1.get('status'),
    'A2_paired_validation_recompute': a2.get('status'),
    'A3_occupancy_analysis': a3.get('status'),
    'A4_compute_profiles': a4.get('status'),
    'A5_new_upsampler_baseline': 'NOT_RUN_DEADLINE_AND_PREREGISTRATION_GATE',
    'A6_factorial_ablation': 'NOT_RUN',
    'K8.3_WeedsGalore_test': 'BLOCKED_ABORTED_RERUN_UNAUTHORIZED',
    'WeedMap': 'NOT_AUTHORIZED',
    'paper_claim_boundary': [
        'Do not claim first, SOTA, external-domain validation, or held-out gains for new models.',
        'Treat all Exp00-07 performance evidence as validation evidence.',
        'Describe finite Sinkhorn as approximate marginal matching, not exact conservation.',
        'Do not infer causality from occupancy/marginal associations.',
    ],
}
gate_path = OUT / 'q2_contribution_gate.json'
gate_path.write_text(json.dumps(gate, indent=2), encoding='utf-8')

archive_path = Path('/kaggle/working') / f'q2_audit_{RUN_ID}.zip'
with ZipFile(archive_path, 'w', compression=ZIP_DEFLATED) as archive:
    for path in sorted(OUT.rglob('*')):
        if path.is_file():
            archive.write(path, arcname=str(path.relative_to(PROJECT)))

print(json.dumps(gate, indent=2))
print('Evidence archive:', archive_path)
print('Do not add test outputs or alter the existing Exp08 marker/protocol.')
```

## Gửi lại để paper closeout

Gửi ZIP `q2_audit_<run-id>.zip` cùng với log cell nếu có lỗi. Tôi sẽ kiểm tra hashes, 5-seed paired CI, source/config khớp nhau và claim boundary trước khi cập nhật paper evidence. Nếu A2 thiếu artifacts, gửi nguyên trạng `PARTIAL/FAIL`; không chạy lại test hay thay bằng kết quả một seed.

**Kỳ vọng cuối phiên:** A0 source evidence được duyệt thủ công; A1 synthetic tests pass; A2 tạo bảng chuẩn hóa paired 5-seed validation; A3/A4 báo rõ mức coverage của artifacts có sẵn. Baseline mới và test held-out không thuộc mục tiêu của ngày cuối này.
