# Exp10 - Occupancy-Guided Intra-Cell Residual Refinement

## Mục tiêu và claim gate

Exp09 không xác nhận OccCond vượt LLR gốc: E9-M1 chỉ đạt `84,519%` so với E9-B0 `84,406%`, chênh lệch `+0,113 pp`, CI95% `[-0,084; +0,317] pp`, H1 `INCONCLUSIVE`. Exp10 vì vậy không thay toàn bộ decoder cuối. Mục tiêu nghiên cứu là kiểm tra một refinement nhỏ có thể đưa validation mIoU vượt `86,00%` hay không; đây là target chưa đạt, không phải kết quả định trước.

Baseline chính của paper vẫn là SegFormer MiT-B0 5-band + LLR của bản Word với `85,49 +/- 0,30%` trên test lịch sử. Exp10 chỉ được dùng để đề xuất phiên bản cải tiến sau khi có evidence validation và test độc lập phù hợp. Không dùng validation mIoU của Exp10 để thay thế số test lịch sử.

**Chưa được duyệt để chạy nguyên xi.** Tài liệu này là specification và runbook kiểm tra; các cell phải dừng ở source contract/unit-test gate nếu implementation chưa được commit/push. Không chạy lại trong tài liệu này.

## Giả thuyết và đóng góp cần kiểm chứng

LLR đã tạo feature ở H/2 và classifier + bilinear của nó giữ được chất lượng phân đoạn tổng thể. Sai số còn lại có thể nằm ở vị trí của bằng chứng lớp bên trong mỗi ô H/2. Exp10 giữ nguyên logits cơ sở `Z_0` và học một residual theo bốn sub-cell:

$$
R_{c,j} = r_{c,j} - \frac{1}{4}\sum_{k=1}^{4}r_{c,k},
\qquad
Z_{c,j} = Z_{0,c,j} + \alpha R_{c,j}.
$$

Trong đó `c` là lớp và `j` là một trong bốn pixel con. Zero-mean chỉ bảo toàn trung bình residual logits theo từng lớp trong một ô; nó không tự động bảo toàn occupancy hay xác suất. Vì vậy Exp10 dùng tên **occupancy-guided**, không dùng “occupancy-constrained”.

Occupancy phải tác động thực vào residual, không chỉ được cộng vào placement scores rồi bị triệt tiêu khi trừ mean. Định nghĩa cho M1:

$$
g_c(q)=1+\beta\,(Cq_c-1),\qquad
Z_{c,j}=Z_{0,c,j}+\alpha\,g_c(q)
\left(s_{c,j}-\frac{1}{4}\sum_{k=1}^{4}s_{c,k}\right),
$$

với `C=3`, `beta=0.5` cố định cho lượt Exp10 đầu tiên, `q=softmax(o)` và `s` là placement scores. Không quét nhiều giá trị beta trên validation; `0.5` là lựa chọn thiết kế đăng ký trước, chưa phải giá trị được chứng minh tối ưu. B1 dùng cùng residual branch nhưng đặt `g_c(q)=1`, nên output B1 không phụ thuộc vào occupancy. M1 dùng occupancy để điều khiển cường độ của residual zero-mean theo lớp; đây là guidance, không phải bảo toàn occupancy chính xác. Unit test phải thay đổi `q` trong khi giữ `s` cố định và đo được thay đổi ở output M1.

Contribution nên được gọi là **Occupancy-Guided Intra-Cell Residual Refinement**, không gọi là ensemble hay gộp hai model. Đây vẫn là một mạng SegFormer duy nhất, end-to-end, với một decoder operator mới. Tuy nhiên residual refinement và occupancy conditioning đều có tiền lệ, nên novelty phải được chứng minh bằng cơ chế, identity test, ablation và giới hạn claim; không được suy ra chỉ từ việc ghép module.

## Kiến trúc khóa

- Giữ nguyên MiT-B0, five-band `R,G,B,NIR,RE`, multi-scale fusion, raw LLR ở H/2, classifier bilinear gốc và toàn bộ protocol Exp09.
- `Z_0` là output logits H x W của đường LLR gốc.
- Residual branch đọc feature H/2 và sinh `3 x 4` placement scores cho mỗi ô; reshape về bốn sub-cell.
- Trừ mean trên bốn sub-cell cho từng lớp, sau đó nhân cường độ theo `g(q)` ở M1 trước khi cộng vào `Z_0`.
- Gate `alpha = sigmoid(a)` hoặc `alpha = sigmoid(a) * alpha_max`; khóa `sigmoid(a) = 0.02` lúc khởi tạo.
- Placement phải khởi tạo ngẫu nhiên rất nhỏ, đề xuất Gaussian `std=1e-3` với bias bằng 0, không khởi tạo toàn bộ bằng 0. Identity được kiểm tra riêng bằng cách chủ động đặt residual về 0 hoặc `alpha=0` trong test.
- Inference không được đọc ground truth. Occupancy auxiliary loss chỉ dùng ở training.
- Khi residual branch bằng 0 hoặc gate bằng 0, output phải bằng chính output B0 trong sai số số học.

### Ma trận đối chứng

| ID     | Kiến trúc                                       | Vai trò                     | Trạng thái lượt đầu                        |
| ------ | ----------------------------------------------- | --------------------------- | ------------------------------------------ |
| E10-B0 | LLR gốc + classifier + bilinear x2              | Baseline bắt buộc           | Train trong protocol Exp10; không reuse E9 |
| E10-B1 | B0 + zero-mean residual 2x2, không conditioning | Kiểm tra residual đơn thuần | Bắt buộc                                   |
| E10-M1 | B0 + occupancy-guided zero-mean residual        | Phương pháp chính           | Bắt buộc                                   |
| E10-M2 | B0 + residual allocation kiểu Sinkhorn          | Ablation phụ                | Không chạy ở lượt đầu nếu quota hạn chế    |

B1 và M1 phải có cùng số tham số, cùng occupancy auxiliary loss, cùng initialization recipe và cùng training protocol; khác biệt duy nhất là B1 đặt `g(q)=1`, còn M1 dùng `g(q)`. B0 phải giữ đúng loss baseline với `occ_weight=0.0`. M2 được đăng ký để kiểm tra sau, không được dùng làm lý do chọn phương pháp nếu chưa chạy đủ B0/B1/M1.

## Protocol và tiêu chí quyết định

Giữ nguyên split hash `675667edf3f35f16`, seeds `0..4`, 100 epochs, batch size 8, Adam `lr=3e-4`, class weights `[0.9, 1.0, 1.3]`, weighted CE + Soft Dice (`0.3`), augmentation và checkpoint selection theo validation mIoU của Exp09. Dùng `spectral_guide=False`; không mở test trong giai đoạn validation.

Primary contrasts:

- H1: E10-M1 - E10-B0 mIoU > 0 với CI95% lower bound > 0.
- H2: E10-M1 - E10-B1 mIoU > 0 với CI95% lower bound > 0.
- Crop-IoU guard: CI95% lower bound > `-1.0 pp`.

`IMPROVED_VS_LLR` là `PASS` chỉ khi H1 đạt và Crop-IoU guard đạt. Nếu CI95% upper bound <= 0 thì `FAIL`; nếu không, `INCONCLUSIVE`. `TARGET_86_REACHED` là kết luận độc lập, chỉ là `PASS` nếu mIoU điểm trung bình của phương pháp chính lớn hơn `86.00%` trên split được báo cáo; nếu chỉ là validation thì phải ghi rõ `VALIDATION_ONLY`, không phải claim test. Hai kết luận này phải được lưu riêng trong JSON. Muốn viết claim vượt baseline test cần một test/external evaluation độc lập, được khóa trước khi chạy.

Secondary metrics: Weed IoU, Crop IoU, Weed BIoU (3 px), Small-Weed Recall@IoU 0.5, `cov_small`, parameters, counted GFLOPs, synchronized latency và peak VRAM. Báo cả delta và CI, không chỉ điểm trung bình.

## Source contract trước khi mở Kaggle

Trước khi chạy các cell, source commit phải có tối thiểu:

- `LLRResidualSubcellSegFormer` trong `src/nets/llr_subcell.py`.
- Một head flag ổn định, đề xuất `llr_residual2` và `llr_residual_occ2`, cùng flags `beta` và `residual_alpha_init`; không cần Sinkhorn ở lượt đầu.
- Tích hợp head vào `src/train.py`, `src/ccmc_common.py` và `src/evaluate_ccmc.py`.
- `tests/test_llr_residual.py` phải kiểm tra shape H x W, zero-mean theo sub-cell, zero-residual identity, occupancy influence, B1 occupancy invariance, finite output và không có GT trong forward. Gradient phải được kiểm tra bằng các loss tách riêng, không gộp thành một assertion chung.
- B1/M1 có cùng parameter count. Nếu khác, phải dừng và sửa trước khi train.

Không nên dùng cell Kaggle để âm thầm sửa source hoặc tạo patch khác với commit đã audit. Kaggle phải chạy đúng commit chứa implementation.

### Bốn gate bắt buộc trước khi train

1. **Identity:** đặt residual bằng 0 hoặc `alpha=0` phải cho output trùng B0 trong sai số số học.
2. **Zero-mean:** với mỗi lớp và mỗi ô H/2, tổng residual sau khi trừ mean trên bốn sub-cell phải bằng 0.
3. **Occupancy influence:** giữ placement `s` cố định, thay `q` phải làm output M1 thay đổi có thể đo được; cùng phép thay đổi đó không được làm output B1 thay đổi.
4. **Gradient:** kiểm tra riêng (a) segmentation loss đến placement head, (b) segmentation loss đến occupancy head của M1, (c) occupancy auxiliary loss đến occupancy head, và (d) segmentation loss đến gate `alpha`; forward/inference không được nhận hoặc đọc ground truth. Với B1, segmentation loss đến occupancy head phải bằng 0 hoặc không tồn tại vì B1 không dùng `q` trong output, trong khi occupancy auxiliary loss vẫn phải truyền gradient.

Nếu bất kỳ gate nào thất bại, không train và không đổi tên phương pháp để che lỗi thiết kế. Đặc biệt, test số 3 nhằm ngăn trường hợp cộng `log(q)` trước phép trừ mean rồi occupancy bị triệt tiêu hoàn toàn.

Với placement toàn zero, gradient segmentation đến occupancy head hoặc gate có thể bằng 0 ở bước đầu vì residual chưa có tín hiệu. Đó là lý do placement dùng random `std=1e-3`. Nếu implementation vẫn cho gradient bằng 0 ở initialization thực tế, test phải báo `FAIL` hoặc ghi rõ `ZERO_AT_INIT`, không được ghi `PASS` chung chung.

## Kaggle cells

### E10-00 - Session và output cô lập

```python
from datetime import datetime, timezone
from pathlib import Path
import glob
import os
import subprocess
import sys

PROJECT = Path('/kaggle/working/WeedsGalore_Ipr28')
assert PROJECT.is_dir(), PROJECT
os.chdir(PROJECT)

RUN_ID = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
OUT = PROJECT / 'results' / 'exp10' / RUN_ID
OUT.mkdir(parents=True, exist_ok=False)
(OUT / 'logs').mkdir()
(OUT / 'runs').mkdir()
(OUT / 'validation').mkdir()

if 'DS' not in globals():
    candidates = sorted(glob.glob('/kaggle/input/**/2023-05-25/semantics', recursive=True))
    assert candidates, 'Attach WeedsGalore before running Exp10.'
    DS = str(Path(candidates[0]).parent.parent)
assert Path(DS).is_dir(), DS

BINS = PROJECT / 'results' / 'instance_bins.json'
assert BINS.is_file(), BINS
assert OUT.parent == PROJECT / 'results' / 'exp10', 'Exp10 output must stay under results/exp10.'

commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
assert (PROJECT / 'src' / 'nets' / 'llr_subcell.py').is_file()
assert 'LLRResidualSubcellSegFormer' in (PROJECT / 'src' / 'nets' / 'llr_subcell.py').read_text(encoding='utf-8')
print('Project:', PROJECT)
print('Commit:', commit)
print('Dataset:', DS)
print('Output:', OUT)
print('Validation only; no test inference is allowed in Exp10 cells.')
```

### E10-P0 - Freeze source and check implementation contract

```python
import hashlib
import json
from pathlib import Path

tracked = [
    PROJECT / 'src' / 'nets' / 'llr_subcell.py',
    PROJECT / 'src' / 'train.py',
    PROJECT / 'src' / 'ccmc_common.py',
    PROJECT / 'src' / 'evaluate_ccmc.py',
    PROJECT / 'tests' / 'test_llr_residual.py',
]
for path in tracked:
    assert path.is_file(), f'Missing committed file: {path}'

source_hashes = {}
for path in tracked:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    source_hashes[str(path.relative_to(PROJECT))] = digest

source_text = {
    relative: (PROJECT / relative).read_text(encoding='utf-8')
    for relative in source_hashes
}
assert 'class LLRResidualSubcellSegFormer' in source_text['src/nets/llr_subcell.py']
assert "LLR_RESIDUAL_HEADS = ('llr_residual2', 'llr_residual_occ2')" in source_text[
    'src/nets/llr_subcell.py']
assert 'beta=0.5' in source_text['src/nets/llr_subcell.py']
assert 'residual_alpha_init=0.02' in source_text['src/nets/llr_subcell.py']

# The train/evaluate modules register the names through *LLR_RESIDUAL_HEADS;
# checking the shared tuple avoids a brittle literal-string assertion.
for relative in ('src/train.py', 'src/ccmc_common.py'):
    content = source_text[relative]
    assert 'LLR_RESIDUAL_HEADS' in content, f'Residual head registry missing in {relative}'
    assert 'LLRResidualSubcellSegFormer' in content, (
        f'Residual model integration missing in {relative}')
assert 'DEFINE_float(\'beta\'' in source_text['src/train.py']
assert 'DEFINE_float(\'residual_alpha_init\'' in source_text['src/train.py']
assert 'define_model_flags' in source_text['src/evaluate_ccmc.py']

report = {
    'status': 'PASS_SOURCE_CONTRACT',
    'git_commit': commit,
    'source_sha256': source_hashes,
    'split_hash_expected': '675667edf3f35f16',
    'evaluation_split': 'validation only',
    'test_accessed': False,
}
(OUT / 'e10_p0_source_contract.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
print(json.dumps(report, indent=2))
```

### E10-P1 - Synthetic correctness gate

```python
import subprocess
import sys

commands = [
    [sys.executable, '-m', 'pytest', '-q', 'tests/test_llr_residual.py'],
    [sys.executable, '-m', 'pytest', '-q', 'tests/test_llr_subcell.py'],
]
outputs = []
for index, command in enumerate(commands):
    log_path = OUT / 'logs' / f'e10_p1_tests_{index}.log'
    result = subprocess.run(command, cwd=PROJECT, stdout=log_path.open('w'),
                            stderr=subprocess.STDOUT, text=True)
    output = log_path.read_text(encoding='utf-8')
    outputs.append({'command': ' '.join(command), 'exit_code': result.returncode,
                    'log': str(log_path.relative_to(PROJECT)), 'output': output})
    print(output)
    assert result.returncode == 0, f'Synthetic gate failed: {command}'

(OUT / 'e10_p1_test_summary.json').write_text(
    json.dumps({'status': 'PASS', 'test_data_accessed': False, 'tests': outputs}, indent=2),
    encoding='utf-8',
)
print('E10-P1 PASS')
```

### E10-P2 - Train B0, B1 và M1 với cùng protocol

Cell này giả định các head flags trong Source contract đã được implement. E10-B0 phải được train trong đúng protocol Exp10 với `occ_weight=0.0`; không reuse checkpoint/CSV E9 vì P0 Exp09 đã loại LiteSent reuse do source không tương thích. `--resume=True` chỉ resume đúng run directory của chính model/seed đó.

```python
import json
import subprocess
import sys

p0 = json.loads((OUT / 'e10_p0_source_contract.json').read_text(encoding='utf-8'))
p1 = json.loads((OUT / 'e10_p1_test_summary.json').read_text(encoding='utf-8'))
assert p0['status'] == 'PASS_SOURCE_CONTRACT'
assert p1['status'] == 'PASS'

common = [
    f'--dataset_path={DS}', '--dataset_size_train=104', '--in_channels=5', '--num_classes=3',
    '--batch_size=8', '--num_workers=4', '--lr=0.0003',
    '--weight_bg=0.9', '--weight_crop=1.0', '--weight_weed=1.3',
    '--use_dice_loss=True', '--dice_weight=0.3', '--epochs=100',
    '--log_interval=50', '--ckpt_interval=1000000', '--pretrained_backbone=True',
    '--use_multiscale_fusion=True', '--spectral_guided=False', '--spectral_guide=False',
    '--measure_stats=False', '--resume=True', '--ccmc=False', '--segformer=True',
    '--use_low_level_refine=True', '--llr_source=raw',
]
heads = {
    'E10B0': ['--head=bilinear', '--occ_weight=0.0'],
    'E10B1': ['--head=llr_residual2', '--occ_weight=0.3', '--beta=0.5', '--residual_alpha_init=0.02'],
    'E10M1': ['--head=llr_residual_occ2', '--occ_weight=0.3', '--beta=0.5', '--residual_alpha_init=0.02'],
}

for seed in range(5):
    for model in ('E10B0', 'E10B1', 'E10M1'):
        run = f'{model}_s{seed}'
        run_dir = OUT / 'runs' / run
        summary = run_dir / 'run_summary.json'
        if summary.is_file():
            print('Already complete:', run)
            continue
        log_path = OUT / 'logs' / f'train_{run}.log'
        command = [sys.executable, 'src/train.py', *common, *heads[model],
                   f'--seed={seed}', f'--run_name={run}', f'--out_dir={run_dir}']
        print('Training:', run)
        with log_path.open('w', encoding='utf-8') as log:
            result = subprocess.run(command, cwd=PROJECT, stdout=log,
                                    stderr=subprocess.STDOUT, text=True)
        if result.returncode != 0:
            print(log_path.read_text(encoding='utf-8')[-4000:])
            raise RuntimeError(f'{run} failed; preserve log and stop.')
        assert summary.is_file(), f'Missing completed summary: {summary}'
print('E10-P2 complete')
```

### E10-P3 - Validation evaluation và paired analysis

```python
import json
import subprocess
import sys

runs = {}
architectures = {
    'E10B0': ['--head=bilinear', '--occ_weight=0.0'],
    'E10B1': ['--head=llr_residual2', '--occ_weight=0.3', '--beta=0.5', '--residual_alpha_init=0.02'],
    'E10M1': ['--head=llr_residual_occ2', '--occ_weight=0.3', '--beta=0.5', '--residual_alpha_init=0.02'],
}
for model, flags in architectures.items():
    for seed in range(5):
        run = f'{model}_s{seed}'
        ckpt = OUT / 'runs' / run / 'best_val.pth'
        assert ckpt.is_file(), ckpt
        prefix = OUT / 'validation' / f'{run}_val'
        command = [
            sys.executable, 'src/evaluate_ccmc.py', f'--dataset_path={DS}', '--split=val',
            f'--ckpt={ckpt}', '--ccmc=False', '--segformer=True',
            '--use_low_level_refine=True', '--llr_source=raw', '--spectral_guide=False',
            *flags, f'--bins_json={BINS}', '--boundary_px=3', f'--run_name={run}',
            f'--out_prefix={prefix}',
        ]
        subprocess.run(command, cwd=PROJECT, check=True)
        assert Path(str(prefix) + '_per_image.csv').is_file()
        assert Path(str(prefix) + '_summary.json').is_file()
        runs[run] = str(prefix)

manifest = {
    'evaluation_split': 'val',
    'test_accessed': False,
    'runs': runs,
    'models': list(architectures),
}
(OUT / 'e10_p3_validation_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')

contrasts = [('E10M1', 'E10B0'), ('E10M1', 'E10B1'), ('E10B1', 'E10B0')]
for method_a, method_b in contrasts:
    paths_a = [f'{runs[f"{method_a}_s{seed}"]}_per_image.csv' for seed in range(5)]
    paths_b = [f'{runs[f"{method_b}_s{seed}"]}_per_image.csv' for seed in range(5)]
    output = OUT / f'cmp_{method_a}_vs_{method_b}.json'
    command = [
        sys.executable, 'src/scripts/compare_runs.py',
        f'--a={",".join(paths_a)}', f'--b={",".join(paths_b)}',
        f'--name_a={method_a}', f'--name_b={method_b}', '--cluster=image',
        '--seed_bootstrap=True', '--n_boot=10000', '--seed=0', '--gate=none',
        '--metrics=SmallWeedRecall,cov_small,mIoU,IoU_weed,IoU_crop,BIoU_weed,PrecisionWeed',
        f'--out={output}',
    ]
    subprocess.run(command, cwd=PROJECT, check=True)

print('E10-P3 complete; inspect all comparison JSON before making a claim.')
```

### E10-P4 - GO/NO-GO report

```python
import json
from pathlib import Path

comparisons = {}
for name in ('E10M1_vs_E10B0', 'E10M1_vs_E10B1', 'E10B1_vs_E10B0'):
    path = OUT / f'cmp_{name}.json'
    assert path.is_file(), path
    comparisons[name] = json.loads(path.read_text(encoding='utf-8'))

def decision(record):
    miou = record['metrics']['mIoU']
    crop = record['metrics']['IoU_crop']
    if miou['ci95'][0] > 0 and crop['ci95'][0] > -1.0:
        return 'PASS'
    if miou['ci95'][1] <= 0 or crop['ci95'][1] <= -1.0:
        return 'FAIL'
    return 'INCONCLUSIVE'

def target_86(record, method='E10M1'):
    mean = record['metrics']['mIoU']['A_mean']
    return 'PASS' if mean > 86.0 else 'FAIL'

m1_b0 = comparisons['E10M1_vs_E10B0']
m1_b1 = comparisons['E10M1_vs_E10B1']
report = {
    'status': decision(m1_b0),
    'IMPROVED_VS_LLR': decision(m1_b0),
    'TARGET_86_REACHED': target_86(m1_b0),
    'target_miou_percent': 86.0,
    'primary': {
        'H1_E10M1_vs_E10B0': decision(m1_b0),
        'H2_E10M1_vs_E10B1': decision(m1_b1),
    },
    'comparisons': comparisons,
    'scope': 'exploratory validation; no held-out test claim',
    'target_scope': 'validation_only',
    'test_accessed': False,
    'interpretation': (
        'Only PASS for H1 with the Crop-IoU guard supports a validated improvement '
        'over the LLR baseline. A validation mean above 86 is not a test claim.'
    ),
}
(OUT / 'e10_go_no_go.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
print(json.dumps(report['primary'], indent=2))
print('IMPROVED_VS_LLR:', report['IMPROVED_VS_LLR'])
print('TARGET_86_REACHED:', report['TARGET_86_REACHED'])
print('Target >86 is not considered verified until an independent evaluation is locked and run.')
```

## Claim boundary và review checklist

- Không gọi Exp10 là ensemble, model fusion hoặc cộng hai checkpoint.
- Không gọi zero-mean residual là bảo toàn xác suất; chỉ gọi đúng là zero-mean logits theo sub-cell.
- Không báo M1 thắng B0 nếu H1 CI95% không có lower bound > 0.
- Không dùng H2 để thay cho H1.
- Không dùng E9 hoặc Exp10 validation để tuyên bố vượt số test `85,49%` của bản Word.
- Nếu M1 không vượt B0 với CI95% phù hợp, kết luận âm tính vẫn là evidence có giá trị: residual nội ô chưa đủ để cải thiện LLR trong protocol này.
- Nếu đạt validation >86, vẫn cần kiểm tra test/external theo protocol khóa trước; khi đó mới cập nhật paper và nêu rõ split, checkpoint, seed, CI và compute.
