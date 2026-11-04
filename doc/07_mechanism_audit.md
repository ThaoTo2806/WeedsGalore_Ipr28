# Exp07 — Kiểm toán cơ chế MCSD-g

> Trạng thái: **kế hoạch trước khi chạy**, chưa mở test. Exp06 xác nhận MCSD-g 5 vòng trên validation; 1 vòng tăng recall/coverage nhưng giảm mIoU/Crop và khớp occupancy kém hơn. Exp07 phân biệt lợi ích của conditioning theo occupancy với lợi ích của ràng buộc Sinkhorn, rồi kiểm tra trade-off theo lớp và occupancy của từng ô.

## Câu hỏi và phạm vi

1. Ràng buộc Sinkhorn có cải thiện chất lượng phân đoạn so với một baseline cũng được điều kiện hóa theo occupancy nhưng không giải bài toán vận chuyển không?
2. Sai số marginal tập trung ở lớp nào, đặc biệt trong các ô có Weed occupancy nhỏ?
3. 1 vòng tăng Small-Weed Recall nhờ giảm bỏ sót thật, hay chủ yếu phân bổ quá nhiều xác suất Weed và đổi lấy false positive/Crop error?

Exp07 chỉ dùng **validation**. Cấu hình đề xuất vẫn là MCSD-g, 5 vòng, τ=1, `occ_weight=0.3`; không chọn 1 hoặc 20 vòng làm cấu hình mới từ kết quả inference. Các kết quả train với số vòng khác nhau, nếu thực hiện, phải được ghi nhãn **EXPLORATORY**, tách khỏi kiểm định chính và không dùng để chọn test configuration.

## E7.0 — Khóa protocol và tái lập

- Dùng bin instance đã khóa `[45, 147]` px, validation split hiện tại, `boundary_px=3`, seed bootstrap paired + image cluster.
- Giữ nguyên loss, augmentation, optimizer, 100 epoch, checkpoint theo val mIoU, decoder/backbone và seed 0–4. Ghi commit, split hash, phiên bản thư viện và cấu hình trong artifacts.
- Không thay MCSD-g đã xác nhận. Unit tests phải xác minh forward/backward `occ_conditioned` với spectral guide, xác suất mỗi sub-pixel tổng bằng 1, `P(c,j)=q_c` khi placement scores bằng nhau, marginal lệch khi scores không đều, segmentation-only gradient tới occupancy/placement/guide, và cùng seed tạo state dict khởi tạo khớp MCSD-g.
- Không chạy test ở bất kỳ cell nào của Exp07.

## E7.1 — Occupancy-conditioned baseline không Sinkhorn

Thêm head/variant riêng, ví dụ `head=occ_conditioned`, dùng cùng occupancy head, placement head, spectral guide 5-band và occupancy loss như MCSD-g. Khác biệt chính là cách tạo xác suất cho mỗi sub-pixel. Với occupancy dự đoán `q_c` và placement score `s_cj`, dùng:

```text
log P(c, j) = log_softmax_c(log(q_c) + s(c, j) / tau)
```

Chuẩn hóa theo lớp độc lập ở từng sub-pixel; **không** chạy Sinkhorn và không ép tổng xác suất lớp qua 16 sub-pixel phải khớp `16*q`. Khi placement scores bằng nhau, đầu ra vẫn theo `q`; khi scores khác nhau, marginal có thể lệch. Đây là occupancy-conditioned, unconstrained sub-pixel normalization. Nó liên quan về mặt toán học tới một bước normalization có prior occupancy, không mô tả hai công thức như không liên quan. Đây không phải PS cũ: PS có occupancy auxiliary loss nhưng occupancy head không điều kiện hóa placement output.

### Train và so sánh

- Train `OccCond-g` seed 0–4, cùng init recipe, 5-band guide, `tau=1`, occupancy loss 0,3 và mọi thiết lập còn lại của MCSD-g. Ghi rõ đây là baseline mới; không tái sử dụng tên PS.
- Đánh giá paired trên validation: `OccCond-g` so với MCSD-g, L0 và PS hiện có. So sánh chính là MCSD-g − OccCond-g; các so sánh còn lại là bối cảnh.
- Báo cáo Small-Weed Recall, `cov_small`, mIoU, Crop/Weed IoU, BIoU, false positive Weed trên GT không-Weed, tổng Weed probability và marginal error. Dùng CI 95% phân tầng seed + ảnh; không chọn biến thể theo test.
- Estimand chính là trung bình paired delta MCSD-g − OccCond-g trên 5 seed, với hierarchical bootstrap paired seed + image-cluster percentile CI 95% (`compare_runs.py --seed_bootstrap=True`). Exp07 không chọn winner và không có gate PASS/FAIL. Chỉ mô tả lợi ích downstream của iterative occupancy-constrained coupling nếu marginal MAE giảm, mIoU delta có CI 95% hoàn toàn trên 0, và hai điều kiện non-inferiority đều đạt: cận dưới CI của Crop IoU delta > −1 pp và cận dưới CI của Small-Weed Recall delta > −1 pp. Biên −1 pp áp dụng trên thang percentage-point cho từng metric. Đây là tiêu chí nghiêm ngặt đã khóa trước khi chạy. Ngay cả khi đạt, kết luận giới hạn ở “iterative occupancy-constrained coupling improves segmentation quality over direct occupancy-conditioned normalization under the evaluated protocol”; không kết luận exact mass conservation là nguyên nhân, vì Sinkhorn hữu hạn vòng chỉ gần bảo toàn marginal.

## E7.2 — Sai số marginal theo lớp và occupancy

Mở rộng `src/scripts/mass_check.py` hoặc thêm script audit để đo trên checkpoint MCSD-g seed 0–4 ở inference 1, 5, 20 vòng. Với mỗi lớp `c`, mỗi ô 4×4 có:

```text
q_pred(c) = softmax(occ_logits)(c)
q_out(c)  = average_pool_4x4(P_c)
residual(c) = q_out(c) - q_pred(c)
```

Ghi theo lớp và theo tầng occupancy GT của từng ô: 0/16, 1–3/16, 4–7/16, 8–16/16 pixel của lớp. Đồng thời, riêng Weed, phân tầng theo occupancy dự đoán đã khóa trước: `[0, 0.01)`, `[0.01, 0.05)`, `[0.05, 0.15)`, `[0.15, 1]`. Nhóm GT 1–3 Weed pixel/ô và nhóm `q_pred_weed < 0.01` là trọng tâm kiểm toán. Cho từng lớp/tầng, báo số ô, signed bias, MAE, p95 absolute error và max absolute error; bootstrap CI theo ảnh cho metric tổng hợp. Đánh giá đồng thời `q_pred` so với occupancy GT để tách lỗi occupancy head khỏi lỗi coupling Sinkhorn.

Không gộp các đại lượng sau thành một “mass error” duy nhất:

- `q_out - q_pred`: mức độ đầu ra tuân theo occupancy dự đoán; đây là sai số marginal gần đúng của Sinkhorn ở MCSD-g và độ lệch unconstrained ở OccCond-g.
- `q_pred - q_GT`: lỗi occupancy head so với tỉ lệ lớp thật trong ô.
- Tổng xác suất Weed / tổng pixel Weed GT: độ lệch tổng lượng dự đoán so với nhãn; không đo bảo toàn theo occupancy head.

## E7.3 — Phân tích vì sao 1 vòng tăng recall

Đây là phân tích giải thích, không phải xác nhận một cơ chế nhân quả. Dùng cùng checkpoint 5-vòng cho mọi lượt suy luận; ghép output theo seed, ảnh và instance. Với 1 vs 5 vòng, phân rã thay đổi thành:

- instance TP/FN theo recall@IoU 0,5 và recall@0,3; false positive được đo ở mức pixel bằng Background/Crop→Weed, không diễn giải thành FP-instance;
- coverage instance và số pixel dự đoán Weed trên vùng GT không-Weed;
- confusion Weed→Background, Background/Crop→Weed, Weed pixel precision, Crop→Weed rate, Crop IoU và boundary IoU;
- các đại lượng trên theo GT Weed occupancy 1–3/16, 4–7/16, 8–16/16 trong ô và theo ngày bay.

Tính `Precision_Weed = TP_Weed / (TP_Weed + FP_Weed)` từ confusion matrix pixel và `Crop→Weed rate = cm_12 / (cm_10 + cm_11 + cm_12)`; báo thêm số pixel Crop→Weed. Đối chiếu instance theo cùng GT instance ID để đếm cỏ nhỏ mới được match ở 1 vòng mà chưa match ở 5 vòng, và cỏ nhỏ bị mất match theo chiều ngược lại. Kiểm tra mối liên hệ giữa marginal residual Weed, tổng lượng Weed dự đoán, số instance được khôi phục và phần suy giảm Crop/mIoU. Báo kết quả theo từng seed/ảnh, không suy luận từ một ví dụ trực quan. Giả thuyết cần kiểm tra là một vòng chưa hội tụ làm lệch row marginals, từ đó phân bổ nhiều xác suất Weed hơn cho một số ô; nếu pixel false positives hoặc Crop→Weed tăng cùng recall thì mô tả trade-off sensitivity/specificity, không gọi đó là cải thiện phát hiện thuần túy. Pixel false positives không được gọi là FP instance.

## E7.4 — Thử nghiệm train với số vòng khác nhau (thăm dò)

Chỉ chạy nếu E7.1–E7.3 cho thấy câu hỏi này còn cần thiết. Nếu chạy, đánh dấu tên run, bảng và report là **EXPLORATORY — TRAINING ITERATION ABLATION**. Đăng ký trước số seed, thứ tự train, metric và cách diễn giải trước khi chạy; giữ nguyên mọi thứ ngoài `sinkhorn_iters`. Không dùng kết quả để thay cấu hình MCSD-g đã xác nhận hoặc để tuyên bố confirmatory. Phân biệt rõ với E6 inference-only, vốn chỉ đổi số vòng trên cùng checkpoint.

## Kaggle cells

Các cell dưới đây không chạy test. Trước khi chạy, commit/push code mới (`occ_conditioned` head, `mcsd_audit.py`, `mcsd_iteration_audit.py`) rồi clone đúng commit đó trên Kaggle. Gắn dataset WeedsGalore, checkpoint datasets dùng ở Exp06 (`ccmc-ckpts`, `exp03-ckpts`, `exp04-ckpts`, `exp05-ckpts`, `exp06-ckpts`) và một dataset `exp06-results` có chứa các file đã giải nén `results/exp06/*_val_per_image.csv`/`*_val_summary.json`. Cell sẽ tái dùng validation artifacts Exp06, không evaluate lại các model cũ. Biến `DS` cần trỏ tới thư mục gốc `weedsgalore-dataset`, như các notebook Exp01–06.

### Cell K7.0 — Setup, test và checkpoint check

```python

# ============================================================
# Cell K7.0 — Setup, test và checkpoint check
# ============================================================

import glob
import json
import os
import shutil
import subprocess
import sys

# ------------------------------------------------------------
# 1. Project setup
# ------------------------------------------------------------

PROJECT = '/kaggle/working/WeedsGalore_Ipr28'
os.chdir(PROJECT)

if 'DS' not in globals():
    candidates = glob.glob(
        '/kaggle/input/**/weedsgalore-dataset',
        recursive=True
    )
    assert candidates, (
        'Cannot find weedsgalore-dataset under /kaggle/input'
    )
    DS = candidates[0]

assert os.path.isdir(DS), f'Invalid dataset path: {DS}'

EXP = 'exp07'
OUT = f'results/{EXP}'
BINS = 'results/instance_bins.json'

os.makedirs(f'{OUT}/logs', exist_ok=True)


# ------------------------------------------------------------
# 2. Helper function
# ------------------------------------------------------------

def run_checked(args, log_path):
    with open(log_path, 'w') as log_file:
        subprocess.run(
            args,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            check=True
        )


# ------------------------------------------------------------
# 3. Unit tests
# ------------------------------------------------------------

print('=' * 60)
print('STEP 1: Running unit and integration tests')
print('=' * 60)

run_checked(
    [
        sys.executable,
        '-m',
        'pytest',
        '-q',
        'tests/test_ccmc_sr.py'
    ],
    f'{OUT}/logs/unit_tests.log'
)

with open(f'{OUT}/logs/unit_tests.log') as f:
    test_output = f.read().lower()

print(test_output)

assert ' skipped' not in test_output, (
    'HF-backed integration tests were skipped; '
    'do not start training'
)

print('PASS: Unit and integration tests')


# ------------------------------------------------------------
# 4. Link Exp06 checkpoints
# ------------------------------------------------------------

print('\n' + '=' * 60)
print('STEP 2: Checking Exp06 checkpoints')
print('=' * 60)

checkpoint_candidates = glob.glob(
    '/kaggle/input/**/runs/*/best_val.pth',
    recursive=True
)

for path in checkpoint_candidates:
    run_dir = os.path.basename(os.path.dirname(path))
    target_dir = os.path.join('runs', run_dir)

    os.makedirs(target_dir, exist_ok=True)

    target = os.path.join(target_dir, 'best_val.pth')

    if not os.path.exists(target):
        os.symlink(path, target)

required = [
    f'{model}_s{seed}'
    for model in ('L0', 'MCSDg', 'PS')
    for seed in range(5)
]

missing = [
    run for run in required
    if not os.path.isfile(f'runs/{run}/best_val.pth')
]

assert not missing, (
    f'Missing Exp06 reference checkpoints: {missing}'
)

print('PASS: All 15 Exp06 checkpoints verified')


# ------------------------------------------------------------
# 5. Exp06 validation results
# ------------------------------------------------------------

print('\n' + '=' * 60)
print('STEP 3: Checking Exp06 validation results')
print('=' * 60)

# Actual existing directory in Kaggle working
EXP06_SOURCE = (
    '/kaggle/working/WeedsGalore_Ipr28/'
    'doc/exp06_results/results/exp06'
)

assert os.path.isdir(EXP06_SOURCE), (
    f'Exp06 results directory not found: {EXP06_SOURCE}'
)

required_results = [
    f'{model}_s{seed}_val_{suffix}'
    for model in (
        'L0',
        'PS',
        'MCSDg',
        'MCSDgIt1',
        'MCSDgIt20'
    )
    for seed in range(5)
    for suffix in (
        'per_image.csv',
        'summary.json'
    )
]

missing_results = [
    name for name in required_results
    if not os.path.isfile(
        os.path.join(EXP06_SOURCE, name)
    )
]

assert not missing_results, (
    f'Missing Exp06 validation files: {missing_results}'
)

print('PASS: All 50 source validation files found')


# ------------------------------------------------------------
# 6. Copy Exp06 reference results into Exp07
# ------------------------------------------------------------

EXP06_RESULTS = f'{OUT}/exp06_reference'

os.makedirs(EXP06_RESULTS, exist_ok=True)

for filename in required_results:
    shutil.copy2(
        os.path.join(EXP06_SOURCE, filename),
        os.path.join(EXP06_RESULTS, filename)
    )

copied_files = [
    f for f in required_results
    if os.path.isfile(
        os.path.join(EXP06_RESULTS, f)
    )
]

assert len(copied_files) == 50, (
    f'Expected 50 copied files, got {len(copied_files)}'
)

print('PASS: Exp06 reference files copied:', len(copied_files))
print('Reference directory:', EXP06_RESULTS)


# ------------------------------------------------------------
# 7. Generate instance bins
# ------------------------------------------------------------

print('\n' + '=' * 60)
print('STEP 4: Generating instance bins')
print('=' * 60)

run_checked(
    [
        sys.executable,
        'src/scripts/make_instance_bins.py',
        f'--dataset_path={DS}',
        f'--out={BINS}'
    ],
    f'{OUT}/logs/bins.log'
)

assert os.path.isfile(BINS), (
    f'Instance bins file not generated: {BINS}'
)

with open(BINS) as f:
    bins = json.load(f)

assert bins['edges'] == [45.0, 147.0], (
    f'Unexpected instance bin edges: {bins["edges"]}'
)

print('PASS: Instance bins verified:', bins['edges'])


# ------------------------------------------------------------
# 8. Final checkpoint
# ------------------------------------------------------------

print('\n' + '=' * 60)
print('CELL K7.0 — ALL CHECKS PASSED')
print('=' * 60)

print('Project:', PROJECT)
print('Dataset:', DS)
print('Experiment:', EXP)
print('Output:', OUT)
print('Checkpoints verified:', len(required))
print('Exp06 validation files:', len(copied_files))
print('Instance bin edges:', bins['edges'])
print('Unit test log:', f'{OUT}/logs/unit_tests.log')
print('Bins log:', f'{OUT}/logs/bins.log')

print('\nREADY FOR NEXT EXP07 SETUP STEP')
```

### Cell K7.1 — Train OccCond-g, seed 0–4

```python
COMMON = (f'--dataset_path={DS} --dataset_size_train=104 --in_channels=5 --num_classes=3 '
		  '--batch_size=8 --num_workers=4 --lr=0.0003 '
		  '--weight_bg=0.9 --weight_crop=1.0 --weight_weed=1.3 '
		  '--use_dice_loss=True --dice_weight=0.3 --epochs=100 --log_interval=50 '
		  '--ckpt_interval=1000000 --pretrained_backbone=True --spectral_guided=False '
		  '--use_multiscale_fusion=True --resume=True')
ARCH = ('--ccmc=False --segformer=True --use_low_level_refine=False '
		'--head=occ_conditioned --spectral_guide=True --mcsd_tau=1.0 --occ_weight=0.3')

for seed in range(5):
	run = f'OccCondg_s{seed}'
	if not os.path.exists(f'runs/{run}/run_summary.json'):
		args = [sys.executable, 'src/train.py', *COMMON.split(), *ARCH.split(),
		        f'--seed={seed}', f'--run_name=OccCondg__s{seed}', f'--out_dir=runs/{run}']
		run_checked(args, f'{OUT}/logs/train_{run}.log')
	assert os.path.isfile(f'runs/{run}/run_summary.json'), f'{run}: missing run_summary.json'
	assert os.path.isfile(f'runs/{run}/best_val.pth'), f'{run}: missing best_val.pth'
	with open(f'runs/{run}/run_summary.json') as f:
		summary = json.load(f)
	assert summary['best_epoch'] >= 0 and summary['best_val_miou'] > 0, f'{run}: invalid summary'
	with open(f'runs/{run}/history.jsonl') as f:
		history = [json.loads(line) for line in f if line.strip()]
	assert len(history) == 100 and history[-1]['epoch'] == 99, f'{run}: incomplete training history'
```

### Cell K7.2 — Evaluate, compare

```python
EVAL_ARCH = '--ccmc=False --head=occ_conditioned --spectral_guide=True'

def evaluate(model, seed, arch, tag=None, extra=''):
	ckpt = f'runs/{model}_s{seed}/best_val.pth'
	if not os.path.isfile(ckpt):
		raise FileNotFoundError(ckpt)
	name = f'{tag or model}_s{seed}'
	args = [sys.executable, 'src/evaluate_ccmc.py', f'--dataset_path={DS}', '--split=val',
	        f'--ckpt={ckpt}', *arch.split(), *extra.split(), f'--bins_json={BINS}',
	        '--boundary_px=3', f'--run_name={name}', f'--out_prefix={OUT}/{name}_val']
	run_checked(args, f'{OUT}/logs/eval_{name}.log')
	assert os.path.isfile(f'{OUT}/{name}_val_per_image.csv'), f'{name}: missing per-image CSV'
	assert os.path.isfile(f'{OUT}/{name}_val_summary.json'), f'{name}: missing summary JSON'

for seed in range(5):
	evaluate('OccCondg', seed, EVAL_ARCH)

def evaluation_csv(model, seed):
	root = OUT if model == 'OccCondg' else EXP06_RESULTS
	path = os.path.join(root, f'{model}_s{seed}_val_per_image.csv')
	if not os.path.isfile(path):
		raise FileNotFoundError(path)
	return path

def compare(a, b):
	common = list(range(5))
	files_a = ','.join(evaluation_csv(a, seed) for seed in common)
	files_b = ','.join(evaluation_csv(b, seed) for seed in common)
	out_path = f'{OUT}/cmp_{a}_vs_{b}.json'
	args = [sys.executable, 'src/scripts/compare_runs.py', f'--a={files_a}', f'--b={files_b}',
	        f'--name_a={a}', f'--name_b={b}', '--gate=none', '--seed_bootstrap=True',
	        '--metrics=SmallWeedRecall,recall30_small,cov_small,mIoU,IoU_crop,IoU_weed,BIoU_weed,PrecisionWeed,CropToWeedRate',
	        f'--out={out_path}']
	run_checked(args, f'{OUT}/logs/cmp_{a}_vs_{b}.log')
	assert os.path.isfile(out_path), f'{a} vs {b}: comparison JSON missing'
	with open(out_path) as f:
		assert 'PrecisionWeed' in json.load(f)['metrics']

for a, b in (('MCSDg', 'OccCondg'), ('MCSDg', 'L0'), ('OccCondg', 'L0'),
			 ('MCSDg', 'PS'), ('MCSDgIt1', 'MCSDg'), ('MCSDgIt20', 'MCSDg')):
	compare(a, b)
```

### Cell K7.3 — Audit marginal theo lớp/occupancy

```python
audit_specs = [('MCSDg', 'mcsd', (1, 5, 20)), ('OccCondg', 'occ_conditioned', (5,))]
for model, head, iteration_counts in audit_specs:
	for seed in range(5):
		ckpt = f'runs/{model}_s{seed}/best_val.pth'
		for iters in iteration_counts:
			prefix = f'{OUT}/audit_{model}_s{seed}_it{iters}'
			args = [sys.executable, 'src/scripts/mcsd_audit.py', f'--dataset_path={DS}',
			        f'--ckpt={ckpt}', f'--head={head}', '--spectral_guide=True',
			        f'--sinkhorn_iters={iters}', '--mcsd_tau=1.0', '--bootstrap_samples=2000',
			        f'--out_prefix={prefix}']
			run_checked(args, f'{OUT}/logs/audit_{model}_s{seed}_it{iters}.log')
			for suffix in ('_marginal_by_image.csv', '_predweed_by_image.csv', '_marginal_summary.json'):
				assert os.path.isfile(prefix + suffix), f'Missing audit output: {prefix + suffix}'
```

Cell này sinh CSV theo từng ảnh/lớp/occupancy và JSON summary gồm signed residual, MAE, p95/max, occupancy-head MAE cùng CI bootstrap theo ảnh. Các sai số được ghi ở dạng fraction 0–1. Giữ cả các file này trong zip kết quả.

### Cell K7.4 — Delta per-image cho 1 vòng so với 5 vòng

```python
for seed in range(5):
	ckpt = f'runs/MCSDg_s{seed}/best_val.pth'
	prefix = f'{OUT}/iteration_audit_MCSDg_s{seed}_it1_vs_it5'
	args = [sys.executable, 'src/scripts/mcsd_iteration_audit.py',
	        f'--dataset_path={DS}', f'--ckpt={ckpt}', f'--bins_json={BINS}',
	        '--head=mcsd', '--spectral_guide=True', '--mcsd_tau=1.0',
	        '--iters_a=1', '--iters_b=5', '--bootstrap_samples=2000',
	        f'--out_prefix={prefix}']
	run_checked(args, f'{OUT}/logs/iteration_audit_MCSDg_s{seed}.log')
	for suffix in ('_per_image.csv', '_small_instances.csv', '_summary.json'):
		assert os.path.isfile(prefix + suffix), f'Missing iteration audit output: {prefix + suffix}'
```

Script xuất Weed precision, pixel false positives, Crop→Weed rate và confusion matrix; CSV instance ghép trên cùng GT instance ID để đếm đối tượng nhỏ mới được phát hiện ở 1 vòng và đối tượng bị mất so với 5 vòng. Point estimate và paired bootstrap CI theo ảnh nằm trong từng JSON; paired seed/image CI tổng hợp cho Precision, Crop→Weed và recall được tính từ CSV evaluation bằng `compare_runs.py` ở K7.2.

### Cell K7.5 — Đóng gói kết quả

```python
import csv, shutil, tempfile, zipfile

for run_dir in sorted(glob.glob('runs/OccCondg_s*')):
	if os.path.islink(f'{run_dir}/best_val.pth') or not os.path.exists(f'{run_dir}/history.jsonl'):
		continue
	dst = f'{OUT}/runs_meta/{os.path.basename(run_dir)}'
	os.makedirs(dst, exist_ok=True)
	for filename in ('history.jsonl', 'run_config.json', 'run_summary.json'):
		if os.path.exists(f'{run_dir}/{filename}'):
			shutil.copy(f'{run_dir}/{filename}', dst)

expected_runs = [f'OccCondg_s{seed}' for seed in range(5)]
collector_metrics = (
	'mIoU', 'IoU_bg', 'IoU_crop', 'IoU_weed', 'BIoU_crop', 'BIoU_weed',
	'SmallWeedRecall', 'recall30_small', 'cov_small', 'recall50_medium',
	'recall50_large', 'recall50_all', 'n_small', 'n_medium', 'n_large'
)
for run in expected_runs:
	assert os.path.isfile(f'{OUT}/runs_meta/{run}/run_summary.json'), f'{run}: missing run metadata'
	assert os.path.isfile(f'{OUT}/{run}_val_summary.json'), f'{run}: missing evaluation summary'
	with open(f'{OUT}/{run}_val_summary.json') as f:
		summary = json.load(f)
	assert 'metrics' in summary, f'{run}: evaluation summary has no metrics'
	missing_metrics = [key for key in collector_metrics if key not in summary['metrics']]
	assert not missing_metrics, f'{run}: missing collector metric keys {missing_metrics}'

# The collector expects the full evaluate_ccmc metric schema. Stage only these five
# validation summaries so reference/audit metadata cannot be mistaken for evaluations.
collect_log = f'{OUT}/logs/collect.log'
try:
	with tempfile.TemporaryDirectory(prefix='exp07_collect_') as collect_dir:
		for run in expected_runs:
			shutil.copy2(f'{OUT}/{run}_val_summary.json', collect_dir)
		run_checked(
			[sys.executable, 'src/scripts/collect_results.py',
			 f'--results_dir={collect_dir}', f'--runs_dir={OUT}/runs_meta',
			 f'--out={OUT}/metrics.csv'],
			collect_log
		)
except subprocess.CalledProcessError:
	with open(collect_log) as f:
		print(f.read())
	raise

assert os.path.isfile(f'{OUT}/metrics.csv'), 'Collector did not create metrics.csv'
with open(f'{OUT}/metrics.csv', newline='') as f:
	metric_rows = list(csv.DictReader(f))
assert len(metric_rows) == 5, f'Expected 5 OccCondg evaluation rows, got {len(metric_rows)}'
new_ckpts = ' '.join(sorted(glob.glob('runs/OccCondg_s*/best_val.pth')))
assert len(new_ckpts.split()) == 5
with zipfile.ZipFile('/kaggle/working/exp07_results.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
	for root, _, files in os.walk(OUT):
		for filename in files:
			path = os.path.join(root, filename)
			archive.write(path, arcname=path)
with zipfile.ZipFile('/kaggle/working/exp07_occcond_ckpts.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
	for path in new_ckpts.split():
		archive.write(path, arcname=path)
assert os.path.isfile('/kaggle/working/exp07_results.zip')
assert os.path.isfile('/kaggle/working/exp07_occcond_ckpts.zip')
print('/kaggle/working/exp07_results.zip')
print('/kaggle/working/exp07_occcond_ckpts.zip')
```

Tải `exp07_results.zip` về repo, giải nén vào `doc/exp07_results/` và gửi lại cùng trạng thái 5 run `OccCondg`. Không mở test. Giữ `exp07_occcond_ckpts.zip` làm đầu vào tái lập cho các thí nghiệm tiếp theo.

## Rà soát novelty (làm song song, ghi lại trước test)

Ưu tiên bốn nhánh: **fractional class occupancy / label proportions**, **sub-pixel mapping** trong viễn thám, **constrained decoding / area-preserving segmentation**, **optimal transport / Sinkhorn** trong dense prediction. Tìm trên Scopus, IEEE Xplore, Web of Science và Google Scholar; lưu ngày tìm, chuỗi truy vấn, khoảng năm, tiêu chí loại/trích và danh sách paper liên quan.

Lập bảng đối chiếu mỗi công trình theo: đầu vào occupancy có/không; dự đoán sub-pixel; ràng buộc marginal/khối lượng; OT/Sinkhorn; end-to-end differentiable; nhiệm vụ/dataset; multispectral; metric và kết quả. Đặc biệt phân biệt sub-pixel mapping hậu xử lý từ fractional abundance với decoder phân đoạn end-to-end. Chưa hoàn tất rà soát thì không dùng claim “first” hoặc “novel”.

## Exp08 — Dataset thứ hai

Kiểm tra WeedMap trước; chỉ chọn nếu metadata xác nhận phổ, nhãn và độ phân giải phù hợp. Nếu không, chọn dataset đa phổ khác có thông tin tương đương cho R/G/B/NIR/RE hoặc ghi rõ band mismatch và giới hạn chuyển giao. Chốt dataset trước khi truy cập test labels.

Protocol cần khóa trước khi train:

- xác nhận sensor/bands, radiometric scaling, GSD, scene/date/site, annotation semantics và license; ánh xạ nhãn cố định thành Background/Crop/Weed;
- chia train/val/test theo field/site/flight group nếu metadata cho phép, tránh tile/patch cùng orthomosaic rơi vào nhiều split;
- định nghĩa small weed theo diện tích vật lý nếu GSD đáng tin, nếu không dùng quantile chỉ từ train annotations. Không mang nguyên ngưỡng 45/147 px sang dataset có GSD khác;
- paired 5 seed MCSD-g (5 vòng, cấu hình khóa) và L0 cùng backbone/input bands, loss, augmentation, epoch/checkpoint policy; ghi rõ mọi khác biệt bắt buộc do dataset;
- báo recall@IoU 0,5, coverage, mIoU, per-class IoU/BIoU, seed variability và chi phí; chỉ chọn checkpoint bằng validation.

Nếu dataset không có test split độc lập hoặc không đủ nhãn instance/occupancy để tính metric tương thích, nêu giới hạn trước khi chạy và không trình bày nó như xác nhận độc lập tương đương WeedsGalore.

## Khóa cấu hình và mở test một lần

Không mở test ngay sau Exp06. Trước tiên hoàn tất Exp07, quyết định phương pháp cuối cùng, kiểm toán novelty và khóa protocol Exp08. Sau đó lập manifest có commit, checkpoint hashes, preprocessing, split IDs/hash, metric/CI, so sánh, lệnh chạy và quy tắc xử lý lỗi. Chỉ chạy test một lần cho cấu hình đã khóa trên WeedsGalore và dataset thứ hai (nếu test protocol đã khóa); không điều chỉnh model/hyperparameter hoặc chọn paper result từ test. Nếu phát sinh sửa lỗi ảnh hưởng output, ghi rõ và coi lần chạy mới là deviation cần giải trình.
