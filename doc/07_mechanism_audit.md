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
- Không thay MCSD-g đã xác nhận. Mọi code mới phải có unit test cho forward shape, xác suất mỗi sub-pixel tổng bằng 1, gradient hữu hạn, occupancy conditioning và khác biệt với Sinkhorn khi placement không đồng đều.
- Không chạy test ở bất kỳ cell nào của Exp07.

## E7.1 — Occupancy-conditioned baseline không Sinkhorn

Thêm head/variant riêng, ví dụ `head=occ_conditioned`, dùng cùng occupancy head, placement head, spectral guide 5-band và occupancy loss như MCSD-g. Khác biệt duy nhất là cách tạo xác suất cho mỗi sub-pixel. Với occupancy dự đoán `q_c` và placement score `s_cj`, dùng:

```text
log P(c, j) = log_softmax_c(log(q_c) + s(c, j) / tau)
```

Chuẩn hóa theo lớp độc lập ở từng sub-pixel; **không** chạy Sinkhorn và không ép tổng xác suất lớp qua 16 sub-pixel phải khớp `16*q`. Khi placement scores bằng nhau, đầu ra vẫn theo `q`; khi scores khác nhau, marginal có thể lệch. Đây là occupancy-conditioned control, không phải PS cũ: PS có occupancy auxiliary loss nhưng occupancy head không điều kiện hóa placement output.

### Train và so sánh

- Train `OccCond-g` seed 0–4, cùng init recipe, 5-band guide, `tau=1`, occupancy loss 0,3 và mọi thiết lập còn lại của MCSD-g. Ghi rõ đây là baseline mới; không tái sử dụng tên PS.
- Đánh giá paired trên validation: `OccCond-g` so với MCSD-g, L0 và PS hiện có. So sánh chính là MCSD-g − OccCond-g; các so sánh còn lại là bối cảnh.
- Báo cáo Small-Weed Recall, `cov_small`, mIoU, Crop/Weed IoU, BIoU, false positive Weed trên GT không-Weed, tổng Weed probability và marginal error. Dùng CI 95% phân tầng seed + ảnh; không chọn biến thể theo test.
- Exp07 là audit cơ chế, không chọn winner và không có gate PASS/FAIL. Chỉ claim lợi ích downstream của Sinkhorn nếu marginal MAE giảm, mIoU delta MCSD-g − OccCond-g có CI 95% hoàn toàn trên 0, Crop IoU không kém quá 1 pp (CI dưới lớn hơn −1 pp), và Small-Weed Recall không kém quá 1 pp (CI dưới lớn hơn −1 pp). Nếu điều kiện không đạt, chỉ báo cáo các estimate/CI và khả năng áp đặt marginal; không claim Sinkhorn cải thiện accuracy.

## E7.2 — Sai số marginal theo lớp và occupancy

Mở rộng `src/scripts/mass_check.py` hoặc thêm script audit để đo trên checkpoint MCSD-g seed 0–4 ở inference 1, 5, 20 vòng. Với mỗi lớp `c`, mỗi ô 4×4 có:

```text
q_pred(c) = softmax(occ_logits)(c)
q_out(c)  = average_pool_4x4(P_c)
residual(c) = q_out(c) - q_pred(c)
```

Ghi theo lớp và theo tầng occupancy GT của từng ô: 0/16, 1–3/16, 4–7/16, 8–16/16 pixel của lớp. Nhóm 1–3 Weed pixel/ô là vùng occupancy nhỏ cần chú ý; ngưỡng được cố định trước khi chạy. Cho từng lớp/tầng, báo số ô, signed bias, MAE, p95 absolute error và max absolute error; bootstrap CI theo ảnh cho metric tổng hợp. Đánh giá đồng thời `q_pred` so với occupancy GT để tách lỗi occupancy head khỏi lỗi coupling Sinkhorn.

Không gộp các đại lượng sau thành một “mass error” duy nhất:

- `q_out - q_pred`: mức độ đầu ra tuân theo occupancy dự đoán; đây là sai số marginal của Sinkhorn.
- `q_pred - q_GT`: lỗi occupancy head so với tỉ lệ lớp thật trong ô.
- Tổng xác suất Weed / tổng pixel Weed GT: độ lệch tổng lượng dự đoán so với nhãn; không đo bảo toàn theo occupancy head.

## E7.3 — Phân tích vì sao 1 vòng tăng recall

Đây là phân tích giải thích, không phải xác nhận một cơ chế nhân quả. Dùng cùng checkpoint 5-vòng cho mọi lượt suy luận; ghép output theo seed, ảnh và instance. Với 1 vs 5 vòng, phân rã thay đổi thành:

- instance TP/FN theo recall@IoU 0,5 và recall@0,3; false positive được đo ở mức pixel bằng Background/Crop→Weed, không diễn giải thành FP-instance;
- coverage instance và số pixel dự đoán Weed trên vùng GT không-Weed;
- confusion Weed→Background, Background/Crop→Weed, Crop IoU và boundary IoU;
- các đại lượng trên theo GT Weed occupancy 1–3/16, 4–7/16, 8–16/16 trong ô và theo ngày bay.

Kiểm tra mối liên hệ giữa marginal residual Weed, tổng lượng Weed dự đoán, số instance được khôi phục và phần suy giảm Crop/mIoU. Báo kết quả theo từng seed/ảnh, không suy luận từ một ví dụ trực quan. Giả thuyết cần kiểm tra là một vòng chưa hội tụ làm lệch row marginals, từ đó phân bổ nhiều xác suất Weed hơn cho một số ô; nếu false positive hoặc Crop→Weed tăng cùng recall thì mô tả đúng trade-off, không gọi đó là cải thiện phát hiện thuần túy.

## E7.4 — Thử nghiệm train với số vòng khác nhau (thăm dò)

Chỉ chạy nếu E7.1–E7.3 cho thấy câu hỏi này còn cần thiết. Nếu chạy, đánh dấu tên run, bảng và report là **EXPLORATORY — TRAINING ITERATION ABLATION**. Đăng ký trước số seed, thứ tự train, metric và cách diễn giải trước khi chạy; giữ nguyên mọi thứ ngoài `sinkhorn_iters`. Không dùng kết quả để thay cấu hình MCSD-g đã xác nhận hoặc để tuyên bố confirmatory. Phân biệt rõ với E6 inference-only, vốn chỉ đổi số vòng trên cùng checkpoint.

## Kaggle cells

Các cell dưới đây không chạy test. Trước khi chạy, commit/push code mới (`occ_conditioned` head và `mcsd_audit.py`) rồi clone đúng commit đó trên Kaggle. Gắn dataset WeedsGalore cùng các checkpoint datasets dùng ở Exp06: `ccmc-ckpts`, `exp03-ckpts`, `exp04-ckpts`, `exp05-ckpts`, `exp06-ckpts`. Biến `DS` cần trỏ tới thư mục gốc `weedsgalore-dataset`, như các notebook Exp01–06.

### Cell K7.0 — Setup, test và checkpoint check

```python
import glob, json, os, shutil, subprocess, sys

os.chdir('/kaggle/working/WeedsGalore_Ipr28')
assert os.path.isdir(DS), f'Invalid dataset path: {DS}'
EXP = 'exp07'
OUT = f'results/{EXP}'
BINS = 'results/instance_bins.json'
os.makedirs(f'{OUT}/logs', exist_ok=True)

# Stop here if the new synthetic and model integration tests fail.
subprocess.run([sys.executable, '-m', 'pytest', '-q', 'tests/test_ccmc_sr.py'], check=True)

for path in glob.glob('/kaggle/input/**/runs/*/best_val.pth', recursive=True):
	run_dir = os.path.basename(os.path.dirname(path))
	target_dir = os.path.join('runs', run_dir)
	os.makedirs(target_dir, exist_ok=True)
	target = os.path.join(target_dir, 'best_val.pth')
	if not os.path.exists(target):
		os.symlink(path, target)

required = [f'{model}_s{seed}' for model in ('L0', 'MCSDg', 'PS') for seed in range(5)]
missing = [run for run in required if not os.path.isfile(f'runs/{run}/best_val.pth')]
assert not missing, f'Missing Exp06 reference checkpoints: {missing}'

!python src/scripts/make_instance_bins.py --dataset_path={DS} --out={BINS} > {OUT}/logs/bins.log 2>&1
with open(BINS) as f:
	assert json.load(f)['edges'] == [45.0, 147.0]
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
	if os.path.exists(f'runs/{run}/run_summary.json'):
		print('skip (done)', run)
		continue
	!python src/train.py {COMMON} {ARCH} --seed={seed} --run_name=OccCondg__s{seed} --out_dir=runs/{run} 2>&1 | tee {OUT}/logs/train_{run}.log | grep --line-buffered -E "Saved best|Traceback|Error" | tail -n 2
```

### Cell K7.2 — Evaluate, compare

```python
EVAL_ARCH = {
	'L0': '--ccmc=True --prior_mode=none --use_low_level_refine=False',
	'PS': '--ccmc=False --head=pixelshuffle --spectral_guide=False',
	'MCSDg': '--ccmc=False --head=mcsd --spectral_guide=True',
	'OccCondg': '--ccmc=False --head=occ_conditioned --spectral_guide=True',
}

def evaluate(model, seed, arch, tag=None, extra=''):
	ckpt = f'runs/{model}_s{seed}/best_val.pth'
	if not os.path.isfile(ckpt):
		raise FileNotFoundError(ckpt)
	name = f'{tag or model}_s{seed}'
	!python src/evaluate_ccmc.py --dataset_path={DS} --split=val --ckpt={ckpt} {arch} {extra} --bins_json={BINS} --boundary_px=3 --run_name={name} --out_prefix={OUT}/{name}_val > {OUT}/logs/eval_{name}.log 2>&1

for model in ('L0', 'PS', 'MCSDg', 'OccCondg'):
	for seed in range(5):
		evaluate(model, seed, EVAL_ARCH[model])

for iters in (1, 20):
	for seed in range(5):
		evaluate('MCSDg', seed, EVAL_ARCH['MCSDg'], tag=f'MCSDgIt{iters}',
				 extra=f'--sinkhorn_iters={iters}')

def compare(a, b):
	common = [seed for seed in range(5)
			  if os.path.isfile(f'{OUT}/{a}_s{seed}_val_per_image.csv')
			  and os.path.isfile(f'{OUT}/{b}_s{seed}_val_per_image.csv')]
	if len(common) != 5:
		raise RuntimeError(f'{a} vs {b}: expected 5 paired seeds, got {common}')
	files_a = ','.join(f'{OUT}/{a}_s{seed}_val_per_image.csv' for seed in common)
	files_b = ','.join(f'{OUT}/{b}_s{seed}_val_per_image.csv' for seed in common)
	!python src/scripts/compare_runs.py --a={files_a} --b={files_b} --name_a={a} --name_b={b} --gate=none --seed_bootstrap=True --out={OUT}/cmp_{a}_vs_{b}.json 2>&1 | tee {OUT}/logs/cmp_{a}_vs_{b}.log

for a, b in (('MCSDg', 'OccCondg'), ('MCSDg', 'L0'), ('OccCondg', 'L0'),
			 ('MCSDg', 'PS'), ('MCSDgIt1', 'MCSDg'), ('MCSDgIt20', 'MCSDg')):
	compare(a, b)
```

### Cell K7.3 — Audit marginal theo lớp/occupancy

```python
for seed in range(5):
	ckpt = f'runs/MCSDg_s{seed}/best_val.pth'
	for iters in (1, 5, 20):
		prefix = f'{OUT}/audit_MCSDg_s{seed}_it{iters}'
		!python src/scripts/mcsd_audit.py --dataset_path={DS} --ckpt={ckpt} --head=mcsd --spectral_guide=True --sinkhorn_iters={iters} --mcsd_tau=1.0 --bootstrap_samples=2000 --out_prefix={prefix} > {OUT}/logs/audit_MCSDg_s{seed}_it{iters}.log 2>&1
```

Cell này sinh CSV theo từng ảnh/lớp/occupancy và JSON summary gồm signed residual, MAE, p95/max, occupancy-head MAE cùng CI bootstrap theo ảnh. Các sai số được ghi ở dạng fraction 0–1. Giữ cả các file này trong zip kết quả.

### Cell K7.4 — Delta per-image cho 1 vòng so với 5 vòng

```python
import pandas as pd

columns = ['n_small', 'tp50_small', 'tp30_small', 'cov_small', 'cm_02', 'cm_12', 'cm_20',
		   'cm_21', 'inter_2', 'union_2', 'binter_2', 'bunion_2']
for seed in range(5):
	one = pd.read_csv(f'{OUT}/MCSDgIt1_s{seed}_val_per_image.csv')
	five = pd.read_csv(f'{OUT}/MCSDg_s{seed}_val_per_image.csv')
	merged = one[['name', *columns]].merge(five[['name', *columns]], on='name',
											suffixes=('_it1', '_it5'), validate='one_to_one')
	for column in columns:
		merged[f'delta_{column}'] = merged[f'{column}_it1'] - merged[f'{column}_it5']
	for metric in ('tp50', 'tp30'):
		merged[f'fn{metric[2:]}_small_it1'] = merged['n_small_it1'] - merged[f'{metric}_small_it1']
		merged[f'fn{metric[2:]}_small_it5'] = merged['n_small_it5'] - merged[f'{metric}_small_it5']
	merged.to_csv(f'{OUT}/per_image_it1_vs_it5_s{seed}.csv', index=False)
```

`tp50_small`/`tp30_small` và `n_small` cho TP/FN instance tổng hợp theo ảnh; `cm_02/cm_12` theo dõi Background/Crop→Weed pixel, còn `cm_20/cm_21` theo dõi Weed→Background/Crop pixel. Evaluator hiện không đếm FP-instance riêng, nên không suy diễn pixel false positives thành số weed instance giả.

### Cell K7.5 — Đóng gói kết quả

```python
import shutil

for run_dir in sorted(glob.glob('runs/OccCondg_s*')):
	if os.path.islink(f'{run_dir}/best_val.pth') or not os.path.exists(f'{run_dir}/history.jsonl'):
		continue
	dst = f'{OUT}/runs_meta/{os.path.basename(run_dir)}'
	os.makedirs(dst, exist_ok=True)
	for filename in ('history.jsonl', 'run_config.json', 'run_summary.json'):
		if os.path.exists(f'{run_dir}/{filename}'):
			shutil.copy(f'{run_dir}/{filename}', dst)

!python src/scripts/collect_results.py --results_dir={OUT} --runs_dir=runs --out={OUT}/metrics.csv > {OUT}/logs/collect.log 2>&1
new_ckpts = ' '.join(sorted(glob.glob('runs/OccCondg_s*/best_val.pth')))
assert len(new_ckpts.split()) == 5
!zip -qr /kaggle/working/exp07_results.zip {OUT}
!zip -qr /kaggle/working/exp07_occcond_ckpts.zip {new_ckpts}
!ls -lh /kaggle/working/exp07_*.zip
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
