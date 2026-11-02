# Exp01 — Baseline L0 + chẩn đoán H1 (Sprint D1–D2, Plan P0–P1, Gate G1)

## Mục tiêu (khóa trước khi xem kết quả)

| Câu hỏi | Bằng chứng tạo ra | Tiêu chí đọc kết quả |
|---|---|---|
| Baseline L0 có tái lập được không? Phương sai giữa seed bao nhiêu? | L0 seed 0 và 1 (val), `history.jsonl` | Chênh mIoU giữa seed ghi vào báo cáo; dùng để hiểu độ lớn noise khi so L3 sau này |
| Kiểm toán 82,16 so với 83,53 | L0 trong pipeline mới (val) so với các dòng cũ của README | Ghi rõ khác loss/split/decoder; không gộp hai số |
| **Diag-D0:** cỏ nhỏ có bị bỏ sót nhiều hơn không? | Recall@0.5 / @0.3 / coverage theo bin (small/medium/large), theo ngày, ma trận nhầm lẫn | Recall_small thấp hơn rõ so với recall_large, CI bootstrap không cắt 0 |
| **Diag-D2:** sau SR, cỏ nhỏ có khó giải mã tuyến tính hơn cỏ lớn không? | AUROC pre-SR (avg) so với post-SR theo stage 1–3; contrast small − large | Contrast > 0 với CI > 0 ở ít nhất một stage → ủng hộ H1 |
| LLR (LiteSent) có giúp chủ yếu cỏ nhỏ không? | LiteSent seed 0 so với L0 seed 0, paired bootstrap | Thông tin bổ trợ: nếu LLR tăng mạnh recall nhỏ thì mất chi tiết không gian là có thật |
| Chi phí tính toán (quy ước thống nhất) | `profile_*.json` cho L0, LiteSent, L3, K5, K6 (kiến trúc) | Kiểm tra lại các số 3,49 G và 17,09 G bằng FlopCounter (có tính matmul attention) |

**G1 (kế hoạch mục 7):** ĐẠT nếu D0 cho thấy nhóm nhỏ suy giảm rõ **và** D2 cho contrast dương có CI > 0 ở ít nhất một stage. Diag-D1 (quét `R`) chỉ mang tính thăm dò nên **NOT RUN** trong Exp01.

Thời lượng ước tính trên T4: mỗi lần train khoảng 15–30 phút (sẽ đo lại bằng `epoch_time_s`), probe khoảng 5 phút, profile khoảng 2 phút cho mỗi mô hình.

---

## Bước 0 — Chạy toàn bộ cell trong [00_setup_kaggle.md](00_setup_kaggle.md) (S1 → S6)

Unit test phải pass trước khi chạy tiếp.

## Cell E1.0 — Biến dùng chung

```python
import os
EXP = 'exp01'
OUT = f'results/{EXP}'
BINS = 'results/instance_bins.json'   # bin cố định cho MỌI thí nghiệm sau này
os.makedirs(f'{OUT}/logs', exist_ok=True)
COMMON = (f"--dataset_path={DS} --dataset_size_train=104 --in_channels=5 --num_classes=3 "
          "--batch_size=8 --num_workers=4 --lr=0.0003 "
          "--weight_bg=0.9 --weight_crop=1.0 --weight_weed=1.3 --use_dice_loss=True --dice_weight=0.3 "
          "--epochs=100 --log_interval=50 --ckpt_interval=1000000 "
          "--pretrained_backbone=True --spectral_guided=False --use_multiscale_fusion=True --resume=True")
L0_ARCH = "--ccmc=True --prior_mode=none --use_low_level_refine=False"
LITE_ARCH = "--ccmc=False --segformer=True --use_low_level_refine=True"
print(COMMON)
```

## Cell E1.1 — Bin kích thước cỏ (tam phân vị của train, khóa trước khi xem mô hình)

```python
!python src/scripts/make_instance_bins.py --dataset_path={DS} --out={BINS} 2>&1 | tee {OUT}/logs/bins.log | head -n 60
!cp {BINS} {OUT}/instance_bins.json
```

Cần kiểm tra trong output: `instance format` (mode/dtype), `source` (`official` hoặc `cc`), số instance của mỗi bin trên val (nếu `n_small < 20` thì bin đó chỉ mang tính thăm dò).

## Cell E1.2 — Train L0 seed 0 và seed 1

```python
for seed in (0, 1):
    run = f'L0_s{seed}'
    !python src/train.py {COMMON} {L0_ARCH} --seed={seed} --run_name=L0__s{seed} --out_dir=runs/{run} 2>&1 | tee {OUT}/logs/train_{run}.log | grep --line-buffered -E "Val mIoU|Saved best|Traceback|Error|FLOPs|Total Parameters"
```

> Nếu phiên Kaggle bị ngắt: chạy lại đúng cell này. `--resume=True` sẽ đọc `runs/<run>/last.pth` và train tiếp.

## Cell E1.3 — Train LiteSent seed 0 (cùng pipeline, để làm Diag-D0 trên checkpoint có LLR)

```python
!python src/train.py {COMMON} {LITE_ARCH} --seed=0 --run_name=LiteSent__s0 --out_dir=runs/LiteSent_s0 2>&1 | tee {OUT}/logs/train_LiteSent_s0.log | grep --line-buffered -E "Val mIoU|Saved best|Traceback|Error|FLOPs|Total Parameters"
```

> Nếu thiếu quota: bỏ cell này và ghi `NOT RUN` khi gửi kết quả. L0 và probe là ưu tiên.

## Cell E1.4 — Diag-D0: đánh giá validation (IoU, Boundary IoU, recall theo bin, theo ngày)

```python
for seed in (0, 1):
    !python src/evaluate_ccmc.py --dataset_path={DS} --split=val --ckpt=runs/L0_s{seed}/best_val.pth {L0_ARCH} --bins_json={BINS} --run_name=L0__s{seed} --out_prefix={OUT}/L0_s{seed}_val 2>&1 | tee {OUT}/logs/eval_L0_s{seed}.log
if os.path.exists('runs/LiteSent_s0/best_val.pth'):
    !python src/evaluate_ccmc.py --dataset_path={DS} --split=val --ckpt=runs/LiteSent_s0/best_val.pth --ccmc=False --use_low_level_refine=True --bins_json={BINS} --run_name=LiteSent__s0 --out_prefix={OUT}/LiteSent_s0_val 2>&1 | tee {OUT}/logs/eval_LiteSent_s0.log
```

## Cell E1.5 — Diag-D0: recall cỏ nhỏ so với cỏ lớn trong cùng mô hình (bootstrap theo ảnh)

```python
import csv, json, numpy as np
def small_vs_large(path, n_boot=10000, seed=0):
    rows = list(csv.DictReader(open(path)))
    g = lambda k: np.array([float(r[k]) for r in rows])
    out = {}
    for t in ('50', '30'):
        tp_s, n_s, tp_l, n_l = g(f'tp{t}_small'), g('n_small'), g(f'tp{t}_large'), g('n_large')
        point = 100 * (tp_l.sum() / n_l.sum() - tp_s.sum() / n_s.sum())
        rng = np.random.default_rng(seed)
        W = rng.multinomial(len(rows), np.full(len(rows), 1 / len(rows)), size=n_boot)
        with np.errstate(invalid='ignore', divide='ignore'):
            boot = 100 * ((W @ tp_l) / (W @ n_l) - (W @ tp_s) / (W @ n_s))
        boot = boot[np.isfinite(boot)]
        out[f'recall{t}_large_minus_small'] = {'delta': point, 'ci95': np.percentile(boot, [2.5, 97.5]).tolist()}
    return out
d0 = {}
for name in ['L0_s0', 'L0_s1', 'LiteSent_s0']:
    p = f'{OUT}/{name}_val_per_image.csv'
    if os.path.exists(p):
        d0[name] = small_vs_large(p)
print(json.dumps(d0, indent=2))
json.dump(d0, open(f'{OUT}/d0_small_vs_large.json', 'w'), indent=2)
```

## Cell E1.6 — LiteSent so với L0 (paired bootstrap, seed 0, mang tính bổ trợ)

```python
if os.path.exists(f'{OUT}/LiteSent_s0_val_per_image.csv'):
    !python src/scripts/compare_runs.py --a={OUT}/LiteSent_s0_val_per_image.csv --b={OUT}/L0_s0_val_per_image.csv --name_a=LiteSent --name_b=L0 --out={OUT}/cmp_LiteSent_vs_L0_s0.json 2>&1 | tee {OUT}/logs/cmp_LiteSent_vs_L0.log
```

## Cell E1.7 — Diag-D2: linear probe trước/sau SR (stage 1–3)

```python
for seed in (0, 1):
    !python src/scripts/probe_sr.py --dataset_path={DS} --ckpt=runs/L0_s{seed}/best_val.pth {L0_ARCH} --bins_json={BINS} --out_prefix={OUT}/d2_probe_L0_s{seed} 2>&1 | tee {OUT}/logs/probe_L0_s{seed}.log | tail -n 60
```

## Cell E1.8 — Profile (cùng quy ước: 1×5×600×600, fp32, batch 1, T4)

```python
profiles = {
    'L0': L0_ARCH,
    'LiteSent': LITE_ARCH.replace('--segformer=True ', ''),
    'L3': '--ccmc=True --prior_mode=learned_weed --injection=ccmc_sr --use_low_level_refine=False',
    'K5': '--ccmc=True --prior_mode=learned_weed --injection=pre_sr_gate --use_low_level_refine=False',
    'K6': '--ccmc=True --prior_mode=learned_weed --injection=post_sr_gate --use_low_level_refine=False',
}
for label, arch in profiles.items():
    !python src/scripts/profile_model.py {arch} --label={label} --out={OUT}/profile_{label}.json 2>&1 | tail -n 3
```

## Cell E1.9 — Gom kết quả và đóng gói để gửi review

```python
import shutil, glob
for run_dir in sorted(glob.glob('runs/*')):
    dst = f'{OUT}/runs_meta/{os.path.basename(run_dir)}'
    os.makedirs(dst, exist_ok=True)
    for f in ('history.jsonl', 'run_config.json', 'run_summary.json'):
        if os.path.exists(f'{run_dir}/{f}'):
            shutil.copy(f'{run_dir}/{f}', dst)
shutil.copytree('results/env', f'{OUT}/env', dirs_exist_ok=True)
!python src/scripts/collect_results.py --results_dir={OUT} --runs_dir=runs --out={OUT}/metrics.csv
!cd /kaggle/working/WeedsGalore_Ipr28 && zip -qr /kaggle/working/{EXP}_results.zip {OUT} {BINS}
!cd /kaggle/working/WeedsGalore_Ipr28 && zip -qr /kaggle/working/{EXP}_ckpts.zip runs/*/best_val.pth
!ls -lh /kaggle/working/*.zip
```

## Gửi lại để review

1. Tải `exp01_results.zip` (tab Output của notebook), giải nén **tại thư mục gốc repo local** để có `results/exp01/...` và `results/instance_bins.json`.
2. Giữ `exp01_ckpts.zip`. Nên tạo Kaggle Dataset riêng (ví dụ `ccmc-ckpts`) từ output này để các thí nghiệm sau dùng lại checkpoint L0 (oracle intervention, probe L3 so với L0).
3. Commit `results/instance_bins.json` vào repo: mọi thí nghiệm sau phải dùng **cùng** file bin này.
4. Nhắn: "đã có kết quả exp01" kèm trạng thái `DONE / PARTIAL / NOT RUN` của từng cell (E1.2–E1.8). Nếu có lỗi, gửi kèm file `results/exp01/logs/*.log` tương ứng.
