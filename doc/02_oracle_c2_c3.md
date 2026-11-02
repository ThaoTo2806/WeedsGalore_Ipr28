# Exp02 — Oracle ceiling C2/C3 + đối chứng leakage C3-K5/C3-K6 (Sprint D3, Plan P3a, Gate G2-screen)

> Chỉ chạy sau khi Exp01 đã review (G1 = PASS có điều kiện, xem [REVIEW_REPORT.md](REVIEW_REPORT.md) mục 3).

## Mục tiêu (khóa trước khi xem kết quả)

| Câu hỏi | So sánh | Quy tắc |
|---|---|---|
| **G2-screen:** prior GT-Weed có hơn GT-Veg khi cùng toán tử CCMC-SR? | C3 − C2, 2 seed paired | `NO-GO` nếu delta Small-Weed Recall ≤ 0; `SCREEN-PASS` nếu CI > 0; còn lại `INCONCLUSIVE`. SCREEN-PASS **không** phải bằng chứng H2 |
| Cùng một mức leakage, vị trí/toán tử có quan trọng không? | C3 − C3-K5 (gate trước SR, không chuẩn hóa cửa sổ); C3 − C3-K6 (gate sau LayerNorm của SR) | Chỉ mô tả, có GT leakage. Dự đoán đăng ký trước: C3 ≈ C3-K5 (LayerNorm triệt tiêu thành phần khối lượng của cửa sổ); C3 > C3-K6 nếu tái trọng số **bên trong** cửa sổ là phần có ích |
| Biên độ trần (mô tả) | C3 − L0 | **Không** dùng làm gate; không tính HR làm tiêu chí |
| Cơ chế nhạy với prior sai đến mức nào? | C3 seed 0 với prior GT bị nhiễu (dilate/erode/drop/shift) lúc suy luận | Mô tả, để dự báo rủi ro của L3 |

Thước đo chính là `SmallWeedRecall` (recall@IoU 0,5, bin nhỏ ≤ 45 px), như đã đăng ký. Đọc kèm `cov_small` (coverage ≥ 50 %): hai chỉ số phải cùng chiều thì mới diễn giải.

Ngân sách: mỗi run khoảng 17 phút trên T4 (Exp01 đo 10 s/epoch). Có 8 run train (C2, C3, C3-K5, C3-K6 × 2 seed), tổng khoảng 2,5 giờ GPU, cộng đánh giá khoảng 15 phút.

---

## Bước 0 — Chạy S1–S4 trong [00_setup_kaggle.md](00_setup_kaggle.md), sau đó S6 (unit test). S5 chỉ cần khi đổi phiên bản `transformers`.

## Bước 0b — Đưa checkpoint Exp01 lên Kaggle

Upload file `exp01_ckpts.zip` thành Kaggle Dataset (ví dụ tên `ccmc-ckpts-exp01`), rồi Add Input vào notebook. Checkpoint này dùng để đánh giá lại L0/LiteSent với Boundary IoU 3 px; Exp01 dùng độ rộng 17 px, gần như bằng cả vật thể nên BIoU ≈ IoU.

## Cell E2.0 — Biến dùng chung và khôi phục checkpoint Exp01

```python
import os, glob, json, zipfile
os.chdir('/kaggle/working/WeedsGalore_Ipr28')
EXP = 'exp02'
OUT = f'results/{EXP}'
BINS = 'results/instance_bins.json'
os.makedirs(f'{OUT}/logs', exist_ok=True)
COMMON = (f"--dataset_path={DS} --dataset_size_train=104 --in_channels=5 --num_classes=3 "
          "--batch_size=8 --num_workers=4 --lr=0.0003 "
          "--weight_bg=0.9 --weight_crop=1.0 --weight_weed=1.3 --use_dice_loss=True --dice_weight=0.3 "
          "--epochs=100 --log_interval=50 --ckpt_interval=1000000 "
          "--pretrained_backbone=True --spectral_guided=False --use_multiscale_fusion=True --resume=True")
L0_ARCH = "--ccmc=True --prior_mode=none --use_low_level_refine=False"
ARCH = {
    'C2':   "--ccmc=True --prior_mode=gt_veg  --injection=ccmc_sr      --use_low_level_refine=False",
    'C3':   "--ccmc=True --prior_mode=gt_weed --injection=ccmc_sr      --use_low_level_refine=False",
    'C3K5': "--ccmc=True --prior_mode=gt_weed --injection=pre_sr_gate  --use_low_level_refine=False",
    'C3K6': "--ccmc=True --prior_mode=gt_weed --injection=post_sr_gate --use_low_level_refine=False",
}
# Exp01 checkpoints (zip hoặc thư mục đã giải nén trong /kaggle/input)
for z in glob.glob('/kaggle/input/**/exp01_ckpts.zip', recursive=True):
    zipfile.ZipFile(z).extractall('.')
for p in glob.glob('/kaggle/input/**/runs/*/best_val.pth', recursive=True):
    dst = os.path.join('runs', os.path.basename(os.path.dirname(p)))
    os.makedirs(dst, exist_ok=True)
    if not os.path.exists(f'{dst}/best_val.pth'):
        os.symlink(p, f'{dst}/best_val.pth')
print('Exp01 ckpts:', sorted(glob.glob('runs/*/best_val.pth')))
```

## Cell E2.1 — Bin kích thước (phải trùng Exp01: edges = [45, 147])

```python
!python src/scripts/make_instance_bins.py --dataset_path={DS} --out={BINS} > {OUT}/logs/bins.log 2>&1
edges = json.load(open(BINS))['edges']
print('edges', edges)
assert edges == [45.0, 147.0], 'Bin khác Exp01 -> dừng lại và gửi logs/bins.log'
!cp {BINS} {OUT}/instance_bins.json
```

## Cell E2.2 — Train C2, C3, C3-K5, C3-K6 (seed 0 rồi seed 1)

Thứ tự ưu tiên khi thiếu quota: C3_s0 → C2_s0 → C3K5_s0 → C3_s1 → C2_s1 → C3K6_s0 → C3K5_s1 → C3K6_s1.

```python
ORDER = [('C3', 0), ('C2', 0), ('C3K5', 0), ('C3', 1), ('C2', 1), ('C3K6', 0), ('C3K5', 1), ('C3K6', 1)]
for model, seed in ORDER:
    run = f'{model}_s{seed}'
    if os.path.exists(f'runs/{run}/run_summary.json'):
        print('skip (done)', run); continue
    print('=====', run)
    !python src/train.py {COMMON} {ARCH[model]} --seed={seed} --run_name={model}__s{seed} --out_dir=runs/{run} 2>&1 | tee {OUT}/logs/train_{run}.log | grep --line-buffered -E "Saved best|lambdas|Traceback|Error" | tail -n 3
```

> Nếu phiên bị ngắt, chạy lại cell này: run nào đã xong sẽ được bỏ qua, run dở dang sẽ resume từ `last.pth`.

## Cell E2.3 — Đánh giá validation (Boundary IoU 3 px)

```python
def evaluate(run, arch, extra=''):
    ck = f'runs/{run}/best_val.pth'
    if not os.path.exists(ck):
        print('NOT RUN', run); return
    tag = run + (extra.replace('--prior_noise=', '_noise-').replace(':', '') if extra else '')
    !python src/evaluate_ccmc.py --dataset_path={DS} --split=val --ckpt={ck} {arch} {extra} --bins_json={BINS} --boundary_px=3 --run_name={tag} --out_prefix={OUT}/{tag}_val > {OUT}/logs/eval_{tag}.log 2>&1
    !grep -E "mIoU|small|GT prior" {OUT}/logs/eval_{tag}.log

# L0 / LiteSent đánh giá lại (BIoU 3 px), từ checkpoint Exp01
for seed in (0, 1):
    evaluate(f'L0_s{seed}', L0_ARCH)
evaluate('LiteSent_s0', '--ccmc=False --use_low_level_refine=True')
# Oracle models (GT leakage)
for model, seed in ORDER:
    evaluate(f'{model}_s{seed}', ARCH[model])
```

## Cell E2.4 — Độ nhạy với prior GT bị nhiễu (C3 seed 0, cùng checkpoint)

```python
for noise in ('dilate:3', 'erode:1', 'drop:0.3', 'shift:4'):
    evaluate('C3_s0', ARCH['C3'], f'--prior_noise={noise}')
```

## Cell E2.5 — So sánh paired và gate

```python
def compare(a, b, gate, seeds=(0, 1)):
    # chỉ dùng các seed có ở CẢ HAI mô hình (paired)
    common = [s for s in seeds if os.path.exists(f'{OUT}/{a}_s{s}_val_per_image.csv')
              and os.path.exists(f'{OUT}/{b}_s{s}_val_per_image.csv')]
    if not common:
        print('NOT RUN', a, 'vs', b); return
    fa = ','.join(f'{OUT}/{a}_s{s}_val_per_image.csv' for s in common)
    fb = ','.join(f'{OUT}/{b}_s{s}_val_per_image.csv' for s in common)
    name = f'{a}_vs_{b}'
    !python src/scripts/compare_runs.py --a={fa} --b={fb} --name_a={a} --name_b={b} --gate={gate} --out={OUT}/cmp_{name}.json 2>&1 | tee {OUT}/logs/cmp_{name}.log

compare('C3', 'C2', 'screen')      # G2-screen (một chiều)
compare('C3', 'C3K5', 'g4')        # cùng leakage: CCMC-SR so với gate trước SR
compare('C3', 'C3K6', 'g4')        # cùng leakage: CCMC-SR so với gate sau SR
compare('C3', 'L0', 'none')        # biên độ trần, chỉ mô tả
compare('C2', 'L0', 'none')
```

## Cell E2.6 — Gom kết quả và đóng gói

```python
import shutil
for run_dir in sorted(glob.glob('runs/*')):
    if not os.path.exists(f'{run_dir}/run_config.json'):
        continue
    dst = f'{OUT}/runs_meta/{os.path.basename(run_dir)}'
    os.makedirs(dst, exist_ok=True)
    for f in ('history.jsonl', 'run_config.json', 'run_summary.json'):
        if os.path.exists(f'{run_dir}/{f}'):
            shutil.copy(f'{run_dir}/{f}', dst)
!python src/scripts/collect_results.py --results_dir={OUT} --runs_dir=runs --out={OUT}/metrics.csv
!zip -qr /kaggle/working/{EXP}_results.zip {OUT}
!zip -qr /kaggle/working/{EXP}_ckpts.zip runs/C*/best_val.pth
!ls -lh /kaggle/working/*.zip
```

## Gửi lại để review

Giải nén `exp02_results.zip` vào `doc/exp02_results/` (giống Exp01). Ghi kèm trạng thái `DONE / PARTIAL / NOT RUN` cho 8 run của E2.2. Lưu `exp02_ckpts.zip` thành Kaggle Dataset: Exp03 cần C3 để làm ma trận oracle 2×2.
