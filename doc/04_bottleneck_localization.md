# Exp04 — Định vị nút thắt thật của cỏ nhỏ (decoder 1/2 và nguồn chi tiết), 3 seed

> Bối cảnh: Exp03 kích hoạt được **quy tắc dừng CCMC-SR** (REVIEW mục 5). Ngay cả khi dùng nhãn thật làm prior, toán tử vẫn không vượt L0. Trong khi đó LLR (LiteSent) tăng Small-Weed Recall +14,5 pp (Exp01). Trước khi thiết kế phương pháp mới, cần biết **lợi ích đó đến từ đâu**. Thí nghiệm này cũng đáp ứng hai yêu cầu trong nhận xét của thầy ([paper/README review.md](../paper/README%20review.md), Ưu tiên 2):
> - "đối chứng decoder độ phân giải 1/2 **không có** nhánh raw-input"
> - "so sánh nhánh raw-input với việc dùng trực tiếp feature F1"

## Mục tiêu và cách diễn giải (khóa trước)

| ID | Cấu hình (`--ccmc=False --segformer=True --use_low_level_refine=True`) | Ý nghĩa |
|---|---|---|
| LiteSent | `--llr_source=raw` | Decoder 1/2 + chi tiết từ 5 băng thô |
| D½-none | `--llr_source=none` | Cùng decoder 1/2 và cùng số conv refine, **không** có thông tin chi tiết bổ sung |
| D½-F1 | `--llr_source=f1` | Cùng decoder 1/2, chi tiết lấy từ F1 của MiT (stride 4) thay cho băng thô |

| Kết quả (Small-Weed Recall, 3 seed, bootstrap phân tầng) | Kết luận về nút thắt | Hướng phương pháp tiếp theo |
|---|---|---|
| D½-none ≈ LiteSent | Nút thắt nằm ở **độ phân giải đầu ra** của decoder | Toán tử ở decoder/upsampling |
| LiteSent > D½-F1 ≈ D½-none | Chi tiết cỏ nhỏ **mất ngay ở patch embedding stage 1** (stride 4) và không khôi phục được từ F1 | **Sửa toán tử bên trong MiT ở stage 1**: phương pháp mới vẫn giữ trọng số pretrained, không ghép mô hình |
| D½-F1 ≈ LiteSent > D½-none | F1 vẫn còn thông tin; lỗi do decoder dùng F1 chưa tốt | Toán tử ở decoder dùng F1 |

Ngân sách: 8 run (LiteSent s1–s2, D½-none s0–s2, D½-F1 s0–s2) × khoảng 24 phút ≈ 3,2 giờ GPU. LiteSent s0 lấy lại từ Exp01.

---

## Bước 0 — Commit + push code mới, chạy S1–S4 và S6. Add Input: `ccmc-ckpts`, `exp02-ckpts`, `exp03-ckpts`.

## Cell E4.0 — Biến dùng chung

```python
import os, glob, json
os.chdir('/kaggle/working/WeedsGalore_Ipr28')
EXP = 'exp04'
OUT = f'results/{EXP}'
BINS = 'results/instance_bins.json'
os.makedirs(f'{OUT}/logs', exist_ok=True)
COMMON = (f"--dataset_path={DS} --dataset_size_train=104 --in_channels=5 --num_classes=3 "
          "--batch_size=8 --num_workers=4 --lr=0.0003 "
          "--weight_bg=0.9 --weight_crop=1.0 --weight_weed=1.3 --use_dice_loss=True --dice_weight=0.3 "
          "--epochs=100 --log_interval=50 --ckpt_interval=1000000 "
          "--pretrained_backbone=True --spectral_guided=False --use_multiscale_fusion=True --resume=True")
TRAIN_ARCH = {
    'L0':       "--ccmc=True --prior_mode=none --use_low_level_refine=False",
    'LiteSent': "--ccmc=False --segformer=True --use_low_level_refine=True --llr_source=raw",
    'Dnone':    "--ccmc=False --segformer=True --use_low_level_refine=True --llr_source=none",
    'DF1':      "--ccmc=False --segformer=True --use_low_level_refine=True --llr_source=f1",
}
EVAL_ARCH = {k: v.replace('--segformer=True ', '') for k, v in TRAIN_ARCH.items()}
for p in glob.glob('/kaggle/input/**/runs/*/best_val.pth', recursive=True):
    dst = os.path.join('runs', os.path.basename(os.path.dirname(p)))
    os.makedirs(dst, exist_ok=True)
    if not os.path.exists(f'{dst}/best_val.pth'):
        os.symlink(p, f'{dst}/best_val.pth')
print(sorted(glob.glob('runs/*/best_val.pth')))
!python src/scripts/make_instance_bins.py --dataset_path={DS} --out={BINS} > {OUT}/logs/bins.log 2>&1
assert json.load(open(BINS))['edges'] == [45.0, 147.0]
```

## Cell E4.1 — Train

```python
ORDER = [('Dnone', 0), ('DF1', 0), ('LiteSent', 1), ('Dnone', 1), ('DF1', 1),
         ('LiteSent', 2), ('Dnone', 2), ('DF1', 2)]
for model, seed in ORDER:
    run = f'{model}_s{seed}'
    if os.path.exists(f'runs/{run}/best_val.pth') and (os.path.islink(f'runs/{run}/best_val.pth')
                                                       or os.path.exists(f'runs/{run}/run_summary.json')):
        print('skip (done)', run); continue
    print('=====', run)
    !python src/train.py {COMMON} {TRAIN_ARCH[model]} --seed={seed} --run_name={model}__s{seed} --out_dir=runs/{run} 2>&1 | tee {OUT}/logs/train_{run}.log | grep --line-buffered -E "Saved best|Traceback|Error" | tail -n 2
```

## Cell E4.2 — Đánh giá validation (cả L0 0–2 và LiteSent s0 lấy từ các Exp trước)

```python
def evaluate(run, arch):
    ck = f'runs/{run}/best_val.pth'
    if not os.path.exists(ck):
        print('NOT RUN', run); return
    !python src/evaluate_ccmc.py --dataset_path={DS} --split=val --ckpt={ck} {arch} --bins_json={BINS} --boundary_px=3 --run_name={run} --out_prefix={OUT}/{run}_val > {OUT}/logs/eval_{run}.log 2>&1
    !grep -E "^mIoU|  small" {OUT}/logs/eval_{run}.log

for model in ('L0', 'LiteSent', 'Dnone', 'DF1'):
    for s in (0, 1, 2):
        evaluate(f'{model}_s{s}', EVAL_ARCH[model])
```

## Cell E4.3 — So sánh (bootstrap phân tầng, 3 seed)

```python
def compare(a, b, gate='none', seeds=(0, 1, 2)):
    common = [s for s in seeds if os.path.exists(f'{OUT}/{a}_s{s}_val_per_image.csv')
              and os.path.exists(f'{OUT}/{b}_s{s}_val_per_image.csv')]
    if not common:
        print('NOT RUN', a, b); return
    fa = ','.join(f'{OUT}/{a}_s{s}_val_per_image.csv' for s in common)
    fb = ','.join(f'{OUT}/{b}_s{s}_val_per_image.csv' for s in common)
    !python src/scripts/compare_runs.py --a={fa} --b={fb} --name_a={a} --name_b={b} --gate={gate} --out={OUT}/cmp_{a}_vs_{b}.json 2>&1 | tee {OUT}/logs/cmp_{a}_vs_{b}.log

compare('LiteSent', 'L0')      # tổng lợi ích của LLR (3 seed)
compare('Dnone', 'L0')         # chỉ tăng độ phân giải decoder
compare('LiteSent', 'Dnone')   # giá trị thêm của chi tiết từ băng thô
compare('LiteSent', 'DF1')     # băng thô so với F1
compare('DF1', 'Dnone')        # F1 có thêm gì không
```

## Cell E4.4 — Profile (600×600, cùng quy ước)

```python
for label in ('Dnone', 'DF1', 'LiteSent'):
    !python src/scripts/profile_model.py {EVAL_ARCH[label]} --label={label} --out={OUT}/profile_{label}.json 2>&1 | grep -E "GFLOPs|latency_ms_median|params_M"
```

## Cell E4.5 — Gom và đóng gói

```python
import shutil
for run_dir in sorted(glob.glob('runs/*')):
    if os.path.islink(f'{run_dir}/best_val.pth') or not os.path.exists(f'{run_dir}/history.jsonl'):
        continue
    dst = f'{OUT}/runs_meta/{os.path.basename(run_dir)}'
    os.makedirs(dst, exist_ok=True)
    for f in ('history.jsonl', 'run_config.json', 'run_summary.json'):
        if os.path.exists(f'{run_dir}/{f}'):
            shutil.copy(f'{run_dir}/{f}', dst)
!python src/scripts/collect_results.py --results_dir={OUT} --runs_dir=runs --out={OUT}/metrics.csv > {OUT}/logs/collect.log 2>&1
!zip -qr /kaggle/working/{EXP}_results.zip {OUT}
new_ckpts = ' '.join(p for p in sorted(glob.glob('runs/*/best_val.pth')) if not os.path.islink(p))
!zip -qr /kaggle/working/{EXP}_ckpts.zip {new_ckpts}
!ls -lh /kaggle/working/*.zip
```

## Gửi lại để review

Giải nén `exp04_results.zip` vào `doc/exp04_results/`, upload `exp04_ckpts.zip` thành Kaggle Dataset, rồi báo trạng thái của 8 run.
