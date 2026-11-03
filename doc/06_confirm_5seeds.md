# Exp06 — Xác nhận MCSD-g với seed mới (3, 4) → đủ 5 seed, cùng ablation số vòng Sinkhorn

> Exp05 (REVIEW mục 7): **MCSD-g đạt gate trên val với 3 seed**, Small-Weed Recall +5,81 [+3,25; +8,48]. Tuy vậy MCSD-g được chọn sau khi xem val (có 2 biến thể MCSD/MCSD-g), nên cần **xác nhận độc lập trên seed chưa từng chạy** trước khi khóa cấu hình và mở test.

## Đăng ký trước (khóa trước khi chạy)

- **Cấu hình khóa:** MCSD-g = `--head=mcsd --spectral_guide=True --sinkhorn_iters=5 --mcsd_tau=1.0 --occ_weight=0.3`, cùng protocol train cũ. Không đổi gì.
- **Xác nhận (confirmatory):** chỉ dùng seed 3 và 4, so MCSD-g với L0. ĐẠT nếu Small-Weed Recall delta > 0, CI 95 % > 0, `cov_small` cùng dấu và Crop không giảm quá 1 pp (`--gate=abs --min_delta=0 --min_seeds_pass=2`).
- **Bảng 5 seed (seed 0–4):** MCSD-g, MCSD, PS, L0, D½-none, LiteSent, dùng cho bảng chính của paper (val).
- **Nếu xác nhận ĐẠT:** khóa MCSD-g, sang Exp07 (test **một lần**, ablation, dataset thứ hai).
- **Nếu KHÔNG đạt:** ghi INCONCLUSIVE, không mở test, review lại.
- **Ablation số vòng Sinkhorn lúc suy luận** (cùng checkpoint, không train lại): 1, 5 (mặc định), 20. Chỉ mô tả.

Ngân sách: 12 run × khoảng 20 phút ≈ 4 giờ GPU (thứ tự ưu tiên ở E6.1), cộng khoảng 20 phút đánh giá.

---

## Bước 0 — Commit + push code (có thêm `src/scripts/mass_check.py`), chạy S1–S4 và S6. Add Input: `ccmc-ckpts`, `exp03-ckpts`, `exp04-ckpts`, `exp05-ckpts`.

## Cell E6.0 — Biến dùng chung

```python
import os, glob, json
os.chdir('/kaggle/working/WeedsGalore_Ipr28')
EXP = 'exp06'
OUT = f'results/{EXP}'
BINS = 'results/instance_bins.json'
os.makedirs(f'{OUT}/logs', exist_ok=True)
COMMON = (f"--dataset_path={DS} --dataset_size_train=104 --in_channels=5 --num_classes=3 "
          "--batch_size=8 --num_workers=4 --lr=0.0003 "
          "--weight_bg=0.9 --weight_crop=1.0 --weight_weed=1.3 --use_dice_loss=True --dice_weight=0.3 "
          "--epochs=100 --log_interval=50 --ckpt_interval=1000000 "
          "--pretrained_backbone=True --spectral_guided=False --use_multiscale_fusion=True --resume=True")
SEG = "--ccmc=False --segformer=True"
TRAIN_ARCH = {
    'L0':       "--ccmc=True --prior_mode=none --use_low_level_refine=False",
    'MCSDg':    f"{SEG} --use_low_level_refine=False --head=mcsd --spectral_guide=True",
    'MCSD':     f"{SEG} --use_low_level_refine=False --head=mcsd",
    'PS':       f"{SEG} --use_low_level_refine=False --head=pixelshuffle",
    'Dnone':    f"{SEG} --use_low_level_refine=True --llr_source=none",
    'LiteSent': f"{SEG} --use_low_level_refine=True --llr_source=raw",
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

## Cell E6.1 — Train seed 3, 4 (ưu tiên cặp xác nhận L0/MCSD-g)

```python
ORDER = [('L0', 3), ('MCSDg', 3), ('L0', 4), ('MCSDg', 4), ('MCSD', 3), ('MCSD', 4),
         ('Dnone', 3), ('Dnone', 4), ('LiteSent', 3), ('LiteSent', 4), ('PS', 3), ('PS', 4)]
for model, seed in ORDER:
    run = f'{model}_s{seed}'
    if os.path.exists(f'runs/{run}/run_summary.json'):
        print('skip (done)', run); continue
    print('=====', run)
    !python src/train.py {COMMON} {TRAIN_ARCH[model]} --seed={seed} --run_name={model}__s{seed} --out_dir=runs/{run} 2>&1 | tee {OUT}/logs/train_{run}.log | grep --line-buffered -E "Saved best|Traceback|Error" | tail -n 2
```

## Cell E6.2 — Đánh giá validation, seed 0–4

```python
def evaluate(model, seed, arch, extra='', tag=None):
    ck = f'runs/{model}_s{seed}/best_val.pth'
    if not os.path.exists(ck):
        print('NOT RUN', model, seed); return
    name = f'{tag or model}_s{seed}'
    !python src/evaluate_ccmc.py --dataset_path={DS} --split=val --ckpt={ck} {arch} {extra} --bins_json={BINS} --boundary_px=3 --run_name={name} --out_prefix={OUT}/{name}_val > {OUT}/logs/eval_{name}.log 2>&1
    !grep -E "^mIoU|  small" {OUT}/logs/eval_{name}.log

for model in ('L0', 'MCSDg', 'MCSD', 'PS', 'Dnone', 'LiteSent'):
    for s in range(5):
        evaluate(model, s, EVAL_ARCH[model])
# Ablation: số vòng Sinkhorn lúc suy luận (cùng checkpoint MCSD-g)
for it in (1, 20):
    for s in range(5):
        evaluate('MCSDg', s, EVAL_ARCH['MCSDg'], f'--sinkhorn_iters={it}', tag=f'MCSDgIt{it}')
```

## Cell E6.3 — Xác nhận (seed 3–4) và bảng 5 seed

```python
def compare(a, b, gate='none', seeds=range(5), extra='', name=None):
    common = [s for s in seeds if os.path.exists(f'{OUT}/{a}_s{s}_val_per_image.csv')
              and os.path.exists(f'{OUT}/{b}_s{s}_val_per_image.csv')]
    if not common:
        print('NOT RUN', a, b); return
    fa = ','.join(f'{OUT}/{a}_s{s}_val_per_image.csv' for s in common)
    fb = ','.join(f'{OUT}/{b}_s{s}_val_per_image.csv' for s in common)
    name = name or f'{a}_vs_{b}'
    !python src/scripts/compare_runs.py --a={fa} --b={fb} --name_a={a} --name_b={b} --gate={gate} {extra} --out={OUT}/cmp_{name}.json 2>&1 | tee {OUT}/logs/cmp_{name}.log

# 1) XÁC NHẬN trên seed mới
compare('MCSDg', 'L0', 'abs', seeds=(3, 4), extra='--min_delta=0 --min_seeds_pass=2', name='CONFIRM_MCSDg_vs_L0_s34')
# 2) Bảng 5 seed
compare('MCSDg', 'L0', 'abs')
compare('MCSD', 'L0', 'abs')
compare('MCSD', 'PS', 'g4')
compare('MCSDg', 'MCSD')
compare('PS', 'L0')
compare('Dnone', 'L0')
compare('LiteSent', 'L0')
compare('MCSDg', 'Dnone')
compare('MCSDg', 'LiteSent')
# 3) Ablation số vòng Sinkhorn
compare('MCSDgIt1', 'MCSDg')
compare('MCSDgIt20', 'MCSDg')
```

## Cell E6.4 — Bảo toàn khối lượng theo số vòng Sinkhorn (MCSD-g seed 0)

```python
for it in (1, 5, 20):
    !python src/scripts/mass_check.py --dataset_path={DS} --ckpt=runs/MCSDg_s0/best_val.pth --ccmc=False --head=mcsd --spectral_guide=True --sinkhorn_iters={it} --out={OUT}/mass_check_MCSDg_s0_it{it}.json
```

## Cell E6.5 — Gom và đóng gói

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

Giải nén `exp06_results.zip` vào `doc/exp06_results/`, upload `exp06_ckpts.zip` thành Kaggle Dataset, báo trạng thái của 12 run. **Không chạy test** cho tới khi review xong Exp06.
