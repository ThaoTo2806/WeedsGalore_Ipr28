# Exp03 — Kích hoạt lại cơ chế: oracle C2/C3 với λ học được thật, 3 seed (Sprint D3 làm lại, Gate G2-screen)

> Lý do làm lại: ở Exp02, λ bị **kẹt ở 0,05** (xem [REVIEW_REPORT.md](REVIEW_REPORT.md) mục 4). Với Adam ở lr 3e-4, mỗi bước chỉ đổi logit của λ tối đa khoảng 3e-4, nên sau 1 300 bước λ không thể vượt quá khoảng 0,07. Vì vậy prior gần như không tác động: C3 với prior GT bị nhiễu cho ra **cùng** kết quả với prior sạch. Kết luận G2 của Exp02 không hợp lệ (cơ chế chưa được kích hoạt).

## Mục tiêu (khóa trước khi xem kết quả)

| ID | Cấu hình | Mục đích |
|---|---|---|
| **C3a / C2a** | GT-Weed / GT-Veg, CCMC-SR, λ học được: init 0,5, `--lam_lr=0.01` | G2-screen với cơ chế đã hoạt động |
| **C3f** | GT-Weed, CCMC-SR, λ **cố định 1,0** | Liều tối đa: toán tử có "đòn bẩy" hay không |
| **C3K5a** | GT-Weed, gate trước SR (chuẩn hóa toàn ảnh), λ học được như C3a | Cùng leakage: chuẩn hóa theo cửa sổ có tác dụng hay không |
| L0 s2 | Baseline seed thứ 3 | Đủ 3 seed paired cho mọi so sánh |

**Kiểm tra đầu tiên — cơ chế có hoạt động không (mechanism activity):** trên cùng checkpoint C3a, đánh giá với prior GT và với prior đồng đều (`--prior_override=uniform`). Nếu Small-Weed Recall chênh dưới 1 pp thì toán tử **không có đòn bẩy**, ngay cả với nhãn thật. Khi đó dừng CCMC-SR (dạng hiện tại) và chuyển phương án.

**Quy tắc gate (đã cập nhật, xem REVIEW mục 4.4):**
- Bootstrap phân tầng gồm seed (paired) và ảnh. PASS/SCREEN-PASS cần ít nhất 3 seed.
- `SmallWeedRecall` và `cov_small` phải cùng dấu.
- **G2-screen:** C3a − C2a. Không còn coi là NO-GO nếu cơ chế chưa hoạt động.
- **Trần mô tả:** C3a − L0 và C3f − L0. Nếu C3a/C3f **không** vượt L0 dù có rò rỉ nhãn thì CCMC-SR không đủ năng lực, và không đi tiếp sang L3.

**A/A test (cho phần phương pháp thống kê của paper):** so sánh L0_s0 với L0_s1 (cùng mô hình, khác seed) chỉ bằng bootstrap theo ảnh. Nếu ra "có ý nghĩa" thì chứng tỏ bootstrap chỉ theo ảnh đánh giá thấp phương sai, và đây là lý do bắt buộc dùng bootstrap theo seed.

Ngân sách: 13 run train × khoảng 17 phút ≈ 3,7 giờ GPU. Nếu thiếu quota, chạy theo thứ tự ưu tiên trong E3.2.

---

## Bước 0 — Commit + push code mới, chạy S1–S4 và S6 trong [00_setup_kaggle.md](00_setup_kaggle.md). Add Input: `ccmc-ckpts` (Exp01) và `exp02-ckpts`.

## Cell E3.0 — Biến dùng chung, nối checkpoint cũ

```python
import os, glob, json
os.chdir('/kaggle/working/WeedsGalore_Ipr28')
EXP = 'exp03'
OUT = f'results/{EXP}'
BINS = 'results/instance_bins.json'
os.makedirs(f'{OUT}/logs', exist_ok=True)
COMMON = (f"--dataset_path={DS} --dataset_size_train=104 --in_channels=5 --num_classes=3 "
          "--batch_size=8 --num_workers=4 --lr=0.0003 "
          "--weight_bg=0.9 --weight_crop=1.0 --weight_weed=1.3 --use_dice_loss=True --dice_weight=0.3 "
          "--epochs=100 --log_interval=50 --ckpt_interval=1000000 "
          "--pretrained_backbone=True --spectral_guided=False --use_multiscale_fusion=True --resume=True")
BASE = "--ccmc=True --use_low_level_refine=False"
LAM_LEARN = "--lam_init=0.5 --lam_lr=0.01 --learn_lambda=True"
LAM_FIXED1 = "--lam_init=0.999 --learn_lambda=False"   # sigmoid(logit(0.999)) ≈ 1
ARCH = {
    'L0':    f"{BASE} --prior_mode=none",
    'C3a':   f"{BASE} --prior_mode=gt_weed --injection=ccmc_sr {LAM_LEARN}",
    'C2a':   f"{BASE} --prior_mode=gt_veg  --injection=ccmc_sr {LAM_LEARN}",
    'C3f':   f"{BASE} --prior_mode=gt_weed --injection=ccmc_sr {LAM_FIXED1}",
    'C3K5a': f"{BASE} --prior_mode=gt_weed --injection=pre_sr_gate {LAM_LEARN}",
}
# Cờ chỉ dùng cho train (evaluate_ccmc không có lam_*)
EVAL_ARCH = {k: ' '.join(t for t in v.split() if not t.startswith(('--lam_', '--learn_lambda')))
             for k, v in ARCH.items()}
# Nối checkpoint Exp01/Exp02 (chỉ đọc)
for p in glob.glob('/kaggle/input/**/runs/*/best_val.pth', recursive=True):
    dst = os.path.join('runs', os.path.basename(os.path.dirname(p)))
    os.makedirs(dst, exist_ok=True)
    if not os.path.exists(f'{dst}/best_val.pth'):
        os.symlink(p, f'{dst}/best_val.pth')
print(sorted(glob.glob('runs/*/best_val.pth')))
```

## Cell E3.1 — Bin (phải trùng [45, 147])

```python
!python src/scripts/make_instance_bins.py --dataset_path={DS} --out={BINS} > {OUT}/logs/bins.log 2>&1
assert json.load(open(BINS))['edges'] == [45.0, 147.0]
```

## Cell E3.2 — Train (theo thứ tự ưu tiên)

```python
ORDER = [('L0', 2), ('C3a', 0), ('C2a', 0), ('C3f', 0), ('C3a', 1), ('C2a', 1), ('C3a', 2), ('C2a', 2),
         ('C3K5a', 0), ('C3K5a', 1), ('C3K5a', 2), ('C3f', 1), ('C3f', 2)]
for model, seed in ORDER:
    run = f'{model}_s{seed}'
    if os.path.exists(f'runs/{run}/run_summary.json'):
        print('skip (done)', run); continue
    print('=====', run)
    !python src/train.py {COMMON} {ARCH[model]} --seed={seed} --run_name={model}__s{seed} --out_dir=runs/{run} 2>&1 | tee {OUT}/logs/train_{run}.log | grep --line-buffered -E "Saved best|Traceback|Error" | tail -n 2
    !grep -E "lambdas" {OUT}/logs/train_{run}.log | tail -n 1
```

## Cell E3.3 — Quỹ đạo λ (kiểm tra λ đã thực sự di chuyển)

```python
for h in sorted(glob.glob('runs/*/history.jsonl')):
    rows = [json.loads(l) for l in open(h)]
    if not rows[0]['lambdas']:
        continue
    lam = [r['lambdas'] for r in rows]; best = rows[-1]['best_epoch']
    print(f"{h.split('/')[1]:10s} ep0={[round(v,3) for v in lam[0]]} ep10={[round(v,3) for v in lam[min(10,len(lam)-1)]]} "
          f"ep50={[round(v,3) for v in lam[min(50,len(lam)-1)]]} best(ep{best})={[round(v,3) for v in lam[best]]}")
```

## Cell E3.4 — Đánh giá validation

```python
def evaluate(run, arch, extra='', tag=None):
    ck = f'runs/{run}/best_val.pth'
    if not os.path.exists(ck):
        print('NOT RUN', run); return
    tag = tag or run
    !python src/evaluate_ccmc.py --dataset_path={DS} --split=val --ckpt={ck} {arch} {extra} --bins_json={BINS} --boundary_px=3 --run_name={tag} --out_prefix={OUT}/{tag}_val > {OUT}/logs/eval_{tag}.log 2>&1
    !grep -E "^mIoU|  small" {OUT}/logs/eval_{tag}.log

for s in (0, 1, 2):
    evaluate(f'L0_s{s}', EVAL_ARCH['L0'])
for model, seed in ORDER:
    if model != 'L0':
        evaluate(f'{model}_s{seed}', EVAL_ARCH[model])
```

## Cell E3.5 — Cơ chế có hoạt động không? (cùng checkpoint, chỉ đổi prior lúc suy luận)

```python
for model in ('C3a', 'C3f'):
    for s in (0, 1, 2):
        run = f'{model}_s{s}'
        evaluate(run, EVAL_ARCH[model], '--prior_override=uniform', tag=f'{run}_ovr-uniform')
for noise in ('dilate:3', 'drop:0.3', 'shift:4'):
    evaluate('C3a_s0', EVAL_ARCH['C3a'], f'--prior_noise={noise}', tag=f"C3a_s0_noise-{noise.replace(':','')}")
```

## Cell E3.6 — So sánh (bootstrap phân tầng theo seed + ảnh) và A/A test

```python
def compare(a, b, gate, seeds=(0, 1, 2), suffix_a='', suffix_b='', name=None, extra=''):
    common = [s for s in seeds if os.path.exists(f'{OUT}/{a}_s{s}{suffix_a}_val_per_image.csv')
              and os.path.exists(f'{OUT}/{b}_s{s}{suffix_b}_val_per_image.csv')]
    if not common:
        print('NOT RUN', a, b); return
    fa = ','.join(f'{OUT}/{a}_s{s}{suffix_a}_val_per_image.csv' for s in common)
    fb = ','.join(f'{OUT}/{b}_s{s}{suffix_b}_val_per_image.csv' for s in common)
    name = name or f'{a}{suffix_a}_vs_{b}{suffix_b}'
    !python src/scripts/compare_runs.py --a={fa} --b={fb} --name_a={a}{suffix_a} --name_b={b}{suffix_b} --gate={gate} {extra} --out={OUT}/cmp_{name}.json 2>&1 | tee {OUT}/logs/cmp_{name}.log

# 1) Cơ chế: prior GT so với prior đồng đều trên CÙNG checkpoint
compare('C3a', 'C3a', 'none', suffix_b='_ovr-uniform', name='C3a_activity')
compare('C3f', 'C3f', 'none', suffix_b='_ovr-uniform', name='C3f_activity')
# 2) G2-screen
compare('C3a', 'C2a', 'screen')
# 3) Trần mô tả và đối chứng cùng leakage
compare('C3a', 'L0', 'none'); compare('C3f', 'L0', 'none'); compare('C2a', 'L0', 'none')
compare('C3a', 'C3K5a', 'g4')
# 4) A/A test: cùng mô hình L0, khác seed, chỉ bootstrap theo ảnh
!python src/scripts/compare_runs.py --a={OUT}/L0_s0_val_per_image.csv --b={OUT}/L0_s1_val_per_image.csv --name_a=L0_s0 --name_b=L0_s1 --seed_bootstrap=False --out={OUT}/cmp_AA_L0s0_vs_L0s1.json 2>&1 | tee {OUT}/logs/cmp_AA.log
```

## Cell E3.7 — Gom kết quả và đóng gói

```python
import shutil
for run_dir in sorted(glob.glob('runs/*')):
    if not os.path.exists(f'{run_dir}/history.jsonl') or os.path.islink(f'{run_dir}/best_val.pth'):
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

Giải nén `exp03_results.zip` vào `doc/exp03_results/`, upload `exp03_ckpts.zip` thành Kaggle Dataset, rồi báo trạng thái DONE / PARTIAL / NOT RUN của 13 run.
