# Exp05 — Phương pháp mới: MCSD (Mass-Conserving Sub-cell Decoding), 3 seed

> Thiết kế dựa trực tiếp trên bằng chứng Exp01–Exp04 (REVIEW mục 6):
> - Cỏ nhỏ mất chủ yếu ở **bước dự đoán 1/4 + bilinear ×4**. Decoder 1/2 dày đặc không có input thêm (D½-none) đã lấy lại +11,3 pp Small-Weed Recall, nhưng tốn **+27 GFLOPs** (gấp 3 L0).
> - Diag-D2: phép **trung bình** làm loãng bằng chứng cỏ nhỏ.
>
> MCSD giữ nguyên MiT-B0 pretrained và decoder 1/4 của L0. Chỉ **thay lớp phân loại 1×1 + bilinear** bằng hai head rẻ trên cùng feature 1/4:
> - **(i) occupancy:** tỉ lệ từng lớp trong mỗi ô 4×4 px, học bằng soft occupancy của GT;
> - **(ii) placement:** điểm của từng lớp tại 16 sub-pixel của ô.
>
> Trong mỗi ô, xác suất sub-pixel là **kế hoạch vận chuyển tối ưu entropic (Sinkhorn)** giữa khối lượng lớp (16·q_c) và 16 sub-pixel. Cách này **bảo toàn khối lượng lớp**: một ô dự đoán 15 % là Weed sẽ trao khoảng 2,4 sub-pixel cho Weed, ở đúng chỗ placement cao nhất, thay vì bị argmax của bilinear xóa mất.

## Mục tiêu (khóa trước khi xem kết quả)

| ID | Lệnh (`--ccmc=False --segformer=True --use_low_level_refine=False`) | Vai trò |
|---|---|---|
| **MCSD** | `--head=mcsd` | Phương pháp đề xuất |
| PS | `--head=pixelshuffle` | Đối chứng: **cùng** hai head, **cùng** occupancy loss, chỉ bỏ ràng buộc bảo toàn khối lượng (softmax từng pixel) |
| MCSD-g | `--head=mcsd --spectral_guide=True` | Biến thể: placement có thêm prior full-res từ 5 băng (NIR/RE) |
| L0, D½-none, LiteSent | checkpoint các Exp trước | Tham chiếu |

| Gate | Phép so sánh (3 seed, bootstrap phân tầng) | Điều kiện đi tiếp |
|---|---|---|
| **H-MCSD** (hiệu quả) | MCSD − L0 | Small-Weed Recall ≥ +3 pp, CI > 0, `cov_small` cùng dấu, Crop IoU không giảm quá 1 pp (`--gate=abs`) |
| **H-mech** (cơ chế) | MCSD − PS | Delta > 0, CI > 0 (`--gate=g4`). Nếu không đạt thì lợi ích chỉ do có head sub-pixel, **không** claim bảo toàn khối lượng |
| H-eff (hiệu suất, mô tả) | MCSD so với D½-none | Tỉ lệ lợi ích cỏ nhỏ lấy lại được = (MCSD − L0)/(D½-none − L0), so với tỉ lệ FLOPs tăng thêm. **Mục tiêu Q1:** lấy lại ≥ 50 % lợi ích với ≤ 10 % chi phí thêm của D½-none |
| H-spec (mô tả) | MCSD-g − MCSD | Chỉ báo cáo; không dùng để chọn cấu hình trên test |

Siêu tham số MCSD khóa trước: 5 vòng Sinkhorn, τ = 1, occupancy loss × 0,3, placement khởi tạo bằng 0 (ban đầu sub-pixel nhận đúng tỉ lệ ô). Không tinh chỉnh τ/iters trước khi có kết quả gate.

Ngân sách: 9 run × khoảng 18–20 phút ≈ 3 giờ GPU.

---

## Bước 0 — Commit + push code, chạy S1–S4 và **S6** (có thêm test cho MCSD, phải pass hết). Add Input: `ccmc-ckpts`, `exp03-ckpts`, `exp04-ckpts`.

## Cell E5.0 — Biến dùng chung

```python
import os, glob, json
os.chdir('/kaggle/working/WeedsGalore_Ipr28')
EXP = 'exp05'
OUT = f'results/{EXP}'
BINS = 'results/instance_bins.json'
os.makedirs(f'{OUT}/logs', exist_ok=True)
COMMON = (f"--dataset_path={DS} --dataset_size_train=104 --in_channels=5 --num_classes=3 "
          "--batch_size=8 --num_workers=4 --lr=0.0003 "
          "--weight_bg=0.9 --weight_crop=1.0 --weight_weed=1.3 --use_dice_loss=True --dice_weight=0.3 "
          "--epochs=100 --log_interval=50 --ckpt_interval=1000000 "
          "--pretrained_backbone=True --spectral_guided=False --use_multiscale_fusion=True --resume=True")
SEG = "--ccmc=False --segformer=True --use_low_level_refine=False"
TRAIN_ARCH = {
    'MCSD':  f"{SEG} --head=mcsd",
    'PS':    f"{SEG} --head=pixelshuffle",
    'MCSDg': f"{SEG} --head=mcsd --spectral_guide=True",
}
EVAL_ARCH = {
    'L0':       "--ccmc=True --prior_mode=none --use_low_level_refine=False",
    'Dnone':    "--ccmc=False --use_low_level_refine=True --llr_source=none",
    'LiteSent': "--ccmc=False --use_low_level_refine=True --llr_source=raw",
    'MCSD':     "--ccmc=False --use_low_level_refine=False --head=mcsd",
    'PS':       "--ccmc=False --use_low_level_refine=False --head=pixelshuffle",
    'MCSDg':    "--ccmc=False --use_low_level_refine=False --head=mcsd --spectral_guide=True",
}
for p in glob.glob('/kaggle/input/**/runs/*/best_val.pth', recursive=True):
    dst = os.path.join('runs', os.path.basename(os.path.dirname(p)))
    os.makedirs(dst, exist_ok=True)
    if not os.path.exists(f'{dst}/best_val.pth'):
        os.symlink(p, f'{dst}/best_val.pth')
print(sorted(glob.glob('runs/*/best_val.pth')))
!python src/scripts/make_instance_bins.py --dataset_path={DS} --out={BINS} > {OUT}/logs/bins.log 2>&1
assert json.load(open(BINS))['edges'] == [45.0, 147.0]
```

## Cell E5.1 — Train (theo thứ tự ưu tiên)

```python
ORDER = [('MCSD', 0), ('PS', 0), ('MCSD', 1), ('PS', 1), ('MCSD', 2), ('PS', 2),
         ('MCSDg', 0), ('MCSDg', 1), ('MCSDg', 2)]
for model, seed in ORDER:
    run = f'{model}_s{seed}'
    if os.path.exists(f'runs/{run}/run_summary.json'):
        print('skip (done)', run); continue
    print('=====', run)
    !python src/train.py {COMMON} {TRAIN_ARCH[model]} --seed={seed} --run_name={model}__s{seed} --out_dir=runs/{run} 2>&1 | tee {OUT}/logs/train_{run}.log | grep --line-buffered -E "Saved best|Traceback|Error" | tail -n 2
```

## Cell E5.2 — Đánh giá validation (mô hình mới + tham chiếu)

```python
def evaluate(run, arch):
    ck = f'runs/{run}/best_val.pth'
    if not os.path.exists(ck):
        print('NOT RUN', run); return
    !python src/evaluate_ccmc.py --dataset_path={DS} --split=val --ckpt={ck} {arch} --bins_json={BINS} --boundary_px=3 --run_name={run} --out_prefix={OUT}/{run}_val > {OUT}/logs/eval_{run}.log 2>&1
    !grep -E "^mIoU|  small" {OUT}/logs/eval_{run}.log

for model in ('L0', 'Dnone', 'LiteSent', 'MCSD', 'PS', 'MCSDg'):
    for s in (0, 1, 2):
        evaluate(f'{model}_s{s}', EVAL_ARCH[model])
```

## Cell E5.3 — Gate và so sánh

```python
def compare(a, b, gate='none', seeds=(0, 1, 2)):
    common = [s for s in seeds if os.path.exists(f'{OUT}/{a}_s{s}_val_per_image.csv')
              and os.path.exists(f'{OUT}/{b}_s{s}_val_per_image.csv')]
    if not common:
        print('NOT RUN', a, b); return
    fa = ','.join(f'{OUT}/{a}_s{s}_val_per_image.csv' for s in common)
    fb = ','.join(f'{OUT}/{b}_s{s}_val_per_image.csv' for s in common)
    !python src/scripts/compare_runs.py --a={fa} --b={fb} --name_a={a} --name_b={b} --gate={gate} --out={OUT}/cmp_{a}_vs_{b}.json 2>&1 | tee {OUT}/logs/cmp_{a}_vs_{b}.log

compare('MCSD', 'L0', 'abs')       # H-MCSD
compare('MCSD', 'PS', 'g4')        # H-mech
compare('PS', 'L0')                # head sub-pixel không bảo toàn khối lượng
compare('MCSD', 'Dnone')           # H-eff
compare('MCSD', 'LiteSent')
compare('MCSDg', 'MCSD')           # H-spec
compare('MCSDg', 'L0', 'abs')
```

## Cell E5.4 — Profile (600×600, cùng quy ước)

```python
for label in ('MCSD', 'PS', 'MCSDg'):
    !python src/scripts/profile_model.py {EVAL_ARCH[label]} --label={label} --out={OUT}/profile_{label}.json 2>&1 | grep -E "GFLOPs|latency_ms_median|params_M"
```

## Cell E5.5 — Kiểm tra bảo toàn khối lượng trên val (sai số hàng của Sinkhorn, MCSD s0)

Hai cell: cell thứ nhất ghi script ra file (tránh trùng tên với gói `datasets` của HuggingFace trong kernel), cell thứ hai chạy script.

```python
%%writefile /kaggle/working/WeedsGalore_Ipr28/mass_check.py
import json, sys
import numpy as np, torch
sys.path.insert(0, 'src')
from nets.mcsd import MCSDSegFormer
from ccmc_common import load_checkpoint
from datasets import WeedsGaloreDataset
ds_path, ckpt, out = sys.argv[1], sys.argv[2], sys.argv[3]
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = load_checkpoint(MCSDSegFormer(head='mcsd'), ckpt, dev).to(dev).eval()
ds = WeedsGaloreDataset(ds_path, None, 5, 3, False, 'val', False)
errs, pred_mass, gt_mass = [], 0.0, 0.0
with torch.no_grad():
    for i in range(len(ds)):
        img, _, lab = ds[i]
        x = torch.from_numpy(np.ascontiguousarray(img)).float()[None].to(dev)
        logp, ex = net(x, return_aux=True)
        p = logp.exp()
        q = torch.softmax(ex['occ_logits'].float(), 1)
        errs.append((torch.nn.functional.avg_pool2d(p, 4) - q).abs().max().item())
        pred_mass += p[:, 2].sum().item()
        gt_mass += float((np.asarray(lab) == 2).sum())
res = {'max_row_marginal_error': max(errs), 'mean_row_marginal_error': float(np.mean(errs)),
       'pred_weed_mass_over_gt': pred_mass / gt_mass}
print(res)
json.dump(res, open(out, 'w'), indent=2)
```

```python
!python mass_check.py {DS} runs/MCSD_s0/best_val.pth {OUT}/mass_check_MCSD_s0.json
```

## Cell E5.6 — Gom và đóng gói

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

Giải nén `exp05_results.zip` vào `doc/exp05_results/`, upload `exp05_ckpts.zip` thành Kaggle Dataset, rồi báo trạng thái của 9 run và output của cell S6 (số test pass).
