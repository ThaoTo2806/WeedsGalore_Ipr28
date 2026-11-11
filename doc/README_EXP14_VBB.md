# Exp14-VBB (seed0 only)

A diagnostic-driven **hypothesis**, not a demonstrated new method or guaranteed improvement.

Previous Exp14 ASCD seed0 D4 mIoU = 84.80%; Exp12 B2 LLR seed0 D4 = 84.94%. Previous validation error breakdown: 87.34% errors involve vegetation/background; 89.06% lie in 3px GT boundary band. Those observations motivate this training-only vegetation/background boundary BCE loss. They do **not** prove cause or effectiveness.

## What stays unchanged

- One MiT-B2 SegFormer + raw LLR, 5 bands, 3 classes.
- AMP=True, batch=8, same frozen Exp12 common config, epochs=100, EMA final checkpoint.
- No inference change, no ensemble, no test split access.
- Existing Exp12 and ASCD registries and results untouched.

## Prepare files

Unzip this package at repo root, preserving paths. The Kaggle `src/train.py` must already have the ASCD patch and AMP patch used in the previous experiment. Do not replace train.py with an older attached copy.

## Preflight (Kaggle notebook)

```python
from pathlib import Path
import subprocess, sys
P=Path('/kaggle/working/WeedsGalore_Ipr28')
for cmd in (
    [sys.executable,'src/scripts/exp14_vbb_patch.py'],
    [sys.executable,'src/scripts/exp14_vbb_arms.py'],
    [sys.executable,'-m','pytest','-q','tests/test_exp14_vbb.py'],
    [sys.executable,'-m','py_compile','src/train.py'],
):
    subprocess.run(cmd,cwd=P,check=True)
print('EXP14 VBB PREFLIGHT PASS')
```

## Train/evaluate seed0, baseline first

```python
from pathlib import Path
import subprocess, sys
P=Path('/kaggle/working/WeedsGalore_Ipr28')
DS='/kaggle/input/datasets/thothanh28/weedsgalore-dataset/weedsgalore-dataset'
for arm in ('E14_B2_AMP_REF_VBB','E14_B2_VBB'):
    subprocess.run([sys.executable,'src/scripts/exp14_vbb_seed0.py',f'--dataset_path={DS}',f'--arm={arm}'],cwd=P,check=True)
```

Compare both in **validation D4, final_ema.pth**. Seed0 screening threshold >86 alone does not permit test. No further seeds or test if arm fails screening. Do not select best_val.pth. Note: the baseline control has identical training configuration to reference except auxiliary loss weight=0.

## Caveats

- This **does not** establish scientific novelty. Boundary-focused losses are established, and 26 reused validation images can induce selection bias.
- No guarantee of 82–83% Crop/Weed IoU or >86% mIoU.
- Compare class IoU, BIoU, small weed recall and paired per-image errors; guard against trading Crop/Weed for background.
- ASCD and VBB runs are separate and their results should be preserved.
