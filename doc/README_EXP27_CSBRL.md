# Exp27 — Counterfactual Spectral Boundary Reliability Learning (candidate method)

**Scientific status**: candidate hypothesis, **NOT** verified unprecedented invention, Q2 acceptance, or guaranteed TEST >86%. Prior work already covers spectral attention, band perturbations, boundary losses, uncertainty weighting. The specific combination requires a full related-work study and ablations.

## Files (upload individually into the matching project folders)

```
/kaggle/working/WeedsGalore_Ipr28/
├── src/
│   ├── exp27_losses.py
│   ├── exp27_sbcl.py
│   ├── exp27_csbrl.py
│   └── scripts/
│       ├── exp22_run.py                 # existing, DO NOT replace
│       ├── exp27_run_csbrl.py
│       └── exp27_compare_bootstrap.py
├── tests/
│   └── test_exp27_csbrl.py
└── README_EXP27_CSBRL.md
```

Depends on existing `src/datasets`, `src/exp15_losses.py`, `src/scripts/exp22_run.py`, and dataset split with hash `675667edf3f35f16`. The checkpoint **must** be `/.../P_B2_LLR/seedN/final_ema.pth`.

## Experimental arms

| arm | role | additional loss / gate |
|---|---|---|
| `control` | matched fine-tune | no new loss |
| `sbcl` | spectral boundary consistency | spectral evidence only |
| `csbrl` | proposed mechanism | frozen-teacher band-group sensitivity reliability |
| `stage` | stage-conditioned ablation | same reliability with prespecified date factor |
| `random` | negative control | shuffled joint reliability (same gate distribution) |
| `rgb` | contribution of RGB group | sensitivity to removing RGB group (NIR/RE remain) |
| `ms` | contribution of NIR/RE group | sensitivity to removing NIR/RE (RGB remain) |

For `rgb`/`ms` the arm names refer to **which counterfactual sensitivity is used** in the loss, not which bands remain on the baseline input. The backbone always sees 5 real bands at normal inference.

**Training-only overhead**: the teacher does 3 no-grad passes per batch for `csbrl`, `stage`, `random`, `rgb`, `ms`. Final inference has no additional network or input perturbation. Only one student checkpoint per arm/seed; D4 is the same 8-view averaging used in Exp12.

**Unverified mechanism**: High stability to group attenuation is treated as a reliability proxy. It may downweight useful hard examples. Therefore `sbcl`, `random`, and single-group ablations are essential.

## Kaggle commands

```python
from pathlib import Path
import subprocess, sys
P=Path('/kaggle/working/WeedsGalore_Ipr28')
DS='/kaggle/input/datasets/thothanh28/weedsgalore-dataset/weedsgalore-dataset'
SEED=0
BASE=f'/kaggle/input/datasets/thanhthao28/exp12-backup/exp12/P_B2_LLR/seed{SEED}/final_ema.pth'
OUT=P/f'results/exp27/E27_CSBRL/seed{SEED}'
script=P/'src/scripts/exp27_run_csbrl.py'
assert script.is_file() and Path(BASE).is_file()
subprocess.run([sys.executable,'-m','pytest','-q','tests/test_exp27_csbrl.py'],cwd=P,check=True)
common=[sys.executable,str(script),f'--dataset_path={DS}',f'--base_ckpt={BASE}',f'--out_dir={OUT}',f'--seed={SEED}',
        '--epochs=6','--lr=2e-6','--batch_size=1','--grad_accum=2',
        '--sbcl_weight=0.02','--lovasz_weight=0.10','--attenuation=0.35','--alpha=5.0']
subprocess.run(common+['--mode=preflight'],cwd=P,check=True)
for arm in ['control','sbcl','csbrl','stage','random','rgb','ms']:
    subprocess.run(common+['--mode=fit',f'--arm={arm}'],cwd=P,check=True)
subprocess.run(common+['--mode=val','--val_once','--tta=d4'],cwd=P,check=True)
subprocess.run([sys.executable,str(P/'src/scripts/exp27_compare_bootstrap.py'),f'--out_dir={OUT}','--split=val'],cwd=P,check=True)
```

Results: `OUT/val_exp27_csbrl_compare.json`, `OUT/bootstrap_exp27_val_seed0.json`, `OUT/perimg_exp27_*_val_d4_seed0.csv`.

**Decision locked on VAL (no TEST):** `csbrl` should beat **both** `control` and original `base` in val D4, have positive image-paired CI (or at least a credible improvement worth confirming), and outperform `random` to suggest meaningful reliability. Do not choose the winning arm by TEST score. Later train seeds 1–4 with identical hyperparameters, assess val, and record CI of paired delta. Existing WeedsGalore TEST was exposed during prior Exp12 work, so any Exp27 TEST is exploratory.

Only after locking arms/configuration and validation assessment, optionally perform exploratory TEST once via:

```python
subprocess.run(common+['--mode=test','--test_once','--allow_exploratory_test','--tta=d4'],cwd=P,check=True)
```

TEST report: `OUT/test_exp27_csbrl_exploratory.json`. Do not use it to tune Exp27 or decide which seed to train. The script refuses a second evaluation at the same output path.

## Important research limitations

- No metadata-based true temporal tracking. `stage` uses day labels from train filenames only during training.
- Do not claim this algorithm is unique without a systematic novelty review of band perturbation, counterfactual attribution, boundary consistency and spectral uncertainty research.
- Check exact input band ordering in local repo; this package assumes bands 0:3 are RGB and 3:5 NIR/RE.
- Fixed optimization budget allows fair comparison of loss components, but the no-grad teacher introduces extra training cost. Report training time.
- Validation images are already repeatedly used in past experiments; confirmation on an independent spatial/campaign dataset is important for strong journal claims.
