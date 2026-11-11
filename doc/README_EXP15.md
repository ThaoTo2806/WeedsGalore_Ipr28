# Exp15-A + Exp15-B — ONE Kaggle upload, sequential conditional run

Two **independent**, training-only auxiliary losses applied to same pretrained B2+Raw LLR architecture; no inference heads, test untouched.

## Grounding and scientific claim
- A: class-wise Crop+Weed Lovasz-Softmax surrogate from Berman et al. (CVPR 2018), https://openaccess.thecvf.com/content_cvpr_2018/html/Berman_The_LovaSz-Softmax_Loss_CVPR_2018_paper.html. **Established baseline, NOT novel**.
- B: **proposed, unverified synthesis**: per-training-image five-band class centroid spectral ambiguity multiplied by supervised Crop/Weed interface binary cross-entropy on a downsampled grid. Inspired by PRC-Net's multispectral channel/context reasoning (Irfan et al., Expert Systems with Applications, 2026, doi:10.1016/j.eswa.2026.132842), VISA dual stream band/index decoupling (BAWSeg, Remote Sensing 2026), and remote sensing boundary-guided context (Wang et al., IEEE TIP 2023 doi:10.1109/TIP.2023.3326400). It is **NOT a reimplementation** of these papers, not proven new or superior, and specifically is a loss, NOT a new architecture.
- B GT class prototypes and interface masks exist **only in training**; inference consumes only 5-band input. No validation GT fed to network.

## Target platform
Kaggle `/kaggle/working/WeedsGalore_Ipr28` with working completed Exp14-VBB code including existing `src/train.py`, `configs/exp14/exp14_vbb_arms.json`, `src/evaluate_ccmc.py`, `src/exp12_config.py`, `doc/exp02_results/results/exp02/instance_bins.json`.

## Install — unzip files to PROJECT, not another directory
```python
from pathlib import Path
import zipfile
PROJECT=Path('/kaggle/working/WeedsGalore_Ipr28')
ZIP=Path('/kaggle/input/REPLACE_WITH_YOUR_UPLOADED_DATASET/exp15_ab_package.zip')
with zipfile.ZipFile(ZIP) as z: z.extractall(PROJECT)
```
If zip is placed in `/kaggle/working`, point ZIP there instead.

## Preflight ONCE (safe to rerun)
```python
import subprocess,sys
for script in ('src/scripts/exp15_patch.py','src/scripts/exp15_arms.py'):
    subprocess.run([sys.executable,script],cwd=PROJECT,check=True)
subprocess.run([sys.executable,'-m','pytest','-q','tests/test_exp15_losses.py'],cwd=PROJECT,check=True)
subprocess.run([sys.executable,'-m','py_compile','src/train.py'],cwd=PROJECT,check=True)
print('EXP15 PREFLIGHT PASS')
```

## Run A then B automatically ONLY IF A fails
```python
DS='/kaggle/input/datasets/thothanh28/weedsgalore-dataset/weedsgalore-dataset'
subprocess.run([sys.executable,'src/scripts/exp15_seed0.py',f'--dataset_path={DS}','--mode=auto'],cwd=PROJECT,check=True)
```
`--mode=both` runs both even if A passes; `--mode=a` / `--mode=b` individually. Script **refuses to overwrite partial runs**. Remove a broken run manually only after archiving it and investigating.

Predeclared seed 0 gate: final_ema.pth, 26 val images, D4, mIoU>86 AND Crop IoU>80 AND Weed IoU>80. Regardless of seed0 pass, **test remains sealed**; confirmation seeds and ablations needed. Seeds1-9 not registered here.

Known limitations: B may over-emphasize pixels near Crop/Weed contacts, ignores BG near-boundary (Exp14 diagnostics suggest majority of errors are vegetation/BG), and its absolute effectiveness cannot be guaranteed. Compare A/B to matched Exp14 AMP control D4 84.8686%, Crop 78.1212%, Weed 78.0853%. Report negative results.
