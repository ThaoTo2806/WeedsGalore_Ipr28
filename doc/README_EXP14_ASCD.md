# Exp14 ASCD — validation-only seed-0 pilot (2026-11-11)

**Scientific status:** experimentally unverified proposal; **NOT novelty-certified**, no >86% performance guarantee. A new name and conditional combination are not evidence of priority over papers. Search similar works and compare equations before submission. No inference-time second model, no ensemble, no extra head. Single SegFormer MiT-B2, five input bands, raw LLR, AMP FP16, physical batch=8, 100 epochs from the frozen Exp12 common config. **Test images and labels remain completely sealed.**

## Hypothesis and technical mechanism

ASCD (Asymmetric Spectral Confusion Decoupling) is a training-only loss, added to the existing CE + Dice objective:

`L_train = L_CE+Dice + 0.15 * L_ASCD`.

ASCD computes conditional Crop/Weed CE for GT plant pixels, with `p(wrong plant class)`-based detached hard weights and normalized five-band spectral prototype distance from **training labels** only. The spectral confusion score is `sigmoid((d_own-d_opposite)/1.0)` and its multiplier is `1 + spectral_confusion`. Per-class and near-opposite-plant-interface / interior regions are **averaged separately**, preventing the majority class or largest area from monopolizing the objective. Training-only GT labels are not fed to the deployed model or validation inference. If a tile lacks >=16 pixels of one class, the spectral factor becomes neutral for that tile. All training losses calculated FP32 even with AMP model outputs. Loss contains no extra learnable parameters. Existing network checkpoint can be evaluated with `src/evaluate_ccmc.py` unmodified.

**Limitations:** class prototypes estimated within each training tile may be unrepresentative; raw spectral distances may be affected by illumination; hard-pixel, boundary-conditioned and prototype-based losses already exist individually. Measure separately and check actual paper equations before making any novel method claim.

## Install into the existing Kaggle repo

From the extracted ZIP root (`exp14_ascd_package`), **copy contents to `/kaggle/working/WeedsGalore_Ipr28/`** preserving structure:

```
README_EXP14_ASCD.md
src/exp14_ascd.py
src/scripts/exp14_ascd_patch.py
src/scripts/exp14_ascd_arms.py
src/scripts/exp14_seed0.py
src/scripts/exp14_diagnose.py
tests/test_exp14_ascd.py
```

**Prerequisites**: current `src/train.py` should be your **working Exp13 AMP file**, not old train.py; it must have `scaler.scale(loss).backward()`, `flags.DEFINE_bool('amp', ...)`, and existing raw B2 SegFormer. If earlier `exp14_cw_dbl` patch was already applied, restore your clean AMP train.py first. Snapshot backup `src/train_pre_ascd.py` generated automatically. Never edit Exp12 frozen config or checkpoint/results.

On Kaggle:

```python
from pathlib import Path
import subprocess, sys
PROJECT = Path('/kaggle/working/WeedsGalore_Ipr28')
subprocess.run([sys.executable, 'src/scripts/exp14_ascd_patch.py'],cwd=PROJECT,check=True)
subprocess.run([sys.executable, 'src/scripts/exp14_ascd_arms.py'],cwd=PROJECT,check=True)
subprocess.run([sys.executable, '-m','pytest','-q','tests/test_exp14_ascd.py'],cwd=PROJECT,check=True)
subprocess.run([sys.executable,'-m','py_compile','src/train.py'],cwd=PROJECT,check=True)
```

Inspect / commit source and registry (lock registry before first run). Run full **baseline validation error diagnosis, ASCD seed0 training and validation both no-TTA/D4** in one command:

```python
from pathlib import Path
import subprocess, sys
PROJECT = Path('/kaggle/working/WeedsGalore_Ipr28')
DS = '/kaggle/input/datasets/thothanh28/weedsgalore-dataset/weedsgalore-dataset'
BASELINE = PROJECT/'results/exp12/P_B2_LLR/seed0/final_ema.pth'
assert BASELINE.is_file(), 'Adjust BASELINE to the existing Exp12 B2 seed0 final_ema.pth'
subprocess.run([sys.executable, 'src/scripts/exp14_seed0.py',
   '--dataset_path='+DS,'--baseline_ckpt='+str(BASELINE)],cwd=PROJECT,check=True)
```

If baseline path differs, point to its true `final_ema.pth`, not best-val. The script **requires** the baseline diagnostic and never touches test.

Outputs:

```
results/exp14_ascd/diagnostics/b2_seed0_d4.json
results/exp14_ascd/E14_B2_ASCD/seed0/final_ema.pth
results/exp14_ascd/E14_B2_ASCD/seed0/val_none_summary.json
results/exp14_ascd/E14_B2_ASCD/seed0/val_d4_summary.json
results/exp14_ascd/E14_B2_ASCD/seed0/seed0_screen.json
```

## Gate (pre-registered)

* **Seed0 screening GO only if final EMA validation D4 mIoU >86.00%**. Report Crop IoU, Weed IoU and BIoU (they are different metrics). 82–83% Crop & Weed IoU are **aspirational**, not a mathematical guarantee of >86 or a validated forecast. E.g. if background IoU remains 98.4, Crop=Weed=79.8 is the 86.0 threshold.
* If seed0 >86, run the preregistered matched `E14_B2_AMP_REF` seed0 to isolate AMP and original CE+Dice, **then** expand ASCD dev seeds 1–4 if ASCD has meaningful paired improvement. Ablation arms `E14_B2_AC`, `E14_B2_AC_BI`, and baseline control have been registered but **NOT** run. Because the 26 validation images have already been used extensively for design, seed0 >86 is hypothesis-generating, not scientific confirmation.
* Confirm on seeds 5–9 using the **same locked full configuration and TTA**. Perform ablations on development/validation, not test, before final locking; compare 26 paired images and seeds and report CI. **One-time test opening only after development, ablations and confirmation are complete.** Using test after seed0 to decide on seeds 1–4 contaminates test and makes unbiased test claims invalid.
* If seed0 <=86: stop ASCD extension; keep negative result; do not hunt many thresholds on same 26-image validation.

## Known constraints

* No actual GPU training in this package. CPU unit tests cover gradients/empty classes/interface/spectral ablations; they do not establish GPU fit or mIoU.
* The working Kaggle AMP trainer has a flag `grad_accum_steps` but gradient accumulation is *not implemented*; this Exp14 does not use it and keeps physical batch8.
* PyTorch scaler may skip optimizer updates on nonfinite gradients; current EMA still updates every minibatch. Monitor loss and scaler behavior. For exact resume reproducibility add scaler state to `last.pth` (not included here).
* ASCD performance/novelty requires separate empirical and literature verification.
