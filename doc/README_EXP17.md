# Exp17 — Safety-Gated Spectral Point Refinement (single model)

**Status:** untested scientific hypothesis; **novelty NOT established**. Built upon *PointRend* (Kirillov et al., CVPR 2020) and *SegFix* (Yuan et al., ECCV 2020). No ensemble, no external pretrained second model, no post-hoc use of test annotations.

## Motivation (measured on Exp16 final_ema, val, no TTA)

- Selected point error rate 37.075%, versus global 1.759%.
- Selected points near GT boundaries: 90.691% (GT only used for validation diagnostics).
- 5,717 points corrected; 6,394 previously correct points damaged; net -677.
- Weed FP 85,895 vs FN 33,799. Existing point head increases weed recall at precision cost.

## Hypothesis

Replace the unconditional additive point correction in Exp16 with a **learned, correction-selective safety gate**. One SegFormer-B2 (5 bands) with raw LLR and a point MLP; lightweight gate uses current point's decoder feature, 5-band raw values, coarse class logits, local 5-band contrast, local interior-neighborhood prediction disagreement, and class confidence. Prediction:

`logits_final = logits_coarse + scatter(gate * correction)`

Training-only counterfactual safety objective penalizes edits that increase loss on already correct coarse points and supervises gate with an improvement target from comparing ungated vs coarse cross-entropy at **training** labels. No labels are used to choose gates during inference. Both correction and gating parts are **inside the same model and jointly trained**.

**Not a novelty proof:** prior gated refinements, point rendering, uncertainty gating and boundary correction may overlap. Scientific novelty requires a systematic literature search, independent mechanism audit, ablation and multi-seed confirmation. These are not established by this package.

## Files (put in project root paths)

- `src/nets/exp17_gate.py`
- `src/scripts/exp17_patch.py`
- `src/scripts/exp17_arms.py`
- `src/scripts/exp17_seed0.py`
- `tests/test_exp17_gate.py`
- `README_EXP17.md`

Prerequisites: existing working Exp16 project including fixed `src/nets/exp16_point.py`, `configs/exp16/exp16_arms.json`, current `src/train.py`, `src/ccmc_common.py`, `src/evaluate_ccmc.py`. Run the patch **once**; it is idempotent if run again, and creates `.pre_exp17` backups.

## Preflight Kaggle cell

```python
from pathlib import Path
import subprocess,sys
P=Path('/kaggle/working/WeedsGalore_Ipr28')
for cmd in [
  ['src/scripts/exp17_patch.py'],
  ['src/scripts/exp17_arms.py'],
  ['-m','pytest','-q','tests/test_exp17_gate.py'],
  ['-m','py_compile','src/train.py','src/ccmc_common.py','src/evaluate_ccmc.py'],
]:
  subprocess.run([sys.executable,*cmd],cwd=P,check=True)
print('EXP17 PREFLIGHT PASS')
```

## Train seed0 (no test)

```python
from pathlib import Path
import subprocess,sys
P=Path('/kaggle/working/WeedsGalore_Ipr28')
DS='/kaggle/input/datasets/thothanh28/weedsgalore-dataset/weedsgalore-dataset'
subprocess.run([sys.executable,'src/scripts/exp17_seed0.py',f'--dataset_path={DS}'],cwd=P,check=True)
```

**Pre-registered screening gate:** single model seed0, final_ema.pth, D4 validation mean mIoU >87%, Crop IoU >80% AND Weed IoU >80%. If passes, independent development seeds 1–4, ablation, confirmation seeds 5–9, **test exactly once at the very end**. No performance guarantee. Fail-closed on existing incomplete `results/exp17/E17_B2_SAFETY_GATE/seed0` directory. Archive that directory instead of deleting or silently overwriting.

## Baselines and fair evaluation

Report Exp16 original 84.857% (D4) and Exp15-A 85.006% (D4) on seed0, both single-model. Report Exp17 raw/no TTA and D4; also compare weed precision, weed FP, BIoU and small recall if diagnostics are re-run. Same 104 train / 26 val / 26 unopened test; split hash `675667edf3f35f16`.

## References

- Kirillov et al. **PointRend: Image Segmentation as Rendering**, CVPR 2020. https://openaccess.thecvf.com/content_CVPR_2020/html/Kirillov_PointRend_Image_Segmentation_As_Rendering_CVPR_2020_paper.html
- Yuan et al. **SegFix: Model-Agnostic Boundary Refinement for Segmentation**, ECCV 2020. https://www.microsoft.com/en-us/research/publication/segfix-model-agnostic-boundary-refinement-for-segmentation/
