# Exp16 — Spectral-guided uncertainty point refinement (seed0)

Experimental PointRend-style refinement for 5-band UAV crop/weed. Scientific inspiration: Kirillov et al., **PointRend: Image Segmentation as Rendering**, CVPR 2020, https://openaccess.thecvf.com/content_CVPR_2020/html/Kirillov_PointRend_Image_Segmentation_as_Rendering_CVPR_2020_paper.html . This implementation **is not a faithful reproduction** and **novelty is not verified**.

Design: leave MiT-B2 + Raw LLR intact; add a sparse point MLP that consumes the decoder's half-resolution feature, raw 5-band values and coarse logits. Select top-uncertain pixels for refinement; during training use 75% top-uncertain and 25% random; at inference select top-uncertain without GT. Learn residual correction in a single model. This changes inference head and cost (unlike prior train-only losses). Point selection is stop-gradient, selected corrections are differentiable. Number of refined points fixed at 4096/image; absolute pixel count may be small relative to 600x600; this may *fail* to improve metrics. No guaranteed >86 test.

Important: experiments 13–15 failed to exceed validation 86; the reserve gate is validation D4 **>87**, Crop IoU >80, Weed IoU >80. Seed0 only; test sealed. If seed0 passes, extend seeds 1–4, then 5–9 with a new pre-registered registry and perform test once after model selection.

## Kaggle install

1. Ensure `PROJECT=/kaggle/working/WeedsGalore_Ipr28` has its working Exp15 `src/train.py`, `configs/exp15/exp15_arms.json` and `src/ccmc_common.py`. Do not overwrite previous code with old ZIP.
2. Extract `exp16_point_package.zip` **directly to PROJECT** (ZIP root contains `src/`, `tests/` and this README).
3. Run scripts in order below.

```python
from pathlib import Path
import subprocess,sys
P=Path('/kaggle/working/WeedsGalore_Ipr28')
DS='/kaggle/input/datasets/thothanh28/weedsgalore-dataset/weedsgalore-dataset'
for cmd in [
    ['src/scripts/exp16_patch.py'],
    ['src/scripts/exp16_arms.py'],
    ['-m','pytest','-q','tests/test_exp16_point.py'],
    ['-m','py_compile','src/train.py','src/ccmc_common.py','src/evaluate_ccmc.py'],
    ['src/scripts/exp16_diagnose.py',f'--dataset_path={DS}'],
]:
    subprocess.run([sys.executable,*cmd],cwd=P,check=True)
print('EXP16 PREFLIGHT PASS')
```

Then seed0 train + validation **only**:

```python
from pathlib import Path
import subprocess,sys
P=Path('/kaggle/working/WeedsGalore_Ipr28')
DS='/kaggle/input/datasets/thothanh28/weedsgalore-dataset/weedsgalore-dataset'
subprocess.run([sys.executable,'src/scripts/exp16_seed0.py',f'--dataset_path={DS}'],cwd=P,check=True)
```

Before GPU: check the `shape_and_error.json` readout. If images are already full native 600×600, no claim of restoration by upsampling is justified. Run seeds only if preflight and dataset shape analysis look correct.

## Preflight limitations

- Unit tests run on a synthetic fake SegFormer; they test point selection/shapes/gradients, **not** actual Kaggle pretrained B2 or the dataset.
- Real Kaggle integration can still fail due to different versions of `ccmc_common.py` or `src/train.py`; scripts fail closed rather than silently substitute architecture.
- `src/evaluate_ccmc.py` must load `SpectralPointRefineSegFormer` with the correct `--exp16_point_refine=True` flag, else state dict strict loading will fail.
- `measure_stats` may be slower with the extra dynamic point head. GPU RAM usage is unmeasured.
