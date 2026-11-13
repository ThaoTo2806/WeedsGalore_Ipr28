# EXP28: Reproducibility, ablation, external generalization and qualitative figures

## Status / claims

**This is a runnable evidence-generation package, not completed Q2 evidence.** All numbers in the existing user logs are real outputs; no external dataset training/evaluation or independent SOTA recreation has yet occurred. CSBRL has **not** outperformed same-run control/SBCL/random at a meaningful level in the WeedsGalore seed-0 experiments. Exp12 B2 wins against B0, a **backbone-size** result, not proof of CSBRL novelty.

- Previously exposed WeedsGalore TEST -> **exploratory**. Do not change architecture or tune from TEST.
- Experimental inference uses **one checkpoint** (8 D4 augmented views); no checkpoint ensemble.
- Existing Exp27 seven-arm ablation **seed 0** is insufficient for a robust Q2 method claim; additionally evaluate seeds 1–4 **without changing hyperparameters** if decided prospectively, but expect a null result unless evidence changes.
- Critical: external WeedMap is a different crop/camera. Its custom field split is **not the original published WeedMap leaderboard split**. It is an *external-domain adaptation* study if fine-tuning on WeedMap train, not zero-shot or domain generalization. Report zero-shot pretrained `base` separately.

## Verified outside sources

- [WeedsGalore paper, WACV 2025](https://openaccess.thecvf.com/content/WACV2025/html/Celikkan_WeedsGalore_A_Multispectral_and_Multitemporal_UAV-Based_Dataset_for_Crop_and_WACV_2025_paper.html).
- [WeedMap original multispectral RGB/NIR/RE dataset](https://projects.asl.ethz.ch/datasets/weedmap-2018/).
- [WeedMap channel and label conventions](https://www.research-collection.ethz.ch/server/api/core/bitstreams/3ae97814-1570-4760-95b8-a9a3eac21975/content): `Tiles/RedEdge/<scene>/tile/{R,G,B,NIR,RE}`, indexed GT 0/1/2, ignore 10000.
- [WeedMap 2018 paper](https://www.mdpi.com/2072-4292/10/9/1423).
- [PhenoBench official dataset/benchmark](https://www.phenobench.org/dataset.html): RGB only, not appropriate for evaluating 5-band NIR/RE perturbations without a new RGB-only method.

## Project tree

```
WeedsGalore_Ipr28/
  src/scripts/exp28_paper_report.py
  src/scripts/exp28_figures.py
  src/scripts/exp28_single_image.py
  src/scripts/exp28_weedmap.py
  src/scripts/exp27_run_csbrl.py        # EXISTING from Exp27
  src/exp27_csbrl.py                    # EXISTING
  src/exp27_sbcl.py                     # EXISTING
  src/exp27_losses.py                   # EXISTING
  tests/test_exp28_evidence.py
  configs/LITERATURE_COMPARISONS.csv
```

## A. Preflight

```python
from pathlib import Path
import subprocess,sys
P=Path('/kaggle/working/WeedsGalore_Ipr28')
subprocess.run([sys.executable,'-m','pytest','-q','tests/test_exp28_evidence.py'],cwd=P,check=True)
print('EXP28 PREFLIGHT PASS')
```

## B. WeedsGalore existing Exp12/27 analysis, NO new TEST reading

```python
subprocess.run([sys.executable,'src/scripts/exp28_paper_report.py',f'--project={P}'],cwd=P,check=True)
```

Read `results/exp28/PAPER_EVIDENCE/PAPER_RESULTS_DRAFT.md` and `evidence_summary.json`.

## C. WeedsGalore qualitative figures on fixed TEST images

TEST already exposed, so figures are **exploratory**; use fixed indices 0–4, do not select best outcomes.

```python
DS='/kaggle/input/datasets/thothanh28/weedsgalore-dataset/weedsgalore-dataset'
CK='/kaggle/input/datasets/thanhthao28/exp12-backup/exp12/P_B2_LLR/seed0/final_ema.pth'
subprocess.run([sys.executable,'src/scripts/exp28_figures.py',f'--dataset_path={DS}',f'--checkpoint={CK}',
                '--split=test','--allow_test_figures','--indices=0,1,2,3,4',
                f'--out_dir={P}/results/exp28/WEEDSGALORE_TEST_FIGURES'],cwd=P,check=True)
```

If you want CSBRL qualitative outputs, change `--checkpoint` to `results/exp27/E27_CSBRL/seed0/csbrl/final_ema.pth` and set a separate output directory; do not call it a five-seed average.

## D. Single new 5-band input image demo

Given five PNGs with the same prefix (e.g. `scene_R.png`, `scene_G.png`, `scene_B.png`, `scene_NIR.png`, `scene_RE.png`), run:

```python
subprocess.run([sys.executable,'src/scripts/exp28_single_image.py',
                '--prefix=/kaggle/working/my_image/scene', f'--ckpt={CK}',
                f'--out_dir={P}/results/exp28/DEMO'],cwd=P,check=True)
```

This is an inference illustration, **not a labeled benchmark score**. The normalization expects 8-bit or 16-bit PNG, matching WeedsGalore loading; other sensors need verified calibration.

## E. Second dataset: WeedMap (RedEdge only; **not** Sequoia)

Download the official WeedMap dataset from ETHZ **yourself** and attach as Kaggle Dataset. Required structure (see official documentation):

```
WEEDMAP_ROOT/
  Tiles/RedEdge/000/{groundtruth,mask,tile/R,tile/G,tile/B,tile/NIR,tile/RE}
  Tiles/RedEdge/001/...
  Tiles/RedEdge/002/...
  Tiles/RedEdge/003/...
```

**IMPORTANT:** `exp28_weedmap.py` uses a **prespecified custom group split** (`000,001` train / `002` val / `003` test) and 256×256 resized tiles. Verify bands and indexed labels during `preflight`. This is not directly comparable to Sa et al.'s reported original test metric, nor a demonstration of zero-shot generalization after external fine-tuning. No `10000` pixels are included in loss or metrics.

```python
WEEDMAP='/kaggle/input/weedmap-release/2018-weedMap-dataset-release' # CHANGE to actual mounted root
BASE='/kaggle/input/datasets/thanhthao28/exp12-backup/exp12/P_B2_LLR/seed0/final_ema.pth'
EXT=P/'results/exp28/WEEDMAP_REDEDGE/seed0'
common=[sys.executable,'src/scripts/exp28_weedmap.py',f'--weedmap_root={WEEDMAP}',
        f'--base_ckpt={BASE}',f'--out_dir={EXT}','--seed=0','--epochs=6','--lr=2e-6']
subprocess.run(common+['--mode=preflight'],cwd=P,check=True)
for arm in ['control','sbcl','csbrl','random']:
    subprocess.run(common+['--mode=fit',f'--arm={arm}'],cwd=P,check=True)
subprocess.run(common+['--mode=val'],cwd=P,check=True)
# Only after freezing method; no tuning after seeing validation:
subprocess.run(common+['--mode=test','--allow_external_test'],cwd=P,check=True)
```

For a full experimental paper, repeat external seeds 0..4 (change `--seed` and both input checkpoint path and output root) and report paired *image+seed* uncertainty. Run `--max_train` only for testing code, not paper claims.

**Normalization limitation:** WeedMap uint8/uint16 is scaled using the dtype dynamic range; do not assume this equals WeedsGalore's radiometric calibration. Confirm input histograms/band alignment before making cross-domain claims.

## F. Evidence gate / stopping rules

- **Novelty:** complete systematic literature review incl. 2024–2026, differentiate from perturbation-based consistency/uncertainty gating, and publish explicit mathematical algorithm.
- **Matched ablation:** CSBRL versus control, SBCL, random, RGB-only gate and NIR/RE-only gate. Stage-aware branch is not meaningful on WeedMap unless real acquisition stages exist and are validated.
- **Statistics:** same image IDs and seed IDs, pooled CMs for mIoU; paired image/seed resampling. Fixed CI should not be reinterpreted as a p-value, and benchmark TEST must not guide tuning.
- **Independent evidence:** WeedMap scene-held-out score and example figures, without presenting as the official SOTA protocol.
- **Efficiency:** Params, MACs/FLOPs, wall-clock inference on same GPU, D4 latency (8 forwards), training cost incl. frozen teacher.
- **Failure modes:** cloud/shadows, thin weeds, boundaries, day-level stratification, sensor/domain shift.
- **Reproducibility:** splits, sources, checkpoint SHA256, code commit, random seeds, packages, preprocessing, inference averaging mode.
- **If CSBRL does not beat the ablations with robust CI:** do NOT claim it improves segmentation. A Q2 journal acceptance is never guaranteed.

## G. Paired external bootstrap and same-hardware efficiency

```python
subprocess.run([sys.executable,'src/scripts/exp28_external_bootstrap.py',f'--out_dir={EXT}',
                '--split=val','--draws=10000'],cwd=P,check=True)
# After the one-time frozen TEST run:
subprocess.run([sys.executable,'src/scripts/exp28_external_bootstrap.py',f'--out_dir={EXT}',
                '--split=test','--draws=10000'],cwd=P,check=True)

subprocess.run([sys.executable,'src/scripts/exp28_efficiency.py',f'--ckpt={CK}',
                f'--out_json={P}/results/exp28/efficiency_B2_seed0.json'],cwd=P,check=True)
```

These scripts do not compute FLOPs. For journal manuscript, report FLOPs from a compatible measured profiler **separately** and give exact input size/GPU. The external bootstrap is *image only, one seed*. Repeat external seeds prospectively if feasible.
