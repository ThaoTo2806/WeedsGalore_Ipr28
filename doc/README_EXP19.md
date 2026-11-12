# Exp19 S3A-SegFormer — research prototype

**Hypothesis, not a guaranteed breakthrough or proven novelty.** 5-band MiT-B2 and one Raw LLR decoder. Training-only masked spectral-spatial representation adaptation, followed by the original 100-epoch supervised experiment. The temporary reconstruction head is *discarded*; there is no teacher or ensemble.

## Project structure

```
WeedsGalore_Ipr28/
  README_EXP19.md
  src/
    train_exp19.py                # Generated from actual uploaded train.py; original untouched
    exp19_ssl.py
    scripts/
      exp19_pretrain.py
      exp19_arms.py
      exp19_seed0.py
  tests/test_exp19_ssl.py
```

**Prerequisites:** Existing project with `src/nets/segformer.py`, train/val dataset modules, Exp14–18 dependencies, `configs/exp18/exp18_arms.json`, frozen Exp12 common config, and `doc/exp02_results/results/exp02/instance_bins.json`. The source archive provided contained only a subset of the full project, so this experiment was not GPU-tested with the complete Kaggle filesystem. Execute preflight and short smoke run before a long experiment.

## Usage after putting individual files at indicated locations

```python
from pathlib import Path
import subprocess, sys
P=Path('/kaggle/working/WeedsGalore_Ipr28')
subprocess.run([sys.executable,'-m','pytest','-q','tests/test_exp19_ssl.py'],cwd=P,check=True)
subprocess.run([sys.executable,'-m','py_compile','src/train_exp19.py','src/exp19_ssl.py','src/scripts/exp19_pretrain.py','src/scripts/exp19_arms.py','src/scripts/exp19_seed0.py'],cwd=P,check=True)
print('EXP19 PREFLIGHT PASS')
```

```python
subprocess.run([sys.executable,'src/scripts/exp19_seed0.py', '--dataset_path=/kaggle/input/datasets/thothanh28/weedsgalore-dataset/weedsgalore-dataset'],cwd=P,check=True)
```

For GPU smoke only (creates output directory and must be archived to rerun):

```python
subprocess.run([sys.executable,'src/scripts/exp19_pretrain.py','--dataset_path=/kaggle/input/datasets/thothanh28/weedsgalore-dataset/weedsgalore-dataset','--epochs=1'],cwd=P,check=True)
```

The default launcher performs SSL 12 epochs on the **104 train** images, then supervised 100 epochs on the **same 104 train** images, and evaluates the existing 26-image validation with no TTA and D4 using the current `evaluate_ccmc.py`. The `--epochs=1` SSL smoke check is not an equivalent experimental run; archive the resulting `results/exp19/ssl_seed0` folder before using the default launcher if you run that smoke.

## Protocol and research limitations

- No validation labels or test data used during SSL; segmentation train labels used only in stage B.
- Single SegFormer-B2 trained once per seed and `final_ema.pth` evaluated with fixed protocol.
- No test until preregistered multi-seed confirmation. Val D4 >87%, Crop >80%, Weed >80% is a **screening threshold**, not a guarantee.
- Masked multispectral reconstruction and spectral index consistency have related prior work. Do **not** claim originality without literature review and ablations.
- Source-only smoke/unit tests are not full GPU validation. Hardware, dependency, and dataset path must be verified on Kaggle.
- SSL crop 256x256, 55% masked spatial patches (16px), 25% whole-band dropout; teacher-free masked L1 and NDVI/NDRE relation consistency; `nvidia/mit-b2` initialization.
- For a fair ablation: compare Exp19 against identical pretrained-backbone initialization without SSL, holding *all* supervised choices fixed; then evaluate multiple seeds if seed0 is promising.
