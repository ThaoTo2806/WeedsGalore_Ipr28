# EXP12 — integration package (reviewed against user-uploaded sources, NOT verified GitHub HEAD)

## Exact folders

| Uploaded file | Copy to repo |
|---|---|
| `exp12_arms.json` | `configs/exp12/exp12_arms.json` |
| `exp12_common_config.json` | `configs/exp12/exp12_common_config.json` |
| `exp12_config.py` | `src/exp12_config.py` |
| `exp12_sched.py` | `src/exp12_sched.py` |
| `exp12_utils.py` | `src/exp12_utils.py` |
| `segformer.py` | `src/nets/segformer.py` |
| `llr_subcell.py` | `src/nets/llr_subcell.py` |
| `mcsd.py` | `src/nets/mcsd.py` |
| `ccmc_common.py` | `src/ccmc_common.py` |
| `train.py` | `src/train.py` |
| `evaluate_ccmc.py` | `src/evaluate_ccmc.py` |

**Do not overwrite other files** or use this as a complete checkout; repo requires existing `src/nets/__init__.py`, `src/datasets`, `src/ccmc_metrics.py`, `src/nets/sc_llr.py` and other dependencies. Commit to a NEW branch, inspect diffs and run pytest first.

## Key SCI mismatch — choose and preregister BEFORE training

Historical `S0_B0/S1_B0/S2_B0` compare **without LLR** (1/4-grid, `occ_conditioned`, `mcsd`, spectral_guide=True), consistent with historical OccCond-g/MCSD-g. They do NOT establish an algorithm-on-LLR claim. The separate `L12_B0_REF/L12_B0_OCC2/L12_B0_MCSD2` (and B1/B2) are **2x2 RAW LLR** experiments newly appended. They are different methods and may not reproduce the historical small-weed gain. Do NOT combine these in one SCI claim. Decide which hypothesis is your SCI primary before first experiment; update preregistration and budget accordingly. Never claim test >86% guaranteed.

## Kaggle — copy to existing checkout in a separate branch

Do NOT run E12.7. In Kaggle notebook, change PROJECT and ZIP:

```python
from pathlib import Path
import subprocess, zipfile, os
PROJECT=Path('/kaggle/working/WeedsGalore_Ipr28')
ZIP=Path('/kaggle/input/exp12-reviewed-patch/exp12_reviewed_patch.zip')  # upload ZIP as Kaggle input
assert PROJECT.joinpath('src','train.py').is_file()
subprocess.run(['git','-C',str(PROJECT),'switch','-c','exp12-reviewed'],check=True)
with zipfile.ZipFile(ZIP) as z:
    for member in z.namelist():
        if member.startswith(('src/','configs/exp12/','tests/test_exp12_review.py','docs/')):
            target=PROJECT/member
            target.parent.mkdir(parents=True,exist_ok=True)
            # Review diffs before replacing existing tracked files; do not silently overwrite current repo!
            if not target.exists(): z.extract(member, PROJECT)
            else: print('EXISTS — REVIEW BEFORE REPLACE',member)
print('Review existing files; do not train with unmerged patch.')
```

Merge code with the current Git HEAD manually, then:

```bash
python -m compileall -q src/train.py src/ccmc_common.py src/exp12_config.py src/exp12_utils.py src/exp12_sched.py src/evaluate_ccmc.py src/nets/segformer.py src/nets/llr_subcell.py src/nets/mcsd.py
python src/scripts/exp12_preflight.py
python -m pytest -q tests/test_exp12_review.py
python src/train.py --helpshort
python src/evaluate_ccmc.py --helpshort
```

E12.0: profile B0/B1/B2 at 600/900 only after verifying pretrained checkpoint loading and no OOM; use no TEST data. E12.1–E12.6 after code, protocol, runtime and storage checks. The v2 README cell templates are not executable end-to-end with this partial source set: evaluation compare_runs, instance metrics, offline checkpoint manifest and test-lock need additional verified source files.

## Important corrections and limitations

- New code explicitly checks gradient-accumulation flag; it refuses !=1 until effective batch and deviation are preregistered. Do not force B2 900 batch8 through OOM.
- Eval adds registered single-model TTA, inference scale and weed-bias probe, with TEST prohibition for nonzero weed bias.
- `exp12_config.py --check_runs` now fails when no configs match, instead of silently reporting success.
- This archive does not contain live GitHub HEAD; remote fetch failed. No actual WeedsGalore training or mIoU results were produced.
- Ensure all 26 validation images, 5 seeds and paired bootstrap before claims. Do not compare historical test 85.49 directly with new validation.
