# Exp12 reviewed patch — Kaggle runbook (source files uploaded Nov 10 2026)

**NOT a guaranteed >86% model. NOT a replacement for checking the current Git HEAD.**

This patch is based on the uploaded `train(2).py`, `segformer.py`, `ccmc_common(2).py`,
`evaluate_ccmc.py`, `mcsd.py`, `llr_subcell(3).py`, `exp12_config.py`,
`exp12_sched.py`, `exp12_utils.py`, and two JSON config files.
The remote GitHub URL was not fetchable here; compare against your actual Kaggle repo.

## 1. Folder tree

```
WeedsGalore_Ipr28/
  configs/exp12/exp12_common_config.json
  configs/exp12/exp12_arms.json
  src/exp12_config.py
  src/exp12_sched.py
  src/exp12_utils.py
  src/ccmc_common.py
  src/train.py
  src/evaluate_ccmc.py
  src/nets/segformer.py
  src/nets/llr_subcell.py
  src/nets/mcsd.py
  src/scripts/exp12_preflight.py
  src/scripts/exp12_profile.py
  tests/test_exp12_review.py
  docs/INSTALL_AND_KAGGLE.md
  README_EXP12_KAGGLE_REVIEWED.md
```

Existing dependencies that must stay in repo: `src/nets/__init__.py`,
`src/nets/sc_llr.py`, `src/datasets`, `src/ccmc_metrics.py`,
`src/scripts/compare_runs.py`, existing Exp10/Exp11 outputs.

## 2. Kaggle cells BEFORE training

### Cell A: clone/existing checkout and patch staging

```python
from pathlib import Path
import subprocess, shutil, zipfile
PROJECT=Path('/kaggle/working/WeedsGalore_Ipr28')
assert (PROJECT/'src/train.py').is_file()
print(subprocess.check_output(['git','-C',str(PROJECT),'rev-parse','HEAD'],text=True).strip())
# Use an existing clean feature branch or create a new branch ONCE.
# Upload exp12_reviewed_patch.zip as Kaggle Dataset, set ZIP path to its real location.
ZIP=Path('/kaggle/input/exp12-review/exp12_reviewed_patch.zip')
assert ZIP.exists(),ZIP
STAGE=Path('/kaggle/working/exp12_staging')
STAGE.mkdir(parents=True,exist_ok=True)
with zipfile.ZipFile(ZIP) as z:z.extractall(STAGE)
for rel in ['src/train.py','src/ccmc_common.py','src/evaluate_ccmc.py','src/nets/segformer.py']:
    print('REVIEW:',rel,'uploaded file:',(STAGE/rel).exists(),'repo file:',(PROJECT/rel).exists())
# DO NOT blindly overwrite tracked files. Compare and selectively merge source.
```

### Cell B: after reviewing and merging, validate

```python
import subprocess
assert subprocess.run(['python','-m','compileall','-q','src','configs/exp12'],cwd=PROJECT).returncode==0
assert subprocess.run(['python','src/scripts/exp12_preflight.py'],cwd=PROJECT).returncode==0
assert subprocess.run(['python','-m','pytest','-q','tests/test_exp12_review.py'],cwd=PROJECT).returncode==0
print('Static / config tests PASS. Still need real HF encoder/GPU smoke tests.')
```

### Cell C: E12.0 profile without accessing labels

```python
from pathlib import Path
import subprocess, sys
for bb in ['b0','b1','b2']:
 for sz in [600,900]:
  args=[sys.executable,'src/scripts/exp12_profile.py',f'--backbone={bb}',f'--size={sz}',
        '--batch_size=1','--steps=3',f'--out=results/exp12/profile_{bb}_{sz}_bs1.json']
  p=subprocess.run(args,cwd=PROJECT)
  if p.returncode: print('PROFILE FAILED (do not assume this arm is feasible):',bb,sz)
# Run a separate batch-size=8 profile for each candidate before registering full batch 8.
```

## 3. E12.1 / E12.2 training — ONLY after freezing config and registration

```python
import json,subprocess,sys
from pathlib import Path
PROJECT=Path('/kaggle/working/WeedsGalore_Ipr28')
sys.path.insert(0,str(PROJECT/'src'))
from exp12_config import load_json,build_cli,check_run_config
common_path=PROJECT/'configs/exp12/exp12_common_config.json'
arms_path=PROJECT/'configs/exp12/exp12_arms.json'
common=load_json(common_path);arms=load_json(arms_path)
DS='/kaggle/input/REPLACE_WITH_YOUR_WEEDSGALORE_DATASET'  # SET YOUR EXISTING verified DS
assert Path(DS).is_dir()

def train_one(arm_id,seed):
 assert seed in arms['arms'][arm_id]['seeds']
 assert seed in range(5), 'Development only. Confirmation seeds 5–9 need a locked E12.6 protocol.'
 out=PROJECT/'results'/'exp12'/arm_id/f'seed{seed}'
 out.mkdir(parents=True,exist_ok=True)
 if (out/'run_summary.json').exists():
  assert not check_run_config(out/'run_config.json',common,arms)
  print('Already done:',arm_id,seed);return
 cmd=[sys.executable,'src/train.py']+build_cli(common,arms,arm_id,seed,DS,str(out),f'{arm_id}_s{seed}',str(common_path),str(arms_path))
 subprocess.run(cmd,cwd=PROJECT,check=True)
 assert not check_run_config(out/'run_config.json',common,arms)
 assert (out/'final_ema.pth').is_file()
 print('PASS',arm_id,seed)
# After B0 smoke/test and prereg review: train_one('P_B0_LLR',0)
```

### VALIDATION inference example (after training)

```python
from exp12_config import eval_cli
arm='P_B0_LLR';seed=0
ckpt=PROJECT/'results'/'exp12'/arm/f'seed{seed}'/'final_ema.pth'
assert ckpt.exists()
cmd=[sys.executable,'src/evaluate_ccmc.py']+eval_cli(arms,arm,str(ckpt),DS,'val',
 str(PROJECT/'results/instance_bins.json'),str(PROJECT/'results/exp12'/arm/f'seed{seed}'/'val'),
 f'{arm}_s{seed}',str(arms_path),tta='none')
subprocess.run(cmd,cwd=PROJECT,check=True)
```

Do not call `split=test` with these cells. The E12.7 test gate is **not implemented/approved in this package**; use an independently reviewed single-run manifest and persistent marker protocol.

## 4. Caveats / scientific boundaries

- `S0/S1/S2` are historical **without LLR**. `L12_*` are distinct newly proposed **with RAW LLR**. Do not present historical OccCond-g +10.55 pp as evidence the new LLR arm achieves it.
- `grad_accum_steps != 1` deliberately refuses to run until effective batch size and protocol deviation are implemented. At 900x900, an OOM remains possible even at batch 1.
- No blind hyperparameter sweeps on the 26-image validation set; no test opened by this patch.
- Current code uses pretrained `nvidia/mit-b*` weights; make HF weights available on Kaggle or enable permitted network access.
- The repository source on GitHub was not retrieved from this environment; verify `git diff` and actual commit manually.
- E12 P3 metrics and SCI cross-backbone comparison still require the verified `ccmc_metrics.py` and `compare_runs.py` from the repo.
