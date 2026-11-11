# Exp13 v2 — VCF-Context (exploratory, single model)

This is **one SegFormer 5-band model with one hierarchical output**, NOT an ensemble of models.
The change relative to teacher's original VCF is a vegetation-gated, parallel multi-dilation crop-vs-weed context branch (dilations 1/3/5) with residual skip. The vegetation head and factorised posterior stay intact; CE+Dice/FSP remain unchanged.
No evidence yet that this achieves 86% or that this operator is novel relative to papers.

## Install on an already working Exp13 repo

Copy:
- `exp13_v2.py` -> `src/nets/exp13_v2.py`
- `exp13_v2_apply_patch.py` -> `src/scripts/exp13_v2_apply_patch.py`
- `exp13_v2_make_arms.py` -> `src/scripts/exp13_v2_make_arms.py`
- `test_exp13_v2.py` -> `tests/test_exp13_v2.py`

Then in repo root:
```
python src/scripts/exp13_v2_apply_patch.py
python src/scripts/exp13_v2_make_arms.py
python -m pytest -q tests/test_exp13.py tests/test_exp13_v2.py
python -m compileall -q src/nets/exp13.py src/nets/exp13_v2.py src/ccmc_common.py src/train.py
```
Review `git diff` and commit on a new branch before training.

New registry `configs/exp13/exp13_v2_arms.json` preserves old arms and adds `P13_B2_VCFCTX` and `P13_B3_VCFCTX` with development seeds 0–4 only. Training code MUST pass this registry to build_cli and eval_cli, using `results/exp13_v2` to avoid overwriting Exp13 results. Never run test on these exploratory arms.

## Operational rule

Start with B2 VCFCTX seed 0 and val none/D4, compare to frozen Exp12 B2 + raw LLR seed 0 (84.54 none, 84.94 D4) and original VCF (84.16 none, 84.84 D4). If no improvement, stop this branch. If improvements appear, plan 5 seeds with paired cluster bootstrap; do not claim >86 from seed0.

**Caveat:** Package static checks do not verify compatibility with the user's private Kaggle repo. Verify patch, unit tests, CLI, training backward, output shape and memory before expensive training.
