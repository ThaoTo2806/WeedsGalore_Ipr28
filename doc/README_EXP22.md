# Exp22 — Reward-Guided Regional Correction (contextual bandit)

## What it does

A **single combined system** comprises one **frozen** 5-band SegFormer-B2 + Raw LLR baseline and a tiny MLP regional policy. Every 24×24 region takes one action: keep, crop+0.6, crop−0.6, weed+0.6, weed−0.6 on the corresponding class logit. Inference sees **only** image bands and model probabilities; GT is used **only** to compute RL rewards on the original 104 train masks.

The policy uses **REINFORCE**: actions are sampled from a categorical distribution and optimized on actually obtained region reward; an action-independent, all-actions counterfactual reward mean reduces variance. This is a one-step contextual bandit (a legitimate simple RL formulation), **not PPO**. It is also much closer to learnable cost-sensitive postprocessing than a new segmentation backbone. Novelty and gains are not guaranteed.

The 104 train images are split deterministically: 84 policy-fitting images, 20 policy-only holdout images. The **base checkpoint was trained on the whole 104-image train set**. Thus holdout is **not independent for the combined system**; it can only test policy-level generalization and is susceptible to optimistic bias. Validation is optionally run once, after committing the policy. Test is not supported by these scripts.

## Installation

Copy `src/nets/exp22_bandit.py`, `src/scripts/exp22_run.py`, `tests/test_exp22_bandit.py` into the corresponding directories of `WeedsGalore_Ipr28`. It expects the same `src/nets/segformer.py` and `src/datasets/weedsgalore/weedsgalore.py` as the supplied source archive.

You **must** supply a real `final_ema.pth` from **Exp12 or Exp15-A B2 3-class LLR**. Do not use the Exp21 6-logit checkpoint or Exp20 ICR checkpoint. `load_state_dict(..., strict=True)` ensures architecture compatibility. Uses `transformers`, `torch`, `numpy`, `matplotlib`, and project original modules.

Set `BASE` to a real checkpoint (check `is_file()`). Run `pytest -q tests/test_exp22_bandit.py`. Then run `cache_train`, `fit`, and finally `val --val_once` if policy-holdout reward is positive and meaningful. Cache may be resumed safely; cache is fingerprinted to the chosen checkpoint before fitting. **Do not swap checkpoint while cache is building**. Prefer fixed seed 0 and avoid further hyperparameter search on validation.

Reward is `(corrected pixels - 1.5 * damaged pixels)/tile_pixels - 0.015 * 1[action != keep]`. Because tile action uses shared bias, results may be negative even when there are uncertain pixels. This is intentional; the policy should learn to abstain. A positive policy-holdout reward is not evidence of better val mIoU. For a scientifically convincing RL contribution, compare the *same MLP and same actions* against supervised oracle-action imitation and always-keep on a fully independent split.

## Important limitations

- The reinforcement-learning policy only **re-ranks existing class logits**; it cannot recover a missing weed or improve spatial boundaries if the logit margin is too large. It is **not** a feature-level critic/refiner.
- D4 averages eight corrected logit predictions; the baseline is evaluated using the same D4 logits inside this script. This is paired and fair, but verify parity with original `evaluate_ccmc.py` before reporting publication metrics.
- `val_already_run.json` prevents accidental multiple validation passes by this script; it is not a security mechanism.
- All reported improvements are exploratory until independently confirmed. No test evaluation is included.
