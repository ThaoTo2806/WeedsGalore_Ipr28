# EXP23: Controlled two-arm growth-stage-aware pixel correction

## Arms

- **nodate**: frozen Exp15-A 5-band SegFormer-B2 + 16-input-channel lightweight pixel correction head; date channel always zero.
- **stage**: identical architecture, train budget, seed, labels and correction actions, but date input is the known normalized acquisition date (25 May, 30 May, 6 June, 15 June 2023).

Both use five actions (KEEP, CROP ±0.6, WEED ±0.6). No spatial temporal matching is claimed. **Growth-stage conditioning is not tracking a particular plant**. Date may be used only if acquisition date is available at inference; unknown dates are rejected.

The correction head learns per-pixel oracle actions from *TRAIN ground truth* and a frozen base. This is **supervised counterfactual decision learning**, not PPO/DQN or multi-step reinforcement learning. All cached labels remain TRAIN only. The separate policy holdout of 20 train images is *not* independent of the full SegFormer (it already saw all 104). Validation is evaluated one time after both arms have been fitted. Never train on validation or test labels. Test remains unopened.

## Kaggle placement

    project/src/nets/exp23_pixel.py
    project/src/scripts/exp23_run.py
    project/src/scripts/exp22_run.py  (existing Exp22 helper)
    project/tests/test_exp23_pixel.py

## Commands

Use exact Exp15-A checkpoint and dataset paths. `OUT` common between both arms.

    python src/scripts/exp23_run.py --mode=cache --dataset_path="$DS" --base_ckpt="$BASE" --out_dir="$OUT"
    python src/scripts/exp23_run.py --mode=fit --arm=nodate --dataset_path="$DS" --base_ckpt="$BASE" --out_dir="$OUT" --epochs=20
    python src/scripts/exp23_run.py --mode=fit --arm=stage --dataset_path="$DS" --base_ckpt="$BASE" --out_dir="$OUT" --epochs=20
    python src/scripts/exp23_run.py --mode=val --val_once --tta=d4 --dataset_path="$DS" --base_ckpt="$BASE" --out_dir="$OUT"

Files: `OUT/cache_train/` (104 caches, roughly 600+ MB), `OUT/nodate/policy_best.pth`, `OUT/stage/policy_best.pth`, `OUT/val_exp23_compare.json`, `OUT/val_exp23_per_image.csv`.

### Scientific caveats

- A train-only oracle gain of +2.44 pp does not predict validation improvement.
- A positive delta on validation may reflect selection bias after many experiments on these 26 images.
- Stage-aware gains need a matched stage-shuffled/random-date control and multiple seeds before attribution to real growth knowledge.
- A 5-band snapshot at one date cannot reconstruct the history of a particular maize plant without reliable aligned earlier images.
- Date may correlate with illumination/sensor conditions, rather than biological growth; avoid attributing gains solely to growth without further controls.
- MiT-B0 publication benchmark is **test**, so validation mIoU cannot be claimed to beat it.
