# Exp18: train-only object-centric spectral hard-negative curriculum

Single 5-band SegFormer-B2 + Raw LLR, **no ensemble**, point head OFF.

## Frozen stages (100 epochs, seed 0)
- Epoch 0–34: 384×384 crops with center preference 55% GT Weed, 35% GT Crop, 10% background.
- Epoch 35–69: 384×384 crops. 35% online train-only background→weed false positive center when available, 35% Weed, 23% Crop, 7% background (exact fallback depends on available hard centers).
- Epoch 70–99: full 600×600 images, same B2 model and weights, no new model.

Hard negatives come from **current single model's predictions on unaugmented TRAIN split** every five epochs. No validation label, no test image or label is used for sampling or mining. Train-only spatial D4 augmentation applied synchronously across bands and masks for cropped stages. Training uses original baseline CE+Dice. Checkpoint `final_ema.pth`; validation D4 is evaluated once after training. Never test unless a separate confirmation protocol passes.

## Files
```
WeedsGalore_Ipr28/
  src/exp18_curriculum.py
  src/scripts/exp18_patch.py
  src/scripts/exp18_arms.py
  src/scripts/exp18_seed0.py
  tests/test_exp18_curriculum.py
  README_EXP18.md
```

## Kaggle preflight
1. Restore the complete working Exp17 project (including Exp16 integration), then copy this package into project root.
2. Run `python src/scripts/exp18_patch.py` **only once** (idempotent).
3. Run `python src/scripts/exp18_arms.py`.
4. Run `python -m pytest -q tests/test_exp18_curriculum.py` and `python -m py_compile src/train.py src/exp18_curriculum.py`.
5. Run `python src/scripts/exp18_seed0.py --dataset_path=...`.

## Scientific caution
This is a hypothesis, not guaranteed improvement or proven novelty. Existing object-centric and hard-negative sampling precedents must be cited. Report separate ablations of each stage if seed0 warrants it. Avoid choosing the final test model from single lucky validation seed.

## Safety and reproducibility
The patch has an anchor check and creates `src/train.py.pre_exp18`. Its training wrapper never overrides `val_dataset`; it always requires exactly 104 TRAIN images. Existing incomplete output directories are never overwritten automatically. `split_hash` stays `675667edf3f35f16`.
