# Exp21 — Single-network hierarchical weed-subtype supervision

Hypothesis: original fine labels (0 BG, 1 crop, 2–5 four weed groups) may provide a richer gradient than early merging all weeds. A single MiT-B2+Raw LLR outputs 6 fine logits; exactly merges the 4 weed subclass probabilities for **3-class evaluation**. Training keeps the frozen 3-class weighted CE + Dice objective plus 0.20× auxiliary 6-class weighted CE. No second model, test unexamined.

## Important caveats
- The dataset loader returns `(image, original_fine_label, merged_3class_label)`. Audit **train only** to verify IDs 0–5. IDs 2–5 should not be named as specific species without documentation.
- This is **not** guaranteed novel: training on original 6 classes has been reported by WeedsGalore work. The proposed *hierarchical joint supervision + exact probabilistic aggregation* is a testable variant, not established novelty.
- Compared with the 3-class baseline, this has a different classifier parameter count and extra supervision. Use a six-class-only ablation and paired seeds if encouraging.
- Source copies correspond to user-provided `exp19_source_for_integration.zip`. `train_exp21.py` is standalone and will import the project's existing Exp14–17 support modules. Do not overwrite `src/train.py`.
- Test is intentionally NOT run. Validation has been reused extensively; interpret any gains cautiously.

## Files
- `src/nets/exp21_hier.py` — 6-to-3 logsumexp merge
- `src/train_exp21.py` — source-matched training script
- `src/evaluate_exp21.py` — source-matched D4 evaluation script
- `src/scripts/exp21_label_audit.py` — train-only six-ID audit
- `src/scripts/exp21_arms.py` — frozen arm
- `src/scripts/exp21_seed0.py` — train seed0 + val none/D4
- `tests/test_exp21_hier.py` — 3 unit tests

## Kaggle steps
1. Place files under `/kaggle/working/WeedsGalore_Ipr28/` preserving paths.
2. `python src/scripts/exp21_label_audit.py --dataset_path=... --project=/kaggle/working/WeedsGalore_Ipr28`
3. `python src/scripts/exp21_arms.py`; `python -m pytest -q tests/test_exp21_hier.py`.
4. `python src/scripts/exp21_seed0.py --dataset_path=...`

If label audit fails with missing IDs, do not train: this dataset is incompatible with the pre-registered 6-class objective.
