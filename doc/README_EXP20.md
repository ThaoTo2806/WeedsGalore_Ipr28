# Exp20 — Intra-Image Crop-Referenced SegFormer (ICR)

A *research candidate*, **not a proven novel method** and not guaranteed to surpass the published MiT-B0 test result of 85.49 ± 0.30%.

- One MiT-B2 5-band SegFormer with existing Raw LLR. The intra-image crop reference and reliability abstention are learned within the SAME model. No ensemble, teacher model, test-label reference, or crop GT at inference.
- Implementation has a confidence-weighted crop feature prototype computed on each image, a 5-band spectral prototype, and a tiny learned 2-class residual for Crop vs Weed. When crop evidence is missing, reference contribution fades to zero. Last layer zero-initialized to preserve baseline predictions before training.
- 104 train / 26 validation / 26 test; fixed split hash `675667edf3f35f16`.
- 100 supervised epochs seed0 on train, validation without TTA & D4. Checkpoint `final_ema.pth`.
- **Test** is separate, never automatic, may be triggered ONCE explicitly using `--accept_exploratory_test`; doing so invalidates held-out independence for any later model selection. It is unsuitable to claim a final confirmatory result if you use it for development.

## Place files

```
WeedsGalore_Ipr28/
  src/nets/exp20_icr.py
  src/train_exp20.py
  src/evaluate_exp20.py
  src/scripts/exp20_arms.py
  src/scripts/exp20_seed0.py
  src/scripts/exp20_test_once.py
  tests/test_exp20_icr.py
  README_EXP20.md
```

Needs the user's existing project (the supplied `exp19_source_for_integration.zip` source), including Exp18 registry and Exp16/17 dependencies already present. No patch of `src/train.py` and `src/evaluate_ccmc.py`.

## Kaggle

```
cd /kaggle/working/WeedsGalore_Ipr28
python src/scripts/exp20_arms.py
python -m pytest -q tests/test_exp20_icr.py
python src/scripts/exp20_seed0.py --dataset_path=/kaggle/input/datasets/thothanh28/weedsgalore-dataset/weedsgalore-dataset
```

OPTIONAL ONE-TIME TEST ONLY IF YOU ACCEPT IT IS EXPLORATORY:

```
python src/scripts/exp20_test_once.py --dataset_path=/kaggle/input/datasets/thothanh28/weedsgalore-dataset/weedsgalore-dataset --accept_exploratory_test
```

For fair comparison to the DOC, MiT-B2 accuracy and resource cost differ from MiT-B0; seed0 validation cannot be directly compared with 5-seed test mean. Publish novelty only after literature comparison and controlled ablation.
