# Exp28 — Paper evidence audit (generated, NOT proof of novelty)

**Status:** WeedsGalore TEST was previously opened; all Exp27 TEST comparisons are exploratory.
**External dataset:** WeedMap RedEdge evaluation must be run separately.

## VAL / D4 / seed 0

| Arm | mIoU % | Δ vs base pp | paired 95% CI pp | Δ vs control pp |
|---|---:|---:|---|---:|
| base | 84.94007 | — | — | — |
| control | 84.67241 | -0.26766 | [-0.467, +0.019] | +nan |
| sbcl | 84.67725 | -0.26282 | [-0.461, +0.024] | +0.00484 |
| csbrl | 84.67560 | -0.26447 | [-0.462, +0.022] | +0.00319 |
| stage | 84.67557 | -0.26450 | [-0.462, +0.023] | +0.00316 |
| random | 84.67725 | -0.26282 | [-0.461, +0.024] | +0.00484 |
| rgb | 84.67565 | -0.26442 | [-0.462, +0.022] | +0.00324 |
| ms | 84.67529 | -0.26478 | [-0.463, +0.023] | +0.00288 |

Paired bootstrap resamples 26 images; seed variation not included for Exp27 seed 0.

## TEST / D4 / seed 0

| Arm | mIoU % | Δ vs base pp | paired 95% CI pp | Δ vs control pp |
|---|---:|---:|---|---:|
| base | 86.30620 | — | — | — |
| control | 86.36896 | +0.06276 | [-0.045, +0.138] | +nan |
| sbcl | 86.37010 | +0.06390 | [-0.044, +0.140] | +0.00114 |
| csbrl | 86.37066 | +0.06446 | [-0.043, +0.141] | +0.00170 |
| stage | 86.37087 | +0.06467 | [-0.042, +0.141] | +0.00191 |
| random | 86.37028 | +0.06408 | [-0.044, +0.140] | +0.00132 |
| rgb | 86.37026 | +0.06406 | [-0.043, +0.140] | +0.00130 |
| ms | 86.37057 | +0.06437 | [-0.043, +0.141] | +0.00160 |

Paired bootstrap resamples 26 images; seed variation not included for Exp27 seed 0.

## Scientific interpretation

No superiority claim for CSBRL unless matched ablations, paired effects, and independent external evaluation support it.
Do not claim SOTA without same dataset, split, label mapping, modality, and evaluation protocol.
Do not choose TEST examples or checkpoints based on TEST performance.

## Missing evidence

- Exp12 val/prob P_B2_LLR: need five seed CSVs
- Exp12 val/prob P_B0_LLR: need five seed CSVs
- Exp12 test/logit P_B2_LLR: need five seed CSVs
- Exp12 test/logit P_B0_LLR: need five seed CSVs