# Candidate paper manuscript — DRAFT, not submission ready

## Working title
Counterfactual Group-Spectral Reliability for Boundary-Consistent Crop–Weed Segmentation: A Controlled Evaluation on Multispectral UAV Data

**Scientific status (November 13, 2026):** Current evidence supports a *methodological hypothesis*, **not** validated superiority of counterfactual spectral gating. The manuscript must not call CSBRL state-of-the-art, statistically superior, or wholly novel without stronger literature and experiment checks.

## Abstract (evidence-grounded draft)

Multispectral UAV segmentation of crops and weeds remains challenging at foreground boundaries. A strong single-checkpoint SegFormer with five spectral bands, a raw low-level refinement pathway and dihedral test-time augmentation attained 86.25% mIoU averaged over five development seeds on the exploratory 26-image WeedsGalore test split. Paired image-and-seed bootstrap comparison against a corresponding MiT-B0 configuration indicated an mIoU advantage of approximately 0.72 percentage points (95% interval 0.46–1.00); this difference reflects backbone configuration rather than the proposed reliability mechanism. We further investigate counterfactual group-spectral boundary reliability learning (CSBRL), which estimates a supervision weight from a frozen teacher's sensitivity to attenuating RGB versus NIR/red-edge bands. In a one-seed study, the CSBRL branch reached 86.371% exploratory test mIoU, but differed from matched control fine-tuning by only 0.0017 percentage points and did not exceed the corresponding baseline on validation. Therefore no accuracy benefit attributable to CSBRL has yet been established. An external RedEdge five-band sugar-beet experiment on WeedMap, spatially disjoint field splits and matched ablations are planned to test transferability. [DO NOT submit the abstract unchanged: update once external and multi-seed results are available.]

## 1. Introduction

- Motivation: precision weed mapping and multispectral sources of vegetation-boundary ambiguity.
- Defensible empirical observation: 95.11% of training misclassification pixels were BG↔Vegetation; approximately 92.20% of that group were within 0–3 pixels from *ground-truth* vegetation boundaries in user's audit.
- Research question: does a teacher-based counterfactual response to groupwise spectral attenuation identify trustworthy boundary supervision better than uniform or randomized gates?
- Main risk: all ingredients have precursors. Novelty hinges on a substantive specific mechanism and demonstrated added value, not its name.

## 2. Related work — required coverage before submission

1. WeedsGalore original WACV 2025 paper and official segmentation baselines.
2. WeedMap (Sa et al., Remote Sensing 2018) and spectral band studies.
3. SegFormer; low-level detail refinement; multispectral adapters.
4. Boundary-aware loss and vegetation-mask consistency.
5. Teacher/student uncertainty maps, dropout, spectral perturbation, group importance and confidence-weighted consistency.
6. Multi-temporal/phenology-aware weed segmentation.

**Novelty search unresolved.** Search databases including Scopus, Web of Science, Google Scholar, IEEE Xplore, ACM DL, Crossref and arXiv for 'counterfactual spectral perturbation', 'band-group sensitivity', 'boundary uncertainty-weighted consistency', 'multispectral teacher reliability' and study exact mechanisms. No claim of worldwide first implementation is justified by the sources reviewed so far.

## 3. Method

### 3.1 Baseline

Single SegFormer MiT-B2, input R/G/B/NIR/RE, raw low-level refinement and multiscale fusion. D4 inference uses eight symmetry views of one model. Five independently trained seeds are **replications**, not checkpoint ensembling.

### 3.2 Teacher reliability gate

For a frozen teacher $f_T$, define $p_V(X)=p(\mathrm{Crop}|X)+p(\mathrm{Weed}|X)$. Let $X_{-MS}$ and $X_{-RGB}$ be input variants attenuated toward per-image group-channel means (not zero masking). Define

$\Delta_{MS}=|p_V(X)-p_V(X_{-MS})|$ and $\Delta_{RGB}=|p_V(X)-p_V(X_{-RGB})|$.

$R=\exp[-\alpha(\Delta_{MS}+\Delta_{RGB})]$ acts as a *stability gate*, not as an empirically calibrated probability of correctness. A GT-defined narrow foreground-boundary region and detached spectral edge magnitude gate boundary consistency loss. Every GT operation is TRAIN only.

### 3.3 Supervision

Compare equal-budget control; uniform spectral boundary loss (SBCL); proposed CSBRL; shuffled random gate preserving distribution; RGB-only and NIR/red-edge-only gate; stage-modulated variant. The same frozen teacher serves only training and is not used at inference. Initial Exp27: 6 epoch, encoder frozen, decoder refined. Report whether each branch changes parameters, gradients and predictions.

## 4. Experimental protocol

### 4.1 WeedsGalore

- Spatially separated split of 104/26/26 images, three classes, five bands. Existing 26-image test has been used in development; its comparisons must be labeled exploratory.
- Evaluate identical D4/logit averaging, image-level pooled confusion matrices; report IoU of each class and mean.
- Five seeds; paired *image+seed* bootstrap for differences, and image-only bootstrap for fixed seeds.
- Choose hyperparameters with train/validation only. No threshold/candidate selection on test.

### 4.2 WeedMap external study

- Five-band **RedEdge** camera only, same class semantics 0/1/2, 10000 ignored.
- Explicit custom whole-scene partition 000,001 train; 002 validation; 003 held-out test. This differs from official WeedMap published test; no cross-paper rank claim.
- Report (1) unadapted zero-shot WeedsGalore base, (2) WeedMap fine-tuned control, (3) SBCL, (4) CSBRL, (5) random gate, with paired metrics. All use same band order and resizing. Sensor calibration remains a threat to validity.
- Do **not** present a WeedMap adaptation result as pure zero-shot transfer.

### 4.3 SOTA / related benchmarks

A numeric row is directly comparable only if dataset release, exact scene/image split, label mapping, class count, sensor bands, input resolution, metric aggregation and TTA protocol match. Otherwise clearly label as **reference only**.

### 4.4 Qualitative figures

- WeedsGalore TEST: indices 0..4 selected *before* looking at per-image accuracy. Show RGB, GT, prediction, overlay and error map in a fixed five-panel layout, separately for base and CSBRL.
- Provide examples from external WeedMap held-out scene, with identically formatted panels if the experiment succeeds.
- Avoid cherry-picking and preserve manifest of selected image names and checkpoint hash.

## 5. Results — REAL NUMBERS ONLY

### 5.1 WeedsGalore, five seed Exp12

| Configuration | Test mIoU (%) | Interval / caveat |
|---|---:|---|
| B0 + raw LLR | 85.53 | Image bootstrap 95% CI [83.13, 86.75], 5 seeds |
| B2 + raw LLR + D4 | 86.25 | Image+seed bootstrap CI [83.78, 87.43]; exploratory |
| B2 − B0 paired | +0.72 pp | Image+seed bootstrap CI [+0.46, +1.00] |

These numbers come from the user's Exp12 bootstrap logs, not new Exp28 experiments. Pairing is matched for local B0/B2 models, not the original old Word baseline.

### 5.2 WeedsGalore, Exp27 matched seed 0

| Arm | Val mIoU | Exploratory test mIoU | Δtest vs control |
|---|---:|---:|---:|
| Base B2 | 84.94007 | 86.30620 | — |
| Fine-tune control | 84.67241 | 86.36896 | 0 |
| SBCL | 84.67725 | 86.37010 | +0.00114 pp |
| CSBRL | 84.67560 | 86.37066 | +0.00170 pp |
| Random gate | 84.67725 | 86.37028 | +0.00132 pp |
| Stage | 84.67557 | 86.37087 | +0.00191 pp |

**Interpretation: no evidence of practically meaningful improvement specifically from CSBRL.** The test improvement shared by all fine-tune arms cannot be credited to CSBRL.

### 5.3 WeedMap external (PENDING)

INSERT results from `weedmap_val_report.json` and `weedmap_test_report.json` after source download, train and fixed test evaluation. Do not invent values.

### 5.4 Efficiency (PENDING)

Report single-model checkpoint parameter counts, same-GPU no-TTA and 8-view D4 latency, training FLOPs/teacher overhead, and memory. Do not reuse MiT-B0 3.66M parameter cost for MiT-B2.

## 6. Threats to validity

- Multiple methods have been evaluated after previous WeedsGalore test exposure.
- Small 26-image test gives a wide image-level CI and high variation in weed prevalence.
- Exp27 is one seed only; current hypothesis may fail over seeds.
- Ground-truth-defined boundary analysis does not translate to inference-time oracle access.
- WeedMap uses a distinct crop and sensor; fixed dtype scaling is not validated radiometric harmonization.
- Published SOTA using different datasets or splits is not a direct controlled comparator.

## 7. Conclusion (conditional; DO NOT claim novelty until confirmed)

The present data establishes robust B2-versus-B0 advantages on the local WeedsGalore pipeline but does not yet show a distinct accuracy gain from counterfactual spectral reliability weighting. Additional matched multi-seed and field-disjoint external experiments are needed to determine whether CSBRL is suitable as a publishable methodological contribution.
