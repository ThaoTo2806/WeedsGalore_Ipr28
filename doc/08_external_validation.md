# Exp08 — External-domain validation and one-time test lock

> **Status (2026-11-05): NOT READY TO OPEN TEST.** K8.0/K8.0b/K8.0c/K8.1 outputs are saved in `doc/exp123.ipynb`; no WeedMap inference/model predictions or WeedsGalore test evaluation have been run. Official iMap codes are sourced. Across 204 aligned iMaps, code 1 is absent globally, but valid-area interpretation depends on the unresolved mask polarity. The 8-bit WeedMap tiles and 16-bit WeedsGalore inputs produce different one-image `plt.imread` ranges; physical radiometry compatibility is unverified. Synthetic loader/metric helpers and tests have been prepared; their actual Kaggle execution and an end-to-end checkpoint evaluator are not yet accepted. Exp08 remains frozen external-domain validation, **not** method development.

## Readiness verdict

| Requirement                                                 | Current evidence                                                                                                                                                                                         | Status / action before test                                                                                                                                                                                                                   |
| ----------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Novelty of both decoder directions                          | Initial Google/Scholar web snippets and cited papers; no full-text systematic review or Scopus/Web of Science query log                                                                                  | **INCOMPLETE.** Treat novelty as a narrow hypothesis; finish and archive the review before any “novel/first” claim.                                                                                                                           |
| WeedMap five-band data, labels, metadata, and test identity | Inventory has 204 tiles per band; archive SHA-256 `f3a5…e7f5d`. Official PDF defines iMap codes. Saved audit aligns all 204 iMaps/masks and finds no code 1; counts under both mask values are retained. | **EXPORTED iMAP HAS NO CROP CODE.** Do not claim three-class validation. Valid-area support remains conditional because the release says white-valid but observed masks track signal with zero; radiometry/co-registration remain unverified. |
| Second-dataset experiment type                              | The three existing WeedsGalore-trained checkpoints are evaluated without target training or adaptation                                                                                                   | **DECIDED: external-domain, zero-shot confirmation.** It is not a WeedMap method-development experiment. Any later WeedMap training/tuning would require a new protocol and a separate untouched test set.                                    |
| OccCond-g compute cost                                      | K8.1 output saved in `doc/exp123.ipynb`: four forward profiles, Tesla T4, FP32, batch 1, 600×600, 50 warmup/300 timed runs                                                                               | **DONE.** Values are transcribed in the cost table below. FLOP counter omits elementwise work.                                                                                                                                                |
| One-time WeedsGalore test lock                              | iMap semantics/class-support audit and pure metric core are documented; mask/radiometry and a data-loading evaluator remain unaccepted                                                                   | **NO-GO.** K8.2/K8.3 remain unrun until mask/radiometry are resolved, evaluator synthetic tests pass, and the immutable manifest/prechecks are complete.                                                                                      |

### Model identity and scope

The source implements one segmentation model per run, not an ensemble. `MCSDSegFormer` inherits the same pretrained MiT-B0 encoder and SegFormer multi-scale decoder as the L0 path; its RGB patch projection is expanded to five channels by retaining the RGB filters and initializing NIR/RE from their RGB-filter mean. OccCond-g/MCSD-g add occupancy and placement heads (and an optional small raw-band guide); they do not delete pretrained backbone layers. The method change is in the dense prediction head/coupling, not in combining several pretrained models. This is verified in `src/nets/segformer.py` and `src/nets/mcsd.py`.

Method development and selection belong to the existing WeedsGalore train/validation experiments. Exp08's second dataset is only a frozen transfer check of L0, OccCond-g, and MCSD-g. Based on Exp07 validation, OccCond-g is the primary candidate for the stated small-weed recall/coverage objective; MCSD-g is the constrained-coupling comparator, and L0 is the reference. Report all three regardless of outcome. Do not use WeedMap to change this choice.

## Decision and scope

Freeze **OccCond-g as the primary method candidate** because it had the stronger validation evidence for the stated small-weed objective: versus L0, Small-Weed Recall +10.55 [6.39; 14.84] pp and coverage +7.55 [2.97; 12.47] pp, with mIoU +0.88 [0.42; 1.54] pp. Keep **MCSD-g as the constrained-decoding comparator/ablation**, not as a second co-primary claim: Exp07 found similar mIoU versus OccCond-g but lower recall/coverage and higher marginal fidelity, Weed precision, and Boundary IoU. Freeze L0, OccCond-g, and MCSD-g as the complete comparison set with five existing validation-selected checkpoints per method. Do not select or alter any method using WeedMap or test data. This choice is based on the stated recall objective and Exp07 validation only; the full novelty review remains a prerequisite to opening test.

Use the official **WeedMap RedEdge-M dataset 003** only as a candidate second-dataset external evaluation. WeedMap covers sugar-beet fields in Switzerland/Germany, unlike WeedsGalore maize fields. Do not substitute Sequoia set 005: it does not match the five-channel input contract. The ETHZ ASL page confirms the broad availability of RedEdge/Sequoia imagery, tiles/orthomosaics, reflectance TIFFs, and background/crop/weed annotations, but the exact downloaded 003 files and compatibility are not verified here. Before inference, record the exact original-paper passage/table/page identifying 003 as test, release DOI/version, metadata for all five aligned bands and radiometry, label codes/instances, license, and tile/site grouping. If any gate fails, stop and do not silently substitute another set.

**Official annotation semantics (verified from ETH Zurich Research Collection release document, p. 2):** `GroundTruth_iMap` code `0=Background`, `1=Crop`, `2=Weed`, `10000=Non-class`. Treat `10000` as ignore and exclude it from the confusion matrix; do not infer semantics from the RGB render color. The release describes iMap as grayscale 16-bit and the color render as black/green/red for Background/Crop/Weed. Citation/source: [ETH Zurich WeedMap release PDF](https://www.research-collection.ethz.ch/server/api/core/bitstreams/3ae97814-1570-4760-95b8-a9a3eac21975/content), p. 2; dataset DOI `10.3929/ethz-c-000788571`.

**Latest preflight interpretation:** the authoritative indexed-map table defines `10000` as Non-class/ignore, regardless of RGB rendering. The saved audit paired 204/204 iMap and mask files with no shape mismatch and found no code `1` anywhere in the exported iMaps. All-iMap counts: background `0=34,239,045`, crop `1=0`, weed `2=343,926`, ignore `10000=668,229`. With the audit's `mask==0` branch, counts were `0=12,000,413`, `2=343,799`, `10000=668,227`; with `mask==255`, `0=22,238,632`, `2=127`, `10000=2`. The mask distribution was 102 all-255, 57 mixed, and 45 all-0 frames. The release says white is valid, whereas the notebook's signal audit associated nonzero imagery with mask zero. Thus full-index iMaps contain no Crop annotation, but which weed/background pixels are valid remains unresolved. Do not report three-class mIoU/Crop IoU; a future approved transfer analysis would be weed-only and explicitly limited.

The saved K8.0c compared one predeclared image per domain (`WeedMap frame0020` and `WeedsGalore 2023-05-25_0109`). WeedsGalore files were 16-bit grayscale PNGs (`uint16`); WeedMap tile PNGs were 8-bit grayscale (`uint8`). Both became float32 through `plt.imread`, but sample distributions differed (NIR median/P99: 0.5098/0.7285 vs 0.2784/0.5922). These are descriptive statistics for one image per dataset, not calibration evidence and not a normalization recipe. Crossref's paper abstract says the orthomosaic channels are radiometrically calibrated/aligned, but the accessible material does not specify the transform from reflectance product to 8-bit tile PNG. This is a hard radiometry blocker until the export transform is sourced.

The class/mask audit is preserved in `doc/exp123.ipynb`; K8.0/K8.0b/K8.0c outputs are under `doc/exp08_results/results/exp08/`. The downloaded `weedmap_003_preflight.json` remains an old `PENDING` snapshot; do not treat it as the reviewed decision artifact. K8.0 now writes a separate inventory. DataCite lists the dataset as “In Copyright - Non-Commercial Use Permitted”; verify reuse conditions before redistributing data/derived outputs.

**External-domain claim boundary:** this tests frozen models on WeedMap 003 without adaptation. It supports transfer evidence only for the inspected WeedMap test collection, not generalization to all crops/sites. If radiometric alignment requires fitting statistics on WeedMap, or if set 003 is not an independent testing partition, stop before inference and preregister a different experiment. The current repository evaluator only instantiates `WeedsGaloreDataset`; passing a WeedMap path to `src/evaluate_ccmc.py` does not evaluate WeedMap. A separate loader/evaluator and tests for band order, label mapping, tile alignment, nodata, and metric aggregation are prerequisites.

## Preliminary novelty review (2026-11-04)

This is a traceable initial screen, not a completed Scopus/Web of Science review. Do not use “first” or “novel” claims yet.

| Work                                                                                                                                                                                                                               | Relation to the two decoder directions                                                                                    | Initial distinction/question to verify                                                                                                                                                                          |
| ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Sa et al., “WeedMap: A Large-Scale Semantic Weed Mapping Framework Using Aerial Multispectral Imaging and Deep Neural Network for Precision Farming,” Remote Sensing 10(9), 1423 (2018), [DOI](https://doi.org/10.3390/rs10091423) | Direct application/dataset prior art: multispectral background/crop/weed mapping and tiling                               | Does not establish sub-cell fractional occupancy coupling from the public abstract; inspect full method before claiming distinction                                                                             |
| He et al., “Deep Subpixel Mapping Based on Semantic Information Modulated Network for Urban Land Use Mapping,” IEEE TGRS 59 (2021), 10628–10646, [DOI](https://doi.org/10.1109/TGRS.2021.3050824)                                  | **Close to occupancy-conditioned subpixel decoding:** coarse semantic information guides fine-resolution class allocation | Different remote-sensing land-use task and mapping setup; compare its coarse class-proportion supervision, fine-grid allocation, and end-to-end pathway equation-by-equation                                    |
| Arun et al., “CNN based sub-pixel mapping for hyperspectral images,” Neurocomputing (2018), [title search](https://scholar.google.com/scholar?q=%22CNN+based+sub-pixel+mapping+for+hyperspectral+images%22)                        | Deep-learning subpixel mapping using coarse/fractional information is a close conceptual family                           | Verify the exact objective and whether fractions constrain fine-pixel labels; do not infer novelty from title alone                                                                                             |
| Classical subpixel mapping and fractional-abundance literature (Atkinson; sparse subpixel mapping; pixel-swapping)                                                                                                                 | Closest to class fractions/occupancy as a constraint on fine-grid class counts                                            | Often starts with coarse abundance estimates and performs a mapping/post-processing step rather than the same trainable decoder; quantify the distinction and cite primary sources                              |
| Kim, Oh, Ye, “OTSeg: Multi-prompt Sinkhorn Attention for Zero-Shot Semantic Segmentation,” ECCV 2024, [arXiv:2403.14183](https://arxiv.org/abs/2403.14183)                                                                         | Sinkhorn/OT already appears in semantic segmentation                                                                      | OTSeg aligns pixel embeddings and text prompts for zero-shot segmentation; it is not the same local per-cell class-marginal constraint. This blocks a broad claim that Sinkhorn in segmentation is itself novel |

**Provisional novelty judgement:** neither broad direction is new by itself. Fractional occupancy, sub-pixel allocation, and coarse-to-fine class assignment have a substantial remote-sensing sub-pixel mapping literature; He et al.'s SIMNet is a close learned semantic-guidance precedent, and Arun et al. is a directly relevant hyperspectral CNN precedent. Occupancy-conditioned placement must be compared mathematically against these methods, including supervision and whether class proportions constrain the fine-grid output. Sinkhorn/OT is also established in segmentation (including OTSeg), so the only defensible candidate distinction is the _local per-cell transport formulation_: predicted multi-class occupancy supplies row marginals, each fine pixel is a unit column, and learned placement scores allocate mass for agricultural multispectral weed segmentation. Even this distinction is **unverified** until prior work on class-count/area constraints, proportion-constrained segmentation, and differentiable sub-pixel mapping is read in full. Finite Sinkhorn iterations only approximately satisfy row marginals; avoid “exact mass conservation.”

For the completed review, save date, database/platform, exact query string, filters/time span, export or result IDs, inclusion/exclusion reason, and a paper-by-paper method matrix. Search Scopus, IEEE Xplore, Web of Science, and Google Scholar for both directions and their synonyms; add SIMNet, CNN-based hyperspectral sub-pixel mapping, classical fractional-abundance/pixel-swapping methods, and OT-based dense prediction. Until that is done, phrase the contribution as a proposed task-specific formulation and empirical comparison, not a novelty/priority claim.

Searches actually screened: `"subpixel mapping" deep learning remote sensing fractional abundance`; `"Deep Subpixel Mapping" semantic information modulated network urban land use`; `"fractional class occupancy" segmentation subpixel decoding`; `Sinkhorn semantic segmentation optimal transport decoder`; and WeedMap sensor/split queries. Google web/Scholar snippets and official ETHZ/IEEE/arXiv pages were used. A full novelty review must now search Scopus, IEEE Xplore, Web of Science, and Google Scholar for both **occupancy-conditioned subpixel decoding** and **Sinkhorn-constrained decoding**, save exact queries/date/exported records, and add proportion-constrained decoding, class-count constraints, differentiable subpixel mapping, area-preserving segmentation, and unmixing-guided mapping. The final claim should be limited to a verified combination/application if overlap remains.

## Locked WeedMap protocol

### Dataset identity and input gate

1. Select the official RedEdge-M testing partition **003** only. Save the downloaded archive name, DOI/version, archive SHA-256, license, dataset index, flight/site metadata, and the original source paper’s statement that 003 is a test partition. The archive listing identifies this as `Tiles/RedEdge/003.zip`; do not identify index 003 by substring matches against frame filenames.
2. Verify five co-registered bands and channel mapping by names and metadata. Model order is exactly `R, G, B, NIR, RE`; map camera-native files by labels, never by directory ordering. Confirm units, bit depth, reflectance calibration, clipping/scale, image shape, nodata handling, and tile-to-label alignment.
3. Use the official indexed-map semantics `background=0, crop=1, weed=2, non-class=10000`; map `10000` to ignore and exclude it from the confusion matrix. Keep the official RGB color rendering separate from indexed codes. Verify the mask valid-pixel polarity against the release definition before using it. Any archive-wide class-presence check is data-integrity-only, not a model/threshold selection step. If no valid Crop pixels are annotated in the official 003 evaluation area, report only the preregistered Weed-only metrics (`WeedRecall`, `WeedPrecision`, `IoU_weed`) and explicitly omit three-class `mIoU`, Crop IoU, and claims of full three-class segmentation validation. Do not treat absence in the four inspected frames as proof about all of 003.
4. Check for same-orthomosaic/overlapping tiles and official split leakage. If 003 is a held-out test subset from a documented flight/site, use its existing split. Keep tile groups intact in any uncertainty analysis.
5. Do not compute WeedMap normalization quantiles, class weights, or thresholds from its test images/labels. Use only a fixed documented transform compatible with the source model. If the source and WeedMap radiometry cannot be aligned without target-fitted statistics, stop: the external test is not ready.

### Methods, checkpoints, and comparisons

- Methods: L0, OccCond-g, MCSD-g.
- Checkpoints: five fixed seeds (0–4) per method, selected by WeedsGalore validation only. Freeze SHA-256 for all 15 `best_val.pth` files before reading WeedMap predictions.
- No WeedMap training/fine-tuning/BN adaptation, pseudo-labeling, calibration, or checkpoint selection.
- All three methods use identical WeedMap tiles, channel mapping, radiometric transform, and output dimensions.
- Primary endpoint/contrast: pixel-level `WeedRecall`, OccCond-g − L0, over valid annotated pixels, with `10000` excluded as ignore and mask polarity applied only after it is source-verified. Secondary weed outcomes: `WeedPrecision` and `IoU_weed`; official-instance recall@IoU 0.5 only if instance IDs are present and validated. If valid Crop (`iMap=1`) pixels are present in the official evaluation area, also report per-class IoU, Crop IoU, and three-class mIoU. If valid Crop is absent, do not report three-class mIoU/Crop IoU as full three-class results; report the weed-only metrics and state the annotation-coverage limitation. Secondary contrasts: MCSD-g − L0 and MCSD-g − OccCond-g. Do not reuse WeedsGalore `[45,147]` pixel-size bins or claim `SmallWeedRecall` without a preregistered physical-area threshold and validated instances.
- Report point estimates and 95% paired hierarchical bootstrap CI (seed, then spatial group/tile). Cluster at orthomosaic/flight when multiple independent units exist. If set 003 is only one flight/orthomosaic, state that intervals quantify within-collection tile variability and do not estimate between-site uncertainty. The bootstrap seed, replicate count, resampling order, and cluster identifiers must be frozen in the manifest.
- Dataset-level external evaluation is descriptive/confirmatory of frozen transfer, not a new method-development gate. Report all three methods, even if a method loses.

### WeedsGalore test lock and one-time release

Before opening WeedsGalore test, freeze a JSON manifest with repository commit, split hash and explicit split IDs/hash, source `run_config.json` hashes, checkpoint SHA-256 values for L0/OccCond-g/MCSD-g seeds 0–4, WeedMap archive/data identity and preflight decision, fixed input transform, class map, metrics, bootstrap seed/replicates/cluster unit, and the primary/secondary contrasts above. Include realized profile artifact hashes when available. Commit the manifest and confirm no code/config changes after it is created. If WeedMap compatibility or its evaluator is unresolved, record WeedMap as `NOT_AUTHORIZED`; do not present K8.3 as a two-dataset run.

The Kaggle test cell below creates a `test_started` marker **before** its first WeedsGalore test evaluation. If a run fails or the kernel is interrupted after the marker is written, do not rerun it as though nothing happened; preserve logs and record the deviation. It evaluates only the locked 15 checkpoints and then runs the locked WeedsGalore comparisons. It does **not** evaluate WeedMap. No test result may alter the manifest or method choice.

## Kaggle cells

**Kaggle only — do not run tests locally.** Copy the cells below from the committed/pushed documentation into the Kaggle notebook after updating the project checkout. Current order: K8.0 inventory (already done), K8.0b ZIP documentation (already done), K8.0c single-frame encoding (already done), K8.0d full-index class/mask audit (run only if not already saved), K8.0e revised synthetic loader/metric tests (run on Kaggle after code push), K8.0f backup. K8.1 profiling is already DONE and need not be rerun. K8.2/K8.3 are strictly NO-GO pending independently verified mask/radiometry/split/grouping, checkpoint integration and evaluator acceptance evidence. All diagnostic artifacts remain separate from the reviewed preflight and locked test manifest.

### Cell K8.0 — WeedMap RedEdge-M 003 inventory (no inference)

```python
import glob, hashlib, json, os, pathlib, sys
from collections import Counter
from PIL import Image

PROJECT = '/kaggle/working/WeedsGalore_Ipr28'
os.chdir(PROJECT)
EXP = 'exp08'
OUT = f'results/{EXP}'
os.makedirs(f'{OUT}/logs', exist_ok=True)

WM_CANDIDATES = ['/kaggle/working/weedmap-2018'] + glob.glob(
    '/kaggle/input/**/weedmap-2018', recursive=True)
WM_ROOT = next((path for path in WM_CANDIDATES if os.path.isdir(path)), None)
assert WM_ROOT, f'WeedMap root not found: {WM_CANDIDATES}'
WM_ROOT = pathlib.Path(WM_ROOT)
INDEX_ROOT = WM_ROOT / 'Tiles' / 'RedEdge' / '003'
INDEX_ARCHIVE = WM_ROOT / 'Tiles' / 'RedEdge' / '003.zip'
assert INDEX_ROOT.is_dir(), f'Extracted RedEdge index 003 not found: {INDEX_ROOT}'
assert INDEX_ARCHIVE.is_file(), f'RedEdge index 003 archive not found: {INDEX_ARCHIVE}'

def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

all_files = [p for p in INDEX_ROOT.rglob('*') if p.is_file()]
print('Mounted root:', WM_ROOT)
print('Dataset index root:', INDEX_ROOT)
print('Index 003 archive sha256:', sha256_file(INDEX_ARCHIVE))
print('File count:', len(all_files))
print('Extension counts:', Counter(p.suffix.lower() for p in all_files))
print('Index 003 subdirectories:')
for path in sorted(p for p in INDEX_ROOT.rglob('*') if p.is_dir()):
    print(' ', path.relative_to(INDEX_ROOT))

metadata_files = [p for p in all_files if p.suffix.lower() in ('.txt', '.md', '.xml', '.json', '.csv', '.yaml', '.yml')]
for path in metadata_files[:80]:
    print('METADATA', path, 'sha256=', sha256_file(path))

label_candidates = [p for p in all_files if 'groundtruth' in p.name.lower()
                    and p.suffix.lower() == '.png']
assert label_candidates, 'No GroundTruth file under dataset 003; inspect archive layout before proceeding'
label_samples = []
for path in label_candidates[:10]:
    with Image.open(path) as label_image:
        print('LABEL', path, 'mode=', label_image.mode, 'size=', label_image.size,
              'bands=', label_image.getbands(), 'format=', label_image.format)
        print('  palette/info keys:', sorted(label_image.info.keys()))
        label_samples.append({'path': str(path), 'mode': label_image.mode,
                              'size': list(label_image.size), 'bands': list(label_image.getbands()),
                              'format': label_image.format,
                              'info_keys': sorted(label_image.info.keys())})

tile_root = INDEX_ROOT / '003' / 'tile'
assert tile_root.is_dir(), f'Index 003 tile root not found: {tile_root}'
channel_dirs = [tile_root / band for band in ('R', 'G', 'B', 'NIR', 'RE')]
missing_channels = [str(path) for path in channel_dirs if not path.is_dir()]
assert not missing_channels, f'Missing required RedEdge channels: {missing_channels}'
rededge_candidates = [path for channel in channel_dirs for path in channel.glob('*.png')]
print('Index 003 per-band tile counts:', {path.name: len(list(path.glob('*.png')))
                                        for path in channel_dirs})
archive_hashes = {str(INDEX_ARCHIVE): sha256_file(INDEX_ARCHIVE)}
inventory = {
    'source_page': 'https://projects.asl.ethz.ch/datasets/weedmap-2018/',
    'paper_doi': '10.3390/rs10091423',
    'dataset_doi': '10.3929/ethz-c-000788571',
    'dataset_root': str(WM_ROOT),
    'dataset_index_root': str(INDEX_ROOT),
    'file_count': len(all_files),
    'dataset_003_rededge_candidate_count': len(rededge_candidates),
    'rededge_sample_paths': [str(path) for path in rededge_candidates[:100]],
    'label_samples': label_samples,
    'metadata_files': [str(path) for path in metadata_files],
    'archive_hashes': archive_hashes,
    'dataset_identity': {
        'dataset_index': '003',
        'sensor': 'MicaSense RedEdge-M',
        'tile_archive': str(INDEX_ARCHIVE),
        'tile_archive_sha256': sha256_file(INDEX_ARCHIVE),
    },
    'mask_validity_policy': 'PENDING: release says white is valid; inspected mask/signal relation appeared opposite',
    'verified_class_map': {'0': 'background', '1': 'crop', '2': 'weed', '10000': 'ignore'},
    'verified_label_semantics': {
        'verified': True,
        'codes': {'0': 'background', '1': 'crop', '2': 'weed', '10000': 'non-class_ignore'},
        'source': 'ETH Zurich Research Collection WeedMap release PDF, page 2',
        'source_url': 'https://www.research-collection.ethz.ch/server/api/core/bitstreams/3ae97814-1570-4760-95b8-a9a3eac21975/content',
    },
    'verified_mask_validity': None,
    'verified_class_support': None,
    'verified_input_transform': None,
    'inventory_status': 'STRUCTURE_ONLY_NOT_AUTHORIZATION',
    'source_test_split_citation': None,
    'verified_grouping': None,
    'manual_review_fields': ['official partition 003 and source citation',
                             'five co-registered RedEdge-M bands and channel order',
                             'radiometric calibration/PNG encoding and fixed source-compatible transform',
                             'mask validity polarity resolved against release',
                             'indexed semantics verified; class support and instance availability documented',
                             'license/reuse conditions and tile/orthomosaic grouping reviewed'],
    'preflight_note': 'Inventory only. Copy to a separate reviewed preflight artifact; do not authorize from this file.',
}
inventory_path = f'{OUT}/weedmap_003_inventory.json'
with open(inventory_path, 'w') as f:
    json.dump(inventory, f, indent=2)
print('Saved inventory:', inventory_path)
print('Existing reviewed preflight was not modified.')
print('Inventory only. Do not evaluate model predictions or tune from these labels.')
```

K8.0 chỉ tạo lại `results/exp08/weedmap_003_inventory.json`; nó không ghi vào `weedmap_003_preflight.json`. Sau review, tạo/cập nhật riêng file preflight từ inventory và gắn evidence/hash cho `dataset_identity`, test-split citation, `verified_class_support`, `verified_mask_validity`, `verified_input_transform`, `verified_grouping`, evaluator acceptance và protocol freeze. Chỉ đổi `manual_review_decision` sang `GO` khi tất cả gate đạt. K8.2 sẽ từ chối nếu thiếu bất kỳ evidence nào. Không chạy pixel statistics nhằm chọn model/threshold; kiểm tra class support chỉ là integrity check và phải báo cáo mọi kết quả.

### Cell K8.0b — Read WeedMap annotation documentation/source only (no label pixels)

Run this after K8.0 if you need to repeat the archive search in Kaggle. The previous cell scanned only already-extracted files and therefore returned zero when documentation was inside the per-index ZIP. This version scans text files in the extracted root and text members of `Tiles/RedEdge/003.zip`; it never opens PNG/TIFF entries. The authoritative code table is already recorded from the release PDF, so an empty archive-text search does not undo that verification.

```python
import hashlib
import os
import pathlib
import re
import glob
import zipfile

root_candidates = [pathlib.Path('/kaggle/working/weedmap-2018')]
root_candidates += [pathlib.Path(path) for path in glob.glob('/kaggle/input/**/weedmap-2018', recursive=True)]
WM_ROOT = next((path for path in root_candidates if path.is_dir()), None)
AUDIT_LOG = pathlib.Path('results/exp08/logs/weedmap_annotation_docs_audit.log')
AUDIT_LOG.parent.mkdir(parents=True, exist_ok=True)
assert WM_ROOT is not None, f'WeedMap root not found in candidates: {root_candidates}'

text_extensions = {
    '.txt', '.md', '.rst', '.html', '.htm', '.xml', '.json', '.yaml', '.yml',
    '.csv', '.m', '.py', '.tex', '.ini', '.cfg'
}
search_pattern = re.compile(
    r'10000|ground.?truth.?imap|ground.?truth.?color|background|crop|weed|'
    r'ignore|unlabel|non.?class|annotation|label.?map|class.?code',
    re.IGNORECASE,
)
exact_code_pattern = re.compile(r'(?<!\d)10000(?!\d)|GroundTruth[_ -]?(?:iMap|color)', re.IGNORECASE)

def sha256_file(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

def emit(message=''):
    print(message, flush=True)
    log.write(message + '\n')

text_files = sorted(
    path for path in WM_ROOT.rglob('*')
    if path.is_file() and path.suffix.lower() in text_extensions
    and path.stat().st_size <= 4 * 1024 * 1024
)
zip_candidates = sorted({
    path for path in [WM_ROOT / 'Tiles' / 'RedEdge' / '003.zip'] + list(WM_ROOT.rglob('003.zip'))
    if path.is_file() and 'rededge' in str(path).lower()
})
exact_hits = 0
files_with_hits = 0
with AUDIT_LOG.open('w', encoding='utf-8', buffering=1) as log:
    emit('WeedMap annotation documentation/source audit')
    emit(f'Root: {WM_ROOT}')
    emit('Scope: text files only; no PNG/TIFF/mask pixels opened.')
    emit(f'Text files scanned: {len(text_files)} (max 4 MiB each)')
    sources = []
    for path in text_files:
        try:
            sources.append((str(path.relative_to(WM_ROOT)), path.read_text(
                encoding='utf-8-sig', errors='replace').splitlines(), sha256_file(path)))
        except OSError as error:
            emit(f'SKIP {path.relative_to(WM_ROOT)}: {error}')
    for archive in zip_candidates:
        emit(f'ZIP {archive.relative_to(WM_ROOT)} sha256={sha256_file(archive)}')
        try:
            with zipfile.ZipFile(archive) as bundle:
                members = [item for item in bundle.infolist()
                           if pathlib.PurePosixPath(item.filename).suffix.lower() in text_extensions
                           and 0 < item.file_size <= 4 * 1024 * 1024]
                emit(f'  text members eligible: {len(members)}')
                for item in members:
                    with bundle.open(item) as stream:
                        content = stream.read(4 * 1024 * 1024 + 1)
                    if len(content) > 4 * 1024 * 1024:
                        continue
                    lines = content.decode('utf-8-sig', errors='replace').splitlines()
                    sources.append((f'{archive.relative_to(WM_ROOT)}::{item.filename}', lines,
                                    hashlib.sha256(content).hexdigest()))
        except (OSError, zipfile.BadZipFile) as error:
            emit(f'  ZIP READ ERROR: {error}')
    for source_name, lines, source_hash in sources:
        matches = [(index, line.strip()) for index, line in enumerate(lines, start=1)
                   if search_pattern.search(line)]
        exact_matches = [(index, line) for index, line in matches
                         if exact_code_pattern.search(line)]
        if not matches:
            continue
        files_with_hits += 1
        exact_hits += len(exact_matches)
        emit(f'\nTEXT SOURCE {source_name} sha256={source_hash}')
        selected = exact_matches if exact_matches else matches[:5]
        if exact_matches:
            context_lines = set()
            for index, _ in exact_matches:
                context_lines.update(range(max(1, index - 2), min(len(lines), index + 2) + 1))
            selected = [(index, lines[index - 1].strip()) for index in sorted(context_lines)]
        else:
            selected = matches[:5]
        for index, line in selected[:20]:
            emit(f'  L{index}: {line[:500]}')
    emit('\nARCHIVE TEXT SEARCH: COMPLETE; this is supplementary and does not determine label semantics.')
    emit('Official semantics remain those in the cited ETH Zurich release PDF, page 2.')
    emit(f'Files with relevant text: {files_with_hits}; exact code/name hits: {exact_hits}')
    emit(f'Saved audit: {AUDIT_LOG}')
print('Saved audit:', AUDIT_LOG, flush=True)
```

Review any additional archive excerpts and add their paths/hashes to the preflight for provenance. The class-code mapping itself is verified by the official release PDF; do not infer mask polarity or radiometry from the annotation color map. Three-class metric availability remains conditional on valid Crop support in the official evaluation area.

### Cell K8.0c — Input encoding/range inspection (imagery only; no transform fitting)

This compares PNG mode/dtype/range as loaded by the current WeedsGalore `plt.imread` pipeline against one corresponding WeedMap 003 tile per band. The WeedMap sample is predeclared as `frame0020`, selected from the prior imagery inventory because all five bands contain signal; it is not selected using labels or model output. Quantiles below describe this one image only, not a radiometric calibration or normalization estimate. The cell does not read labels/masks or fit/choose normalization statistics.

```python
import glob
import os
from pathlib import Path

import matplotlib.image as mpimg
import numpy as np
from PIL import Image

PROJECT = Path('/kaggle/working/WeedsGalore_Ipr28')
os.chdir(PROJECT)
AUDIT_LOG = Path('results/exp08/logs/radiometry_encoding_audit.log')
AUDIT_LOG.parent.mkdir(parents=True, exist_ok=True)
WM_CANDIDATES = [Path('/kaggle/working/weedmap-2018')]
WM_CANDIDATES += [Path(path) for path in glob.glob('/kaggle/input/**/weedmap-2018', recursive=True)]
WM_ROOT = next((path for path in WM_CANDIDATES if path.is_dir()), None)
assert WM_ROOT is not None, f'Cannot find WeedMap root: {WM_CANDIDATES}'
WG_CANDIDATES = sorted(glob.glob('/kaggle/input/**/2023-05-25/images', recursive=True))
if WG_CANDIDATES:
    WG_ROOT = Path(WG_CANDIDATES[0]).parent.parent
elif Path('/kaggle/working/weedsgalore-dataset/2023-05-25/images').is_dir():
    WG_ROOT = Path('/kaggle/working/weedsgalore-dataset')
else:
    raise FileNotFoundError('Cannot find WeedsGalore images under Kaggle input/working')

WM_TILE_ROOT = WM_ROOT / 'Tiles' / 'RedEdge' / '003' / '003' / 'tile'
assert WM_TILE_ROOT.is_dir(), f'WeedMap tile root missing: {WM_TILE_ROOT}'
bands = ('R', 'G', 'B', 'NIR', 'RE')
wg_files = sorted((WG_ROOT / '2023-05-25' / 'images').glob('*_R.png'))
assert wg_files, f'No WeedsGalore sample images under {WG_ROOT}'
wg_stem = wg_files[0].name[:-len('_R.png')]
wm_stem = 'frame0020'
assert (WM_TILE_ROOT / 'R' / f'{wm_stem}.png').is_file(), (
    f'Predeclared WeedMap radiometry sample is missing: {wm_stem}')

def report_image(domain, band, path):
    with Image.open(path) as image:
        raw = np.asarray(image)
        mode = image.mode
        size = image.size
    if domain == 'WeedMap-003-RedEdge':
        assert size == (480, 360), f'Unexpected predeclared frame size: {path} {size}'
        assert int(raw.max()) > 0, f'Predeclared frame is blank in {band}: {path}'
    loaded = mpimg.imread(path)
    finite = loaded[np.isfinite(loaded)]
    quantiles = np.percentile(finite, [0, 1, 50, 99, 100])
    record = {
        'domain': domain,
        'band': band,
        'path': str(path),
        'png_mode': mode,
        'raw_dtype': str(raw.dtype),
        'raw_shape': list(raw.shape),
        'size_wh': list(size),
        'raw_min_max': [int(raw.min()), int(raw.max())],
        'plt_imread_dtype': str(loaded.dtype),
        'plt_imread_min_p01_p50_p99_max': [float(v) for v in quantiles],
    }
    print(record, flush=True)
    audit.write(str(record) + '\n')

with AUDIT_LOG.open('w', encoding='utf-8', buffering=1) as audit:
    print('WeedsGalore root:', WG_ROOT, '| sample:', wg_stem, flush=True)
    print('WeedMap root:', WM_TILE_ROOT, '| sample:', wm_stem, flush=True)
    audit.write(f'WeedsGalore root: {WG_ROOT} | sample: {wg_stem}\n')
    audit.write(f'WeedMap root: {WM_TILE_ROOT} | sample: {wm_stem}\n')
    for band in bands:
        wg_path = WG_ROOT / '2023-05-25' / 'images' / f'{wg_stem}_{band}.png'
        wm_path = WM_TILE_ROOT / band / f'{wm_stem}.png'
        assert wg_path.is_file(), f'Missing WeedsGalore {band}: {wg_path}'
        assert wm_path.is_file(), f'Missing WeedMap {band}: {wm_path}'
        with Image.open(wm_path) as image:
            assert image.size == (480, 360), f'Unexpected WeedMap frame0020 size for {band}: {image.size}'
        report_image('WeedsGalore', band, wg_path)
        report_image('WeedMap-003-RedEdge', band, wm_path)
    print('No labels/masks were opened. Do not use these summaries to fit or tune a transform.', flush=True)
    print('Radiometric compatibility remains PENDING until source calibration/encoding documentation is verified.', flush=True)
    audit.write('No labels/masks were opened. No transform fitted. Radiometry remains PENDING.\n')
print('Saved audit:', AUDIT_LOG, flush=True)
```

The source loader in `src/datasets/weedsgalore/weedsgalore.py` calls `plt.imread` on each PNG and stacks `R,G,B,NIR,RE`; `src/train.py` converts batches to the device without an explicit `/255`, standardization, or per-band normalization. Thus the effective numeric scale depends on the stored PNG bit depth and `plt.imread` behavior. Matching numeric ranges after `plt.imread` is necessary but not sufficient: the WeedMap PNGs may be display-stretched/quantized rather than calibrated reflectance. Do not authorize inference until the release/source documents establish the tile PNG radiometry or supply a fixed transform to align it with the WeedsGalore training inputs.

**Radiometry gate:** the WeedMap landing page distinguishes per-channel tile PNGs from orthomosaic reflectance TIFFs. The paper's statement that channels are aligned/calibrated across the map does not, by itself, specify the numerical encoding of each exported tile PNG or show that it matches WeedsGalore's PNG input scale. Before inference, identify the tile PNG export/reflectance conversion rule from the release materials, or document a deterministic mapping from the calibrated reflectance product to the checkpoint's source-domain numeric scale. Do not estimate per-band normalization from WeedMap test tiles. If the release only provides display-stretched PNGs without a documented invertible/fixed mapping, stop the zero-shot evaluation rather than claiming radiometric transfer.

### Cell K8.0d — Full-index class support and mask-polarity integrity audit (no inference)

This is a data-integrity audit of all index-003 iMap/mask pairs. It reports class counts for the complete iMap and separately under `mask==0` and `mask==255`; it does not decide which mask value is valid, tune a method, or run inference. Use the authoritative iMap codes from the ETH release: 0 background, 1 crop, 2 weed, 10000 non-class/ignore. Save the output and compare the mask branches against the documented mask policy. This is the evidence source for determining whether three-class metrics are defined on the evaluated partition.

```python
import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path('/kaggle/working/weedmap-2018/Tiles/RedEdge/003/003')
OUT = Path('results/exp08')
LOG = OUT / 'logs' / 'class_support_mask_integrity.log'
REPORT = OUT / 'weedmap_003_class_support_integrity.json'
OUT.mkdir(parents=True, exist_ok=True)
LOG.parent.mkdir(parents=True, exist_ok=True)
BANDS = ('R', 'G', 'B', 'NIR', 'RE')
CODE_NAMES = {0: 'background', 1: 'crop', 2: 'weed', 10000: 'non_class_ignore'}
frames = sorted(path.stem for path in (ROOT / 'tile' / 'R').glob('frame*.png'))
assert len(frames) == 204, f'Expected 204 R frames in index 003, found {len(frames)}'

all_counts = Counter()
counts_by_mask = {0: Counter(), 255: Counter()}
mask_patterns = Counter()
imap_modes = Counter()
missing_pairs = []
shape_mismatches = []
frame_rows = []
band_frame_sets = {
    band: {path.stem for path in (ROOT / 'tile' / band).glob('frame*.png')}
    for band in BANDS
}
for band, band_frames in band_frame_sets.items():
    assert band_frames == set(frames), f'{band} frame IDs differ from R'

with LOG.open('w', encoding='utf-8', buffering=1) as log:
    def emit(message):
        print(message, flush=True)
        log.write(message + '\n')

    emit('WeedMap 003 class-support/mask integrity audit; no model inference.')
    emit('Counts are reported separately for mask==0 and mask==255; neither is presumed valid.')
    for frame in frames:
        imap_path = ROOT / 'groundtruth' / f'003_{frame}_GroundTruth_iMap.png'
        mask_path = ROOT / 'mask' / f'{frame}.png'
        if not imap_path.is_file() or not mask_path.is_file():
            missing_pairs.append(frame)
            continue
        with Image.open(imap_path) as image:
            imap = np.asarray(image)
            imap_mode = image.mode
        with Image.open(mask_path) as image:
            mask = np.asarray(image)
        if imap.ndim != 2 or mask.ndim != 2 or imap.shape != mask.shape:
            shape_mismatches.append({'frame': frame, 'imap_shape': list(imap.shape),
                                     'mask_shape': list(mask.shape)})
            continue
        unknown = set(np.unique(imap).tolist()) - set(CODE_NAMES)
        assert not unknown, f'Unexpected iMap codes in {frame}: {sorted(unknown)}'
        imap_modes[imap_mode] += 1
        mask_values, mask_counts = np.unique(mask, return_counts=True)
        mask_patterns[tuple(int(value) for value in mask_values)] += 1
        frame_counts = Counter({int(code): int(count) for code, count in
                                zip(*np.unique(imap, return_counts=True))})
        all_counts.update(frame_counts)
        for mask_value in (0, 255):
            values, counts = np.unique(imap[mask == mask_value], return_counts=True)
            counts_by_mask[mask_value].update({int(code): int(count) for code, count in zip(values, counts)})
        frame_rows.append({'frame': frame, 'imap_mode': imap_mode,
                           'mask_counts': {str(int(value)): int(count) for value, count
                                           in zip(mask_values, mask_counts)},
                           'class_counts': {str(code): frame_counts.get(code, 0) for code in CODE_NAMES}})

    assert not missing_pairs, f'Missing iMap/mask pairs: {missing_pairs}'
    assert not shape_mismatches, f'iMap/mask shape mismatches: {shape_mismatches[:10]}'
    crop_count = all_counts.get(1, 0)
    result = {
        'dataset_index': '003',
        'tile_archive': '/kaggle/working/weedmap-2018/Tiles/RedEdge/003.zip',
        'frames_expected': len(frames),
        'frames_aligned': len(frame_rows),
        'band_frame_counts': {band: len(ids) for band, ids in band_frame_sets.items()},
        'imap_modes': dict(imap_modes),
        'mask_patterns_by_frame': {str(key): value for key, value in mask_patterns.items()},
        'class_codes': {str(code): name for code, name in CODE_NAMES.items()},
        'all_imap_counts': {str(code): int(all_counts.get(code, 0)) for code in CODE_NAMES},
        'counts_if_mask_zero_valid': {str(code): int(counts_by_mask[0].get(code, 0)) for code in CODE_NAMES},
        'counts_if_mask_255_valid': {str(code): int(counts_by_mask[255].get(code, 0)) for code in CODE_NAMES},
        'crop_code_present_in_any_imap_pixel': bool(crop_count),
        'mask_validity_decision': 'PENDING; both hypotheses retained',
        'inference_run': False,
        'per_frame': frame_rows,
    }
    with REPORT.open('w', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2)
    emit(f'Frames aligned: {len(frame_rows)} / {len(frames)}')
    emit(f'iMap modes: {dict(imap_modes)}')
    emit(f'Mask patterns by frame: {dict(mask_patterns)}')
    emit(f'All iMap class counts: {result["all_imap_counts"]}')
    emit(f'Counts if mask==0 is valid: {result["counts_if_mask_zero_valid"]}')
    emit(f'Counts if mask==255 is valid: {result["counts_if_mask_255_valid"]}')
    emit(f'Crop code 1 present anywhere: {bool(crop_count)}')
    emit('No mask polarity selected; no model inference performed.')
    emit(f'Saved report: {REPORT}')
    emit(f'Saved log: {LOG}')
```

### Cell K8.0e — Run synthetic WeedMap loader/metric tests on Kaggle (no real dataset/inference)

**Only Kaggle, never local.** First commit/push the revised `src/weedmap_evaluation.py` and `tests/test_weedmap_evaluation.py`, then pull that exact commit into the Kaggle clone (or restart the notebook against the updated checkout). This test cell must operate on the revised files from the review package; the earlier 78-line metric-only implementation and its 7 tests are not equivalent. Check that you are in the project root and the source/test files exist. The tests create temporary synthetic PNGs under pytest's temporary directory; they do **not** open WeedMap/WeedsGalore imagery, load a checkpoint, or evaluate predictions from a held-out test set.

This cell saves the **actual Kaggle** pytest transcript and a hash-bound synthetic-core test report. It does *not* set `weedmap_evaluator_acceptance.json` to PASS, choose `valid_mask_code`, provide radiometric calibration evidence, or authorize K8.2/K8.3. Do not fabricate acceptance metadata from successful unit tests.

```python
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT = Path('/kaggle/working/WeedsGalore_Ipr28')
assert PROJECT.is_dir(), f'Project not found: {PROJECT}'
os.chdir(PROJECT)
source_path = Path('src/weedmap_evaluation.py')
test_path = Path('tests/test_weedmap_evaluation.py')
assert source_path.is_file() and test_path.is_file(), 'Push/clone both revised Python files first'

# Prevent accidentally testing the older metric-only files.
source_text = source_path.read_text(encoding='utf-8')
test_text = test_path.read_text(encoding='utf-8')
for symbol in ('def discover_frames(', 'def load_tile(', 'def resolve_valid_pixels(',
               'def aggregate_predictions('):
    assert symbol in source_text, f'Expected revised evaluator primitive missing: {symbol}'
assert test_text.count('def test_') >= 14, 'Expected revised synthetic test suite (14 tests)'

OUT = Path('results/exp08')
LOGS = OUT / 'logs'
LOGS.mkdir(parents=True, exist_ok=True)
log_path = LOGS / 'weedmap_synthetic_core_tests.log'
report_path = OUT / 'weedmap_synthetic_core_test_report.json'

def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

command = [sys.executable, '-m', 'pytest', '-q', '-ra', str(test_path)]
print('$', ' '.join(command), flush=True)
with log_path.open('w', encoding='utf-8', buffering=1) as log:
    proc = subprocess.Popen(command, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in proc.stdout:
        print(line, end='', flush=True)
        log.write(line)
    exit_code = proc.wait()

report = {
    'status': 'SYNTHETIC_CORE_PASS' if exit_code == 0 else 'SYNTHETIC_CORE_FAIL',
    'authorization': 'NOT_AUTHORIZED_FOR_WEEDMAP_INFERENCE_OR_K8_2_K8_3',
    'timestamp_utc': datetime.now(timezone.utc).isoformat(),
    'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
    'command': command,
    'pytest_exit_code': exit_code,
    'source_file': str(source_path),
    'source_sha256': sha256(source_path),
    'test_file': str(test_path),
    'test_sha256': sha256(test_path),
    'log_file': str(log_path),
    'log_sha256': sha256(log_path),
    'scope': 'Synthetic loader, band order, mask-code handling, labels, confusion and aggregation only',
    'not_verified': ['radiometry', 'official mask polarity', 'official test split',
                     'spatial grouping', 'checkpoint inference integration'],
}
report_path.write_text(json.dumps(report, indent=2), encoding='utf-8')
print('Saved synthetic test report:', report_path, flush=True)
print('Saved pytest log:', log_path, flush=True)
assert exit_code == 0, 'Synthetic tests failed: inspect the saved pytest log'
print('Synthetic code tests PASS; K8.2/K8.3 remain NO-GO.', flush=True)
```

Expected output for the revised uploaded test suite: **`14 passed`**. The exact Kaggle output, not a local run, is the evidence. If fewer tests are collected, a test fails, or the code is stale, stop; do not create a PASS acceptance report.

### Cell K8.0f — Backup Exp08 audit/test artifacts (no inference)

After K8.0e, back up the Exp08 inventory, class/mask audit, radiometry inspection, synthetic-test report and logs. This does not change any reviewed preflight or open any test. The ZIP does not contain checkpoints, test predictions, or other `results/` experiments.

```python
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

PROJECT = Path('/kaggle/working/WeedsGalore_Ipr28')
exp08 = PROJECT / 'results' / 'exp08'
backup = Path('/kaggle/working/exp08_preflight_synthetic_backup.zip')
assert exp08.is_dir(), f'Exp08 outputs not found: {exp08}'
with ZipFile(backup, 'w', compression=ZIP_DEFLATED) as archive:
    for path in sorted(exp08.rglob('*')):
        if path.is_file():
            archive.write(path, arcname=str(path.relative_to(PROJECT)))
print('Saved:', backup)
print('Size MB:', round(backup.stat().st_size / 1024**2, 2))
from IPython.display import FileLink, display
display(FileLink(str(backup)))
```

### Cell K8.1 — Profile L0, OccCond-g, MCSD-g, PS (600×600)

```python
import json, os, subprocess, sys

OUT = 'results/exp08'
LOGS = f'{OUT}/logs'
os.makedirs(LOGS, exist_ok=True)

def run_live(args, log_path):
    print('\n$', ' '.join(args), flush=True)
    with open(log_path, 'w', buffering=1) as log_file:
        process = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, bufsize=1)
        for line in process.stdout:
            print(line, end='', flush=True)
            log_file.write(line)
        code = process.wait()
    if code != 0:
        raise subprocess.CalledProcessError(code, args)

profiles = {
    'L0': '--ccmc=False --head=bilinear --use_low_level_refine=False',
    'PS': '--ccmc=False --head=pixelshuffle --spectral_guide=False',
    'OccCondg': '--ccmc=False --head=occ_conditioned --spectral_guide=True --mcsd_tau=1.0',
    'MCSDg': '--ccmc=False --head=mcsd --spectral_guide=True --sinkhorn_iters=5 --mcsd_tau=1.0',
}
for label, arch in profiles.items():
    out_json = f'{OUT}/profile_{label}.json'
    args = [sys.executable, 'src/scripts/profile_model.py', *arch.split(),
            '--in_channels=5', '--num_classes=3', '--use_multiscale_fusion=True',
            '--size=600', '--warmup=50', '--runs=300', f'--label={label}', f'--out={out_json}']
    run_live(args, f'{LOGS}/profile_{label}.log')
    assert os.path.isfile(out_json), f'{label} profile JSON missing'
    with open(out_json) as f:
        result = json.load(f)
    print(label, 'params_M=', result['params_M'],
          'GFLOPs=', result['GFLOPs_flopcounter'],
          'median_ms=', result['latency_ms_median'],
          'peak_mem_MB=', result['peak_mem_MB'], flush=True)

cost_rows = []
for label in profiles:
    with open(f'{OUT}/profile_{label}.json') as f:
        result = json.load(f)
    cost_rows.append({
        'method': label,
        'params_M': result['params_M'],
        'GFLOPs': result['GFLOPs_flopcounter'],
        'GMACs': result['GMACs'],
        'latency_ms_median': result['latency_ms_median'],
        'latency_ms_p10': result['latency_ms_p10'],
        'latency_ms_p90': result['latency_ms_p90'],
        'peak_mem_MB': result['peak_mem_MB'],
        'device': result['device'],
        'input': result['input'],
        'precision': result['precision'],
    })
cost_path = f'{OUT}/compute_cost_table.json'
with open(cost_path, 'w') as f:
    json.dump({'protocol': '600x600, batch=1, fp32, warmup=50, synchronized runs=300',
               'profiles': cost_rows}, f, indent=2)
print('Saved compute-cost table:', cost_path, flush=True)
for row in cost_rows:
    print(row, flush=True)
```

K8.1 completed on Tesla T4 with batch 1, FP32, input 600×600, 50 warmup iterations, and 300 synchronized timed runs. These are forward-only profiles, not training measurements. The FLOP counter omits elementwise operations such as Sinkhorn normalization, while latency includes them. All four saved outputs are in `doc/exp123.ipynb`.

| Model     | Parameters (M) | Counted GFLOPs |    GMACs | Median latency (ms) [p10, p90] | Peak memory (MB) |
| --------- | -------------: | -------------: | -------: | -----------------------------: | ---------------: |
| L0        |       3.455619 |      13.268002 | 6.634001 |        14.225 [14.162, 14.308] |          145.366 |
| PS        |       3.461811 |      13.544482 | 6.772241 |        14.470 [14.335, 14.591] |          145.390 |
| OccCond-g |       3.462614 |      14.097442 | 7.048721 |        15.799 [15.714, 15.899] |          145.396 |
| MCSD-g    |       3.462614 |      14.097442 | 7.048721 |        17.833 [17.693, 17.982] |          145.396 |

### Cell K8.2 — Lock the WeedsGalore test manifest

Run this only after the method set, metrics, and exact checkpoint set are final. It does not read test labels or run test.

```python
import hashlib, json, os, subprocess

OUT = 'results/exp08'
os.makedirs(OUT, exist_ok=True)
EXPECTED_COMMIT = 'REPLACE_WITH_PUSHED_COMMIT'
actual_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
assert actual_commit == EXPECTED_COMMIT, (actual_commit, EXPECTED_COMMIT)

def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

specs = {
    'L0': '--ccmc=True --prior_mode=none --use_low_level_refine=False',
    'OccCondg': '--ccmc=False --head=occ_conditioned --spectral_guide=True --mcsd_tau=1.0',
    'MCSDg': '--ccmc=False --head=mcsd --spectral_guide=True --sinkhorn_iters=5 --mcsd_tau=1.0',
}
checkpoints = {}
for model in specs:
    checkpoints[model] = {}
    for seed in range(5):
        path = f'runs/{model}_s{seed}/best_val.pth'
        assert os.path.isfile(path), f'Missing frozen checkpoint: {path}'
        checkpoints[model][str(seed)] = {'path': path, 'sha256': sha256_file(path)}

profile_files = {name: f'results/exp08/profile_{name}.json'
                 for name in ('L0', 'PS', 'OccCondg', 'MCSDg')}
profiles = {}
for name, path in profile_files.items():
    assert os.path.isfile(path), f'Missing locked-cost profile: {path}'
    profiles[name] = {'path': path, 'sha256': sha256_file(path)}

run_configs = {}
for model in specs:
    run_configs[model] = {}
    for seed in range(5):
        path = f'runs/{model}_s{seed}/run_config.json'
        assert os.path.isfile(path), f'Missing run config: {path}'
        with open(path) as f:
            config = json.load(f)
        assert config['split_hash'] == '675667edf3f35f16', f'Split mismatch: {path}'
        flags = config['flags']
        assert int(flags['seed']) == seed, f'Seed/config mismatch: {path}'
        assert int(flags['in_channels']) == 5 and int(flags['num_classes']) == 3, (
            f'Input/class config mismatch: {path}')
        assert bool(flags['use_multiscale_fusion']) is True, f'Decoder config mismatch: {path}'
        expected_head = {'L0': 'bilinear', 'OccCondg': 'occ_conditioned', 'MCSDg': 'mcsd'}[model]
        assert flags['head'] == expected_head, f'Method/head mismatch: {path}'
        assert bool(flags['ccmc']) is (model == 'L0'), f'Encoder path mismatch: {path}'
        assert bool(flags['spectral_guide']) is (model != 'L0'), f'Spectral guide mismatch: {path}'
        run_configs[model][str(seed)] = {'path': path, 'sha256': sha256_file(path)}

preflight_path = f'{OUT}/weedmap_003_preflight.json'
assert os.path.isfile(preflight_path), 'Run and review K8.0 before locking the manifest'
with open(preflight_path) as f:
    weedmap_preflight = json.load(f)
assert weedmap_preflight['manual_review_decision'] == 'GO', (
    'WeedMap is not authorized: finish archive, radiometry, labels, split, and leakage review')
required_preflight_fields = ('dataset_identity', 'verified_input_transform', 'verified_class_map',
                            'verified_label_semantics', 'verified_mask_validity',
                            'verified_class_support', 'verified_grouping',
                            'source_test_split_citation')
missing_preflight = [key for key in required_preflight_fields
                     if not weedmap_preflight.get(key)]
assert not missing_preflight, f'Incomplete reviewed WeedMap preflight fields: {missing_preflight}'
assert weedmap_preflight['verified_label_semantics'].get('verified') is True, (
    'Official WeedMap indexed-label semantics have not been verified')
assert weedmap_preflight['verified_mask_validity'].get('verified') is True, (
    'WeedMap mask-validity convention is unresolved')
assert weedmap_preflight['verified_input_transform'].get('fixed_without_target_fitting') is True, (
    'WeedMap radiometric transform is not source-compatible and fixed')
assert weedmap_preflight['verified_class_support'].get('status') in ('crop_present', 'crop_absent'), (
    'Record whether valid Crop labels are present before selecting the metric branch')
weedmap_class_support = weedmap_preflight['verified_class_support']['status']
weedmap_metrics = ['WeedRecall', 'WeedPrecision', 'IoU_weed']
if weedmap_class_support == 'crop_present':
    weedmap_metrics.extend(['IoU_bg', 'IoU_crop', 'mIoU'])

evidence_path = f'{OUT}/weedmap_evaluator_acceptance.json'
assert os.path.isfile(evidence_path), 'WeedMap evaluator acceptance evidence is missing'
with open(evidence_path) as f:
    evaluator_evidence = json.load(f)
assert evaluator_evidence.get('status') == 'PASS', 'WeedMap evaluator is not accepted'
evaluator_path = evaluator_evidence.get('evaluator_path')
test_report_path = evaluator_evidence.get('test_report_path')
assert evaluator_path and os.path.isfile(evaluator_path), 'Accepted evaluator source is missing'
assert test_report_path and os.path.isfile(test_report_path), 'Evaluator test report is missing'
assert sha256_file(evaluator_path) == evaluator_evidence.get('evaluator_sha256'), (
    'Evaluator source hash does not match acceptance report')
assert sha256_file(test_report_path) == evaluator_evidence.get('test_report_sha256'), (
    'Evaluator test report hash does not match acceptance report')
assert evaluator_evidence.get('tests_passed', 0) > 0 and evaluator_evidence.get('tests_failed', 1) == 0, (
    'Evaluator acceptance requires passing synthetic tests and zero failures')
assert evaluator_evidence.get('mask_policy_evidence'), 'Mask policy source/audit evidence is missing'
mask_evidence_path = evaluator_evidence.get('mask_policy_evidence_path')
radiometry_evidence_path = evaluator_evidence.get('radiometry_evidence_path')
assert mask_evidence_path and os.path.isfile(mask_evidence_path), 'Mask-policy evidence artifact is missing'
assert radiometry_evidence_path and os.path.isfile(radiometry_evidence_path), (
    'Radiometry evidence artifact is missing')
assert sha256_file(mask_evidence_path) == evaluator_evidence.get('mask_policy_evidence_sha256'), (
    'Mask-policy evidence hash does not match acceptance report')
assert sha256_file(radiometry_evidence_path) == evaluator_evidence.get('radiometry_evidence_sha256'), (
    'Radiometry evidence hash does not match acceptance report')
assert evaluator_evidence.get('synthetic_checks', {}).get('ignore_10000_excluded') is True
assert evaluator_evidence.get('synthetic_checks', {}).get('crop_absence_branch') is True
assert evaluator_evidence.get('synthetic_checks', {}).get('mask_validity') is True
assert evaluator_evidence.get('synthetic_checks', {}).get('fixed_band_order') is True
assert evaluator_evidence.get('radiometry_evidence'), 'Fixed transform provenance is missing'
assert evaluator_evidence.get('statistical_protocol_frozen') is True, (
    'WeedMap metric/bootstrap protocol is not frozen')
protocol_path = evaluator_evidence.get('statistical_protocol_path')
assert protocol_path and os.path.isfile(protocol_path), 'Frozen statistical protocol artifact is missing'
assert sha256_file(protocol_path) == evaluator_evidence.get('statistical_protocol_sha256'), (
    'Statistical protocol hash does not match acceptance report')

assert set(checkpoints) == {'L0', 'OccCondg', 'MCSDg'}
assert all(set(seeds) == {str(seed) for seed in range(5)} for seeds in checkpoints.values()), (
    'Expected exactly seeds 0-4 for each of the three methods')
assert set(run_configs) == set(checkpoints)
assert all(set(seeds) == {str(seed) for seed in range(5)} for seeds in run_configs.values()), (
    'Expected one source run config for every locked checkpoint')

source_files = (
    'src/nets/mcsd.py', 'src/nets/segformer.py', 'src/ccmc_common.py',
    'src/ccmc_metrics.py', 'src/evaluate_ccmc.py', 'src/scripts/compare_runs.py',
    'src/train.py', 'src/scripts/make_instance_bins.py',
    'src/datasets/weedsgalore/weedsgalore.py',
    evaluator_path,
)
source_hashes = {}
for path in source_files:
    assert os.path.isfile(path), f'Missing locked source file: {path}'
    source_hashes[path] = sha256_file(path)

bins_path = 'results/instance_bins.json'
assert os.path.isfile(bins_path), f'Missing instance bins: {bins_path}'
with open(bins_path) as f:
    bins_data = json.load(f)
assert bins_data['edges'] == [45.0, 147.0], f'Unexpected instance-bin edges: {bins_data["edges"]}'

manifest = {
    'protocol': 'one-time WeedsGalore test evaluation; frozen external-method comparison set',
    'commit': actual_commit,
    'source_commit': actual_commit,
    'source_files': source_hashes,
    'evaluator_acceptance': {
        'path': evidence_path,
        'sha256': sha256_file(evidence_path),
        'test_report_path': test_report_path,
        'test_report_sha256': sha256_file(test_report_path),
        'mask_evidence_path': mask_evidence_path,
        'mask_evidence_sha256': sha256_file(mask_evidence_path),
        'radiometry_evidence_path': radiometry_evidence_path,
        'radiometry_evidence_sha256': sha256_file(radiometry_evidence_path),
        'statistical_protocol_path': protocol_path,
        'statistical_protocol_sha256': sha256_file(protocol_path),
    },
    'split_hash': '675667edf3f35f16',
    'checkpoint_selection': 'best_val only; selected on validation before test lock',
    'methods': specs,
    'checkpoints': checkpoints,
    'run_configs': run_configs,
    'compute_profiles': profiles,
    'instance_bins': {'path': 'results/instance_bins.json',
                      'sha256': sha256_file('results/instance_bins.json'),
                      'edges': [45.0, 147.0]},
    'evaluation': {
        'split': 'test', 'input_channels': ['R', 'G', 'B', 'NIR', 'RE'],
        'boundary_px': 3, 'instance_bins': [45.0, 147.0],
        'primary': 'SmallWeedRecall', 'co_primary': 'cov_small', 'gate': 'none',
        'bootstrap': {'replicates': 10000, 'cluster': 'image', 'paired_seed_resampling': True, 'seed': 0},
        'metrics': ['mIoU', 'IoU_bg', 'IoU_crop', 'IoU_weed', 'BIoU_crop', 'BIoU_weed',
                'SmallWeedRecall', 'recall30_small', 'cov_small', 'PrecisionWeed', 'CropToWeedRate'],
        'contrasts': [['OccCondg', 'L0'], ['MCSDg', 'L0'], ['MCSDg', 'OccCondg']],
        'no_test_based_selection': True,
    },
    'weedmap_protocol': {
        'role': 'zero-shot external-domain confirmation of frozen WeedsGalore checkpoints; not method development',
        'dataset': 'WeedMap official RedEdge-M testing dataset 003',
        'preflight_path': preflight_path,
        'preflight_sha256': sha256_file(preflight_path),
        'sensor_channels': ['R', 'G', 'B', 'NIR', 'RE'],
        'label_semantics': weedmap_preflight['verified_label_semantics'],
        'mask_validity': weedmap_preflight['verified_mask_validity'],
        'class_support': weedmap_preflight['verified_class_support'],
        'input_transform': weedmap_preflight['verified_input_transform'],
        'metrics': weedmap_metrics,
        'primary_endpoint': 'WeedRecall',
        'primary_contrast': ['OccCondg', 'L0'],
        'secondary_metrics_common': ['WeedPrecision', 'IoU_weed'],
        'secondary_metrics_if_crop_present': ['mIoU', 'IoU_bg', 'IoU_crop'],
        'secondary_metrics_if_crop_absent': ['WeedRecall', 'WeedPrecision', 'IoU_weed'],
        'instance_metric': 'recall50_only_if_official_instance_ids_verified',
        'ignore_code': 10000,
        'mask_policy_status': 'PENDING; resolve documented white-valid rule against observed mask/signal relation',
        'secondary_contrasts': [['MCSDg', 'L0'], ['MCSDg', 'OccCondg']],
        'bootstrap': {'replicates': 10000, 'seed': 0, 'paired_seed_resampling': True,
                  'spatial_cluster': 'orthomosaic_or_flight_then_tile'},
        'evaluator_status': 'PASS',
        'evaluator_acceptance_path': evidence_path,
        'evaluator_acceptance_sha256': sha256_file(evidence_path),
        'required_preflight': 'channel alignment, reflectance scale, labels, official split, and archive hash verified before inference',
    },
}
lock_path = f'{OUT}/test_lock_manifest.json'
assert not os.path.exists(lock_path), 'Manifest already exists; do not overwrite a locked manifest'
with open(lock_path, 'w') as f:
    json.dump(manifest, f, indent=2)
print('LOCKED:', lock_path, flush=True)
print(json.dumps(manifest, indent=2), flush=True)
```

Commit/push the finalized source first and set `EXPECTED_COMMIT` to that source commit for K8.2. K8.2 records `source_commit` and hashes of the files that affect evaluation. Then commit/push the generated manifest as a separate commit. K8.3 requires the source commit to be an ancestor of current HEAD and verifies every locked source-file hash; it does not incorrectly require HEAD to equal the earlier source commit. The start marker records both current HEAD (the manifest commit) and the immutable manifest hash.

### Cell K8.3 — One-time WeedsGalore test (only after manifest is committed)

This is the single release operation across 15 frozen checkpoints, not 15 model-selection trials. It releases the WeedsGalore test only. K8.2 additionally requires independently implemented and synthetically tested WeedMap evaluator acceptance before it can create the combined final manifest. All environment, data, checkpoint, CLI, manifest, and evaluator-acceptance checks run before the one-time marker. If any check fails, no marker is written and no test process starts. Once the marker is written, interruption/failure consumes the one-time run; preserve the evidence and do not silently rerun.

```python
import glob, hashlib, json, os, subprocess, sys
from pathlib import Path

OUT = 'results/exp08'
LOGS = f'{OUT}/logs'
manifest_path = f'{OUT}/test_lock_manifest.json'
marker_path = f'{OUT}/test_started.json'
os.makedirs(LOGS, exist_ok=True)
assert os.path.isfile(manifest_path), 'Committed test manifest is missing'
assert not os.path.exists(marker_path), 'One-time test marker exists; rerun is prohibited'

with open(manifest_path) as f:
    manifest = json.load(f)

def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

manifest_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
source_commit = manifest['source_commit']
subprocess.run(['git', 'merge-base', '--is-ancestor', source_commit, manifest_commit], check=True)
subprocess.run(['git', 'ls-files', '--error-unmatch', manifest_path], check=True,
               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
committed_manifest = subprocess.check_output(
    ['git', 'show', f'HEAD:{manifest_path}'], stderr=subprocess.STDOUT)
assert hashlib.sha256(committed_manifest).hexdigest() == sha256_file(manifest_path), (
    'Working manifest differs from the committed manifest')
for path, expected_hash in manifest['source_files'].items():
    assert os.path.isfile(path), f'Locked source file missing: {path}'
    assert sha256_file(path) == expected_hash, f'Locked source changed: {path}'

DS_CANDIDATES = sorted(glob.glob('/kaggle/input/**/2023-05-25/semantics', recursive=True))
if DS_CANDIDATES:
    DS = str(Path(DS_CANDIDATES[0]).parent.parent)
elif Path('/kaggle/working/weedsgalore-dataset/2023-05-25/semantics').is_dir():
    DS = '/kaggle/working/weedsgalore-dataset'
else:
    raise FileNotFoundError('WeedsGalore dataset not found before test marker')
assert os.path.isdir(DS), f'Invalid WeedsGalore dataset path: {DS}'

split_dir = Path('src/datasets/weedsgalore/splits')
split_digest = hashlib.sha256()
for split_name in ('train.txt', 'val.txt', 'test.txt'):
    split_path = split_dir / split_name
    assert split_path.is_file(), f'Split file missing: {split_path}'
    split_digest.update(split_path.read_bytes())
assert split_digest.hexdigest()[:16] == manifest['split_hash'], 'WeedsGalore split hash changed'
test_ids = [line.strip() for line in (split_dir / 'test.txt').read_text().splitlines() if line.strip()]
assert test_ids, 'WeedsGalore test split is empty'
missing_test_images = []
for image_id in test_ids:
    image_dir = Path(DS) / image_id[:10] / 'images'
    for band in manifest['evaluation']['input_channels']:
        path = image_dir / f'{image_id}_{band}.png'
        if not path.is_file():
            missing_test_images.append(str(path))
assert not missing_test_images, f'Missing test image inputs (first 20): {missing_test_images[:20]}'

bins = manifest['instance_bins']
assert os.path.isfile(bins['path']), f'Instance-bin file missing: {bins["path"]}'
assert sha256_file(bins['path']) == bins['sha256'], 'Instance-bin artifact changed'
with open(bins['path']) as f:
    bins_data = json.load(f)
assert bins_data['edges'] == bins['edges'] == [45.0, 147.0], 'Instance-bin edges changed'

assert set(manifest['checkpoints']) == {'L0', 'OccCondg', 'MCSDg'}
for model, seeds in manifest['checkpoints'].items():
    assert set(seeds) == {str(seed) for seed in range(5)}, f'Expected seeds 0-4: {model}'
    assert set(manifest['run_configs'][model]) == {str(seed) for seed in range(5)}
    for seed, item in seeds.items():
        assert os.path.isfile(item['path']), f'Checkpoint missing: {model} seed {seed}'
        assert sha256_file(item['path']) == item['sha256'], f'Checkpoint changed: {model} seed {seed}'
        run_config = manifest['run_configs'][model][seed]
        assert os.path.isfile(run_config['path']), f'Run config missing: {model} seed {seed}'
        assert sha256_file(run_config['path']) == run_config['sha256'], (
            f'Run config changed: {model} seed {seed}')

for name, item in manifest['compute_profiles'].items():
    assert os.path.isfile(item['path']), f'Compute profile missing: {name}'
    assert sha256_file(item['path']) == item['sha256'], f'Compute profile changed: {name}'

preflight = manifest['weedmap_protocol']
assert os.path.isfile(preflight['preflight_path']), 'Reviewed WeedMap preflight missing'
assert sha256_file(preflight['preflight_path']) == preflight['preflight_sha256'], (
    'WeedMap preflight changed after lock')
acceptance = manifest['evaluator_acceptance']
with open(acceptance['path']) as f:
    acceptance_report = json.load(f)
assert acceptance_report.get('status') == 'PASS', 'WeedMap evaluator acceptance is no longer PASS'
assert manifest['weedmap_protocol']['evaluator_status'] == 'PASS'
for key in ('path', 'test_report_path', 'mask_evidence_path', 'radiometry_evidence_path'):
    assert os.path.isfile(acceptance[key]), f'Evaluator evidence missing: {acceptance[key]}'
for path_key, hash_key in (('path', 'sha256'), ('test_report_path', 'test_report_sha256'),
                           ('mask_evidence_path', 'mask_evidence_sha256'),
                           ('radiometry_evidence_path', 'radiometry_evidence_sha256'),
                           ('statistical_protocol_path', 'statistical_protocol_sha256')):
    assert sha256_file(acceptance[path_key]) == acceptance[hash_key], (
        f'Evaluator acceptance evidence changed: {acceptance[path_key]}')

def check_cli(args, log_path):
    result = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    with open(log_path, 'w', encoding='utf-8') as log_file:
        log_file.write(result.stdout)
    print(result.stdout, end='', flush=True)
    if result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode, args, output=result.stdout)
    return result.stdout

evaluate_help = check_cli([sys.executable, 'src/evaluate_ccmc.py', '--helpshort'],
                          f'{LOGS}/precheck_evaluate_cli.log')
assert all(flag in evaluate_help for flag in ('confirm_test_locked', 'dataset_path', 'split', 'ckpt'))
compare_help = check_cli([sys.executable, 'src/scripts/compare_runs.py', '--helpshort'],
                         f'{LOGS}/precheck_compare_cli.log')
assert all(flag in compare_help for flag in ('--a', '--b', '--cluster', '--metrics', '--out'))
assert not any(Path(OUT).glob('*_test_per_image.csv')), 'Test outputs already exist; do not start a duplicate run'

def run_live(args, log_path):
    print('\n$', ' '.join(args), flush=True)
    with open(log_path, 'w', buffering=1) as log_file:
        process = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, bufsize=1)
        for line in process.stdout:
            print(line, end='', flush=True)
            log_file.write(line)
        code = process.wait()
    if code != 0:
        raise subprocess.CalledProcessError(code, args)

# All setup checks are complete; from this point an interrupted/erroring test consumes the lock.
with open(marker_path, 'x') as f:
    json.dump({'started': True, 'source_commit': source_commit,
               'manifest_commit': manifest_commit,
               'manifest_sha256': sha256_file(manifest_path)}, f, indent=2)

arch_by_model = manifest['methods']
for model, seeds in manifest['checkpoints'].items():
    for seed, item in seeds.items():
        name = f'{model}_s{seed}_test'
        args = [sys.executable, 'src/evaluate_ccmc.py', f'--dataset_path={DS}',
                '--split=test', '--confirm_test_locked=True', f'--ckpt={item["path"]}',
                *arch_by_model[model].split(), '--bins_json=results/instance_bins.json',
                '--boundary_px=3', f'--run_name={name}', f'--out_prefix={OUT}/{name}']
        run_live(args, f'{LOGS}/eval_{name}.log')
        assert os.path.isfile(f'{OUT}/{name}_per_image.csv')
        assert os.path.isfile(f'{OUT}/{name}_summary.json')

for model_a, model_b in manifest['evaluation']['contrasts']:
    files_a = ','.join(f'{OUT}/{model_a}_s{seed}_test_per_image.csv' for seed in range(5))
    files_b = ','.join(f'{OUT}/{model_b}_s{seed}_test_per_image.csv' for seed in range(5))
    comparison = f'{OUT}/cmp_TEST_{model_a}_vs_{model_b}.json'
    args = [sys.executable, 'src/scripts/compare_runs.py', f'--a={files_a}', f'--b={files_b}',
            f'--name_a={model_a}', f'--name_b={model_b}', '--gate=none', '--cluster=image',
            '--seed_bootstrap=True', '--n_boot=10000',
            '--metrics=SmallWeedRecall,recall30_small,cov_small,mIoU,IoU_crop,IoU_weed,BIoU_weed,PrecisionWeed,CropToWeedRate',
            f'--out={comparison}']
    run_live(args, f'{LOGS}/cmp_TEST_{model_a}_vs_{model_b}.log')
    assert os.path.isfile(comparison)

print('ONE-TIME WEEDSGALORE TEST PACKAGE COMPLETE. Do not rerun or select from test.', flush=True)
```

> Do not execute K8.3 until the WeedMap archive preflight and the final method/metric/manifest review are complete. If the test lock is created before these prerequisites, stop and resolve before opening test.
