# Exp08 — External-domain validation and one-time test lock

> **Status (2026-11-04): NOT READY TO OPEN TEST.** K8.0 preflight and K8.1 profiling have saved outputs in `doc/exp123.ipynb`; no WeedMap inference or model predictions have been run, and no WeedsGalore test result has been opened. Four annotation frames show a consistent stored-code/render-color structure, but official semantics for code `10000` remain **PENDING**. Exp08 is frozen external-domain validation, **not** method development. No fine-tuning, calibration, threshold selection, architecture changes, or hyperparameter search on WeedMap.

## Readiness verdict

| Requirement                                                 | Current evidence                                                                                                                                                                              | Status / action before test                                                                                                                                                                                |
| ----------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Novelty of both decoder directions                          | Initial Google/Scholar web snippets and cited papers; no full-text systematic review or Scopus/Web of Science query log                                                                       | **INCOMPLETE.** Treat novelty as a narrow hypothesis; finish and archive the review before any “novel/first” claim.                                                                                        |
| WeedMap five-band data, labels, metadata, and test identity | K8.0 confirms the expected RedEdge channel/mask and annotation file structure. Four frames show `iMap` codes 0/2/10000 with black/red/green render colors; code 1 was absent in those frames. | **STRUCTURE PASS; SEMANTICS PENDING.** This does not define code 10000, prove code 1 means crop, verify radiometry/co-registration, or establish split independence.                                       |
| Second-dataset experiment type                              | The three existing WeedsGalore-trained checkpoints are evaluated without target training or adaptation                                                                                        | **DECIDED: external-domain, zero-shot confirmation.** It is not a WeedMap method-development experiment. Any later WeedMap training/tuning would require a new protocol and a separate untouched test set. |
| OccCond-g compute cost                                      | K8.1 output saved in `doc/exp123.ipynb`: four forward profiles, Tesla T4, FP32, batch 1, 600×600, 50 warmup/300 timed runs                                                                    | **DONE.** Values are transcribed in the cost table below. FLOP counter omits elementwise work.                                                                                                             |
| One-time WeedsGalore test lock                              | K8.2/K8.3 provide a preliminary lock and live logging; WeedMap label semantics and radiometry remain unresolved, and K8.3 only runs the WeedsGalore test                                      | **BLOCKED ON WEEDMAP.** Keep WeedsGalore test closed until full preflight, a dedicated evaluator with tests, and an immutable manifest are ready.                                                          |

### Model identity and scope

The source implements one segmentation model per run, not an ensemble. `MCSDSegFormer` inherits the same pretrained MiT-B0 encoder and SegFormer multi-scale decoder as the L0 path; its RGB patch projection is expanded to five channels by retaining the RGB filters and initializing NIR/RE from their RGB-filter mean. OccCond-g/MCSD-g add occupancy and placement heads (and an optional small raw-band guide); they do not delete pretrained backbone layers. The method change is in the dense prediction head/coupling, not in combining several pretrained models. This is verified in `src/nets/segformer.py` and `src/nets/mcsd.py`.

Method development and selection belong to the existing WeedsGalore train/validation experiments. Exp08's second dataset is only a frozen transfer check of L0, OccCond-g, and MCSD-g. Based on Exp07 validation, OccCond-g is the primary candidate for the stated small-weed recall/coverage objective; MCSD-g is the constrained-coupling comparator, and L0 is the reference. Report all three regardless of outcome. Do not use WeedMap to change this choice.

## Decision and scope

Freeze **OccCond-g as the primary method candidate** because it had the stronger validation evidence for the stated small-weed objective: versus L0, Small-Weed Recall +10.55 [6.39; 14.84] pp and coverage +7.55 [2.97; 12.47] pp, with mIoU +0.88 [0.42; 1.54] pp. Keep **MCSD-g as the constrained-decoding comparator/ablation**, not as a second co-primary claim: Exp07 found similar mIoU versus OccCond-g but lower recall/coverage and higher marginal fidelity, Weed precision, and Boundary IoU. Freeze L0, OccCond-g, and MCSD-g as the complete comparison set with five existing validation-selected checkpoints per method. Do not select or alter any method using WeedMap or test data. This choice is based on the stated recall objective and Exp07 validation only; the full novelty review remains a prerequisite to opening test.

Use the official **WeedMap RedEdge-M dataset 003** only as a candidate second-dataset external evaluation. WeedMap covers sugar-beet fields in Switzerland/Germany, unlike WeedsGalore maize fields. Do not substitute Sequoia set 005: it does not match the five-channel input contract. The ETHZ ASL page confirms the broad availability of RedEdge/Sequoia imagery, tiles/orthomosaics, reflectance TIFFs, and background/crop/weed annotations, but the exact downloaded 003 files and compatibility are not verified here. Before inference, record the exact original-paper passage/table/page identifying 003 as test, release DOI/version, metadata for all five aligned bands and radiometry, label codes/instances, license, and tile/site grouping. If any gate fails, stop and do not silently substitute another set.

**Latest preflight interpretation:** the observed per-frame mapping is structurally consistent (`iMap 0` renders black, `2` red, and `10000` green in the checked frames), but this is not semantic documentation. Do not treat `10000` as Crop or Ignore, and do not treat absent code `1` as proof of a class definition. DataCite lists this release as in-copyright with non-commercial use permitted; confirm the exact reuse terms before redistribution of data or derived artifacts.

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

1. Select the official RedEdge-M testing partition **003** only. Save the downloaded archive name, DOI/version, archive SHA-256, license, dataset index, flight/site metadata, and the original source paper’s statement that 003 is a test partition.
2. Verify five co-registered bands and channel mapping by names and metadata. Model order is exactly `R, G, B, NIR, RE`; map camera-native files by labels, never by directory ordering. Confirm units, bit depth, reflectance calibration, clipping/scale, image shape, nodata handling, and tile-to-label alignment.
3. Verify semantic mapping to `background=0, crop=1, weed=2`, ignored labels, and whether an official weed instance map exists. Save unique class codes/names and counts as a data-integrity artifact only; do not use test frequencies to select thresholds or methods.
4. Check for same-orthomosaic/overlapping tiles and official split leakage. If 003 is a held-out test subset from a documented flight/site, use its existing split. Keep tile groups intact in any uncertainty analysis.
5. Do not compute WeedMap normalization quantiles, class weights, or thresholds from its test images/labels. Use only a fixed documented transform compatible with the source model. If the source and WeedMap radiometry cannot be aligned without target-fitted statistics, stop: the external test is not ready.

### Methods, checkpoints, and comparisons

- Methods: L0, OccCond-g, MCSD-g.
- Checkpoints: five fixed seeds (0–4) per method, selected by WeedsGalore validation only. Freeze SHA-256 for all 15 `best_val.pth` files before reading WeedMap predictions.
- No WeedMap training/fine-tuning/BN adaptation, pseudo-labeling, calibration, or checkpoint selection.
- All three methods use identical WeedMap tiles, channel mapping, radiometric transform, and output dimensions.
- Primary endpoint/contrast: pixel-level `WeedRecall`, OccCond-g − L0, aggregated over the complete official test partition. This matches the validation-based method objective without assuming WeedMap instance IDs or a transferable “small weed” pixel bin. Secondary outcomes: `IoU_weed`, `mIoU`, per-class IoU, Weed pixel precision, confusion matrix, and official-instance recall@IoU 0.5 only if instance IDs are present and validated. Secondary contrasts: MCSD-g − L0 and MCSD-g − OccCond-g. Do not reuse WeedsGalore `[45,147]` pixel-size bins because WeedMap GSD/crop differ. Do not claim comparable `SmallWeedRecall` unless a physical-area threshold is preregistered from source-domain information and WeedMap instance labels are verified.
- Report point estimates and 95% paired hierarchical bootstrap CI (seed, then spatial group/tile). Cluster at orthomosaic/flight when multiple independent units exist. If set 003 is only one flight/orthomosaic, state that intervals quantify within-collection tile variability and do not estimate between-site uncertainty. The bootstrap seed, replicate count, resampling order, and cluster identifiers must be frozen in the manifest.
- Dataset-level external evaluation is descriptive/confirmatory of frozen transfer, not a new method-development gate. Report all three methods, even if a method loses.

### WeedsGalore test lock and one-time release

Before opening WeedsGalore test, freeze a JSON manifest with repository commit, split hash and explicit split IDs/hash, source `run_config.json` hashes, checkpoint SHA-256 values for L0/OccCond-g/MCSD-g seeds 0–4, WeedMap archive/data identity and preflight decision, fixed input transform, class map, metrics, bootstrap seed/replicates/cluster unit, and the primary/secondary contrasts above. Include realized profile artifact hashes when available. Commit the manifest and confirm no code/config changes after it is created. If WeedMap compatibility or its evaluator is unresolved, record WeedMap as `NOT_AUTHORIZED`; do not present K8.3 as a two-dataset run.

The Kaggle test cell below creates a `test_started` marker **before** its first WeedsGalore test evaluation. If a run fails or the kernel is interrupted after the marker is written, do not rerun it as though nothing happened; preserve logs and record the deviation. It evaluates only the locked 15 checkpoints and then runs the locked WeedsGalore comparisons. It does **not** evaluate WeedMap. No test result may alter the manifest or method choice.

## Kaggle cells

Run in order only after committing/pushing the source and review decisions. The cells below stream subprocess stdout to the Kaggle notebook output and simultaneously save it to `results/exp08/logs/`. The `run_live` helper's `print(..., flush=True)` is intentional so progress appears in Kaggle output while each process is running. K8.0 is inventory only; the WeedMap evaluation cell is not implemented because the repository has no WeedMap loader/evaluator. Do not run any test cell until that implementation and its tests are reviewed and the manifest is locked.

### Cell K8.0 — WeedMap RedEdge-M 003 preflight (no inference)

```python
import glob, hashlib, json, os, pathlib, sys
from collections import Counter
from PIL import Image

PROJECT = '/kaggle/working/WeedsGalore_Ipr28'
os.chdir(PROJECT)
EXP = 'exp08'
OUT = f'results/{EXP}'
os.makedirs(f'{OUT}/logs', exist_ok=True)

# Set this to the mounted archive's extracted root; do not guess a Sequoia directory.
WM_ROOT = '/kaggle/input/weedmap-2018'
assert os.path.isdir(WM_ROOT), f'WeedMap root not found: {WM_ROOT}'

def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

all_files = [p for p in pathlib.Path(WM_ROOT).rglob('*') if p.is_file()]
print('Mounted root:', WM_ROOT)
print('File count:', len(all_files))
print('Extension counts:', Counter(p.suffix.lower() for p in all_files))
print('Directories containing 003 or RedEdge:')
for path in sorted({p.parent for p in all_files if '003' in str(p) or 'rededge' in str(p).lower()}):
    print(' ', path)
print('Candidate image/label files:')
for path in all_files:
    low = str(path).lower()
    if ('003' in low and any(token in low for token in ('groundtruth', 'imap', 'rededge'))):
        print(path)

metadata_files = [p for p in all_files if p.suffix.lower() in ('.txt', '.md', '.xml', '.json', '.csv', '.yaml', '.yml')]
for path in metadata_files[:80]:
    print('METADATA', path, 'sha256=', sha256_file(path))

label_candidates = [p for p in all_files if '003' in str(p) and 'groundtruth' in p.name.lower()]
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

rededge_candidates = [p for p in all_files if '003' in str(p) and 'rededge' in str(p).lower()]
assert rededge_candidates, 'Dataset 003 RedEdge files not found; do not substitute Sequoia'
print('RedEdge-M candidate count:', len(rededge_candidates))
archive_candidates = [p for p in all_files if p.suffix.lower() in ('.zip', '.tar', '.gz')
                      and ('003' in p.name or 'weedmap' in p.name.lower())]
archive_hashes = {str(path): sha256_file(path) for path in archive_candidates}
preflight = {
    'source_page': 'https://projects.asl.ethz.ch/datasets/weedmap-2018/',
    'paper_doi': '10.3390/rs10091423',
    'dataset_root': WM_ROOT,
    'file_count': len(all_files),
    'dataset_003_rededge_candidate_count': len(rededge_candidates),
    'rededge_sample_paths': [str(path) for path in rededge_candidates[:100]],
    'label_samples': label_samples,
    'metadata_files': [str(path) for path in metadata_files],
    'archive_hashes': archive_hashes,
    'manual_review_decision': 'PENDING',
    'manual_review_fields': ['official partition 003 verified from source docs',
                             'five co-registered RedEdge-M bands and order verified',
                             'radiometric scale fixed without test-fitted statistics',
                             'label mapping/license/instance availability verified',
                             'tile/orthomosaic grouping and leakage reviewed'],
    'preflight_note': 'No inference performed. Change decision to GO only after reviewing all fields.',
}
with open(f'{OUT}/weedmap_003_preflight.json', 'w') as f:
    json.dump(preflight, f, indent=2)
print('Saved:', f'{OUT}/weedmap_003_preflight.json')
print('Archive inspection only. Do not evaluate model predictions or tune from these labels.')
```

K8.0 chỉ inventory/kiểm tra cấu trúc; không đủ để xác nhận radiometric compatibility. Trong cell cũ, điều kiện `'003' in str(path)` có thể bắt nhầm index khác nếu tên frame chứa chuỗi `003` (ví dụ `frame0030`); không dùng riêng danh sách path đó làm bằng chứng index 003. Lưu console output, xác minh index từ thành phần thư mục đúng, và dùng K8.0b để rà tài liệu/mã nguồn dạng text. Trước khi đặt `manual_review_decision` thành `GO`, bổ sung bằng chứng cho `dataset_identity`, `verified_input_transform`, `verified_class_map`, `verified_label_semantics`, `verified_grouping`, và `source_test_split_citation`; K8.2 sẽ từ chối nếu thiếu trường nào. Không chạy thêm pixel statistics theo nhãn test. Nếu radiometry/co-registration không thể xác minh từ tài liệu nguồn, dừng trước inference.

### Cell K8.0b — Read WeedMap annotation documentation/source only (no label pixels)

Run this after K8.0. It reads only text files bundled with the mounted release (README/docs, XML/JSON metadata, and MATLAB/Python source); it does not open PNG/TIFF masks or count any labels. It prints each matching excerpt to Kaggle output and saves the same audit transcript. A missing definition is a **PENDING** result, not evidence that 10000 means Ignore or Crop.

```python
import hashlib
import os
import pathlib
import re

WM_ROOT = pathlib.Path('/kaggle/input/weedmap-2018')
AUDIT_LOG = pathlib.Path('results/exp08/logs/weedmap_annotation_docs_audit.log')
AUDIT_LOG.parent.mkdir(parents=True, exist_ok=True)
assert WM_ROOT.is_dir(), f'WeedMap root not found: {WM_ROOT}'

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
exact_hits = 0
files_with_hits = 0
with AUDIT_LOG.open('w', encoding='utf-8', buffering=1) as log:
    emit('WeedMap annotation documentation/source audit')
    emit(f'Root: {WM_ROOT}')
    emit('Scope: text files only; no PNG/TIFF/mask pixels opened.')
    emit(f'Text files scanned: {len(text_files)} (max 4 MiB each)')
    for path in text_files:
        try:
            lines = path.read_text(encoding='utf-8-sig', errors='replace').splitlines()
        except OSError as error:
            emit(f'SKIP {path.relative_to(WM_ROOT)}: {error}')
            continue
        matches = [(index, line.strip()) for index, line in enumerate(lines, start=1)
                   if search_pattern.search(line)]
        exact_matches = [(index, line) for index, line in matches
                         if exact_code_pattern.search(line)]
        if not matches:
            continue
        files_with_hits += 1
        exact_hits += len(exact_matches)
        relative = path.relative_to(WM_ROOT)
        emit(f'\nFILE {relative} sha256={sha256_file(path)}')
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
    emit('\nAUDIT RESULT: PENDING')
    emit('Only mark semantics verified if an official release document or author source explicitly defines code 10000 and the class/ignore policy.')
    emit(f'Files with relevant text: {files_with_hits}; exact code/name hits: {exact_hits}')
    emit(f'Saved audit: {AUDIT_LOG}')
print('Saved audit:', AUDIT_LOG, flush=True)
```

Review the printed excerpts and cite the exact release document/source path and line/page in the preflight JSON. If the archive only repeats the RGB rendering or does not define `10000`, keep `label_semantics` **PENDING**, ask the dataset maintainers or obtain an authoritative annotation specification, and do not build a three-class WeedMap evaluator yet.

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
        run_configs[model][str(seed)] = {'path': path, 'sha256': sha256_file(path)}

preflight_path = f'{OUT}/weedmap_003_preflight.json'
assert os.path.isfile(preflight_path), 'Run and review K8.0 before locking the manifest'
with open(preflight_path) as f:
    weedmap_preflight = json.load(f)
assert weedmap_preflight['manual_review_decision'] == 'GO', (
    'WeedMap is not authorized: finish archive, radiometry, labels, split, and leakage review')
required_preflight_fields = ('dataset_identity', 'verified_input_transform', 'verified_class_map',
                            'verified_label_semantics', 'verified_grouping',
                            'source_test_split_citation')
missing_preflight = [key for key in required_preflight_fields
                     if not weedmap_preflight.get(key)]
assert not missing_preflight, f'Incomplete reviewed WeedMap preflight fields: {missing_preflight}'

manifest = {
    'protocol': 'one-time WeedsGalore test evaluation; frozen external-method comparison set',
    'commit': actual_commit,
    'split_hash': '675667edf3f35f16',
    'checkpoint_selection': 'best_val only; selected on validation before test lock',
    'methods': specs,
    'checkpoints': checkpoints,
    'run_configs': run_configs,
    'compute_profiles': profiles,
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
        'primary_endpoint': 'WeedRecall',
        'primary_contrast': ['OccCondg', 'L0'],
                'secondary_metrics': ['mIoU', 'IoU_bg', 'IoU_crop', 'IoU_weed', 'PrecisionWeed',
                                    'secondary_metrics': ['mIoU', 'IoU_bg', 'IoU_crop', 'IoU_weed', 'PrecisionWeed',
                                                  'instance_recall50_if_official_ids_verified'],
        'secondary_contrasts': [['MCSDg', 'L0'], ['MCSDg', 'OccCondg']],
        'bootstrap': {'replicates': 10000, 'seed': 0, 'paired_seed_resampling': True,
                  'spatial_cluster': 'orthomosaic_or_flight_then_tile'},
        'evaluator_status': 'NOT_IMPLEMENTED; do not infer from this manifest alone',
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

Preserve the manifest checksum (K8.3 records it in the start marker). Run K8.2 and K8.3 on the same pushed source commit; do not create a follow-up commit between them, since that would invalidate the locked commit check. Replace `EXPECTED_COMMIT` with the finalized pushed source commit before K8.2. If the manifest itself must be committed, change the integrity check to verify the locked source-file hashes rather than comparing HEAD to the pre-manifest commit.

### Cell K8.3 — One-time WeedsGalore test (only after manifest is committed)

This is a single release operation across 15 frozen checkpoints, not 15 model-selection trials. It releases only the WeedsGalore test; the WeedMap evaluator is not implemented. A marker is written **before** the first test process. If interrupted or failed after the marker, preserve outputs/logs and record the deviation; do not silently rerun.

```python
import hashlib, json, os, subprocess, sys

OUT = 'results/exp08'
LOGS = f'{OUT}/logs'
manifest_path = f'{OUT}/test_lock_manifest.json'
marker_path = f'{OUT}/test_started.json'
os.makedirs(LOGS, exist_ok=True)
assert os.path.isfile(manifest_path)
assert not os.path.exists(marker_path), 'One-time test marker exists; rerun is prohibited'

with open(manifest_path) as f:
    manifest = json.load(f)
actual_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
assert actual_commit == manifest['commit'], 'Code commit differs from locked manifest'

def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

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

for model, seeds in manifest['checkpoints'].items():
    for seed, item in seeds.items():
        assert sha256_file(item['path']) == item['sha256'], f'Checkpoint changed: {model} seed {seed}'
        run_config = manifest['run_configs'][model][seed]
        assert sha256_file(run_config['path']) == run_config['sha256'], (
            f'Run config changed: {model} seed {seed}')

for name, item in manifest['compute_profiles'].items():
    assert sha256_file(item['path']) == item['sha256'], f'Compute profile changed: {name}'

preflight = manifest['weedmap_protocol']
assert sha256_file(preflight['preflight_path']) == preflight['preflight_sha256'], (
    'WeedMap preflight changed after lock')

with open(marker_path, 'x') as f:
    json.dump({'started': True, 'commit': actual_commit,
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
