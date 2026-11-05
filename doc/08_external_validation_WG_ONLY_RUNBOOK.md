# Exp08 Kaggle Runbook: WeedsGalore-only K8.3

## Scope

This runbook replaces the confusing mixed WeedMap/WeedsGalore path for the current deadline.

- WeedMap status: `NOT_AUTHORIZED`
- WeedMap is excluded from the current test protocol.
- Existing WeedMap audit files are preserved and must not be overwritten.
- Current test scope: one-time WeedsGalore test only.
- No WeedMap inference, no WeedMap label evaluation, and no target-domain tuning.

Locked method set:

- `L0`
- `OccCondg`
- `MCSDg`
- Seeds `0, 1, 2, 3, 4`

Do not run the old combined WeedMap-dependent K8.2/K8.3 cells.

## Session startup

1. Open a new Kaggle session with GPU if required.
2. Clone or load the repository at `/kaggle/working/WeedsGalore_Ipr28`.
3. Attach the checkpoint datasets containing the seven roots used by the provenance audit:
   - `ccmc-ckpts`
   - `exp02-ckpts`
   - `exp03-ckpts`
   - `exp04-ckpts`
   - `exp05-ckpts`
   - `exp06-ckpts`
   - `exp07-occcond-ckpts`
4. Confirm the WeedsGalore dataset is available. The expected root is normally `/kaggle/working/weedsgalore-dataset` or a Kaggle Input dataset containing `2023-05-25/images`, `semantics`, and `instances`.
5. Change directory to the project:

```python
from pathlib import Path
import os

PROJECT = Path('/kaggle/working/WeedsGalore_Ipr28')
assert PROJECT.is_dir(), PROJECT
os.chdir(PROJECT)
print(PROJECT)
```

Do not open test images or labels during setup.

## Already completed

These artifacts are already available from the previous work and should not be recreated unless missing:

- K8.0 through K8.0r diagnostic evidence.
- K8.1 compute profiles.
- `checkpoint_provenance_verified.json` with `15/15` checkpoint SHA-256 PASS.
- Strict loading plus synthetic forward PASS.
- THOP buffer accounting PASS.
- WeedMap mask/radiometry evidence, with both gates still pending.

The reported checkpoint audit found `58` `.pth` files across the seven attached roots, but exactly the required frozen set passed:

- `15/15` checkpoint hashes PASS.
- `15/15` config hashes PASS after restoration.
- No missing source artifacts.
- No SHA mismatch.
- No test inference.

## Step 1: Restore frozen run paths

Run the checkpoint/config restoration cell named:

`K8.2b — Restore frozen run paths after Kaggle restart`

Before running it, update the cell as follows:

- Keep the seven Kaggle Input roots and provenance file.
- Keep checkpoint/config SHA-256 verification.
- Keep the `test_started.json` and `test_lock_manifest.json` stop checks.
- Remove any hard-coded `provenance["commit"]` assertion. Commit identity is not part of this restore audit.
- Do not retrain or select another checkpoint.

Expected result:

```text
Original checkpoint/config sources: 15/15 VERIFIED
RESTORE PASS: 15/15
Checkpoint contents unchanged.
No test inference performed.
K8.3: NOT AUTHORIZED.
```

The restore cell may create symlinks for `best_val.pth` and copy `run_config.json`. It must not modify checkpoint contents.

## Step 2: Freeze the WeedMap amendment

Run the cell named:

`K8.1a — Freeze WeedMap NOT_AUTHORIZED amendment`

Expected artifact:

```text
results/exp08/weedmap_scope_amendment_not_authorized.json
```

Required fields:

- `weedmap_status = NOT_AUTHORIZED`
- Reasons for unresolved mask validity, radiometric compatibility, and evaluator acceptance.
- Links to preserved WeedMap audit artifacts.
- `test_data_opened = false`
- `inference_run = false`

If the amendment file already exists, stop. Do not overwrite it.

## Step 3: Run WG-only preflight and lock manifest

Run the cell named:

`K8.2-WG-only — Preflight and lock the WeedsGalore-only manifest`

This cell may inspect only metadata and hashes. It must not open test images or labels.

It checks:

- `15` frozen checkpoints and `15` `run_config.json` files.
- SHA-256 for all locked artifacts.
- No duplicate/conflicting artifact version.
- WeedsGalore dataset root and required directory structure.
- Split metadata and split hash `675667edf3f35f16`.
- Evaluator CLI help and comparison CLI help.
- Required source-file hashes.
- No existing `test_started.json`.

Expected artifact:

```text
results/exp08/test_lock_manifest.json
```

The manifest must contain:

- `repository_head`
- `dataset_root`
- `split_hash`
- checkpoint and config paths/hashes
- source-file hashes
- methods, metrics, contrasts, and bootstrap settings
- `protocol_amendment.weedmap_status = NOT_AUTHORIZED`
- `weedmap_protocol.status = NOT_AUTHORIZED`

Expected status:

```text
WG-only preflight PASS
Manifest locked
No test images/labels opened
No test_started.json created
```

If any assertion fails, stop and repair the environment. Do not create a partial manifest.

## Step 4: Commit the manifest before K8.3

The manifest must be committed to the repository before the one-time test.

Use a normal non-interactive Git commit from the project root. Do not change source code, checkpoint files, configs, metrics, or manifest contents after this commit.

Verify:

```python
from pathlib import Path
import subprocess

manifest = Path('results/exp08/test_lock_manifest.json')
assert manifest.is_file()
subprocess.run(
    ['git', 'ls-files', '--error-unmatch', str(manifest)],
    check=True,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True,
)
print('Committed manifest is tracked.')
```

Do not run K8.3 if the manifest is untracked or modified after the commit.

## Step 5: One-time WeedsGalore test

Run only:

`K8.3-WG-only — One-time WeedsGalore test`

This cell performs all checks before creating the marker:

- Manifest is tracked and unchanged.
- Locked source hashes match.
- All 15 checkpoint/config hashes match.
- Locked dataset root exists.
- Split hash matches.
- Instance-bin artifact matches.
- No prior test outputs exist.

Only after all checks pass does it create:

```text
results/exp08/test_started.json
```

After the marker is created, the cell evaluates exactly 15 frozen checkpoints and creates the locked pairwise comparisons:

- `OccCondg` vs `L0`
- `MCSDg` vs `L0`
- `MCSDg` vs `OccCondg`

It then writes evaluation logs and result files under `results/exp08/`.

This is a one-time release. If the process is interrupted or fails after the marker is created:

- Do not rerun it silently.
- Preserve the marker and all logs.
- Record the failure and deviation.

## Do not run

Do not run these paths for the current protocol:

- WeedMap model inference.
- WeedMap end-to-end evaluation.
- Any transform fitting or normalization from WeedMap test tiles.
- Any checkpoint selection after seeing test results.
- The archived combined WeedMap/WeedsGalore K8.3 cell.
- Any old K8.2 cell requiring `weedmap_003_preflight.json` with `manual_review_decision = GO`.
- Any cell requiring `weedmap_evaluator_acceptance.json` with `status = PASS`.

## Final evidence to download

After K8.3-WG-only completes, download or archive:

```text
results/exp08/test_lock_manifest.json
results/exp08/weedmap_scope_amendment_not_authorized.json
results/exp08/test_started.json
results/exp08/logs/
results/exp08/*_test_per_image.csv
results/exp08/*_test_summary.json
results/exp08/cmp_TEST_*.json
```

Report wording for the paper:

> WeedsGalore held-out test results were obtained in a one-time, preregistered WG-only evaluation of three validation-selected methods across five seeds. WeedMap external validation was not authorized because mask polarity, radiometric compatibility, and end-to-end evaluator acceptance remained unresolved; all WeedMap audit artifacts were preserved and no WeedMap inference was performed.
