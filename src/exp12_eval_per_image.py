"""Exp12 per-image audit (Kaggle, needs torch + project).

(a) Re-run the *same* D4 pipeline as the exploratory TEST cell on VALIDATION and compare it with
    the project's evaluate_ccmc.py D4 summaries (probability averaging). Both averaging rules are
    computed in one pass: 'prob' (softmax then mean, = evaluate_ccmc/exp12_utils) and 'logit'
    (mean of logits, = the exploratory TEST cell).
(b) Save PER-IMAGE 3x3 confusion matrices for every (arm, seed, split, mode), so that
    exp12_bootstrap.py can resample IMAGES (the real source of uncertainty with 26 images).

No training, no checkpoint selection. TEST is refused unless --allow_test is given, and on TEST only
ONE averaging rule (--test_mode, default 'logit' = the rule already reported) is stored, so the two
rules are never compared on TEST.

Example (Kaggle):
  python exp12_eval_per_image.py --dataset_path=$DS --arms=P_B2_LLR,P_B0_LLR --splits=val
  python exp12_eval_per_image.py --dataset_path=$DS --arms=P_B2_LLR,P_B0_LLR --splits=val,test --allow_test
"""
import argparse
import csv
import gc
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

SCRIPT_VERSION = 'exp12-per-image-audit-1'
ARM_BACKBONE = {'P_B0_LLR': 'b0', 'P_B1_LLR': 'b1', 'P_B2_LLR': 'b2'}
CM_COLS = [f'cm{g}{p}' for g in range(3) for p in range(3)]
D4_VIEWS = [(k, flip) for flip in (False, True) for k in range(4)]   # the 8 symmetries
DEFAULT_ROOTS = ['/kaggle/working/WeedsGalore_Ipr28/results/exp12',
                 '/kaggle/input/datasets/thanhthao28',
                 '/kaggle/input/datasets/thothanh28']


# ---------------------------------------------------------------- torch-free helpers
def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def miou_from_cm(cm):
    """Pooled 3x3 confusion matrix (rows = GT, cols = prediction) -> (mIoU, per-class IoU) in %."""
    cm = np.asarray(cm, dtype=np.float64)
    tp = np.diag(cm)
    den = cm.sum(0) + cm.sum(1) - tp
    iou = np.divide(tp, den, out=np.zeros(3), where=den > 0) * 100.0
    return float(iou.mean()), iou


def cm_from_arrays(gt, pred):
    gt = np.asarray(gt, dtype=np.int64).ravel()
    pred = np.asarray(pred, dtype=np.int64).ravel()
    valid = (gt >= 0) & (gt < 3) & (pred >= 0) & (pred < 3)
    return np.bincount(3 * gt[valid] + pred[valid], minlength=9).reshape(3, 3)


def find_ckpts(roots, project, arm, seeds):
    """final_ema.pth of EXACTLY this arm and seed; project copy preferred over backups."""
    cands = []
    for root in roots:
        root = Path(root)
        if root.exists():
            cands.extend(root.rglob('final_ema.pth'))
    found = {}
    for seed in seeds:
        m = [p for p in cands if arm in p.parts and p.parent.name == f'seed{seed}']
        m = sorted(set(m), key=lambda p: (not str(p).startswith(str(project)), str(p)))
        if not m:
            raise FileNotFoundError(f'Missing checkpoint for {arm} seed{seed} under {list(map(str, roots))}')
        found[seed] = m[0]
    return found


def find_reference(ckpt_dir, split, tta):
    """A summary written by evaluate_ccmc.py next to the checkpoint: (path, mIoU) or None."""
    for p in sorted(Path(ckpt_dir).glob('*summary.json')):
        try:
            d = json.loads(p.read_text(encoding='utf-8'))
        except (OSError, ValueError):
            continue
        if d.get('split') == split and d.get('tta') == tta and 'metrics' in d:
            return p, float(d['metrics']['mIoU'])
    return None


def write_rows(path, arm, seed, split, mode, rows):
    with Path(path).open('w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['arm', 'seed', 'split', 'mode', 'image_idx', 'name'] + CM_COLS)
        for idx, name, cm in rows:
            w.writerow([arm, seed, split, mode, idx, name] + [int(v) for v in np.asarray(cm).ravel()])


def read_rows_cm(path):
    with Path(path).open(encoding='utf-8') as f:
        return [np.array([int(r[c]) for c in CM_COLS]).reshape(3, 3) for r in csv.DictReader(f)]


# ---------------------------------------------------------------- torch part (lazy imports)
def transform_d4(x, k, flip):
    import torch
    if flip:
        x = torch.flip(x, dims=(-1,))
    return torch.rot90(x, k=k, dims=(-2, -1))


def inverse_d4(x, k, flip):
    import torch
    x = torch.rot90(x, k=(-k) % 4, dims=(-2, -1))
    return torch.flip(x, dims=(-1,)) if flip else x


def self_check_d4():
    import torch
    t = torch.arange(3 * 5 * 7, dtype=torch.float32).reshape(1, 3, 5, 7)
    for k, flip in D4_VIEWS:
        assert torch.equal(t, inverse_d4(transform_d4(t, k, flip), k, flip)), (k, flip)
    print('D4 inverse check PASS', flush=True)


def load_model(ckpt, backbone, device):
    import torch
    from nets.segformer import SegFormer5Band          # project module (main() puts <project>/src on sys.path)
    model = SegFormer5Band(num_classes=3, in_channels=5, pretrained_backbone=True,
                           use_low_level_refine=True, use_multiscale_fusion=True,
                           llr_source='raw', backbone=backbone)
    state = torch.load(ckpt, map_location='cpu', weights_only=False)
    if isinstance(state, dict) and 'model' in state and isinstance(state['model'], dict):
        state = state['model']
    state = {k: v for k, v in state.items() if not k.endswith(('total_ops', 'total_params'))}
    model.load_state_dict(state, strict=True)           # architecture must match exactly
    return model.to(device).eval()


def predict_d4_both(model, x):
    """One pass over the 8 D4 views -> (argmax of mean logits, argmax of mean probabilities)."""
    import torch
    logit_sum = prob_sum = None
    with torch.inference_mode():
        for k, flip in D4_VIEWS:
            out = model(transform_d4(x, k, flip)).float()
            lg = inverse_d4(out, k, flip)
            pr = inverse_d4(torch.softmax(out, dim=1), k, flip)
            logit_sum = lg if logit_sum is None else logit_sum + lg
            prob_sum = pr if prob_sum is None else prob_sum + pr
    return (logit_sum / 8.0).argmax(1), (prob_sum / 8.0).argmax(1)


def evaluate_split(model, ds, device, modes, max_images):
    import torch
    out = {m: [] for m in modes}
    n = len(ds) if not max_images else min(max_images, len(ds))
    for i in range(n):
        img, _, label = ds[i]                       # same 3-class label as evaluate_ccmc.py
        x = torch.from_numpy(np.ascontiguousarray(img)).float()[None].to(device)
        gt = np.ascontiguousarray(label)
        p_logit, p_prob = predict_d4_both(model, x)
        preds = {'logit': p_logit[0].cpu().numpy(), 'prob': p_prob[0].cpu().numpy()}
        name = str(ds.img_list[i])
        for m in modes:
            out[m].append((i, name, cm_from_arrays(gt, preds[m])))
        if (i + 1) % 5 == 0 or i + 1 == n:
            print(f'    {i + 1}/{n}', flush=True)
    return out


# ---------------------------------------------------------------- main
def git_commit(project):
    try:
        return subprocess.check_output(['git', '-C', str(project), 'rev-parse', 'HEAD'], text=True).strip()
    except Exception:
        return None


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--project', default='/kaggle/working/WeedsGalore_Ipr28')
    ap.add_argument('--dataset_path', required=True)
    ap.add_argument('--out_dir', default='')
    ap.add_argument('--arms', default='P_B2_LLR')
    ap.add_argument('--seeds', default='0,1,2,3,4')
    ap.add_argument('--ckpt_roots', nargs='*', default=DEFAULT_ROOTS)
    ap.add_argument('--splits', default='val')
    ap.add_argument('--allow_test', action='store_true', help='required to evaluate the TEST split')
    ap.add_argument('--val_modes', default='prob,logit')
    ap.add_argument('--test_mode', default='logit', choices=['logit', 'prob'])
    ap.add_argument('--ref_tol', type=float, default=0.05, help='max |prob-mode - evaluate_ccmc| in pp')
    ap.add_argument('--max_images', type=int, default=0, help='smoke test only (0 = all 26 images)')
    return ap.parse_args(argv)


def main(argv=None):
    a = parse_args(argv)
    project = Path(a.project)
    arms = [s for s in a.arms.split(',') if s]
    seeds = [int(s) for s in a.seeds.split(',') if s]
    splits = [s for s in a.splits.split(',') if s]
    for arm in arms:
        if arm not in ARM_BACKBONE:
            raise SystemExit(f'arm {arm!r} not supported; choose from {sorted(ARM_BACKBONE)}')
    if 'test' in splits and not a.allow_test:       # checked BEFORE torch/data are touched
        raise SystemExit('Refusing TEST: pass --allow_test (test is already open; no tuning on it).')
    import torch
    sys.path.insert(0, str(project / 'src'))
    from datasets import WeedsGaloreDataset
    out_dir = Path(a.out_dir) if a.out_dir else project / 'results/exp12/PER_IMAGE_AUDIT'
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    self_check_d4()

    datasets = {}
    for split in splits:
        ds = WeedsGaloreDataset(str(a.dataset_path), None, 5, 3, False, split, False)
        if not a.max_images:
            assert len(ds) == 26, f'expected 26 {split} images, got {len(ds)}'
        datasets[split] = ds

    meta = {'script': SCRIPT_VERSION, 'git_commit': git_commit(project), 'arms': {}}
    ckpt_dirs = {}
    for arm in arms:
        ckpts = find_ckpts(a.ckpt_roots, project, arm, seeds)
        meta['arms'][arm] = {str(s): {'path': str(p), 'sha256': sha256_file(p)} for s, p in ckpts.items()}
        ckpt_dirs[arm] = ckpts
        print(f'{arm}: ' + ', '.join(f'seed{s}={meta["arms"][arm][str(s)]["sha256"][:12]}' for s in seeds), flush=True)
    (out_dir / 'audit_manifest.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')

    for arm in arms:
        for seed in seeds:
            todo = []
            for split in splits:
                for mode in (a.val_modes.split(',') if split == 'val' else [a.test_mode]):
                    f = out_dir / f'perimg_{arm}_{split}_{mode}_seed{seed}.csv'
                    if not f.exists():
                        todo.append((split, mode, f))
            if not todo:
                print(f'{arm} seed{seed}: all files exist, reusing', flush=True)
                continue
            print(f'\n=== {arm} seed{seed} ===', flush=True)
            model = load_model(ckpt_dirs[arm][seed], ARM_BACKBONE[arm], device)
            for split in sorted({t[0] for t in todo}):
                modes = [t[1] for t in todo if t[0] == split]
                print(f'  {split}: modes={modes}', flush=True)
                res = evaluate_split(model, datasets[split], device, modes, a.max_images)
                for _, mode, f in [t for t in todo if t[0] == split]:
                    write_rows(f, arm, seed, split, mode, res[mode])
                    m, iou = miou_from_cm(sum(c for _, _, c in res[mode]))
                    print(f'  -> {f.name}: mIoU {m:.3f} (bg {iou[0]:.2f} crop {iou[1]:.2f} weed {iou[2]:.2f})', flush=True)
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if 'val' in splits and {'prob', 'logit'} <= set(a.val_modes.split(',')):
        print('\n' + '=' * 72 + '\nPIPELINE CONSISTENCY CHECK (VALIDATION, D4)\n' + '=' * 72)
        for arm in arms:
            rows, ok = [], True
            for seed in seeds:
                v = {m: miou_from_cm(sum(read_rows_cm(out_dir / f'perimg_{arm}_val_{m}_seed{seed}.csv')))[0]
                     for m in ('prob', 'logit')}
                ref = find_reference(Path(ckpt_dirs[arm][seed]).parent, 'val', 'd4')
                rows.append((seed, ref[1] if ref else None, v['prob'], v['logit']))
            print(f'\n{arm}\nseed  evaluate_ccmc  script-prob  script-logit  prob-ref  logit-prob')
            for seed, ref, pr, lg in rows:
                if ref is None:
                    print(f'{seed:>4}  {"n/a":>13}  {pr:11.3f}  {lg:12.3f}  {"n/a":>8}  {lg - pr:+10.3f}')
                else:
                    ok &= abs(pr - ref) <= a.ref_tol
                    print(f'{seed:>4}  {ref:13.3f}  {pr:11.3f}  {lg:12.3f}  {pr - ref:+8.3f}  {lg - pr:+10.3f}')
            have = [r for r in rows if r[1] is not None]
            if have:
                print(f'  prob-mode reproduces evaluate_ccmc within {a.ref_tol} pp for all seeds: {"PASS" if ok else "FAIL"}')
            else:
                print('  no evaluate_ccmc *_summary.json found next to the checkpoints: reference comparison skipped')
            d = np.mean([r[3] - r[2] for r in rows])
            print(f'  mean(logit - prob) on VAL = {d:+.3f} pp  -> '
                  + ('averaging rule is immaterial' if abs(d) < 0.1 else 'averaging rule matters; report which one'))
    print('\nDONE. Next: python exp12_bootstrap.py --per_image_dir', out_dir)


if __name__ == '__main__':
    main()
