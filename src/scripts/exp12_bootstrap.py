"""Image-level paired bootstrap from per-image confusion matrices (torch-free: numpy only).

Reads perimg_<arm>_<split>_<mode>_seed<N>.csv written by exp12_eval_per_image.py.
mIoU is computed exactly like evaluate_ccmc.py / the exploratory TEST cell: from the confusion matrix
POOLED over images. Each bootstrap draw resamples the 26 images with replacement (the SAME images for
every seed and every arm, hence paired), pools the confusion matrices, computes mIoU per seed and
averages over seeds. Optional --resample_seeds also resamples the seeds (paired across arms).

Why: the seed-only t-interval (e.g. 86.07-86.43) ignores the dominant uncertainty, the 26 test images.

Example:
  python exp12_bootstrap.py --per_image_dir=.../PER_IMAGE_AUDIT --split=test --mode=logit \
      --arms=P_B2_LLR,P_B0_LLR --thresholds=86
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

CM_COLS = [f'cm{g}{p}' for g in range(3) for p in range(3)]
CLASSES = ('bg', 'crop', 'weed')


def iou_from_cm(cm):
    """cm (..., 3, 3) rows = GT -> IoU (..., 3) in %."""
    cm = np.asarray(cm, dtype=np.float64)
    tp = np.diagonal(cm, axis1=-2, axis2=-1)
    den = cm.sum(-1) + cm.sum(-2) - tp
    return np.divide(tp, den, out=np.zeros_like(tp), where=den > 0) * 100.0


def load_arm(per_image_dir, arm, split, mode, seeds=None):
    """-> (names, seeds, cm array (S, N, 3, 3)); images sorted by name and identical across seeds."""
    files = sorted(Path(per_image_dir).glob(f'perimg_{arm}_{split}_{mode}_seed*.csv'))
    if not files:
        raise SystemExit(f'No per-image files for arm={arm} split={split} mode={mode} in {per_image_dir}')
    found, names_ref = {}, None
    for f in files:
        with f.open(encoding='utf-8') as fh:
            rows = sorted(csv.DictReader(fh), key=lambda r: r['name'])
        seed = int(rows[0]['seed'])
        if seeds is not None and seed not in seeds:
            continue
        names = [r['name'] for r in rows]
        if names_ref is None:
            names_ref = names
        elif names != names_ref:
            raise SystemExit(f'{f.name}: image list differs from the other seeds of {arm}')
        found[seed] = np.array([[int(r[c]) for c in CM_COLS] for r in rows]).reshape(len(rows), 3, 3)
    if seeds is not None and set(seeds) != set(found):
        raise SystemExit(f'{arm}: requested seeds {sorted(seeds)} but found {sorted(found)}')
    order = sorted(found)
    return names_ref, order, np.stack([found[s] for s in order])


def bootstrap(cms, n_boot, rng_seed, resample_seeds=False):
    """cms: list of arrays (S, N, 3, 3), one per arm (same S and N).
    Returns point estimates and bootstrap draws: per-arm mean-over-seeds mIoU (B,) and class IoU (B, 3)."""
    S, N = cms[0].shape[:2]
    rng = np.random.default_rng(rng_seed)
    W = rng.multinomial(N, np.full(N, 1.0 / N), size=n_boot).astype(np.float64)       # (B, N)
    idx = rng.integers(0, S, size=(n_boot, S)) if resample_seeds else None
    draws, points = [], []
    for cm in cms:
        points.append(iou_from_cm(cm.sum(1)).mean(0))                                  # (3,) mean over seeds
        pooled = np.einsum('bn,snij->bsij', W, cm.astype(np.float64))                  # (B, S, 3, 3)
        iou = iou_from_cm(pooled)                                                      # (B, S, 3)
        if idx is not None:
            iou = np.take_along_axis(iou, idx[:, :, None], axis=1)
        draws.append(iou.mean(1))                                                      # (B, 3)
    return points, draws


def summarize(x, point, thresholds=()):
    lo, hi = np.percentile(x, [2.5, 97.5])
    out = {'point': float(point), 'boot_mean': float(x.mean()), 'ci95_low': float(lo), 'ci95_high': float(hi)}
    for t in thresholds:
        out[f'P(>{t:g})'] = float((x > t).mean())
    return out


def seed_t_interval(cm_seeds):
    from scipy.stats import t as student_t
    v = iou_from_cm(cm_seeds.sum(1)).mean(-1)                      # mIoU per seed
    if len(v) < 2:
        return None
    half = float(student_t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v)))
    return {'mean': float(v.mean()), 'sd': float(v.std(ddof=1)), 'ci95_low': float(v.mean() - half),
            'ci95_high': float(v.mean() + half)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--per_image_dir', required=True)
    ap.add_argument('--split', required=True, choices=['val', 'test'])
    ap.add_argument('--mode', default='logit', choices=['logit', 'prob'])
    ap.add_argument('--arms', required=True, help='target arm first; optional baseline second, e.g. P_B2_LLR,P_B0_LLR')
    ap.add_argument('--seeds', default='')
    ap.add_argument('--n_boot', type=int, default=10000)
    ap.add_argument('--rng_seed', type=int, default=20261113)
    ap.add_argument('--thresholds', default='86')
    ap.add_argument('--resample_seeds', action='store_true')
    ap.add_argument('--out_json', default='')
    a = ap.parse_args(argv)

    arms = [s for s in a.arms.split(',') if s]
    seeds = [int(s) for s in a.seeds.split(',') if s] or None
    ths = [float(t) for t in a.thresholds.split(',') if t]
    loaded = [load_arm(a.per_image_dir, arm, a.split, a.mode, seeds) for arm in arms]
    for arm, (names, sd, _) in zip(arms[1:], loaded[1:]):
        if names != loaded[0][0] or sd != loaded[0][1]:
            raise SystemExit(f'{arm}: images/seeds differ from {arms[0]}; paired bootstrap needs identical sets')
    names, sd, _ = loaded[0]
    cms = [x[2] for x in loaded]
    points, draws = bootstrap(cms, a.n_boot, a.rng_seed, a.resample_seeds)

    report = {'split': a.split, 'mode': a.mode, 'n_images': len(names), 'seeds': sd, 'n_boot': a.n_boot,
              'rng_seed': a.rng_seed, 'resample_seeds': a.resample_seeds, 'arms': {}, 'deltas': {}}
    print(f'\nsplit={a.split} mode={a.mode} images={len(names)} seeds={sd} draws={a.n_boot}'
          f' ({"images+seeds" if a.resample_seeds else "images only, seeds fixed"})')
    for i, arm in enumerate(arms):
        m_draw, m_point = draws[i].mean(1), points[i].mean()
        r = {'mIoU': summarize(m_draw, m_point, ths), 'seed_t_interval': seed_t_interval(cms[i])}
        for c, name in enumerate(CLASSES):
            r[f'IoU_{name}'] = summarize(draws[i][:, c], points[i][c])
        report['arms'][arm] = r
        s = r['mIoU']
        print(f'\n{arm}: mIoU {s["point"]:.2f}  image-bootstrap 95% CI [{s["ci95_low"]:.2f}, {s["ci95_high"]:.2f}]'
              + ''.join(f'  P(>{t:g})={s[f"P(>{t:g})"]:.3f}' for t in ths))
        if r['seed_t_interval']:
            t = r['seed_t_interval']
            print(f'   seed-only t-interval (what the TEST cell printed): {t["mean"]:.2f} ± {t["sd"]:.2f}'
                  f' [{t["ci95_low"]:.2f}, {t["ci95_high"]:.2f}]')
        for name in CLASSES[1:]:
            c = r[f'IoU_{name}']
            print(f'   IoU_{name}: {c["point"]:.2f} [{c["ci95_low"]:.2f}, {c["ci95_high"]:.2f}]')
    for j in range(1, len(arms)):
        d_draw = draws[0].mean(1) - draws[j].mean(1)
        d = summarize(d_draw, points[0].mean() - points[j].mean())
        d['P(delta<=0)'] = float((d_draw <= 0).mean())
        report['deltas'][f'{arms[0]}-{arms[j]}'] = d
        print(f'\nDELTA {arms[0]} - {arms[j]}: {d["point"]:+.2f} pp  95% CI [{d["ci95_low"]:+.2f}, {d["ci95_high"]:+.2f}]'
              f'  P(delta<=0)={d["P(delta<=0)"]:.4f}')
    out = Path(a.out_json) if a.out_json else Path(a.per_image_dir) / f'bootstrap_{a.split}_{a.mode}.json'
    out.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('\nSaved:', out)
    return report


if __name__ == '__main__':
    main()
