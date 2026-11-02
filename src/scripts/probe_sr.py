"""Diag-D2: linear decodability of Weed windows before vs after K/V spatial reduction.

For MiT stages 1-3 (first block), each full SR window (kernel = stride = R) gives three
C-dim vectors: (a) average pooling of the SR input, (b) the SR-conv output at that window
(before LayerNorm), (c) max pooling of the SR input (control). Windows are labelled by the
LARGEST Weed instance bin they contain (none / small / medium / large; train tertiles).

Same linear probe (StandardScaler + balanced logistic regression) is fit on TRAIN windows
and evaluated on VAL windows: AUROC, AP, recall at precision >= 0.8.
H1 contrast per stage: [AUROC(a) - AUROC(b)]_small - [AUROC(a) - AUROC(b)]_large, with
an image-level bootstrap CI on validation. This is RELATIVE linear decodability, not an
absolute information measure (plan 6.1).

python src/scripts/probe_sr.py --dataset_path=... --ckpt=L0/best_val.pth --ccmc=True \
    --prior_mode=none --bins_json=results/instance_bins.json --out_prefix=results/d2_probe_L0
"""

import csv
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from absl import app, flags

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ccmc_common import build_model, define_model_flags, load_checkpoint  # noqa: E402
from ccmc_metrics import area_bin, instance_areas, load_bins, load_instance_map, weed_instances  # noqa: E402
from datasets import WeedsGaloreDataset  # noqa: E402

FLAGS = flags.FLAGS
flags.DEFINE_string('dataset_path', 'weedsgalore-dataset', 'dataset directory')
flags.DEFINE_string('ckpt', None, 'checkpoint to probe (L0 first; later L3 with the same protocol)')
flags.DEFINE_string('bins_json', 'results/instance_bins.json', 'instance bins')
flags.DEFINE_list('probe_stages', ['1', '2', '3'], 'stages to probe (1-based)')
flags.DEFINE_string('out_prefix', 'results/d2_probe', 'output prefix')
flags.DEFINE_integer('n_boot', 1000, 'bootstrap replicates over validation images')
flags.DEFINE_float('probe_C', 1.0, 'inverse L2 strength of the logistic probe')
flags.DEFINE_float('fixed_precision', 0.8, 'recall at this precision')
define_model_flags()

FEATURES = ('pre_avg', 'post_sr', 'pre_max')
BINS = ('small', 'medium', 'large')


def window_categories(inst, edges, size, ratio, device):
    """(h, w) int map: 0 none, 1 small, 2 medium, 3 large (largest bin in the window)."""
    ids, areas = instance_areas(inst)
    bin_of = np.zeros(inst.max() + 1, dtype=np.int64)
    if len(ids):
        bin_of[ids] = area_bin(areas, edges) + 1
    pix_bin = torch.from_numpy(bin_of[inst]).float()[None, None].to(device)
    height, width = size
    hc, wc = (height // ratio) * ratio, (width // ratio) * ratio
    cat = None
    for b in (1, 2, 3):
        ind = (pix_bin == b).float()
        ind = F.adaptive_max_pool2d(ind, (height, width))[..., :hc, :wc]
        ind = F.max_pool2d(ind, ratio, ratio)[0, 0]
        cat = torch.where(ind > 0, torch.full_like(ind, b), cat if cat is not None else torch.zeros_like(ind))
    return cat.long().cpu().numpy()


def collect(net, split, edges, stages, device):
    dataset = WeedsGaloreDataset(dataset_path=FLAGS.dataset_path, dataset_size=None,
                                 in_bands=FLAGS.in_channels, num_classes=3, is_training=False,
                                 split=split, augmentation=False)
    captured = {}
    handles = []
    hf = net.encoder.encoder
    ratios = list(net.encoder.config.sr_ratios)
    for s in stages:
        sr = hf.block[s - 1][0].attention.self.sr

        def hook(module, args, output, s=s):
            captured[s] = (args[0].detach(), output.detach())
        handles.append(sr.register_forward_hook(hook))
    store = {s: {f: [] for f in FEATURES} for s in stages}
    cats = {s: [] for s in stages}
    img_ids = {s: [] for s in stages}
    with torch.no_grad():
        for idx in range(len(dataset)):
            name = str(dataset.img_list[idx])
            img, _, label = dataset[idx]
            x = torch.from_numpy(np.ascontiguousarray(img)).float().unsqueeze(0).to(device)
            if FLAGS.ccmc and FLAGS.prior_mode in ('gt_veg', 'gt_weed'):
                lab = torch.from_numpy(np.ascontiguousarray(label)).long().unsqueeze(0).to(device)
                net(x, prior_mask=lab)
            else:
                net(x)
            inst, _ = weed_instances(np.asarray(label), load_instance_map(FLAGS.dataset_path, name))
            for s in stages:
                x_in, x_out = captured[s]
                r = ratios[s - 1]
                h, w = x_out.shape[-2:]
                crop = x_in[..., :h * r, :w * r]
                feats = {'pre_avg': F.avg_pool2d(crop, r, r), 'post_sr': x_out,
                         'pre_max': F.max_pool2d(crop, r, r)}
                for f in FEATURES:
                    store[s][f].append(feats[f][0].flatten(1).T.float().cpu().numpy())
                cat = window_categories(inst, edges, x_in.shape[-2:], r, device)
                assert cat.shape == (h, w), (cat.shape, (h, w))
                cats[s].append(cat.ravel())
                img_ids[s].append(np.full(cat.size, idx))
    for hnd in handles:
        hnd.remove()
    out = {}
    for s in stages:
        out[s] = {f: np.concatenate(store[s][f]) for f in FEATURES}
        out[s]['cat'] = np.concatenate(cats[s])
        out[s]['img'] = np.concatenate(img_ids[s])
    return out


def recall_at_precision(y, score, precision):
    from sklearn.metrics import precision_recall_curve
    p, r, _ = precision_recall_curve(y, score)
    ok = p >= precision
    return float(r[ok].max()) if ok.any() else 0.0


def main(_):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score, roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = load_checkpoint(build_model(FLAGS), FLAGS.ckpt, device).to(device).eval()
    edges, _ = load_bins(FLAGS.bins_json)
    stages = [int(s) for s in FLAGS.probe_stages]
    train = collect(net, 'train', edges, stages, device)
    val = collect(net, 'val', edges, stages, device)

    tasks = {f'{b}_vs_none': (i + 1, 'none') for i, b in enumerate(BINS)}
    tasks['small_vs_rest'] = (1, 'rest')  # negatives include windows with larger Weed only
    rows, scores = [], {}
    for s in stages:
        for task, (pos_cat, neg_kind) in tasks.items():
            def select(d):
                pos = d['cat'] == pos_cat
                neg = d['cat'] == 0 if neg_kind == 'none' else d['cat'] != pos_cat
                keep = pos | neg
                return keep, pos[keep].astype(int)
            k_tr, y_tr = select(train[s])
            k_va, y_va = select(val[s])
            if y_tr.sum() < 5 or y_va.sum() < 2 or (1 - y_va).sum() < 2:
                continue
            for feat in FEATURES:
                clf = make_pipeline(StandardScaler(), LogisticRegression(
                    C=FLAGS.probe_C, max_iter=3000, class_weight='balanced'))
                clf.fit(train[s][feat][k_tr], y_tr)
                sc = clf.decision_function(val[s][feat][k_va])
                scores[(s, task, feat)] = (y_va, sc, val[s]['img'][k_va])
                rows.append({'stage': s, 'task': task, 'feature': feat,
                             'n_pos_train': int(y_tr.sum()), 'n_neg_train': int((1 - y_tr).sum()),
                             'n_pos_val': int(y_va.sum()), 'n_neg_val': int((1 - y_va).sum()),
                             'auroc': float(roc_auc_score(y_va, sc)),
                             'ap': float(average_precision_score(y_va, sc)),
                             f'recall@p{FLAGS.fixed_precision}': recall_at_precision(
                                 y_va, sc, FLAGS.fixed_precision)})

    # H1 contrast with image-level bootstrap on validation
    rng = np.random.default_rng(0)
    n_val_images = int(max(v['img'].max() for v in val.values())) + 1
    contrasts = {}
    for s in stages:
        keys = [(s, f'{b}_vs_none', f) for b in ('small', 'large') for f in ('pre_avg', 'post_sr')]
        if not all(k in scores for k in keys):
            contrasts[s] = 'NOT ENOUGH WINDOWS'
            continue

        def contrast(sel_images=None):
            vals = {}
            for k in keys:
                y, sc, im = scores[k]
                if sel_images is not None:
                    counts = np.bincount(sel_images, minlength=n_val_images)
                    w = counts[im]
                    keep = w > 0
                    y, sc, w = y[keep], sc[keep], w[keep]
                    if y.min() == y.max():
                        return np.nan
                    vals[k] = roc_auc_score(y, sc, sample_weight=w)
                else:
                    vals[k] = roc_auc_score(y, sc)
            drop_small = vals[(s, 'small_vs_none', 'pre_avg')] - vals[(s, 'small_vs_none', 'post_sr')]
            drop_large = vals[(s, 'large_vs_none', 'pre_avg')] - vals[(s, 'large_vs_none', 'post_sr')]
            return drop_small - drop_large, drop_small, drop_large

        point = contrast()
        boot = []
        for _ in range(FLAGS.n_boot):
            res = contrast(rng.integers(0, n_val_images, n_val_images))
            if not isinstance(res, float):
                boot.append(res[0])
        boot = np.asarray(boot)
        contrasts[s] = {'drop_small': point[1], 'drop_large': point[2],
                        'contrast_small_minus_large': point[0],
                        'ci95': np.percentile(boot, [2.5, 97.5]).tolist() if len(boot) else None,
                        'n_boot_valid': int(len(boot))}

    os.makedirs(os.path.dirname(FLAGS.out_prefix) or '.', exist_ok=True)
    with open(FLAGS.out_prefix + '.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with open(FLAGS.out_prefix + '_h1.json', 'w') as f:
        json.dump({'ckpt': FLAGS.ckpt, 'contrasts': {str(k): v for k, v in contrasts.items()},
                   'note': 'positive contrast with CI > 0 = small Weed loses more linear '
                           'decodability through SR than large Weed (supports H1)'}, f, indent=2)
    for r in rows:
        print(f"stage {r['stage']} {r['task']:14s} {r['feature']:8s} pos {r['n_pos_val']:5d} "
              f"AUROC {r['auroc']:.3f} AP {r['ap']:.3f}")
    print(json.dumps(contrasts, indent=2, default=float))


if __name__ == '__main__':
    app.run(main)
