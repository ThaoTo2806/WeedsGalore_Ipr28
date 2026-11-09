"""Where do the 'extra plant' predictions of E11-B0 lie? (validation only, 2 dense tiles)

Run on Kaggle from the project root (needs src/, a dataset path, E11B0_s*/best_val.pth):
    python e11_fp_vegetation_check.py --dataset_path=$DS --seeds=0,1,2,3,4 --out=results/exp11/fp_veg_check

For every selected val tile and seed it splits the plant-vs-background errors by
  * NDVI vegetation mask (NDVI = (NIR-R)/(NIR+R) from the dataset bands, Otsu +/-0.1),
  * distance to the nearest GT plant pixel (<=2 px, <=5 px, >5 px).
Diagnostic only: no test split, no training, no model selection.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
from scipy import ndimage as ndi

TILES = ('2023-06-15_0462', '2023-06-15_0956')


def otsu(x, bins=256):
    h, e = np.histogram(x, bins=bins)
    c = (e[:-1] + e[1:]) / 2
    w0 = np.cumsum(h)
    w1 = w0[-1] - w0
    m0 = np.cumsum(h * c) / np.maximum(w0, 1)
    m1 = (np.sum(h * c) - np.cumsum(h * c)) / np.maximum(w1, 1)
    return float(c[np.argmax(w0 * w1 * (m0 - m1) ** 2)])


def analyse(pred, gt, ndvi, deltas=(-0.1, 0.0, 0.1)):
    """pred, gt: HxW int arrays (0 bg, 1 crop, 2 weed); ndvi: HxW float."""
    pp, gp = pred > 0, gt > 0
    fp, fn, tp = pp & ~gp, ~pp & gp, pp & gp
    dist = ndi.distance_transform_edt(~gp)
    out = {'tp': int(tp.sum()), 'fp': int(fp.sum()), 'fn': int(fn.sum()),
           'plant_iou': float(tp.sum() / max(tp.sum() + fp.sum() + fn.sum(), 1)),
           'fp_dist_le2': float((fp & (dist <= 2)).sum() / max(fp.sum(), 1)),
           'fp_dist_le5': float((fp & (dist <= 5)).sum() / max(fp.sum(), 1)),
           'fp_dist_gt5': float((fp & (dist > 5)).sum() / max(fp.sum(), 1)), 'by_threshold': {}}
    th0 = otsu(ndvi)
    for d in deltas:
        veg = ndvi > th0 + d
        fp_on_veg = fp & veg
        # diagnostic only: plant IoU if FP lying on NDVI vegetation were not counted as errors
        iou_ignore = tp.sum() / max(tp.sum() + (fp & ~veg).sum() + fn.sum(), 1)
        out['by_threshold'][f'{th0 + d:.3f}'] = {
            'veg_frac': float(veg.mean()), 'gt_plant_frac': float(gp.mean()),
            'veg_unlabeled_frac_of_veg': float((veg & ~gp).sum() / max(veg.sum(), 1)),
            'fp_on_veg_frac': float(fp_on_veg.sum() / max(fp.sum(), 1)),
            'fn_on_nonveg_frac': float((fn & ~veg).sum() / max(fn.sum(), 1)),
            'plant_iou_ignoring_fp_on_veg_DIAGNOSTIC_ONLY': float(iou_ignore)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset_path', required=True)
    ap.add_argument('--seeds', default='0,1,2,3,4')
    ap.add_argument('--model', default='E11B0')
    ap.add_argument('--out', default='results/exp11/fp_veg_check')
    a = ap.parse_args()
    sys.path.insert(0, 'src')
    import torch
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from ccmc_common import load_checkpoint
    from datasets import WeedsGaloreDataset
    from nets import SegFormer5Band

    os.makedirs(a.out, exist_ok=True)
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ds = WeedsGaloreDataset(dataset_path=a.dataset_path, dataset_size=None, in_bands=5,
                            num_classes=3, is_training=False, split='val', augmentation=False)
    names = [str(n) for n in ds.img_list]
    items = {}
    for t in TILES:
        img, _, lab = ds[names.index(t)]          # val split only
        img = np.ascontiguousarray(img)
        items[t] = (img, np.asarray(lab), (img[3] - img[0]) / (img[3] + img[0] + 1e-9))
    results = {}
    for seed in [int(s) for s in a.seeds.split(',')]:
        found = glob.glob(f'**/{a.model}_s{seed}/best_val.pth', recursive=True) + \
                glob.glob(f'/kaggle/input/**/{a.model}_s{seed}/best_val.pth', recursive=True)
        if not found:
            print('NOT FOUND', a.model, seed)
            continue
        net = SegFormer5Band(num_classes=3, pretrained_backbone=True, in_channels=5,
                             use_low_level_refine=True, use_multiscale_fusion=True, llr_source='raw')
        net = load_checkpoint(net, found[0], dev).to(dev).eval()
        for t, (img, gt, ndvi) in items.items():
            with torch.no_grad():
                pred = net(torch.from_numpy(img).float()[None].to(dev)).argmax(1)[0].cpu().numpy()
            r = analyse(pred, gt, ndvi)
            results[f'{t}_s{seed}'] = r
            th = list(r['by_threshold'])[1]       # Otsu threshold
            b = r['by_threshold'][th]
            print(f"{t} s{seed}: plantIoU {r['plant_iou']:.3f} | FP {r['fp']} "
                  f"on-NDVI-veg {100 * b['fp_on_veg_frac']:.0f}% | <=2px of GT {100 * r['fp_dist_le2']:.0f}% "
                  f"| >5px {100 * r['fp_dist_gt5']:.0f}% | FN on non-veg {100 * b['fn_on_nonveg_frac']:.0f}%")
            if seed == int(a.seeds.split(',')[0]):
                veg = ndvi > float(th)
                pp, gp = pred > 0, gt > 0
                err = np.zeros(gt.shape + (3,))
                err[pp & ~gp & veg] = (1, .6, 0)       # FP on vegetation (orange)
                err[pp & ~gp & ~veg] = (1, 0, 0)       # FP not on vegetation (red)
                err[~pp & gp] = (0, .4, 1)             # FN (blue)
                err[pp & gp] = (.75, .75, .75)         # TP (grey)
                fig, ax = plt.subplots(1, 4, figsize=(20, 5))
                for x, im, ti in zip(ax, (img[3], gp, pp, err), ('NIR', 'GT plant', 'pred plant',
                                                                  'orange FP on NDVI-veg | red FP elsewhere | blue FN')):
                    x.imshow(im, cmap='gray' if im.ndim == 2 else None)
                    x.set_title(ti, fontsize=9)
                    x.axis('off')
                fig.savefig(f'{a.out}/{t}_s{seed}_errors.png', dpi=110, bbox_inches='tight')
                plt.close(fig)
    json.dump(results, open(f'{a.out}/fp_veg_check.json', 'w'), indent=2)
    # mean over seeds, Otsu threshold
    for t in TILES:
        rs = [v for k, v in results.items() if k.startswith(t)]
        if rs:
            f = lambda g: np.mean([g(v) for v in rs])
            print(f'MEAN {t} ({len(rs)} seeds): FP on NDVI-veg '
                  f"{100 * f(lambda v: list(v['by_threshold'].values())[1]['fp_on_veg_frac']):.0f}% | "
                  f"plant IoU {f(lambda v: v['plant_iou']):.3f} -> ignoring FP on veg "
                  f"{f(lambda v: list(v['by_threshold'].values())[1]['plant_iou_ignoring_fp_on_veg_DIAGNOSTIC_ONLY']):.3f}")


if __name__ == '__main__':
    main()
