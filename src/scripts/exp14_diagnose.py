"""Exp14 validation-only pixel error audit for a frozen B2 RawLLR checkpoint.

Reports gt->pred confusion counts, relative error types, and error distance
(near a GT class boundary vs. interior). NO TEST ACCESS. No GT used in model.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
import numpy as np
import torch
from torch.nn import functional as F
from datasets import WeedsGaloreDataset
from nets import SegFormer5Band
from exp12_utils import predict_labels
from ccmc_common import load_checkpoint


def near_gt_border(label, radius):
    # Integer-valued labels, ignoring negative pixels in the final metrics.
    vals = torch.from_numpy(np.maximum(label, 0))[None, None].float()
    k = 2 * radius + 1
    mx = F.max_pool2d(vals, k, stride=1, padding=radius)
    mn = -F.max_pool2d(-vals, k, stride=1, padding=radius)
    return (mx != mn)[0, 0].numpy()


def count_error(cm):
    cm = np.asarray(cm)
    wrong = cm.sum() - np.trace(cm)
    crop_weed = int(cm[1, 2] + cm[2, 1])
    bg_veg = int(cm[0, 1] + cm[0, 2] + cm[1, 0] + cm[2, 0])
    return {'n_pixels': int(cm.sum()), 'n_errors': int(wrong),
            'crop_to_weed': int(cm[1, 2]), 'weed_to_crop': int(cm[2, 1]),
            'crop_weed_errors': crop_weed, 'veg_bg_errors': bg_veg,
            'crop_weed_share_of_errors_pct': round(100 * crop_weed / max(wrong, 1), 3),
            'matrix_gt_rows_pred_cols': cm.astype(int).tolist()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset_path', required=True)
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--tta', choices=['none', 'd4'], default='d4')
    ap.add_argument('--boundary_px', type=int, default=3)
    a = ap.parse_args()
    if a.boundary_px < 1:
        raise SystemExit('boundary_px must >=1')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = SegFormer5Band(num_classes=3, pretrained_backbone=True, in_channels=5,
                           use_low_level_refine=True, use_multiscale_fusion=True,
                           llr_source='raw', backbone='b2')
    net = load_checkpoint(net, a.ckpt, device).to(device).eval()
    dataset = WeedsGaloreDataset(dataset_path=a.dataset_path, dataset_size=None,
                                 in_bands=5, num_classes=3, is_training=False,
                                 split='val', augmentation=False)
    total = {k: np.zeros((3, 3), dtype=np.int64) for k in ('all', 'border', 'interior')}
    per_image = []
    with torch.inference_mode():
        for idx in range(len(dataset)):
            img, _, label = dataset[idx]
            img_tensor = torch.from_numpy(np.ascontiguousarray(img)).float()[None].to(device)
            pred = predict_labels(net, img_tensor, scale=1.0, tta=a.tta, weed_bias=0.0)[0].cpu().numpy()
            label = np.asarray(label)
            assert pred.shape == label.shape
            valid = (label >= 0) & (label < 3)
            border = near_gt_border(label, a.boundary_px)
            row = {'name': str(dataset.img_list[idx])}
            for section, mask in [('all', valid), ('border', valid & border),
                                  ('interior', valid & ~border)]:
                cm = np.bincount(3 * label[mask].astype(np.int64) + pred[mask].astype(np.int64),
                                 minlength=9).reshape(3, 3)
                total[section] += cm
                counts = count_error(cm)
                for key in ('n_errors', 'crop_to_weed', 'weed_to_crop', 'veg_bg_errors'):
                    row[f'{section}_{key}'] = counts[key]
            per_image.append(row)
            print(f'{idx + 1}/{len(dataset)} {row["name"]}', flush=True)
    report = {'split': 'val', 'n_images': len(dataset), 'tta': a.tta,
              'boundary_px': a.boundary_px, 'gt_leakage_in_inference': False,
              'note': 'Border defined as pixels with >1 GT class in the radius-r neighborhood; '
                      'evaluation-only; GT is never fed to the model.',
              'breakdown': {key: count_error(cm) for key, cm in total.items()}}
    report['border_share_of_all_errors_pct'] = round(
        100 * report['breakdown']['border']['n_errors'] /
        max(report['breakdown']['all']['n_errors'], 1), 3)
    p = Path(a.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(report, indent=2), encoding='utf-8')
    with p.with_suffix('.csv').open('w', newline='') as f:
        wr = csv.DictWriter(f, per_image[0].keys())
        wr.writeheader(); wr.writerows(per_image)
    print(json.dumps(report, indent=2))
    print('Saved:', p, p.with_suffix('.csv'))


if __name__ == '__main__':
    main()
