"""MCSD diagnostic: how exactly is class mass conserved per 4x4 cell on validation images?

Reports the deviation between the decoded sub-pixel probabilities averaged over each cell and
the predicted cell fractions q (Sinkhorn row marginals), and the predicted/GT Weed mass ratio.

python src/scripts/mass_check.py --dataset_path=... --ckpt=runs/MCSDg_s0/best_val.pth \
    --head=mcsd --spectral_guide=True --sinkhorn_iters=5 --out=results/mass_check.json
"""

import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from absl import app, flags

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ccmc_common import build_model, define_model_flags, load_checkpoint  # noqa: E402
from datasets import WeedsGaloreDataset  # noqa: E402

FLAGS = flags.FLAGS
flags.DEFINE_string('dataset_path', 'weedsgalore-dataset', 'dataset directory')
flags.DEFINE_string('ckpt', None, 'MCSD checkpoint')
flags.DEFINE_string('out', None, 'output json')
define_model_flags()


def main(_):
    FLAGS.ccmc = False
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = load_checkpoint(build_model(FLAGS), FLAGS.ckpt, device).to(device).eval()
    ds = WeedsGaloreDataset(FLAGS.dataset_path, None, FLAGS.in_channels, 3, False, 'val', False)
    max_err, mean_err, pred_mass, gt_mass = [], [], 0.0, 0.0
    with torch.no_grad():
        for i in range(len(ds)):
            img, _, lab = ds[i]
            x = torch.from_numpy(np.ascontiguousarray(img)).float()[None].to(device)
            logp, extras = net(x, return_aux=True)
            p = logp.exp()
            q = torch.softmax(extras['occ_logits'].float(), 1)
            err = (F.avg_pool2d(p, net.cell) - q).abs()
            max_err.append(err.max().item())
            mean_err.append(err.mean().item())
            pred_mass += p[:, 2].sum().item()
            gt_mass += float((np.asarray(lab) == 2).sum())
    res = {'ckpt': FLAGS.ckpt, 'sinkhorn_iters': FLAGS.sinkhorn_iters,
           'max_abs_fraction_error': float(np.max(max_err)),
           'mean_abs_fraction_error': float(np.mean(mean_err)),
           'pred_weed_mass_over_gt': pred_mass / gt_mass}
    print(json.dumps(res))
    if FLAGS.out:
        with open(FLAGS.out, 'w') as f:
            json.dump(res, f, indent=2)


if __name__ == '__main__':
    app.run(main)
