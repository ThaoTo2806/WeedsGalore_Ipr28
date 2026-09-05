# SPDX-License-Identifier: Apache-2.0
"""Fits a single scalar temperature T on the val split to calibrate a trained
checkpoint's confidence (Guo et al., 2017, "On Calibration of Modern Neural Networks").

logits_calibrated = logits / T

Temperature scaling does NOT change argmax predictions (dividing all logits in a
sample by the same positive constant preserves their ranking) -- so mIoU is
mathematically identical before and after. It only rescales the softmax
distribution's sharpness, which is what ECE measures. This is why it's the
recommended way to fix a high-ECE / good-mIoU checkpoint without retraining and
without risking the accuracy gains you already have.

T is fit by minimizing NLL (cross-entropy) on the val split w.r.t. T alone (all
other weights frozen) via LBFGS -- this is the standard/original method and a good
proxy for minimizing ECE directly, since NLL is smooth (ECE's binning makes it
non-differentiable, so it can't be optimized directly with gradient methods).

Usage (mirrors evaluate.py's flags for building/loading the same checkpoint):
    !python calibrate_temperature.py \
        --dataset_path=/kaggle/working/weedsgalore-dataset \
        --ckpt=/kaggle/working/out_dir_segformer_refine1/best_val.pth \
        --segformer=True --pretrained_backbone=True --spectral_guided=False \
        --in_channels=5 --num_classes=3

Prints the fitted T and the val ECE before/after, e.g.:
    Fitted temperature: T = 1.83
    ECE before (T=1.0): 0.009562
    ECE after  (T=1.83): 0.002341

Then re-run evaluate.py with --temperature=<the printed T> on the split you want to
report (val AND test -- fit T on val only, never on test, to avoid leaking test
information into calibration).
"""

from absl import app, flags
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.classification import MulticlassCalibrationError
from datasets import WeedsGaloreDataset
from torch.utils.data import DataLoader
from nets import (deeplabv3plus_resnet50, deeplabv3plus_resnet50_do, deeplabv3plus_resnet50_attn,
                   deeplabv3plus_resnet34_spectral, SegFormer5Band)

FLAGS = flags.FLAGS

flags.DEFINE_string('dataset_path', '/weedsgalore-dataset', 'dataset directory')
flags.DEFINE_string('ckpt', '/dlv3p_msi_6.pth', 'checkpoint to calibrate')
flags.DEFINE_integer('in_channels', 5, 'options: 3 (RGB), 5 (MSI)')
flags.DEFINE_integer('num_classes', 6, 'options: 3 (uni-weed), 6 (multi-weed)')
flags.DEFINE_integer('ignore_index', -1, 'ignore during loss/ECE calculation')
flags.DEFINE_boolean('spectral_guided', False, 'must match the checkpoint\'s training flags')
flags.DEFINE_boolean('segformer', False, 'must match the checkpoint\'s training flags')
flags.DEFINE_boolean('pretrained_backbone', False, 'must match the checkpoint\'s training flags')
flags.DEFINE_boolean('use_attention', False, 'must match the checkpoint\'s training flags')
flags.DEFINE_integer('lbfgs_iters', 50, 'max LBFGS iterations to fit T')
flags.DEFINE_integer('ece_n_bins', None, 'bins for the reported ECE; defaults to num_classes '
                     '(matches evaluate.py\'s convention) if left unset')


def build_net(device):
    if FLAGS.segformer:
        net = SegFormer5Band(num_classes=FLAGS.num_classes, pretrained_backbone=FLAGS.pretrained_backbone)
    elif FLAGS.spectral_guided:
        net = deeplabv3plus_resnet34_spectral(num_classes=FLAGS.num_classes, pretrained_backbone=False)
    elif FLAGS.use_attention:
        net = deeplabv3plus_resnet50_attn(num_classes=FLAGS.num_classes, pretrained_backbone=False)
    else:
        net = deeplabv3plus_resnet50(num_classes=FLAGS.num_classes, pretrained_backbone=False)
    net = net.to(device)
    if FLAGS.in_channels == 5 and not FLAGS.spectral_guided and not FLAGS.segformer:
        net.backbone.conv1 = nn.Conv2d(FLAGS.in_channels, net.backbone.conv1.out_channels,
                                       kernel_size=7, stride=2, padding=3, bias=False, device=device)
    model_dict = torch.load(FLAGS.ckpt, map_location=device)
    thop_keys = [k for k in model_dict if k.endswith('total_ops') or k.endswith('total_params')]
    for k in thop_keys:
        del model_dict[k]
    net.load_state_dict(model_dict)
    net.eval()
    return net


def main(_):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using {device}")
    net = build_net(device)

    # Collect ALL val logits + labels up front (val sets here are small enough to fit
    # in memory as a single batch for LBFGS; this matches how temperature scaling is
    # normally fit -- one joint optimization over the whole calibration set, not
    # mini-batches).
    dataset = WeedsGaloreDataset(dataset_path=FLAGS.dataset_path, dataset_size=None,
                                 in_bands=FLAGS.in_channels, num_classes=FLAGS.num_classes,
                                 is_training=False, split='val', augmentation=False)
    dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=False, num_workers=1, drop_last=True)

    all_logits, all_labels = [], []
    with torch.no_grad():
        for features, unique_labels, binary_labels in dataloader:
            labels = binary_labels if FLAGS.num_classes == 3 else unique_labels
            features, labels = features.to(device), labels.to(device)
            logits = net(features)  # (1, C, H, W)
            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())
    logits = torch.cat(all_logits, dim=0).to(device)  # (N, C, H, W)
    labels = torch.cat(all_labels, dim=0).to(device)  # (N, H, W)

    n_bins = FLAGS.ece_n_bins or FLAGS.num_classes
    mcce = MulticlassCalibrationError(num_classes=FLAGS.num_classes, n_bins=n_bins, norm='l1',
                                      ignore_index=FLAGS.ignore_index).to(device)
    ece_before = mcce(F.softmax(logits, dim=1), labels).item()

    # Fit T by minimizing NLL on the (frozen) logits -- this is the only trainable
    # parameter; everything else in the network stays exactly as trained.
    log_temperature = torch.zeros(1, device=device, requires_grad=True)  # optimize in log-space -> T=exp(.)>0 always
    optimizer = torch.optim.LBFGS([log_temperature], lr=0.05, max_iter=FLAGS.lbfgs_iters)
    valid_mask = (labels != FLAGS.ignore_index)
    safe_labels = labels.clone()
    safe_labels[~valid_mask] = 0

    def closure():
        optimizer.zero_grad()
        temperature = torch.exp(log_temperature)
        scaled_logits = logits / temperature
        # per-pixel NLL, masked to valid (non-ignored) pixels
        log_probs = F.log_softmax(scaled_logits, dim=1)
        nll = F.nll_loss(log_probs, safe_labels.long(), reduction='none')
        nll = (nll * valid_mask.float()).sum() / valid_mask.float().sum()
        nll.backward()
        return nll

    optimizer.step(closure)
    fitted_T = torch.exp(log_temperature).item()

    with torch.no_grad():
        calibrated_probs = F.softmax(logits / fitted_T, dim=1)
        ece_after = mcce(calibrated_probs, labels).item()

    print(f"\n{'=' * 40}")
    print(f"{'Temperature calibration (val split)':^40}")
    print(f"{'=' * 40}")
    print(f"Fitted temperature: T = {fitted_T:.4f}")
    print(f"ECE before (T=1.0):  {ece_before:.6f}")
    print(f"ECE after  (T={fitted_T:.2f}): {ece_after:.6f}")
    print(f"\nNote: argmax predictions are UNCHANGED by temperature scaling, so mIoU is "
          f"identical before/after -- only confidence calibration (ECE) changes.")
    print(f"\nNext: re-run evaluate.py with --temperature={fitted_T:.4f} on val AND test "
          f"to report calibrated ECE (T was fit on val only, so test ECE is a fair, "
          f"un-leaked estimate).")


if __name__ == '__main__':
    app.run(main)