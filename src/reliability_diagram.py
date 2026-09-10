# SPDX-License-Identifier: Apache-2.0
"""Draws a reliability diagram (confidence vs. accuracy, binned) to visually show model
calibration -- the standard figure accompanying any ECE number in a paper. Plots BOTH
uncalibrated (T=1.0) and temperature-scaled (T=fitted) reliability side by side, so the
figure directly illustrates what temperature scaling did.

A reliability diagram bins predictions by their confidence (max softmax probability),
then for each bin plots:
    - the bin's average confidence (what the model "claims")
    - the bin's actual accuracy (what really happens)
A perfectly calibrated model has bars sitting exactly on the y=x diagonal. The gap
between the bar and the diagonal is what ECE (weighted-average of these gaps) measures.

Usage (mirrors evaluate.py's flags for building/loading the checkpoint):
    !python reliability_diagram.py \
        --dataset_path=/kaggle/working/weedsgalore-dataset \
        --split=test \
        --ckpt=/kaggle/working/out_dir_segformer_refine1/best_val.pth \
        --segformer=True --pretrained_backbone=True --spectral_guided=False \
        --in_channels=5 --num_classes=3 \
        --temperature=1.2215 \
        --out_path=/kaggle/working/reliability_diagram.png
"""

from absl import app, flags
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from datasets import WeedsGaloreDataset
from torch.utils.data import DataLoader
from nets import (deeplabv3plus_resnet50, deeplabv3plus_resnet50_do, deeplabv3plus_resnet50_attn,
                   deeplabv3plus_resnet34_spectral, SegFormer5Band)
import torch.nn as nn

FLAGS = flags.FLAGS

flags.DEFINE_string('dataset_path', '/weedsgalore-dataset', 'dataset directory')
flags.DEFINE_string('split', 'test', 'Options: val, test')
flags.DEFINE_string('ckpt', '', 'checkpoint to evaluate (required)')
flags.DEFINE_integer('in_channels', 5, 'options: 3 (RGB), 5 (MSI)')
flags.DEFINE_integer('num_classes', 3, 'options: 3 (uni-weed), 6 (multi-weed)')
flags.DEFINE_integer('ignore_index', -1, 'ignore during calculation')
flags.DEFINE_boolean('spectral_guided', False, '')
flags.DEFINE_boolean('segformer', False, '')
flags.DEFINE_boolean('pretrained_backbone', False, '')
flags.DEFINE_boolean('use_attention', False, '')
flags.DEFINE_float('temperature', 1.0, 'fitted global temperature (from calibrate_temperature.py)')
flags.DEFINE_integer('n_bins', 15, 'number of confidence bins for the diagram')
flags.DEFINE_string('out_path', 'reliability_diagram.png', 'output figure path')


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


def compute_bin_stats(confidences, correct, n_bins):
    """Returns (bin_centers, bin_accuracy, bin_confidence, bin_counts) for a standard
    equal-width reliability diagram."""
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_acc = np.zeros(n_bins)
    bin_conf = np.zeros(n_bins)
    bin_count = np.zeros(n_bins)
    bin_idx = np.digitize(confidences, bin_edges[1:-1])
    for b in range(n_bins):
        mask = bin_idx == b
        if mask.sum() > 0:
            bin_acc[b] = correct[mask].mean()
            bin_conf[b] = confidences[mask].mean()
            bin_count[b] = mask.sum()
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    return bin_centers, bin_acc, bin_conf, bin_count


def plot_reliability(ax, confidences, correct, n_bins, title, ece):
    bin_centers, bin_acc, bin_conf, bin_count = compute_bin_stats(confidences, correct, n_bins)
    width = 1.0 / n_bins
    has_data = bin_count > 0

    ax.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect calibration')
    ax.bar(bin_centers[has_data], bin_acc[has_data], width=width * 0.9,
          edgecolor='black', color='steelblue', alpha=0.8, label='Accuracy')
    # gap between confidence and accuracy, shown as a hatched overlay (what ECE penalizes)
    gap = bin_conf[has_data] - bin_acc[has_data]
    ax.bar(bin_centers[has_data], gap, bottom=np.minimum(bin_acc[has_data], bin_conf[has_data]),
          width=width * 0.9, edgecolor='red', color='none', hatch='//', label='Gap (over/under-confidence)')

    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel('Confidence'); ax.set_ylabel('Accuracy')
    ax.set_title(f'{title}\nECE = {ece:.4f}')
    ax.legend(loc='upper left', fontsize=8)
    ax.set_aspect('equal')


def main(_):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    assert FLAGS.ckpt, "--ckpt is required"
    net = build_net(device)

    dataset = WeedsGaloreDataset(dataset_path=FLAGS.dataset_path, dataset_size=None, in_bands=FLAGS.in_channels,
                                 num_classes=FLAGS.num_classes, is_training=False, split=FLAGS.split,
                                 augmentation=False)
    dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=False, num_workers=1, drop_last=True)

    all_confidences, all_correct = [], []          # T=1.0 (uncalibrated)
    all_confidences_t, all_correct_t = [], []       # T=fitted (calibrated)
    with torch.no_grad():
        for features, unique_labels, binary_labels in dataloader:
            labels = binary_labels if FLAGS.num_classes == 3 else unique_labels
            features, labels = features.to(device), labels.to(device)
            logits = net(features)

            valid = (labels != FLAGS.ignore_index)

            probs = F.softmax(logits, dim=1)
            conf, pred = probs.max(dim=1)
            correct = (pred == labels).float()
            all_confidences.append(conf[valid].cpu().numpy())
            all_correct.append(correct[valid].cpu().numpy())

            probs_t = F.softmax(logits / FLAGS.temperature, dim=1)
            conf_t, pred_t = probs_t.max(dim=1)
            correct_t = (pred_t == labels).float()  # identical to `correct` (T doesn't change argmax)
            all_confidences_t.append(conf_t[valid].cpu().numpy())
            all_correct_t.append(correct_t[valid].cpu().numpy())

    confidences = np.concatenate(all_confidences)
    correct = np.concatenate(all_correct)
    confidences_t = np.concatenate(all_confidences_t)
    correct_t = np.concatenate(all_correct_t)

    # standard L1 ECE, matching evaluate.py's MulticlassCalibrationError(norm='l1') convention
    def l1_ece(conf, corr, n_bins):
        _, bin_acc, bin_conf, bin_count = compute_bin_stats(conf, corr, n_bins)
        weights = bin_count / bin_count.sum()
        return float(np.sum(weights * np.abs(bin_conf - bin_acc)))

    ece_before = l1_ece(confidences, correct, FLAGS.n_bins)
    ece_after = l1_ece(confidences_t, correct_t, FLAGS.n_bins)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2))
    plot_reliability(axes[0], confidences, correct, FLAGS.n_bins,
                     f'Before calibration (T=1.0)\nsplit={FLAGS.split}', ece_before)
    plot_reliability(axes[1], confidences_t, correct_t, FLAGS.n_bins,
                     f'After temperature scaling (T={FLAGS.temperature:.4f})\nsplit={FLAGS.split}', ece_after)
    plt.tight_layout()
    plt.savefig(FLAGS.out_path, dpi=200, bbox_inches='tight')
    print(f"Saved reliability diagram: {FLAGS.out_path}")
    print(f"ECE before: {ece_before:.6f}  |  ECE after: {ece_after:.6f}")


if __name__ == '__main__':
    app.run(main)