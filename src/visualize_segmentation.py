# SPDX-License-Identifier: Apache-2.0
"""Generates qualitative segmentation comparison figures: for N sample images, shows
RGB input | Ground Truth | Baseline prediction | Current-model prediction side by side.

This is the standalone deliverable that answers "show me the predictions, not just the
numbers" -- required for any segmentation paper. Choose sample indices that include
images with many small/thin weed regions to visually demonstrate where the refined
decoder helps most (that's usually where the numeric IoU gain is concentrated).

Usage (mirrors evaluate.py's flags for building/loading each checkpoint):
    !python visualize_segmentation.py \
        --dataset_path=/kaggle/working/weedsgalore-dataset \
        --split=test \
        --num_classes=3 \
        --num_samples=6 \
        --baseline_ckpt=/kaggle/working/dlv3p_baseline/best_val.pth \
        --baseline_in_channels=5 \
        --current_ckpt=/kaggle/working/out_dir_segformer_refine1/best_val.pth \
        --current_segformer=True --current_pretrained_backbone=True \
        --current_temperature=1.2215 \
        --out_dir=/kaggle/working/qualitative_figs

Saves one PNG per sample to --out_dir, plus one combined grid PNG.
"""

from absl import app, flags
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from datasets import WeedsGaloreDataset
from torch.utils.data import DataLoader
from nets import (deeplabv3plus_resnet50, deeplabv3plus_resnet50_do, deeplabv3plus_resnet50_attn,
                   deeplabv3plus_resnet34_spectral, SegFormer5Band)
import os

FLAGS = flags.FLAGS

flags.DEFINE_string('dataset_path', '/weedsgalore-dataset', 'dataset directory')
flags.DEFINE_string('split', 'test', 'Options: val, test')
flags.DEFINE_integer('num_classes', 3, 'options: 3 (uni-weed), 6 (multi-weed)')
flags.DEFINE_integer('num_samples', 6, 'number of sample images to visualize')
flags.DEFINE_string('out_dir', 'qualitative_figs', 'directory to save figures')
flags.DEFINE_integer('seed', 0, 'random seed for sample selection (fixed = reproducible figures)')

# --- Baseline model (e.g. ResNet50 + ASPP + DeepLabV3+, in_channels=5 via patched conv1) ---
flags.DEFINE_string('baseline_ckpt', '', 'baseline checkpoint path; leave empty to skip baseline column')
flags.DEFINE_integer('baseline_in_channels', 5, 'in_channels the baseline was trained with')
flags.DEFINE_boolean('baseline_spectral_guided', False, '')
flags.DEFINE_boolean('baseline_use_attention', False, '')

# --- Current model (e.g. SegFormer5Band with LowLevelRefine) ---
flags.DEFINE_string('current_ckpt', '', 'current-model checkpoint path (required)')
flags.DEFINE_boolean('current_segformer', True, '')
flags.DEFINE_boolean('current_pretrained_backbone', True, '')
flags.DEFINE_boolean('current_spectral_guided', False, '')
flags.DEFINE_float('current_temperature', 1.0, 'global temperature to apply to current model logits')

# WeedsGalore 3-class palette: 0=background, 1=crop, 2=weed
PALETTE = np.array([
    [0, 0, 0],        # background -> black
    [0, 200, 0],      # crop -> green
    [220, 30, 30],    # weed -> red
], dtype=np.uint8)


def colorize(label_map, num_classes):
    """label_map: (H, W) int array -> (H, W, 3) uint8 RGB using PALETTE (extended with
    a distinct color per extra class if num_classes > 3)."""
    palette = PALETTE
    if num_classes > len(palette):
        rng = np.random.default_rng(42)
        extra = rng.integers(0, 255, size=(num_classes - len(palette), 3)).astype(np.uint8)
        palette = np.concatenate([palette, extra], axis=0)
    return palette[np.clip(label_map, 0, num_classes - 1)]


def build_baseline_net(device):
    if FLAGS.baseline_spectral_guided:
        net = deeplabv3plus_resnet34_spectral(num_classes=FLAGS.num_classes, pretrained_backbone=False)
    elif FLAGS.baseline_use_attention:
        net = deeplabv3plus_resnet50_attn(num_classes=FLAGS.num_classes, pretrained_backbone=False)
    else:
        net = deeplabv3plus_resnet50(num_classes=FLAGS.num_classes, pretrained_backbone=False)
    net = net.to(device)
    if FLAGS.baseline_in_channels == 5 and not FLAGS.baseline_spectral_guided:
        net.backbone.conv1 = nn.Conv2d(FLAGS.baseline_in_channels, net.backbone.conv1.out_channels,
                                       kernel_size=7, stride=2, padding=3, bias=False, device=device)
    _load_ckpt(net, FLAGS.baseline_ckpt, device)
    net.eval()
    return net


def build_current_net(device):
    if FLAGS.current_segformer:
        net = SegFormer5Band(num_classes=FLAGS.num_classes, pretrained_backbone=FLAGS.current_pretrained_backbone)
    elif FLAGS.current_spectral_guided:
        net = deeplabv3plus_resnet34_spectral(num_classes=FLAGS.num_classes, pretrained_backbone=False)
    else:
        net = deeplabv3plus_resnet50(num_classes=FLAGS.num_classes, pretrained_backbone=False)
    net = net.to(device)
    _load_ckpt(net, FLAGS.current_ckpt, device)
    net.eval()
    return net


def _load_ckpt(net, ckpt_path, device):
    model_dict = torch.load(ckpt_path, map_location=device)
    thop_keys = [k for k in model_dict if k.endswith('total_ops') or k.endswith('total_params')]
    for k in thop_keys:
        del model_dict[k]
    net.load_state_dict(model_dict)


def main(_):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(FLAGS.out_dir, exist_ok=True)

    assert FLAGS.current_ckpt, "--current_ckpt is required"
    show_baseline = bool(FLAGS.baseline_ckpt)

    current_net = build_current_net(device)
    baseline_net = build_baseline_net(device) if show_baseline else None

    dataset = WeedsGaloreDataset(dataset_path=FLAGS.dataset_path, dataset_size=None, in_bands=5,
                                 num_classes=FLAGS.num_classes, is_training=False, split=FLAGS.split,
                                 augmentation=False)
    dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=False, num_workers=1, drop_last=True)

    all_samples = list(dataloader)
    rng = np.random.default_rng(FLAGS.seed)
    idxs = rng.choice(len(all_samples), size=min(FLAGS.num_samples, len(all_samples)), replace=False)

    n_cols = 3 + (1 if show_baseline else 0)  # RGB | [Baseline] | Current | GT
    fig, axes = plt.subplots(len(idxs), n_cols, figsize=(3.2 * n_cols, 3.2 * len(idxs)))
    if len(idxs) == 1:
        axes = axes[None, :]

    for row, idx in enumerate(idxs):
        features, unique_labels, binary_labels = all_samples[idx]
        labels = binary_labels if FLAGS.num_classes == 3 else unique_labels
        features, labels = features.to(device), labels.to(device)

        rgb = features[0, :3].detach().cpu().numpy().transpose(1, 2, 0)
        rgb = np.clip((rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8), 0, 1)  # normalize for display
        gt_np = labels[0].detach().cpu().numpy()

        with torch.no_grad():
            current_logits = current_net(features) / FLAGS.current_temperature
            current_pred = current_logits.argmax(dim=1)[0].detach().cpu().numpy()

        col = 0
        axes[row, col].imshow(rgb); axes[row, col].set_title('RGB Input' if row == 0 else '')
        axes[row, col].axis('off'); col += 1

        if show_baseline:
            baseline_features = features if FLAGS.baseline_in_channels == 5 else features[:, :3]
            with torch.no_grad():
                baseline_pred = baseline_net(baseline_features).argmax(dim=1)[0].detach().cpu().numpy()
            axes[row, col].imshow(colorize(baseline_pred, FLAGS.num_classes))
            axes[row, col].set_title('Baseline' if row == 0 else ''); axes[row, col].axis('off'); col += 1

        axes[row, col].imshow(colorize(current_pred, FLAGS.num_classes))
        axes[row, col].set_title('Current Model' if row == 0 else ''); axes[row, col].axis('off'); col += 1

        axes[row, col].imshow(colorize(gt_np, FLAGS.num_classes))
        axes[row, col].set_title('Ground Truth' if row == 0 else ''); axes[row, col].axis('off')

    # Legend (background/crop/weed color key)
    handles = [plt.Rectangle((0, 0), 1, 1, color=np.array(PALETTE[c]) / 255.0)
              for c in range(min(FLAGS.num_classes, len(PALETTE)))]
    class_names = ['background', 'crop', 'weed'][:FLAGS.num_classes]
    fig.legend(handles, class_names, loc='lower center', ncol=len(class_names), bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout()
    out_path = os.path.join(FLAGS.out_dir, f'qualitative_comparison_{FLAGS.split}.png')
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    print(f"Saved combined figure: {out_path}")

    # also save each row individually (useful for picking single-image figures for the paper)
    for row, idx in enumerate(idxs):
        fig_i, axes_i = plt.subplots(1, n_cols, figsize=(3.2 * n_cols, 3.2))
        for c in range(n_cols):
            axes_i[c].imshow(axes[row, c].images[0].get_array())
            axes_i[c].set_title(axes[row, c].get_title())
            axes_i[c].axis('off')
        indiv_path = os.path.join(FLAGS.out_dir, f'sample_{row:02d}_{FLAGS.split}.png')
        plt.savefig(indiv_path, dpi=200, bbox_inches='tight')
        plt.close(fig_i)
    print(f"Saved {len(idxs)} individual sample figures to {FLAGS.out_dir}")


if __name__ == '__main__':
    app.run(main)