# SPDX-FileCopyrightText: 2024 Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences
# SPDX-FileCopyrightText: 2024 Ekin Celikkan <ekin.celikkan@gfz-potsdam.de>
# SPDX-License-Identifier: Apache-2.0

from absl import app, flags
import torch
from torchmetrics.classification import MulticlassJaccardIndex, MulticlassCalibrationError, MulticlassConfusionMatrix
from datasets import WeedsGaloreDataset
from torch.utils.data import DataLoader
import torch.nn as nn
from nets import (deeplabv3plus_resnet50, deeplabv3plus_resnet50_do, deeplabv3plus_resnet50_attn,
                   deeplabv3plus_resnet34_spectral)

FLAGS = flags.FLAGS

flags.DEFINE_string('dataset', 'weedsgalore', 'options: weedsgalore')
flags.DEFINE_string('dataset_path', '/weedsgalore-dataset', 'dataset directory')
flags.DEFINE_string('split', 'test', 'Options: val, test')

flags.DEFINE_string('network', 'deeplabv3plus', 'options: deeplabv3plus')
flags.DEFINE_string('ckpt', '/dlv3p_msi_6.pth', 'checkpoint directory')

flags.DEFINE_integer('in_channels', 5, 'options: 3 (RGB), 5 (MSI)')
flags.DEFINE_integer('num_classes', 6, 'options: 3 (uni-weed), 6 (multi-weed)')
flags.DEFINE_integer('ignore_index', -1, 'ignore during loss and iou calculation')

# --- Must match the flags used at train time for this checkpoint's architecture ---
flags.DEFINE_boolean('spectral_guided', False, 'set True if the checkpoint was trained with '
                     '--spectral_guided=True (ResNet34 + spectral branch + Lite-ASPP). Must match '
                     'the training run exactly, or state_dict loading will fail with key mismatches.')
flags.DEFINE_boolean('use_attention', False, 'set True if the checkpoint was trained with '
                     '--use_attention=True (ResNet50 + CBAM head). Ignored if spectral_guided=True.')
flags.DEFINE_boolean('use_spectral_indices', True, 'set True if the checkpoint used NDVI/NDRE channels')
# NOTE: use_spectral_indices is NOT wired here on purpose. If your local nets/modeling.py already
# supports NDVI/NDRE indices (4ch spectral input) but this checkpoint was trained BEFORE that
# patch (2ch raw NIR/RE), passing use_spectral_indices=True here would build the WRONG shape
# and state_dict loading would fail with a size mismatch on TinySpectralEncoder's first conv.
# Only re-add this flag once you retrain a checkpoint with the NDVI/NDRE-enabled architecture.


def main(_):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using {device}")
    if device.type == 'cuda':
        print(f"Cuda current device: {torch.cuda.current_device()}")
        print(f"Cuda device name: {torch.cuda.get_device_name(0)}")

    # Build the SAME architecture used at train time -- this must mirror train.py's branch order.
    if FLAGS.spectral_guided:
        net = deeplabv3plus_resnet34_spectral(
            num_classes=FLAGS.num_classes, pretrained_backbone=False,
            use_spectral_indices=FLAGS.use_spectral_indices)
        print("Architecture: ResNet34 OS=8 + spectral branch + residual Spectral Gate + Lite-ASPP")
    elif FLAGS.use_attention:
        net = deeplabv3plus_resnet50_attn(num_classes=FLAGS.num_classes, pretrained_backbone=False)
        print("Architecture: ResNet50 + CBAM")
    else:
        net = deeplabv3plus_resnet50(num_classes=FLAGS.num_classes, pretrained_backbone=False)
        print("Architecture: ResNet50 (baseline)")
    net = net.to(device)

    # first conv to fit input channels -- ONLY for the plain ResNet50 path. The spectral-guided
    # backbone keeps a standard 3-channel conv1 (RGB only) and splits off NIR/RE internally, so
    # patching conv1 here would corrupt it / break state_dict loading.
    if FLAGS.in_channels == 5 and not FLAGS.spectral_guided:
        net.backbone.conv1 = nn.Conv2d(FLAGS.in_channels, net.backbone.conv1.out_channels, kernel_size=7, stride=2, padding=3, bias=False, device=device)

    # load checkpoint
    model_weights_dir = FLAGS.ckpt
    model_dict = torch.load(model_weights_dir, map_location=device)
    # thop.profile() (used in train.py's measure_model_stats to count FLOPs) registers
    # 'total_ops'/'total_params' buffers on every submodule to do its counting. These get
    # saved into the checkpoint's state_dict as a side effect, but a freshly-constructed
    # eval model (which never ran through thop) doesn't have them -> strip them here.
    thop_keys = [k for k in model_dict if k.endswith('total_ops') or k.endswith('total_params')]
    if thop_keys:
        print(f"Stripping {len(thop_keys)} thop profiling buffer(s) from checkpoint (harmless, not model weights)")
        for k in thop_keys:
            del model_dict[k]
    net.load_state_dict(model_dict)

    # Dataset and dataloader
    dataset_path = FLAGS.dataset_path
    dataset = WeedsGaloreDataset(dataset_path=dataset_path, dataset_size=None, in_bands=FLAGS.in_channels,
                                        num_classes=FLAGS.num_classes, is_training=False, split=FLAGS.split, augmentation=False)

    dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=False, num_workers=1, collate_fn=None, drop_last=False)
    dataset_iter = iter(dataloader)

    # iou evaluator
    evaluator = MulticlassJaccardIndex(num_classes=FLAGS.num_classes, average=None, ignore_index=FLAGS.ignore_index).to(device)

    # mcce evaluator
    mcce = MulticlassCalibrationError(num_classes=FLAGS.num_classes, n_bins=FLAGS.num_classes, norm='l1', ignore_index=FLAGS.ignore_index).to(device)

    # confusion matrix
    confmat = MulticlassConfusionMatrix(num_classes=FLAGS.num_classes, normalize='true', ignore_index=FLAGS.ignore_index).to(device)

    net.eval()

    # Run evaluation on dataset
    for i, data in enumerate(dataset_iter):
        features, unique_labels, binary_labels = data
        features, unique_labels, binary_labels = features.to(device), unique_labels.to(device), binary_labels.to(device)

        if FLAGS.num_classes == 3:
            labels = binary_labels
        else:
            labels = unique_labels

        with torch.no_grad():
            out = net(features)

        out = torch.nn.functional.softmax(out, dim=1)

        # pred
        _, pred = torch.max(out, 1)

        # calculate metric
        evaluator.update(pred, labels)
        scene_scores = evaluator(pred, labels)
        mcce.update(out, labels)
        confmat.update(pred, labels)


    # print scores over all dataset
    print(f"\n{'=' * 40}")
    print(f"{'Overall scores':^40}")
    print(f"{'=' * 40}")
    scores = evaluator.compute()
    print(f'Split: {FLAGS.split}')
    print(f'mIoU: {scores.mean() * 100:.2f}%')
    print(f'iou bg: {scores[0] * 100:.2f}%')
    print(f'iou crop: {scores[1] * 100:.2f}%')
    for weed_idx, weed_iou in enumerate(scores[2:], start=2):
        print(f'iou weed_{weed_idx-1}: {weed_iou * 100:.2f}%')
    print(f"ECE: {mcce.compute()}")


if __name__ == '__main__':
    app.run(main)