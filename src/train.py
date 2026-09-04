# SPDX-FileCopyrightText: 2025 Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences
# SPDX-FileCopyrightText: 2025 Ekin Celikkan <ekin.celikkan@gfz.de>
# SPDX-License-Identifier: Apache-2.0

from absl import app, flags
import torch
from datasets import WeedsGaloreDataset
from torch.utils.data import DataLoader
from nets import (deeplabv3plus_resnet50, deeplabv3plus_resnet50_do,
                  deeplabv3plus_resnet50_attn, deeplabv3plus_resnet34_spectral,
                  SegFormer5Band)
from pathlib import Path
from torchmetrics.classification import MulticlassJaccardIndex
from torch.utils.tensorboard import SummaryWriter
import os
import time
try:
    from thop import profile
    THOP_AVAILABLE = True
except ImportError:
    THOP_AVAILABLE = False

FLAGS = flags.FLAGS

flags.DEFINE_string('dataset_path', 'weedsgalore-dataset', 'dataset directory')
flags.DEFINE_integer('dataset_size_train', 104, 'dataset size of train set')
flags.DEFINE_integer('in_channels', 5, 'options: 3 (RGB), 5 (MSI)')
flags.DEFINE_integer('num_classes', 6, 'options: 3 (uni-weed), 6 (multi-weed)')
flags.DEFINE_integer('ignore_index', -1, 'ignore during loss and iou calculation')
flags.DEFINE_boolean('dlv3p_do', False, 'set True to use probabilistic variant of DLv3+ with dropout')
flags.DEFINE_boolean('segformer', False, 'set True to use the 5-band SegFormer architecture')
flags.DEFINE_boolean('spectral_guided', True, 'set True to use the spectral-guided architecture: ResNet34 RGB backbone '
                     '+ tiny NIR/RE spectral branch (Spectral Gate, replaces CBAM) + Lite-ASPP')
flags.DEFINE_boolean('use_spectral_indices', True, 'set True to feed the spectral encoder '
                     '[NIR, RE, NDVI-like, NDRE-like] (4ch) instead of raw [NIR, RE] (2ch). Only used when '
                     'spectral_guided=True. Set False to isolate/compare against the raw-band-only variant.')
flags.DEFINE_boolean('use_attention', False, 'set True to insert a CBAM attention module between ASPP and the decoder '
                     '(only used when spectral_guided=False; CBAM itself is currently disabled/commented out, so this '
                     'defaults to False to avoid conflicting with spectral_guided regardless of branch order)')
flags.DEFINE_boolean('pretrained_backbone', True, 'set True to use pretrained ResNet50 backbone')
flags.DEFINE_string('ckpt_resnet', 'ckpts/resnet50-19c8e357.pth', 'ckpt path for pretrained backbone')
flags.DEFINE_integer('batch_size', 2, 'batch size')
flags.DEFINE_integer('num_workers', 4, 'number of subprocesses')
flags.DEFINE_float('lr', 0.001, 'Learning rate')
flags.DEFINE_float('weight_bg', 0.35, 'Cross-entropy weight for background')
flags.DEFINE_float('weight_crop', 1.0, 'Cross-entropy weight for crop')
flags.DEFINE_float('weight_weed', 1.5, 'Cross-entropy weight for weed')
flags.DEFINE_integer('epochs', 10, 'number of epochs for training')
flags.DEFINE_string('out_dir', 'out_dir', 'directory to save logs and ckpts')
flags.DEFINE_integer('log_interval', 25, 'number of iterations to log scalars')
flags.DEFINE_integer('ckpt_interval', 500, 'number of iterations to save ckpts')

def count_parameters(model):
    """Count total trainable and non-trainable parameters"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params
    return total_params, trainable_params, non_trainable_params


def measure_model_stats(model, device, in_channels, input_size=(512, 512)):
    """Measure model parameters, FLOPs, and inference time"""
    # Count parameters
    total_params, trainable_params, non_trainable_params = count_parameters(model)
    print(f"\n{'='*60}")
    print(f"MODEL STATISTICS")
    print(f"{'='*60}")
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    print(f"Non-trainable Parameters: {non_trainable_params:,}")
   
    # Measure FLOPs
    if THOP_AVAILABLE:
        try:
            dummy_input = torch.randn(1, in_channels, input_size[0], input_size[1]).to(device)
            flops, params = profile(model, inputs=(dummy_input,), verbose=False)
            print(f"FLOPs (forward pass): {flops:,.0f} (~{flops/1e9:.2f}G)")
            print(f"FLOPs per pixel: {flops / (input_size[0] * input_size[1]):,.0f}")
        except Exception as e:
            print(f"FLOPs calculation failed: {e}")
    else:
        print("FLOPs calculation requires 'thop' library. Install with: pip install thop")
   
    # Measure inference time
    model.eval()
    with torch.no_grad():
        dummy_input = torch.randn(1, in_channels, input_size[0], input_size[1]).to(device)
        # Warmup
        for _ in range(5):
            _ = model(dummy_input)
       
        # Time measurement
        start_time = time.time()
        num_runs = 100
        for _ in range(num_runs):
            _ = model(dummy_input)
        end_time = time.time()
       
        avg_inference_time = (end_time - start_time) / num_runs * 1000  # in ms
        throughput = 1000 / avg_inference_time  # images per second
       
        print(f"Inference time (avg {num_runs} runs): {avg_inference_time:.2f} ms")
        print(f"Throughput: {throughput:.2f} img/s")
    print(f"{'='*60}\n")



def main(_):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using {device}")
    if device.type == 'cuda':
        print(f"Cuda current device: {torch.cuda.current_device()}")
        print(f"Cuda device name: {torch.cuda.get_device_name(0)}")


    # Dataset
    train_dataset = WeedsGaloreDataset(dataset_path=FLAGS.dataset_path, dataset_size=FLAGS.dataset_size_train, in_bands=FLAGS.in_channels,
                                        num_classes=FLAGS.num_classes, is_training=True, split='train', augmentation=True)
    val_dataset = WeedsGaloreDataset(dataset_path=FLAGS.dataset_path, dataset_size=None, in_bands=FLAGS.in_channels,
                                        num_classes=FLAGS.num_classes, is_training=False, split='val', augmentation=False)

    # Dataloader
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=FLAGS.batch_size, shuffle=True,
                                  num_workers=FLAGS.num_workers, collate_fn=None, drop_last=True)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=FLAGS.batch_size, shuffle=False,
                                num_workers=FLAGS.num_workers, collate_fn=None, drop_last=False)

    # Network
    if FLAGS.segformer:
        net = SegFormer5Band(num_classes=FLAGS.num_classes)
    elif FLAGS.dlv3p_do:
        net = deeplabv3plus_resnet50_do(num_classes=FLAGS.num_classes, pretrained_backbone=FLAGS.pretrained_backbone)  # probabilistic DeepLabv3+
    elif FLAGS.use_attention:
        net = deeplabv3plus_resnet50_attn(num_classes=FLAGS.num_classes, pretrained_backbone=FLAGS.pretrained_backbone)  # DeepLabv3+ (CBAM currently disabled, see _deeplab.py)
    elif FLAGS.spectral_guided:
        net = deeplabv3plus_resnet34_spectral(num_classes=FLAGS.num_classes, pretrained_backbone=FLAGS.pretrained_backbone,
                                              use_spectral_indices=FLAGS.use_spectral_indices)  # ResNet34 RGB + NIR/RE(+NDVI/NDRE) Spectral Gate + Lite-ASPP
    else:
        net = deeplabv3plus_resnet50(num_classes=FLAGS.num_classes, pretrained_backbone=FLAGS.pretrained_backbone)  # (determinsitic) DeepLabv3+

    # Modify first layer
    if FLAGS.in_channels == 5 and not FLAGS.spectral_guided and not FLAGS.segformer:
        net.backbone.conv1 = torch.nn.Conv2d(FLAGS.in_channels, net.backbone.conv1.out_channels, kernel_size=7, stride=2, padding=3, bias=False, device=device)

    # Model to device
    net.to(device=device)

    # Measure model statistics
    measure_model_stats(net, device, in_channels=FLAGS.in_channels)

    # Loss criterion
    class_weights = None
    if FLAGS.num_classes == 3:
        class_weights = torch.tensor(
            [FLAGS.weight_bg, FLAGS.weight_crop, FLAGS.weight_weed], device=device)
    criterion = torch.nn.CrossEntropyLoss(
        weight=class_weights, ignore_index=FLAGS.ignore_index).to(device)

    # Optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=FLAGS.lr)

    # Metric
    evaluator = MulticlassJaccardIndex(num_classes=FLAGS.num_classes, average=None, ignore_index=FLAGS.ignore_index).to(device)
    val_evaluator = MulticlassJaccardIndex(
        num_classes=FLAGS.num_classes, average=None, ignore_index=FLAGS.ignore_index).to(device)
    best_val_miou = float('-inf')

    # Logging
    accum_loss, accum_iter, tot_iter = 0, 0, 0
    os.makedirs(FLAGS.out_dir, exist_ok=True)
    writer = SummaryWriter(f'{FLAGS.out_dir}')
    print(f'Logging to: {FLAGS.out_dir}')
    Path(FLAGS.out_dir).mkdir(parents=True, exist_ok=True)

    torch.autograd.set_detect_anomaly(True)

    # Train
    for epoch in range(FLAGS.epochs):
        net.train()
        train_iter = iter(train_dataloader)
        for i, data in enumerate(train_iter):
            features, unique_labels, binary_labels = data
            if FLAGS.num_classes == 3:
                labels = binary_labels
            else:
                labels = unique_labels
            features, labels = features.to(device), labels.to(device)  # NCHW

            optimizer.zero_grad()
            out = net(features)
            loss = criterion(out, labels.long())
            loss.backward()
            optimizer.step()

            accum_loss += loss.item()
            accum_iter += 1
            tot_iter += 1

            # compute miou
            _, pred = torch.max(out, 1)
            evaluator.update(pred, labels)

            # log scalars
            if tot_iter % FLAGS.log_interval == 0 or tot_iter == 1:
                metrics = evaluator.compute() * 100

                print(f'Epoch: {epoch} iter: {tot_iter}, Loss: {(accum_loss / accum_iter):.2f}')
                print(f'mIoU: {metrics.mean():.2f}%')

                writer.add_scalar('Training Loss', accum_loss / accum_iter, tot_iter)
                writer.add_scalar('miou (%)', metrics.mean(), tot_iter)
                writer.add_scalar('iou_crop (%)', metrics[1], tot_iter)
                for weed_idx, weed_iou in enumerate(metrics[2:], start=2):
                    writer.add_scalar(f'iou_weed_{weed_idx-1} (%)', weed_iou, tot_iter)

                evaluator.reset()
                accum_loss, accum_iter = 0, 0

            # save ckpt
            if tot_iter % FLAGS.ckpt_interval == 0 or tot_iter == 1:
                torch.save(net.state_dict(), f'{FLAGS.out_dir}/{str(epoch)}.pth')
                torch.save(optimizer.state_dict(), f'{FLAGS.out_dir}/optimizer.pth')

        net.eval()
        val_evaluator.reset()
        with torch.no_grad():
            for features, unique_labels, binary_labels in val_dataloader:
                labels = binary_labels if FLAGS.num_classes == 3 else unique_labels
                features, labels = features.to(device), labels.to(device)
                predictions = net(features).argmax(dim=1)
                val_evaluator.update(predictions, labels)

        val_scores = val_evaluator.compute() * 100
        val_miou = val_scores.mean().item()
        print(f'Epoch: {epoch}, Val mIoU: {val_miou:.2f}%')
        writer.add_scalar('val_miou (%)', val_miou, epoch + 1)
        writer.add_scalar('val_iou_bg (%)', val_scores[0], epoch + 1)
        writer.add_scalar('val_iou_crop (%)', val_scores[1], epoch + 1)
        for weed_idx, weed_iou in enumerate(val_scores[2:], start=2):
            writer.add_scalar(f'val_iou_weed_{weed_idx-1} (%)', weed_iou, epoch + 1)

        if val_miou > best_val_miou:
            best_val_miou = val_miou
            torch.save(net.state_dict(), f'{FLAGS.out_dir}/best_val.pth')
            print(f'Saved best_val.pth ({best_val_miou:.2f}%)')

if __name__ == '__main__':
    app.run(main)