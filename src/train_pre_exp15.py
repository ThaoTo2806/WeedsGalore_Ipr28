# SPDX-FileCopyrightText: 2025 Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences
# SPDX-FileCopyrightText: 2025 Ekin Celikkan <ekin.celikkan@gfz.de>
# SPDX-License-Identifier: Apache-2.0

from absl import app, flags
import torch
import torch.nn.functional as F
from datasets import WeedsGaloreDataset
from torch.utils.data import DataLoader
from nets import (deeplabv3plus_resnet50, deeplabv3plus_resnet50_do,
                  deeplabv3plus_resnet50_attn, deeplabv3plus_resnet34_spectral,
                  SegFormer5Band, UNet, CCMCSegFormer)
from nets.ccmc_sr import GT_MODES, aux_target
from nets.mcsd import MCSDSegFormer, occupancy_loss
from nets.llr_subcell import (LLR_RESIDUAL_HEADS, LLR_SUBCELL_HEADS,
                               LLRResidualSubcellSegFormer, LLRSubcellSegFormer)
from nets.sc_llr import SC_LLR_HEADS, SCLLRSegFormer
import exp12_config
from exp14_vbb import vbb_loss
from exp14_ascd import ascd_loss
from exp12_sched import ema_decay_at, lr_factor
from exp12_utils import clean_state, ema_update, predict_labels, rescale_batch
from pathlib import Path
import copy
import hashlib
import json
import platform
import random
import numpy as np
from torchmetrics.classification import MulticlassJaccardIndex
from torch.utils.tensorboard import SummaryWriter
import os
import time
try:
    from thop import profile
    THOP_AVAILABLE = True
except ImportError:
    THOP_AVAILABLE = False


def soft_dice_loss(logits, labels, num_classes, ignore_index=-1, class_weights=None, eps=1e-6):
    """Multi-class soft Dice loss, computed directly on class probabilities (not hard
    predictions), so it is differentiable. Dice = 2*|pred ∩ target| / (|pred| + |target|),
    which is a smooth surrogate for IoU -- unlike CrossEntropy, it optimizes the actual
    region-overlap quantity that mIoU measures, so it pushes directly on small/thin
    minority classes (e.g. weed) instead of being diluted pixel-by-pixel by the
    dominant background class.

    Args:
        logits: (N, C, H, W) raw network output (pre-softmax).
        labels: (N, H, W) integer class labels, may contain ignore_index.
        num_classes: number of classes C.
        ignore_index: label value to exclude from the loss (e.g. unlabeled pixels).
        class_weights: optional (C,) tensor to weight each class's Dice term (use the
            same weights as your CrossEntropy for consistency, e.g. emphasize weed).
        eps: numerical stability constant.
    Returns:
        scalar loss (1 - mean Dice over classes), lower is better, same convention as CE.
    """
    probs = F.softmax(logits, dim=1)  # (N, C, H, W)
    valid_mask = (labels != ignore_index)  # (N, H, W)
    safe_labels = labels.clone()
    safe_labels[~valid_mask] = 0  # placeholder, excluded via valid_mask below
    labels_onehot = F.one_hot(safe_labels.long(), num_classes=num_classes)  # (N, H, W, C)
    labels_onehot = labels_onehot.permute(0, 3, 1, 2).float()  # (N, C, H, W)

    valid_mask = valid_mask.unsqueeze(1).float()  # (N, 1, H, W), broadcasts over classes
    probs = probs * valid_mask
    labels_onehot = labels_onehot * valid_mask

    dims = (0, 2, 3)  # sum over batch + spatial, keep per-class
    intersection = torch.sum(probs * labels_onehot, dims)
    cardinality = torch.sum(probs + labels_onehot, dims)
    dice_per_class = (2. * intersection + eps) / (cardinality + eps)  # (C,)

    if class_weights is not None:
        weights = class_weights / class_weights.sum()
        loss = 1. - torch.sum(dice_per_class * weights)
    else:
        loss = 1. - dice_per_class.mean()
    return loss


FLAGS = flags.FLAGS
_ABSL_FLAGS = set(FLAGS)  # flags defined before this file's own (used by the Exp12 config check)

flags.DEFINE_string('dataset_path', 'weedsgalore-dataset', 'dataset directory')
flags.DEFINE_integer('dataset_size_train', 104, 'dataset size of train set')
flags.DEFINE_integer('in_channels', 5, 'options: 3 (RGB), 5 (MSI)')
flags.DEFINE_integer('num_classes', 6, 'options: 3 (uni-weed), 6 (multi-weed)')
flags.DEFINE_integer('ignore_index', -1, 'ignore during loss and iou calculation')
flags.DEFINE_boolean('dlv3p_do', False, 'set True to use probabilistic variant of DLv3+ with dropout')
flags.DEFINE_boolean('segformer', False, 'set True to use the 5-band SegFormer architecture')
flags.DEFINE_boolean('use_low_level_refine', True, 'use the raw-input LowLevelRefine branch in SegFormer')
flags.DEFINE_boolean('use_multiscale_fusion', True, 'use multi-scale feature fusion in SegFormer')
flags.DEFINE_enum('llr_source', 'raw', ['raw', 'none', 'f1'], 'input of the 1/2-res refine decoder: '
                  'raw bands (LiteSent), none (decoder-1/2 only), f1 (MiT stage-1 feature)')
flags.DEFINE_boolean('unet', False, 'set True to use the U-Net baseline')
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
flags.DEFINE_float('ascd_weight', 0.0, 'Train-only ASCD weight')
flags.DEFINE_float('vbb_weight', 0.0, 'Training-only vegetation boundary loss weight')
flags.DEFINE_integer('vbb_radius', 3, 'Vegetation boundary radius in pixels')
flags.DEFINE_float('ascd_hard_alpha', 1.0, 'Wrong CW class probability weighting')
flags.DEFINE_float('ascd_spectral_alpha', 1.0, 'Spectral confusion weighting')
flags.DEFINE_float('ascd_spectral_temperature', 1.0, 'Spectral confusion temperature')
flags.DEFINE_integer('ascd_interface_radius', 3, 'Crop-Weed interface radius')
flags.DEFINE_bool('amp', False, 'CUDA AMP')
flags.DEFINE_integer('num_workers', 4, 'number of subprocesses')
flags.DEFINE_integer('seed', 0, 'random seed for model initialization, shuffling, and augmentation')
flags.DEFINE_float('lr', 0.001, 'Learning rate')
flags.DEFINE_float('weight_bg', 0.35, 'Cross-entropy weight for background')
flags.DEFINE_float('weight_crop', 1.0, 'Cross-entropy weight for crop')
flags.DEFINE_float('weight_weed', 1.5, 'Cross-entropy weight for weed')
flags.DEFINE_boolean('use_dice_loss', True, 'set True to add soft Dice loss (optimizes region '
                     'overlap / IoU directly) alongside CrossEntropy. Strongly recommended when '
                     'the minority/thin class (weed) is underperforming, since CE alone does not '
                     'optimize overlap and dilutes gradient for small classes.')
flags.DEFINE_float('dice_weight', 0.5, 'weight of the Dice loss term: '
                    'total_loss = (1 - dice_weight) * CE + dice_weight * Dice. '
                    'Only used when use_dice_loss=True.')
flags.DEFINE_integer('epochs', 10, 'number of epochs for training')
flags.DEFINE_string('out_dir', 'out_dir', 'directory to save logs and ckpts')
flags.DEFINE_integer('log_interval', 25, 'number of iterations to log scalars')
flags.DEFINE_integer('ckpt_interval', 500, 'number of iterations to save ckpts')
# --- CCMC-SR (paper/README_CCMC_SR_v2_3_oracle_leakage_reviewed.md) ---
flags.DEFINE_boolean('ccmc', False, 'use CCMCSegFormer (MiT-B0 5-band, prior-modulated K/V SR). '
                     'prior_mode=none gives L0 through the same code path.')
flags.DEFINE_enum('prior_mode', 'none', ['none', 'ndvi', 'learned_veg', 'learned_weed', 'learned_free',
                                         'gt_veg', 'gt_weed'],
                  'L0=none, L1=ndvi, L2=learned_veg, L3=learned_weed, K3=learned_free, '
                  'C2=gt_veg, C3=gt_weed (GT oracle: diagnostic only)')
flags.DEFINE_enum('injection', 'ccmc_sr', ['ccmc_sr', 'none', 'pre_sr_gate', 'post_sr_gate'],
                  'ccmc_sr=proposed, none=K1, pre_sr_gate=K5, post_sr_gate=K6')
flags.DEFINE_enum('prior_perturb', 'none', ['none', 'window_shuffle', 'global_shuffle'],
                  'window_shuffle=K2, global_shuffle=K2b placebo')
flags.DEFINE_list('ccmc_stages', ['2', '3'], 'MiT stages whose K/V SR is modulated (1-based)')
flags.DEFINE_float('ccmc_eps', 0.1, 'floor eps of w = eps + (1-eps) * p')
flags.DEFINE_float('lam_init', 0.05, 'initial lambda = sigmoid(a)')
flags.DEFINE_boolean('learn_lambda', True, 'learn lambda (bounded in (0,1)) or keep it fixed')
flags.DEFINE_float('lam_lr', 0.0, 'separate Adam lr for the lambda logits (0 = same as --lr)')
flags.DEFINE_float('aux_weight', 0.3, 'weight of the prior aux BCE loss (learned_veg/learned_weed)')
# --- MCSD: mass-conserving sub-cell decoding (src/nets/mcsd.py) ---
flags.DEFINE_enum('head', 'bilinear', ['bilinear', 'mcsd', 'pixelshuffle', 'occ_conditioned',
                                       *LLR_SUBCELL_HEADS, *LLR_RESIDUAL_HEADS, *SC_LLR_HEADS],
                  'prediction head; *2 heads replace bilinear on the raw-LLR H/2 feature')
flags.DEFINE_integer('sinkhorn_iters', 5, 'Sinkhorn iterations per cell (mcsd)')
flags.DEFINE_float('mcsd_tau', 1.0, 'temperature of the placement scores (mcsd)')
flags.DEFINE_boolean('spectral_guide', False, 'add a full-resolution placement prior from the raw bands')
flags.DEFINE_float('occ_weight', 0.3, 'weight of the cell-occupancy soft-CE loss (mcsd/pixelshuffle)')
flags.DEFINE_float('beta', 0.5, 'fixed occupancy guidance strength for LLR residual heads')
flags.DEFINE_float('residual_alpha_init', 0.02, 'initial sigmoid gate for LLR residual heads')
flags.DEFINE_integer('sc_radius', 1, 'SC-LLR local radius')
flags.DEFINE_integer('sc_guide_dim', 8, 'SC-LLR spectral embedding size')
flags.DEFINE_integer('sc_embed_dim', 16, 'SC-LLR semantic embedding size')
flags.DEFINE_float('sc_alpha_init', 0.1, 'SC-LLR initial gate')
flags.DEFINE_float('sc_alpha_lr', 0.01, 'SC-LLR gate learning rate')
# --- Exp12 (doc/README_EXP12_PLAN_TEST86_Q2_v3.md) ---
flags.DEFINE_enum('backbone', 'b0', ['b0', 'b1', 'b2'], 'MiT encoder (official nvidia/mit-b*); SegFormer5Band family only')
flags.DEFINE_float('input_scale', 1.0, 'input scale: images bilinear, labels nearest; validation logits are resized '
                   'back to the original size (pure interpolation: adds no spatial information)')
flags.DEFINE_enum('lr_schedule', 'const', ['const', 'cosine'], 'const = legacy; cosine = FSP warmup + cosine decay')
flags.DEFINE_integer('warmup_epochs', 0, 'linear warmup epochs (cosine schedule)')
flags.DEFINE_float('min_lr_ratio', 0.0, 'final lr as a fraction of --lr (cosine schedule)')
flags.DEFINE_float('ema_decay', 0.0, 'EMA of weights (0 = off); final_ema.pth is written at the end')
flags.DEFINE_integer('grad_accum_steps', 1, 'microbatches per optimizer step; must be preregistered when !=1')
flags.DEFINE_string('arm_id', '', 'registered arm in exp12_arms.json; when set, flags are verified against the registry')
flags.DEFINE_string('exp12_common_config', 'configs/exp12/exp12_common_config.json', 'frozen common config')
flags.DEFINE_string('exp12_arms', 'configs/exp12/exp12_arms.json', 'registered arm configs')
flags.DEFINE_string('run_name', '', 'run id written to run_config.json (e.g. L3__seed0)')
flags.DEFINE_boolean('resume', False, 'resume from <out_dir>/last.pth if it exists')
flags.DEFINE_boolean('detect_anomaly', False, 'torch.autograd anomaly detection (very slow, debug only)')
flags.DEFINE_boolean('measure_stats', True, 'print params/THOP FLOPs/latency before training')
TRAIN_FLAGS = sorted(set(FLAGS) - _ABSL_FLAGS)


def split_hash():
    split_dir = Path(__file__).resolve().parent / 'datasets' / 'weedsgalore' / 'splits'
    digest = hashlib.sha256()
    for name in ('train.txt', 'val.txt', 'test.txt'):
        digest.update((split_dir / name).read_bytes())
    return digest.hexdigest()[:16]


def environment_info():
    info = {'python': platform.python_version(), 'torch': torch.__version__,
            'cuda': torch.version.cuda, 'numpy': np.__version__}
    try:
        import transformers
        info['transformers'] = transformers.__version__
    except ImportError:
        pass
    if torch.cuda.is_available():
        info['gpu'] = torch.cuda.get_device_name(0)
    try:
        import subprocess
        info['git_commit'] = subprocess.check_output(
            ['git', 'rev-parse', '--short', 'HEAD'], cwd=Path(__file__).resolve().parent,
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        info['git_commit'] = 'unknown'
    return info

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
    exp12_info = None
    if FLAGS.grad_accum_steps < 1:
        raise SystemExit('--grad_accum_steps must be >=1')
    if FLAGS.arm_id and FLAGS.grad_accum_steps != 1:
        raise SystemExit('Gradient accumulation needs an explicit batch/accumulation arm registration and protocol deviation; not yet supported by the frozen Exp12 common config')
    if FLAGS.backbone != 'b0' and (FLAGS.ccmc or FLAGS.head in SC_LLR_HEADS or FLAGS.unet):
        raise SystemExit('--backbone b1/b2 needs the SegFormer5Band family (no CCMC-SR, SC-LLR or U-Net)')
    if FLAGS.input_scale <= 0 or (FLAGS.input_scale != 1.0 and FLAGS.ccmc):
        raise SystemExit('--input_scale must be > 0 and is not supported with --ccmc=True')
    if FLAGS.arm_id:
        common = exp12_config.load_json(FLAGS.exp12_common_config)
        arms = exp12_config.load_json(FLAGS.exp12_arms)
        defaults = {name: FLAGS[name].default for name in TRAIN_FLAGS}
        errors = exp12_config.verify_flags(FLAGS.flag_values_dict(), common, arms, FLAGS.arm_id,
                                           defaults, TRAIN_FLAGS)
        if errors:
            raise SystemExit('Exp12 configuration check FAILED:\n  ' + '\n  '.join(errors))
        exp12_info = {'arm_id': FLAGS.arm_id,
                      'common_sha256': exp12_config.sha256_file(FLAGS.exp12_common_config),
                      'arms_sha256': exp12_config.sha256_file(FLAGS.exp12_arms)}
        print('Exp12 configuration check passed:', exp12_info)
    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)
    torch.manual_seed(FLAGS.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(FLAGS.seed)
    # Separate generator for data order/augmentation, so models with extra parameters
    # (prior head) see the SAME batches as L0 for the same seed (paired comparison).
    data_generator = torch.Generator()
    data_generator.manual_seed(FLAGS.seed)

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
                                  num_workers=FLAGS.num_workers, collate_fn=None, drop_last=True,
                                  generator=data_generator)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=FLAGS.batch_size, shuffle=False,
                                num_workers=FLAGS.num_workers, collate_fn=None, drop_last=False)

    # Network
    if FLAGS.head in SC_LLR_HEADS:
        if (FLAGS.ccmc or not FLAGS.segformer or not FLAGS.use_low_level_refine
            or FLAGS.llr_source != 'raw' or FLAGS.spectral_guide):
            raise SystemExit('SC-LLR requires --ccmc=False --segformer=True '
                             '--use_low_level_refine=True --llr_source=raw --spectral_guide=False')
        if FLAGS.occ_weight > 0:
            raise SystemExit('SC-LLR has no occupancy loss; use --occ_weight=0.0')
        net = SCLLRSegFormer(
            num_classes=FLAGS.num_classes, in_channels=FLAGS.in_channels,
            pretrained_backbone=FLAGS.pretrained_backbone,
            use_multiscale_fusion=FLAGS.use_multiscale_fusion, head=FLAGS.head,
            sc_radius=FLAGS.sc_radius, sc_guide_dim=FLAGS.sc_guide_dim,
            sc_embed_dim=FLAGS.sc_embed_dim, sc_alpha_init=FLAGS.sc_alpha_init,
        )
    elif FLAGS.head in LLR_RESIDUAL_HEADS:
        if (FLAGS.ccmc or not FLAGS.segformer or not FLAGS.use_low_level_refine
            or FLAGS.llr_source != 'raw' or FLAGS.spectral_guide):
            raise SystemExit('LLR residual heads require --ccmc=False --segformer=True '
                     '--use_low_level_refine=True --llr_source=raw --spectral_guide=False')
        net = LLRResidualSubcellSegFormer(
            num_classes=FLAGS.num_classes,
            in_channels=FLAGS.in_channels,
            use_multiscale_fusion=FLAGS.use_multiscale_fusion,
            pretrained_backbone=FLAGS.pretrained_backbone,
            head=FLAGS.head,
            beta=FLAGS.beta,
            residual_alpha_init=FLAGS.residual_alpha_init,
            backbone=FLAGS.backbone,
        )
    elif FLAGS.head in LLR_SUBCELL_HEADS:
        if (FLAGS.ccmc or not FLAGS.segformer or not FLAGS.use_low_level_refine
            or FLAGS.llr_source != 'raw' or FLAGS.spectral_guide):
            raise SystemExit('LLR 2x2 heads require --ccmc=False --segformer=True '
                     '--use_low_level_refine=True --llr_source=raw --spectral_guide=False')
        net = LLRSubcellSegFormer(
            num_classes=FLAGS.num_classes,
            in_channels=FLAGS.in_channels,
            use_multiscale_fusion=FLAGS.use_multiscale_fusion,
            pretrained_backbone=FLAGS.pretrained_backbone,
            head=FLAGS.head,
            sinkhorn_iters=FLAGS.sinkhorn_iters,
            tau=FLAGS.mcsd_tau,
            backbone=FLAGS.backbone,
        )
    elif FLAGS.head != 'bilinear':
        if FLAGS.ccmc or FLAGS.use_low_level_refine:
            raise SystemExit('--head=mcsd/pixelshuffle needs --ccmc=False --use_low_level_refine=False')
        net = MCSDSegFormer(num_classes=FLAGS.num_classes, in_channels=FLAGS.in_channels,
                            use_multiscale_fusion=FLAGS.use_multiscale_fusion, head=FLAGS.head,
                            sinkhorn_iters=FLAGS.sinkhorn_iters, tau=FLAGS.mcsd_tau,
                            spectral_guide=FLAGS.spectral_guide, backbone=FLAGS.backbone)
    elif FLAGS.ccmc:
        net = CCMCSegFormer(num_classes=FLAGS.num_classes, in_channels=FLAGS.in_channels,
                            use_low_level_refine=FLAGS.use_low_level_refine,
                            use_multiscale_fusion=FLAGS.use_multiscale_fusion,
                            prior_mode=FLAGS.prior_mode, injection=FLAGS.injection,
                            prior_perturb=FLAGS.prior_perturb,
                            stages=[int(s) for s in FLAGS.ccmc_stages], eps=FLAGS.ccmc_eps,
                            lam_init=FLAGS.lam_init, learn_lambda=FLAGS.learn_lambda)
    elif FLAGS.unet:
        net = UNet(in_channels=FLAGS.in_channels, num_classes=FLAGS.num_classes)
    elif FLAGS.segformer:
        net = SegFormer5Band(num_classes=FLAGS.num_classes,
                             pretrained_backbone=FLAGS.pretrained_backbone,
                             in_channels=FLAGS.in_channels,
                             use_low_level_refine=FLAGS.use_low_level_refine,
                             use_multiscale_fusion=FLAGS.use_multiscale_fusion,
                             llr_source=FLAGS.llr_source, backbone=FLAGS.backbone)
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
    if FLAGS.in_channels == 5 and not FLAGS.spectral_guided and not FLAGS.segformer and not FLAGS.ccmc:
        net.backbone.conv1 = torch.nn.Conv2d(FLAGS.in_channels, net.backbone.conv1.out_channels, kernel_size=7, stride=2, padding=3, bias=False, device=device)

    # Model to device
    net.to(device=device)

    # Measure model statistics (GT-oracle models need labels in forward -> skip)
    is_oracle = FLAGS.ccmc and FLAGS.prior_mode in GT_MODES
    if FLAGS.measure_stats and not is_oracle:
        measure_model_stats(net, device, in_channels=FLAGS.in_channels)
    ema_model = None
    if FLAGS.ema_decay > 0:
        ema_model = copy.deepcopy(net).eval()
        for parameter in ema_model.parameters():
            parameter.requires_grad_(False)

    # Loss criterion
    class_weights = None
    if FLAGS.num_classes == 3:
        class_weights = torch.tensor(
            [FLAGS.weight_bg, FLAGS.weight_crop, FLAGS.weight_weed], device=device)
    criterion = torch.nn.CrossEntropyLoss(
        weight=class_weights, ignore_index=FLAGS.ignore_index).to(device)

    # Optimizer. Adam moves a parameter by at most ~lr per step: with lr=3e-4 and ~1300 steps the
    # lambda logit can change by <= ~0.4 (lambda stuck near its init, observed in Exp02), so lambda
    # gets its own learning rate when --lam_lr > 0.
    lam_params = [p for n, p in net.named_parameters() if n.endswith('lam_logit') and p.requires_grad]
    if FLAGS.ccmc and FLAGS.lam_lr > 0 and lam_params:
        lam_ids = {id(p) for p in lam_params}
        other = [p for p in net.parameters() if id(p) not in lam_ids]
        optimizer = torch.optim.Adam([{'params': other},
                                      {'params': lam_params, 'lr': FLAGS.lam_lr}], lr=FLAGS.lr)
    elif FLAGS.head in SC_LLR_HEADS and FLAGS.sc_alpha_lr > 0:
        alpha_params = [net.sc_llr.alpha_logit]
        alpha_ids = {id(parameter) for parameter in alpha_params}
        other_params = [parameter for parameter in net.parameters()
                        if id(parameter) not in alpha_ids]
        optimizer = torch.optim.Adam(
            [{'params': other_params, 'lr': FLAGS.lr},
             {'params': alpha_params, 'lr': FLAGS.sc_alpha_lr}])
    else:
        optimizer = torch.optim.Adam(net.parameters(), lr=FLAGS.lr)

    base_lrs = [group['lr'] for group in optimizer.param_groups]
    steps_per_epoch = len(train_dataloader)
    total_steps = FLAGS.epochs * steps_per_epoch
    warmup_steps = FLAGS.warmup_epochs * steps_per_epoch

    scaler = torch.amp.GradScaler('cuda', enabled=FLAGS.amp)

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

    use_aux = FLAGS.ccmc and FLAGS.prior_mode in ('learned_veg', 'learned_weed') and FLAGS.aux_weight > 0
    use_occ = FLAGS.head not in ('bilinear', *SC_LLR_HEADS) and FLAGS.occ_weight > 0
    bce = torch.nn.BCEWithLogitsLoss()
    run_config = {'run_name': FLAGS.run_name, 'split_hash': split_hash(), 'env': environment_info(),
                  'flags': FLAGS.flag_values_dict(),
                  'num_params': sum(p.numel() for p in net.parameters()),
                  'uses_gt_prior': is_oracle, 'aux_loss': use_aux, 'exp12': exp12_info}
    with open(f'{FLAGS.out_dir}/run_config.json', 'w') as f:
        json.dump(run_config, f, indent=2, default=str)
    history_path = f'{FLAGS.out_dir}/history.jsonl'

    if FLAGS.detect_anomaly:
        torch.autograd.set_detect_anomaly(True)

    start_epoch, best_epoch = 0, -1
    last_path = f'{FLAGS.out_dir}/last.pth'
    if FLAGS.resume and os.path.exists(last_path):
        state = torch.load(last_path, map_location=device, weights_only=False)
        net.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        data_generator.set_state(state['data_generator'])
        torch.set_rng_state(state['torch_rng'])
        if torch.cuda.is_available() and state.get('cuda_rng') is not None:
            torch.cuda.set_rng_state_all(state['cuda_rng'])
        np.random.set_state(state['numpy_rng'])
        random.setstate(state['python_rng'])
        start_epoch, tot_iter = state['epoch'] + 1, state['tot_iter']
        best_val_miou, best_epoch = state['best_val_miou'], state['best_epoch']
        if ema_model is not None and state.get('ema') is not None:
            ema_model.load_state_dict(state['ema'])
        print(f'Resumed from {last_path} at epoch {start_epoch} (best {best_val_miou:.2f}% @ {best_epoch})')

    def forward_train(features, labels, binary_labels):
        if FLAGS.head != 'bilinear':
            return net(features, return_aux=True)
        if not FLAGS.ccmc:
            return net(features), None
        prior_mask = binary_labels if is_oracle else None
        return net(features, prior_mask=prior_mask, return_aux=True)

    def run_validation(model):
        model.eval()
        val_evaluator.reset()
        with torch.no_grad():
            for features, unique_labels, binary_labels in val_dataloader:
                labels = binary_labels if FLAGS.num_classes == 3 else unique_labels
                features, labels = features.to(device), labels.to(device)
                if FLAGS.ccmc:
                    # Oracle (C2/C3) models read GT of the val image: label leakage, diagnostic only.
                    prior_mask = binary_labels.to(device) if is_oracle else None
                    predictions = model(features, prior_mask=prior_mask).argmax(dim=1)
                else:  # predictions are always at the ORIGINAL resolution (labels are not rescaled)
                    predictions = predict_labels(model, features, scale=FLAGS.input_scale)
                val_evaluator.update(predictions, labels)
        return val_evaluator.compute() * 100

    last_val_miou = last_ema_miou = None

    # Train
    for epoch in range(start_epoch, FLAGS.epochs):
        epoch_start = time.time()
        net.train()
        train_iter = iter(train_dataloader)
        epoch_aux, epoch_seg, epoch_steps = 0.0, 0.0, 0
        epoch_sc_entropy = 0.0
        for i, data in enumerate(train_iter):
            features, unique_labels, binary_labels = data
            if FLAGS.num_classes == 3:
                labels = binary_labels
            else:
                labels = unique_labels
            features, labels = features.to(device), labels.to(device)  # NCHW
            binary_labels = binary_labels.to(device)
            features, labels, binary_labels = rescale_batch(features, labels, binary_labels, FLAGS.input_scale)

            optimizer.zero_grad()
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=FLAGS.amp):
                out, extras = forward_train(features, labels, binary_labels)
            ce_loss = criterion(out, labels.long())
            if FLAGS.use_dice_loss:
                dice_loss = soft_dice_loss(out, labels, num_classes=FLAGS.num_classes,
                                           ignore_index=FLAGS.ignore_index, class_weights=class_weights)
                loss = (1 - FLAGS.dice_weight) * ce_loss + FLAGS.dice_weight * dice_loss
            else:
                loss = ce_loss
            epoch_seg += loss.item()
            if use_aux:
                prior_logits = extras['prior_logits']
                target = aux_target(binary_labels, FLAGS.prior_mode, prior_logits.shape[-2:])
                aux_loss = bce(prior_logits.float(), target)
                loss = loss + FLAGS.aux_weight * aux_loss
                epoch_aux += aux_loss.item()
            if use_occ:
                occ_loss = occupancy_loss(extras['occ_logits'], labels, FLAGS.num_classes)
                loss = loss + FLAGS.occ_weight * occ_loss
                epoch_aux += occ_loss.item()
            if FLAGS.head in SC_LLR_HEADS and extras:
                epoch_sc_entropy += float(extras.get('sc_weight_entropy', 0.0))
            epoch_steps += 1
            # EXP14_VBB_TRAIN_ONLY
            if FLAGS.vbb_weight > 0:
                if FLAGS.num_classes != 3 or FLAGS.head != "bilinear" or not FLAGS.segformer:
                    raise ValueError("VBB requires 3-class bilinear SegFormer")
                loss = loss + FLAGS.vbb_weight * vbb_loss(
                    out, labels, radius=FLAGS.vbb_radius, ignore_index=FLAGS.ignore_index)
            # EXP14_ASCD_TRAIN_ONLY
            if FLAGS.ascd_weight > 0:
                if FLAGS.num_classes != 3 or FLAGS.head != "bilinear" or not FLAGS.segformer:
                    raise ValueError("ASCD requires 3-class bilinear SegFormer")
                aux_ascd = ascd_loss(out, labels, features,
                    hard_alpha=FLAGS.ascd_hard_alpha,
                    spectral_alpha=FLAGS.ascd_spectral_alpha,
                    spectral_temperature=FLAGS.ascd_spectral_temperature,
                    interface_radius=FLAGS.ascd_interface_radius,
                    ignore_index=FLAGS.ignore_index)
                loss = loss + FLAGS.ascd_weight * aux_ascd
            scaler.scale(loss).backward()
            factor = lr_factor(tot_iter, total_steps, warmup_steps, FLAGS.lr_schedule, FLAGS.min_lr_ratio)
            for group, base_lr in zip(optimizer.param_groups, base_lrs):
                group['lr'] = base_lr * factor
            scaler.step(optimizer)
            scaler.update()
            if ema_model is not None:
                ema_update(ema_model, net, ema_decay_at(tot_iter, FLAGS.ema_decay))

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

        val_scores = run_validation(net)
        ema_scores = run_validation(ema_model) if ema_model is not None else None
        val_miou = val_scores.mean().item()
        ema_miou = ema_scores.mean().item() if ema_scores is not None else None
        last_val_miou, last_ema_miou = val_miou, ema_miou
        if ema_miou is not None:
            print(f'Epoch: {epoch}, EMA Val mIoU: {ema_miou:.2f}%')
        print(f'Epoch: {epoch}, Val mIoU: {val_miou:.2f}% | ' +
              ' '.join(f'c{c}={s:.2f}' for c, s in enumerate(val_scores.tolist())))
        writer.add_scalar('val_miou (%)', val_miou, epoch + 1)
        if ema_miou is not None:
            writer.add_scalar('val_miou_ema (%)', ema_miou, epoch + 1)
        writer.add_scalar('val_iou_bg (%)', val_scores[0], epoch + 1)
        writer.add_scalar('val_iou_crop (%)', val_scores[1], epoch + 1)
        for weed_idx, weed_iou in enumerate(val_scores[2:], start=2):
            writer.add_scalar(f'val_iou_weed_{weed_idx-1} (%)', weed_iou, epoch + 1)

        if val_miou > best_val_miou:
            best_val_miou = val_miou
            best_epoch = epoch
            torch.save(net.state_dict(), f'{FLAGS.out_dir}/best_val.pth')
            print(f'Saved best_val.pth ({best_val_miou:.2f}%)')

        lambdas = net.encoder.lambdas() if FLAGS.ccmc else []
        record = {'epoch': epoch, 'val_miou': val_miou, 'val_iou': val_scores.tolist(),
                  'train_seg_loss': epoch_seg / max(epoch_steps, 1),
                  'train_aux_loss': epoch_aux / max(epoch_steps, 1) if (use_aux or use_occ) else None,
                  'lambdas': lambdas, 'best_val_miou': best_val_miou, 'best_epoch': best_epoch,
                  'epoch_time_s': time.time() - epoch_start,
                  'lr': optimizer.param_groups[0]['lr'],
                  'val_miou_ema': ema_miou,
                  'val_iou_ema': ema_scores.tolist() if ema_scores is not None else None,
                  'max_mem_gb': (torch.cuda.max_memory_allocated() / 1e9) if torch.cuda.is_available() else None}
        if FLAGS.head in SC_LLR_HEADS:
            record['sc_alpha'] = float(torch.sigmoid(net.sc_llr.alpha_logit).item())
            record['sc_weight_entropy'] = epoch_sc_entropy / max(epoch_steps, 1)
        with open(history_path, 'a') as f:
            f.write(json.dumps(record) + '\n')
        if lambdas:
            print(f'lambdas (stages {FLAGS.ccmc_stages}): ' + ', '.join(f'{v:.4f}' for v in lambdas))

        torch.save({'model': net.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch,
                    'ema': ema_model.state_dict() if ema_model is not None else None,
                    'tot_iter': tot_iter, 'best_val_miou': best_val_miou, 'best_epoch': best_epoch,
                    'data_generator': data_generator.get_state(), 'torch_rng': torch.get_rng_state(),
                    'cuda_rng': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                    'numpy_rng': np.random.get_state(), 'python_rng': random.getstate()}, last_path)

    # FSP: report the FINAL checkpoint (EMA when enabled); best_val.pth is kept only for legacy comparisons.
    torch.save(clean_state(net), f'{FLAGS.out_dir}/final.pth')
    if ema_model is not None:
        torch.save(clean_state(ema_model), f'{FLAGS.out_dir}/final_ema.pth')
    summary = {'run_name': FLAGS.run_name, 'best_val_miou': best_val_miou, 'best_epoch': best_epoch,
               'final_val_miou': last_val_miou, 'final_ema_val_miou': last_ema_miou,
               'final_checkpoint': 'final_ema.pth' if ema_model is not None else 'final.pth',
               'exp12': exp12_info,
               'final_lambdas': net.encoder.lambdas() if FLAGS.ccmc else [],
               'split_hash': run_config['split_hash']}
    with open(f'{FLAGS.out_dir}/run_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary))

if __name__ == '__main__':
    app.run(main)