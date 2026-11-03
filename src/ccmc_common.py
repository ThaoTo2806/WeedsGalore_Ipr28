"""Shared helpers for the CCMC-SR scripts (model flags, model building, checkpoint loading)."""

import torch
from absl import flags

from nets import CCMCSegFormer, SegFormer5Band
from nets.mcsd import MCSDSegFormer


def define_model_flags():
    flags.DEFINE_integer('in_channels', 5, '3 (RGB) or 5 (MSI)')
    flags.DEFINE_integer('num_classes', 3, 'must be 3 for the CCMC-SR study (bg/crop/weed)')
    flags.DEFINE_boolean('ccmc', True, 'checkpoint is a CCMCSegFormer (else SegFormer5Band)')
    flags.DEFINE_boolean('use_low_level_refine', False, 'LLR branch (LiteSent=True, L0/L3=False)')
    flags.DEFINE_boolean('use_multiscale_fusion', True, 'multi-scale decoder fusion')
    flags.DEFINE_enum('llr_source', 'raw', ['raw', 'none', 'f1'], 'refine-decoder input (SegFormer5Band)')
    flags.DEFINE_enum('prior_mode', 'none', ['none', 'ndvi', 'learned_veg', 'learned_weed',
                                             'learned_free', 'gt_veg', 'gt_weed'],
                      'prior mode used at TRAIN time (architecture)')
    flags.DEFINE_enum('injection', 'ccmc_sr', ['ccmc_sr', 'none', 'pre_sr_gate', 'post_sr_gate'],
                      'injection used at train time')
    flags.DEFINE_enum('prior_perturb', 'none', ['none', 'window_shuffle', 'global_shuffle'],
                      'prior perturbation used at train time')
    flags.DEFINE_list('ccmc_stages', ['2', '3'], 'modulated stages (1-based)')
    flags.DEFINE_float('ccmc_eps', 0.1, 'eps floor')
    flags.DEFINE_enum('head', 'bilinear', ['bilinear', 'mcsd', 'pixelshuffle', 'occ_conditioned'],
                      'prediction head')
    flags.DEFINE_integer('sinkhorn_iters', 5, 'Sinkhorn iterations (mcsd)')
    flags.DEFINE_float('mcsd_tau', 1.0, 'placement temperature (mcsd)')
    flags.DEFINE_boolean('spectral_guide', False, 'raw-band placement prior (mcsd)')


def build_model(F):
    if F.head != 'bilinear':
        return MCSDSegFormer(num_classes=F.num_classes, in_channels=F.in_channels,
                             use_multiscale_fusion=F.use_multiscale_fusion, head=F.head,
                             sinkhorn_iters=F.sinkhorn_iters, tau=F.mcsd_tau,
                             spectral_guide=F.spectral_guide)
    if F.ccmc:
        return CCMCSegFormer(num_classes=F.num_classes, in_channels=F.in_channels,
                             use_low_level_refine=F.use_low_level_refine,
                             use_multiscale_fusion=F.use_multiscale_fusion,
                             prior_mode=F.prior_mode, injection=F.injection,
                             prior_perturb=F.prior_perturb,
                             stages=[int(s) for s in F.ccmc_stages], eps=F.ccmc_eps)
    return SegFormer5Band(num_classes=F.num_classes, pretrained_backbone=True,
                          in_channels=F.in_channels,
                          use_low_level_refine=F.use_low_level_refine,
                          use_multiscale_fusion=F.use_multiscale_fusion,
                          llr_source=F.llr_source)


def load_checkpoint(net, path, device):
    state = torch.load(path, map_location=device, weights_only=False)
    if isinstance(state, dict) and 'model' in state and 'optimizer' in state:
        state = state['model']  # last.pth (full training state)
    state = {k: v for k, v in state.items()
             if not (k.endswith('total_ops') or k.endswith('total_params'))}  # thop buffers
    net.load_state_dict(state)
    return net


def name_parts(name):
    """'2023-05-25_0966' -> date '2023-05-25', tile '0966'."""
    return name[:10], name[11:]
