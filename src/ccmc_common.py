"""Shared helpers for the CCMC-SR scripts (model flags, model building, checkpoint loading)."""

import torch
from absl import flags

from nets import CCMCSegFormer, SegFormer5Band
from nets.mcsd import MCSDSegFormer
from nets.llr_subcell import (LLR_RESIDUAL_HEADS, LLR_SUBCELL_HEADS,
                               LLRResidualSubcellSegFormer, LLRSubcellSegFormer)
from nets.sc_llr import SC_LLR_HEADS, SCLLRSegFormer


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
    flags.DEFINE_enum('head', 'bilinear', ['bilinear', 'mcsd', 'pixelshuffle', 'occ_conditioned',
                                           *LLR_SUBCELL_HEADS, *LLR_RESIDUAL_HEADS, *SC_LLR_HEADS],
                      'prediction head')
    flags.DEFINE_integer('sinkhorn_iters', 5, 'Sinkhorn iterations (mcsd)')
    flags.DEFINE_float('mcsd_tau', 1.0, 'placement temperature (mcsd)')
    flags.DEFINE_boolean('spectral_guide', False, 'raw-band placement prior (mcsd)')
    flags.DEFINE_float('beta', 0.5, 'fixed occupancy guidance strength for LLR residual heads')
    flags.DEFINE_float('residual_alpha_init', 0.02, 'initial sigmoid gate for LLR residual heads')
    flags.DEFINE_integer('sc_radius', 1, 'SC-LLR local radius')
    flags.DEFINE_integer('sc_guide_dim', 8, 'SC-LLR spectral embedding size')
    flags.DEFINE_integer('sc_embed_dim', 16, 'SC-LLR semantic embedding size')
    flags.DEFINE_float('sc_alpha_init', 0.1, 'SC-LLR initial gate')
    flags.DEFINE_float('sc_alpha_lr', 0.01, 'SC-LLR gate learning rate')
    flags.DEFINE_float('sc_alpha_override', -1.0, 'SC-LLR evaluation gate override')


def build_model(F):
    if F.head in SC_LLR_HEADS:
        if F.ccmc or not F.use_low_level_refine or F.llr_source != 'raw' or F.spectral_guide:
            raise ValueError('SC-LLR requires raw non-CCMC LLR and spectral_guide=False')
        return SCLLRSegFormer(
            num_classes=F.num_classes, in_channels=F.in_channels,
            use_multiscale_fusion=F.use_multiscale_fusion, head=F.head,
            sc_radius=F.sc_radius, sc_guide_dim=F.sc_guide_dim,
            sc_embed_dim=F.sc_embed_dim, sc_alpha_init=F.sc_alpha_init,
            pretrained_backbone=True)
    if F.head in LLR_RESIDUAL_HEADS:
        if F.ccmc or not F.use_low_level_refine or F.llr_source != 'raw' or F.spectral_guide:
            raise ValueError('LLR residual heads require --ccmc=False --use_low_level_refine=True '
                             '--llr_source=raw --spectral_guide=False')
        return LLRResidualSubcellSegFormer(
            num_classes=F.num_classes, in_channels=F.in_channels,
            use_multiscale_fusion=F.use_multiscale_fusion, head=F.head,
            beta=F.beta, residual_alpha_init=F.residual_alpha_init,
            pretrained_backbone=True,
        )
    if F.head in LLR_SUBCELL_HEADS:
        if F.ccmc or not F.use_low_level_refine or F.llr_source != 'raw' or F.spectral_guide:
            raise ValueError('2x2 LLR heads require --ccmc=False --use_low_level_refine=True '
                             '--llr_source=raw --spectral_guide=False')
        return LLRSubcellSegFormer(
            num_classes=F.num_classes, in_channels=F.in_channels,
            use_multiscale_fusion=F.use_multiscale_fusion, head=F.head,
            sinkhorn_iters=F.sinkhorn_iters, tau=F.mcsd_tau,
            pretrained_backbone=True,
        )
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
