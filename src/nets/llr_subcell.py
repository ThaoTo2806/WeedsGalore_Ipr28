"""Two-by-two sub-cell prediction heads on the existing half-resolution LLR feature."""

import torch
from torch import nn
from torch.nn import functional as F

from .mcsd import occupancy_conditioned_subcell, sinkhorn_subcell
from .segformer import SegFormer5Band

LLR_SUBCELL_HEADS = ('pixelshuffle2', 'occ_conditioned2', 'mcsd2')
LLR_RESIDUAL_HEADS = ('llr_residual2', 'llr_residual_occ2')


class LLRSubcellSegFormer(SegFormer5Band):
    """SegFormer5Band with its raw five-band LLR branch and a 2x2 sub-cell head.

    The inherited ``decode_features`` output is H/2 when raw LLR is enabled. Each
    feature location therefore predicts four probabilities that pixel-shuffle to HxW.
    ``pixelshuffle2`` and the occupancy-conditioned heads share identical occupancy,
    placement, and auxiliary-loss parameters; they differ only in allocation rule.
    """

    def __init__(self, num_classes=3, in_channels=5, use_multiscale_fusion=True,
                 pretrained_backbone=True, head='occ_conditioned2', sinkhorn_iters=5,
                 tau=1.0, decoder_dim=128, encoder=None):
        if head not in LLR_SUBCELL_HEADS:
            raise ValueError(f'head must be one of {LLR_SUBCELL_HEADS}')
        super().__init__(
            num_classes=num_classes,
            decoder_dim=decoder_dim,
            pretrained_backbone=pretrained_backbone,
            in_channels=in_channels,
            use_low_level_refine=True,
            use_multiscale_fusion=use_multiscale_fusion,
            encoder=encoder,
            llr_source='raw',
        )
        self.head = head
        self.num_classes = num_classes
        self.sinkhorn_iters = sinkhorn_iters
        self.tau = tau
        self.subcell_scale = 2

        # Keep the inherited decoder and LLR initialization identical; replace only
        # the final H/2 classifier with occupancy and placement predictors.
        self.classifier = None
        self.occupancy = nn.Conv2d(decoder_dim, num_classes, kernel_size=1)
        self.placement = nn.Conv2d(decoder_dim, num_classes * 4, kernel_size=1)
        nn.init.zeros_(self.placement.weight)
        nn.init.zeros_(self.placement.bias)

    def forward(self, x, return_aux=False):
        features = self.decode_features(x, self.encoder(x))
        batch, _, height, width = features.shape
        scale = self.subcell_scale
        if (height * scale, width * scale) != tuple(x.shape[-2:]):
            raise ValueError(
                f'LLR feature grid {(height, width)} at scale {scale} must match '
                f'input size {tuple(x.shape[-2:])}'
            )

        occ_logits = self.occupancy(features)
        scores = self.placement(features).view(batch, self.num_classes, scale * scale,
                                                height, width)
        if self.head == 'pixelshuffle2':
            log_p = F.log_softmax(scores.float(), dim=1)
        else:
            log_q = F.log_softmax(occ_logits.float(), dim=1)
            if self.head == 'occ_conditioned2':
                log_p = occupancy_conditioned_subcell(log_q, scores, self.tau)
            else:
                log_p = sinkhorn_subcell(log_q, scores, self.sinkhorn_iters, self.tau)

        output = F.pixel_shuffle(
            log_p.reshape(batch, self.num_classes * scale * scale, height, width), scale
        )
        extras = {'occ_logits': occ_logits}
        return (output, extras) if return_aux else output


class LLRResidualSubcellSegFormer(SegFormer5Band):
    """Keep the bilinear LLR prediction and add a gated intra-cell residual."""

    def __init__(self, num_classes=3, in_channels=5, use_multiscale_fusion=True,
                 pretrained_backbone=True, head='llr_residual_occ2', beta=0.5,
                 residual_alpha_init=0.02, decoder_dim=128, encoder=None):
        if head not in LLR_RESIDUAL_HEADS:
            raise ValueError(f'head must be one of {LLR_RESIDUAL_HEADS}')
        if not 0.0 <= beta < 1.0:
            raise ValueError('beta must satisfy 0 <= beta < 1')
        if not 0.0 < residual_alpha_init < 1.0:
            raise ValueError('residual_alpha_init must satisfy 0 < value < 1')
        super().__init__(
            num_classes=num_classes,
            decoder_dim=decoder_dim,
            pretrained_backbone=pretrained_backbone,
            in_channels=in_channels,
            use_low_level_refine=True,
            use_multiscale_fusion=use_multiscale_fusion,
            encoder=encoder,
            llr_source='raw',
        )
        self.head = head
        self.num_classes = num_classes
        self.beta = beta
        self.subcell_scale = 2
        self.branch_scale = 1.0
        self.occupancy = nn.Conv2d(decoder_dim, num_classes, kernel_size=1)
        self.placement = nn.Conv2d(decoder_dim, num_classes * 4, kernel_size=1)
        nn.init.normal_(self.placement.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.placement.bias)
        alpha_logit = torch.logit(torch.tensor(residual_alpha_init))
        self.alpha_logit = nn.Parameter(alpha_logit)

    @property
    def alpha(self):
        return torch.sigmoid(self.alpha_logit)

    def set_branch_scale(self, value):
        self.branch_scale = float(value)

    def forward(self, x, return_aux=False):
        features = self.decode_features(x, self.encoder(x))
        batch, _, height, width = features.shape
        scale = self.subcell_scale
        if (height * scale, width * scale) != tuple(x.shape[-2:]):
            raise ValueError(
                f'LLR feature grid {(height, width)} at scale {scale} must match '
                f'input size {tuple(x.shape[-2:])}'
            )

        base = self.classifier(features)
        base = F.interpolate(base, size=x.shape[-2:], mode='bilinear', align_corners=False)
        occupancy_logits = self.occupancy(features)
        placement = self.placement(features).view(
            batch, self.num_classes, scale * scale, height, width)
        residual = placement - placement.mean(dim=2, keepdim=True)
        if self.head == 'llr_residual_occ2':
            q = F.softmax(occupancy_logits.float(), dim=1)
            gain = 1.0 + self.beta * (self.num_classes * q - 1.0)
            residual = residual * gain.unsqueeze(2)
        cell_residual = residual
        residual = F.pixel_shuffle(
            residual.reshape(batch, self.num_classes * scale * scale, height, width), scale
        )
        output = base + self.branch_scale * self.alpha * residual
        extras = {
            'occ_logits': occupancy_logits,
            'base_logits': base,
            'residual_logits': residual,
            'cell_residual': cell_residual,
            'alpha': self.alpha,
        }
        return (output, extras) if return_aux else output