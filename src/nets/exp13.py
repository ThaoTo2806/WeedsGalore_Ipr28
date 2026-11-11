"""Exp13: Vegetation-Conditioned Factorised head (VCF) on top of the raw-LLR SegFormer.

Motivation (from Exp12 validation, not from the sub-cell experiments):
  * OCC2/MCSD2 (sub-cell placement) did not help: the remaining error is not boundary placement.
  * Large-weed recall@0.5 is only ~50% and Crop IoU ~77%: the dominant error is crop<->weed
    SEMANTIC confusion inside vegetation, not vegetation vs soil (bg IoU is 98.4%).

VCF factorises the 3-class posterior into
    P(bg)   = 1 - v
    P(crop) = v * (1 - c)
    P(weed) = v * c
v = vegetation probability (cheap head), c = P(weed | vegetation) from a DEDICATED
crop-vs-weed branch (dilated depthwise context) that only sees vegetation-gated features.
The output is a per-pixel log-probability map (B,3,H,W), usable directly by CE/Dice
(same convention as the MCSD heads), by predict_labels() and by TTA.

NOTE: hierarchical / factorised heads exist in the literature. Novelty of the COMBINATION
(raw 5-band LLR + vegetation-gated crop/weed branch for UAV multispectral data) must be
verified by a documented literature search before any claim.
"""
import torch
from torch import nn
from torch.nn import functional as F

from .segformer import SegFormer5Band

EXP13_HEADS = ('vcf',)


class VCFSegFormer(SegFormer5Band):
    def __init__(self, num_classes=3, in_channels=5, use_multiscale_fusion=True,
                 pretrained_backbone=True, decoder_dim=128, backbone='b0', encoder=None):
        if num_classes != 3:
            raise ValueError('VCF head is defined for bg/crop/weed (num_classes=3)')
        super().__init__(num_classes=num_classes, decoder_dim=decoder_dim,
                         pretrained_backbone=pretrained_backbone, in_channels=in_channels,
                         use_low_level_refine=True, use_multiscale_fusion=use_multiscale_fusion,
                         encoder=encoder, llr_source='raw', backbone=backbone)
        d = decoder_dim
        self.head = 'vcf'
        self.classifier = None
        self.veg_head = nn.Conv2d(d, 1, 1)
        self.cw_branch = nn.Sequential(
            nn.Conv2d(d, d, 3, padding=2, dilation=2, groups=d, bias=False),
            nn.BatchNorm2d(d), nn.ReLU(inplace=True),
            nn.Conv2d(d, d, 1, bias=False),
            nn.BatchNorm2d(d), nn.ReLU(inplace=True),
        )
        self.cw_head = nn.Conv2d(d, 1, 1)

    def forward(self, x, return_aux=False):
        feats = self.decode_features(x, self.encoder(x))
        v = self.veg_head(feats)
        c = self.cw_head(self.cw_branch(feats * torch.sigmoid(v)))
        size = x.shape[-2:]
        v = F.interpolate(v.float(), size=size, mode='bilinear', align_corners=False)
        c = F.interpolate(c.float(), size=size, mode='bilinear', align_corners=False)
        log_veg = F.logsigmoid(v)
        out = torch.cat([F.logsigmoid(-v),
                         log_veg + F.logsigmoid(-c),
                         log_veg + F.logsigmoid(c)], dim=1)
        return (out, {}) if return_aux else out
