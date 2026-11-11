"""VCF-Context: one SegFormer model, one factorised output, multi-dilation
vegetation-conditioned crop-vs-weed context head. An untested hypothesis,
not an ensemble and not a demonstrated scientific novelty.
"""
import torch
from torch import nn
from torch.nn import functional as F
from .exp13 import VCFSegFormer


class DWContext(nn.Module):
    def __init__(self, dim, dilation):
        super().__init__()
        self.path = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=dilation, dilation=dilation,
                      groups=dim, bias=False),
            nn.BatchNorm2d(dim), nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, 1, bias=False),
            nn.BatchNorm2d(dim), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.path(x)


class VCFContextSegFormer(VCFSegFormer):
    """Single VCF network; replace crop/weed branch by residual multi-scale context."""
    def __init__(self, num_classes=3, in_channels=5, use_multiscale_fusion=True,
                 pretrained_backbone=True, decoder_dim=128, backbone='b0', encoder=None):
        super().__init__(num_classes=num_classes, in_channels=in_channels,
                         use_multiscale_fusion=use_multiscale_fusion,
                         pretrained_backbone=pretrained_backbone,
                         decoder_dim=decoder_dim, backbone=backbone, encoder=encoder)
        d = decoder_dim
        self.head = 'vcfctx'
        self.cw_branch = nn.ModuleList([DWContext(d, k) for k in (1, 3, 5)])
        self.cw_merge = nn.Sequential(
            nn.Conv2d(4 * d, d, 1, bias=False),
            nn.BatchNorm2d(d), nn.ReLU(inplace=True),
        )

    def forward(self, x, return_aux=False):
        feats = self.decode_features(x, self.encoder(x))
        v = self.veg_head(feats)
        gated = feats * torch.sigmoid(v)
        context = self.cw_merge(torch.cat(
            [gated] + [branch(gated) for branch in self.cw_branch], dim=1
        ))
        c = self.cw_head(gated + context)
        size = x.shape[-2:]
        v = F.interpolate(v.float(), size=size, mode='bilinear', align_corners=False)
        c = F.interpolate(c.float(), size=size, mode='bilinear', align_corners=False)
        log_v = F.logsigmoid(v)
        out = torch.cat((F.logsigmoid(-v),
                         log_v + F.logsigmoid(-c),
                         log_v + F.logsigmoid(c)), dim=1)
        return (out, {}) if return_aux else out
