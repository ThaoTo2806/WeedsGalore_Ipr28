"""Exp16 experimental sparse PointRend-style 5-band refinement.
Inspired by PointRend (Kirillov et al., CVPR 2020); not a reproduction.
No GT is consulted at inference. An independent single model is used.
"""
import torch
from torch import nn
import torch.nn.functional as F
from nets.segformer import SegFormer5Band


class SpectralPointRefineSegFormer(SegFormer5Band):
    def __init__(self, *, point_budget=4096, point_hidden=64, **kwargs):
        super().__init__(**kwargs)
        self.point_budget = int(point_budget)
        if self.point_budget < 1:
            raise ValueError('point_budget must be positive')
        # The decoder feature map is 128 channels by default, fused with raw bands
        # and coarse 3-class logits. The residual is initialized at zero.
        features = self.classifier.in_channels
        self.point_mlp = nn.Sequential(
            nn.Linear(features + kwargs.get('in_channels', 5) + kwargs.get('num_classes', 3), point_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(point_hidden, kwargs.get('num_classes', 3)),
        )
        nn.init.zeros_(self.point_mlp[-1].weight)
        nn.init.zeros_(self.point_mlp[-1].bias)

    def forward(self, x):
        b, _, h, w = x.shape
        refined = self.decode_features(x, self.encoder(x))
        coarse_half = self.classifier(refined)
        coarse = F.interpolate(coarse_half, size=(h, w), mode='bilinear', align_corners=False)
        prob = coarse.detach().float().softmax(dim=1)
        # Margin-based uncertainty between two most plausible classes
        top = prob.topk(2, dim=1).values
        uncertainty = -(top[:, 0] - top[:, 1]).flatten(1)
        count = min(self.point_budget, h * w)
        if self.training:
            # 75% uncertain and 25% uniform: prevent sparse training collapse
            uncertain_n = min(count, max(1, count * 3 // 4))
            ids_uncertain = uncertainty.topk(uncertain_n, dim=1).indices
            ids_random = torch.randint(h * w, (b, count - uncertain_n), device=x.device)
            ids = torch.cat([ids_uncertain, ids_random], dim=1)
        else:
            ids = uncertainty.topk(count, dim=1).indices
        # Aligned sampling from full-resolution bands and coarser decoder features.
        y, xx = torch.div(ids, w, rounding_mode='floor'), ids % w
        grid = torch.stack([(xx.float() + .5) / w * 2 - 1,
                            (y.float() + .5) / h * 2 - 1], dim=-1).unsqueeze(2)
        point_feat = F.grid_sample(refined, grid, align_corners=False).squeeze(-1).transpose(1, 2)
        raw_feat = torch.gather(x.flatten(2), 2, ids.unsqueeze(1).expand(-1, x.shape[1], -1)).transpose(1, 2)
        base_logits = torch.gather(coarse.flatten(2), 2,
                                   ids.unsqueeze(1).expand(-1, coarse.shape[1], -1)).transpose(1, 2)
        delta = self.point_mlp(torch.cat([point_feat.float(), raw_feat.float(), base_logits.float()], dim=-1))
        # Sparse differentiable replacement; operates strictly on x, never labels.
        correction = torch.zeros_like(coarse, dtype=torch.float32).flatten(2)
        correction = correction.scatter(2, ids.unsqueeze(1).expand(-1, delta.shape[-1], -1),
                                         delta.transpose(1, 2).to(dtype=correction.dtype))
        return coarse.float() + correction.reshape(b, -1, h, w)
