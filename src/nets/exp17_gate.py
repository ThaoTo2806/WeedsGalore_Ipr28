"""Exp17: single-model, correction-selective spectral/interior gated point head.

Research hypothesis, NOT established novelty. Train labels ONLY enter optional safety loss.
Inspired by PointRend (CVPR 2020) and SegFix (ECCV 2020).
"""
import torch
from torch import nn
import torch.nn.functional as F
from nets.exp16_point import SpectralPointRefineSegFormer


class SpectralSafetyGateSegFormer(SpectralPointRefineSegFormer):
    def __init__(self, *, gate_hidden=48, **kwargs):
        super().__init__(**kwargs)
        f = self.classifier.in_channels
        bands = kwargs.get('in_channels', 5)
        classes = kwargs.get('num_classes', 3)
        assert classes == 3 and bands == 5, 'Exp17 requires 5-band / 3-class setup'
        # Decoder + raw spectra + coarse logits + local spectral edge +
        # mismatch with stable 5x5 interior prediction + max-class confidence.
        self.safety_gate = nn.Sequential(
            nn.Linear(f + bands + classes + 3, gate_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(gate_hidden, 1),
        )
        nn.init.normal_(self.safety_gate[-1].weight, std=0.005)
        nn.init.constant_(self.safety_gate[-1].bias, -1.5)
        self._safety_cache = None

    def forward(self, x):
        b, _, h, w = x.shape
        refined = self.decode_features(x, self.encoder(x))
        coarse_half = self.classifier(refined)
        coarse = F.interpolate(coarse_half, size=(h, w), mode='bilinear', align_corners=False)
        prob = coarse.detach().float().softmax(dim=1)
        top = prob.topk(2, dim=1).values
        uncertainty = -(top[:, 0] - top[:, 1]).flatten(1)
        count = min(self.point_budget, h * w)
        if self.training:
            uncertain_n = min(count, max(1, count * 3 // 4))
            ids = torch.cat([
                uncertainty.topk(uncertain_n, dim=1).indices,
                torch.randint(h * w, (b, count - uncertain_n), device=x.device),
            ], dim=1)
        else:
            ids = uncertainty.topk(count, dim=1).indices
        index = ids.unsqueeze(1)
        yy = torch.div(ids, w, rounding_mode='floor')
        xx = ids % w
        grid = torch.stack([(xx.float() + 0.5) / w * 2 - 1,
                            (yy.float() + 0.5) / h * 2 - 1], dim=-1).unsqueeze(2)
        feature = F.grid_sample(refined, grid, align_corners=False).squeeze(-1).transpose(1, 2).float()
        raw = torch.gather(x.float().flatten(2), 2, index.expand(-1, x.shape[1], -1)).transpose(1, 2)
        base = torch.gather(coarse.float().flatten(2), 2, index.expand(-1, 3, -1)).transpose(1, 2)
        inputs = torch.cat([feature, raw, base], dim=-1)
        delta = self.point_mlp(inputs)
        # Classwise locally stable interior prediction, no ground truth.
        local_prob = F.avg_pool2d(prob, kernel_size=5, stride=1, padding=2)
        interior = torch.gather(local_prob.flatten(2), 2, index.expand(-1, 3, -1)).transpose(1, 2)
        mismatch = (interior - prob.flatten(2).gather(2, index.expand(-1, 3, -1)).transpose(1, 2)).abs().mean(-1, keepdim=True)
        # Local spectral discontinuity using raw 5-band input, scale normalized.
        raw_float = x.float()
        local_raw = F.avg_pool2d(raw_float, kernel_size=3, stride=1, padding=1)
        contrast = (raw_float - local_raw).abs().mean(1, keepdim=True)
        spectral_edge = contrast.flatten(2).gather(2, index).transpose(1, 2)
        confidence = top[:, 0].flatten(1).gather(1, ids).unsqueeze(-1)
        gate_input = torch.cat([inputs, spectral_edge, mismatch, confidence], dim=-1)
        gate = torch.sigmoid(self.safety_gate(gate_input))
        change = (gate * delta).transpose(1, 2).to(dtype=torch.float32)
        correction = torch.zeros_like(coarse, dtype=torch.float32).flatten(2)
        correction = correction.scatter(2, index.expand(-1, 3, -1), change)
        out = coarse.float() + correction.view(b, 3, h, w)
        self._safety_cache = (ids, base, delta, gate) if self.training else None
        return out


def exp17_safety_loss(net, target, *, margin=0.02):
    """Train-only counterfactual safety supervision; no labels in inference.

    Gate gets a supervised signal to avoid changes that raise per-point CE.
    Targets computed from the UNGATED candidate versus coarse, detached.
    We also discourage loss-increasing gated edits, especially at correct coarse points.
    """
    cache = getattr(net, '_safety_cache', None)
    if cache is None:
        raise RuntimeError('Exp17 safety cache is absent: call training forward first')
    ids, base, delta, gate = cache
    y = target.reshape(target.shape[0], -1).gather(1, ids).long()
    valid = (y >= 0) & (y < base.shape[-1])
    yc = y.clamp(0, base.shape[-1] - 1)
    baseline_loss = F.cross_entropy(base.float().transpose(1, 2), yc, reduction='none')
    candidate_loss = F.cross_entropy((base + delta).float().transpose(1, 2), yc, reduction='none')
    actual_loss = F.cross_entropy((base + gate * delta).float().transpose(1, 2), yc, reduction='none')
    # Counterfactual target: only open gate where proposed correction improves CE.
    desirable = ((baseline_loss.detach() - candidate_loss.detach()) > margin).float()
    gate_prob = gate.squeeze(-1).float().clamp(1e-5, 1-1e-5)
    gate_supervision = F.binary_cross_entropy(gate_prob, desirable, reduction='none')
    # Explicit harm penalty at originally correct points.
    correct = base.detach().argmax(-1) == yc
    harm = (actual_loss - baseline_loss).relu() * correct.float()
    valid_f = valid.float()
    norm = valid_f.sum().clamp_min(1.0)
    return ((0.15 * gate_supervision + 1.0 * harm) * valid_f).sum() / norm
