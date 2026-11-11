"""Training-only vegetation/background boundary BCE loss for the 3-class segmentation model.

Uses ground-truth training labels only. No inference-time head or GT access.
"""
import torch
import torch.nn.functional as F


def vbb_loss(logits, labels, radius=3, ignore_index=-1):
    if logits.ndim != 4 or logits.size(1) != 3 or labels.shape != logits.shape[:1] + logits.shape[2:]:
        raise ValueError('expected logits [N,3,H,W], labels [N,H,W]')
    if radius < 1:
        raise ValueError('radius must be >=1')
    valid = labels != ignore_index
    veg = ((labels == 1) | (labels == 2)).float()
    x = veg.unsqueeze(1)
    k = radius * 2 + 1
    # Binary vegetation boundary: local neighborhood contains both vegetation and background.
    vmax = F.max_pool2d(x, k, stride=1, padding=radius)
    vmin = -F.max_pool2d(-x, k, stride=1, padding=radius)
    border = ((vmax - vmin) > 0).squeeze(1) & valid
    if not bool(border.any()):
        return logits.float().sum() * 0.0
    z = logits.float()
    veg_logit = torch.logsumexp(z[:, 1:3], dim=1) - z[:, 0]
    bce = F.binary_cross_entropy_with_logits(veg_logit, veg, reduction='none')
    # Mild hard-pixel weighting, detached to avoid unintended gradient through weights.
    p = torch.sigmoid(veg_logit).detach()
    hard = torch.where(veg.bool(), 1.0 - p, p)
    w = border.float() * (1.0 + hard)
    return (w * bce).sum() / w.sum().clamp_min(1.0)
