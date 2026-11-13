"""Exp26 supervised foreground-boundary loss. No GT required at inference."""
import torch
import torch.nn.functional as F


def foreground_edge(labels, radius=3, ignore_index=-1):
    if labels.ndim != 3 or radius < 1:
        raise ValueError('Expected [B,H,W], radius>=1')
    valid = (labels != ignore_index) & (labels >= 0) & (labels < 3)
    fg = ((labels > 0) & valid).float().unsqueeze(1)
    k = 2 * radius + 1
    dil = F.max_pool2d(fg, k, 1, radius)
    ero = -F.max_pool2d(-fg, k, 1, radius)
    edge = (dil - ero).clamp(0,1).squeeze(1)
    # Ignore pixels do not contribute to gradients.
    return edge * valid.float()


def boundary_ce(logits, labels, class_weights=None, radius=3, ignore_index=-1):
    if logits.ndim != 4 or logits.shape[1] != 3 or tuple(logits.shape[0:1] + logits.shape[-2:]) != tuple(labels.shape):
        raise ValueError('Expected logits [B,3,H,W] and labels [B,H,W]')
    valid = (labels != ignore_index) & (labels >= 0) & (labels < 3)
    target = labels.masked_fill(~valid, 0).long()
    ce = F.cross_entropy(logits.float(), target, weight=class_weights, reduction='none')
    edge = foreground_edge(labels, radius, ignore_index)
    weight = edge * valid.float()
    if class_weights is not None:
        weight = weight * class_weights[target]
    return (ce * edge * valid).sum() / weight.sum().clamp_min(1.0)


def dice_loss(logits, labels, class_weights=None, ignore_index=-1):
    valid = (labels != ignore_index) & (labels >= 0) & (labels < 3)
    target = labels.masked_fill(~valid,0).long()
    probs = logits.float().softmax(1)*valid[:,None]
    onehot = F.one_hot(target,3).permute(0,3,1,2).float()*valid[:,None]
    inter = (probs*onehot).sum((0,2,3))
    total = (probs+onehot).sum((0,2,3))
    dice = (2*inter+1e-6)/(total+1e-6)
    if class_weights is None: return 1-dice.mean()
    w=class_weights.float()/class_weights.float().sum()
    return 1-(dice*w).sum()
