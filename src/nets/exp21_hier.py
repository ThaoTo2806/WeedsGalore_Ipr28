"""Exp21 single-network 6-way fine supervision, evaluated as 3-way BG/Crop/Weed.

No ground truth is used in forward(). Fine logits are trained on official
train semantics; predictions aggregate weed subclass probabilities exactly.
"""
import torch
from torch import nn
from nets.segformer import SegFormer5Band


def fine_to_three_logits(fine):
    if fine.ndim != 4 or fine.shape[1] != 6:
        raise ValueError("Expected N,6,H,W fine logits")
    return torch.cat((fine[:, 0:2], torch.logsumexp(fine[:, 2:6], dim=1, keepdim=True)), dim=1)


class HierarchicalSegFormer(nn.Module):
    def __init__(self, num_classes=3, **kwargs):
        super().__init__()
        if num_classes != 3:
            raise ValueError("Exp21 always outputs exactly 3 evaluation classes")
        self.base = SegFormer5Band(num_classes=6, **kwargs)
        self.fine_logits = None  # graph only during training; not in checkpoint

    def forward(self, x):
        fine = self.base(x)
        self.fine_logits = fine if self.training else None
        return fine_to_three_logits(fine)
