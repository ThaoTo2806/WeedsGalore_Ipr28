"""MCSD: Mass-Conserving Sub-cell Decoding for small-object semantic segmentation.

Motivation (doc/REVIEW_REPORT.md, Exp01-Exp04):
  * the dominant loss of small weeds is at the PREDICTION stage: the decoder predicts one
    class distribution per 1/4-resolution cell (4x4 px) and bilinear upsampling + argmax
    cannot place a 2-3 px weed inside a cell (a dense 1/2-res refine decoder recovers
    +11.3 pp Small-Weed Recall without any extra input, at +27 GFLOPs);
  * Diag-D2: averaging dilutes small-weed evidence, max-like aggregation keeps it.

MCSD replaces "1x1 classifier at 1/4 res + bilinear x4" by two cheap heads on the same
1/4-res decoder feature and an exact, differentiable coupling:
  occupancy  q_c(cell)    : softmax over classes = predicted class FRACTIONS of each cell,
                            supervised with soft occupancy targets (avg-pooled one-hot GT);
  placement  s_c(cell, j) : score of class c at sub-pixel j (r*r sub-pixels per cell).
Inside every cell, the sub-pixel class probabilities P (C x r^2) are the entropic optimal
transport plan between class masses r^2 * q_c (rows) and one unit per sub-pixel (columns):
    P = diag(u) exp(s / tau) diag(v),   sum_j P_cj = r^2 q_c,   sum_c P_cj = 1,
solved with a few log-domain Sinkhorn iterations (columns are exact after the last step,
rows approximately). Hence a cell with predicted Weed fraction 0.15 hands ~2.4 sub-pixels
to Weed, placed where s_weed is highest, instead of losing them to argmax smoothing.

Controls (same heads and losses, only the coupling differs):
  head='pixelshuffle' : softmax over classes of pixel-shuffled placement scores (no mass
                        conservation; occupancy head trained with the same aux loss).
"""

import torch
from torch import nn
from torch.nn import functional as F

from .segformer import SegFormer5Band

HEADS = ('bilinear', 'mcsd', 'pixelshuffle')


def sinkhorn_subcell(log_q, scores, iters=5, tau=1.0):
    """Log-domain Sinkhorn per cell.

    log_q:  (B, C, h, w) log class fractions (log_softmax over C).
    scores: (B, C, R2, h, w) placement scores, R2 = r*r sub-pixels.
    Returns log P (B, C, R2, h, w) with sum_C P = 1 exactly and sum_R2 P ~= R2 * q.
    """
    r2 = scores.shape[2]
    log_k = scores.float() / tau
    log_rows = log_q.float().unsqueeze(2) + torch.log(torch.tensor(float(r2), device=log_k.device))
    log_v = torch.zeros_like(log_k[:, :1])          # (B, 1, R2, h, w): column scalings
    log_u = torch.zeros_like(log_k[:, :, :1])       # (B, C, 1, h, w):  row scalings
    for _ in range(iters):
        log_u = log_rows - torch.logsumexp(log_k + log_v, dim=2, keepdim=True)
        log_v = -torch.logsumexp(log_k + log_u, dim=1, keepdim=True)  # columns sum to 1
    return log_k + log_u + log_v


class MCSDSegFormer(SegFormer5Band):
    """SegFormer-B0 5-band (L0 encoder + multi-scale decoder) with a sub-cell prediction head.

    forward(x)                  -> per-pixel log-probabilities (B, C, H, W) for 'mcsd' and
                                   'pixelshuffle' (usable as logits by CE/Dice), logits for
                                   'bilinear'.
    forward(x, return_aux=True) -> (output, {'occ_logits': (B, C, h, w)})
    """

    def __init__(self, num_classes=3, in_channels=5, use_multiscale_fusion=True,
                 head='mcsd', sinkhorn_iters=5, tau=1.0, spectral_guide=False, decoder_dim=128):
        super().__init__(num_classes=num_classes, decoder_dim=decoder_dim, pretrained_backbone=True,
                         in_channels=in_channels, use_low_level_refine=False,
                         use_multiscale_fusion=use_multiscale_fusion)
        if head not in HEADS:
            raise ValueError(f'head must be one of {HEADS}')
        self.head = head
        self.num_classes = num_classes
        self.sinkhorn_iters = sinkhorn_iters
        self.tau = tau
        # New heads are created AFTER the base decoder (decoder init identical to L0 per seed).
        if head != 'bilinear':
            self.classifier = None
            self.occupancy = nn.Conv2d(decoder_dim, num_classes, 1)
            self.cell = 4  # MiT stage-1 stride: one decoder cell = 4x4 input pixels
            self.placement = nn.Conv2d(decoder_dim, num_classes * self.cell ** 2, 1)
            nn.init.zeros_(self.placement.weight)  # uniform placement at init
            nn.init.zeros_(self.placement.bias)
            self.guide = None
            if spectral_guide:
                # full-resolution placement prior from the raw bands (R,G,B,NIR,RE)
                self.guide = nn.Sequential(
                    nn.Conv2d(in_channels, 16, 3, padding=1, bias=False),
                    nn.BatchNorm2d(16),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(16, num_classes, 1),
                )
                nn.init.zeros_(self.guide[-1].weight)
                nn.init.zeros_(self.guide[-1].bias)

    def forward(self, x, return_aux=False):
        feats = self.decode_features(x, self.encoder(x))
        if self.head == 'bilinear':
            out = F.interpolate(self.classifier(feats), size=x.shape[-2:], mode='bilinear',
                                align_corners=False)
            return (out, {}) if return_aux else out
        r = self.cell
        b, _, h, w = feats.shape
        if (h * r, w * r) != tuple(x.shape[-2:]):
            raise ValueError(f'input {tuple(x.shape[-2:])} must equal {r} x decoder grid {(h, w)}')
        occ_logits = self.occupancy(feats)
        scores = self.placement(feats)                        # (B, C*r*r, h, w)
        if self.guide is not None:
            full = F.pixel_shuffle(scores, r) + self.guide(x)   # (B, C, H, W)
            scores = F.pixel_unshuffle(full, r)
        c = self.num_classes
        scores = scores.view(b, c, r * r, h, w)               # channel = c*r*r + (i*r + j)
        if self.head == 'mcsd':
            log_p = sinkhorn_subcell(F.log_softmax(occ_logits.float(), dim=1), scores,
                                     self.sinkhorn_iters, self.tau)
        else:  # pixelshuffle control: per-pixel softmax, no mass conservation
            log_p = F.log_softmax(scores.float(), dim=1)
        out = F.pixel_shuffle(log_p.reshape(b, c * r * r, h, w), r)
        return (out, {'occ_logits': occ_logits}) if return_aux else out


def occupancy_loss(occ_logits, labels, num_classes):
    """Soft cross-entropy between predicted cell fractions and GT class fractions."""
    onehot = F.one_hot(labels.clamp(min=0).long(), num_classes).permute(0, 3, 1, 2).float()
    valid = (labels >= 0).float().unsqueeze(1)
    target = F.adaptive_avg_pool2d(onehot * valid, occ_logits.shape[-2:])
    mass = target.sum(1, keepdim=True)
    target = target / mass.clamp(min=1e-6)
    loss = -(target * F.log_softmax(occ_logits.float(), dim=1)).sum(1)
    return (loss * (mass.squeeze(1) > 0)).mean()
