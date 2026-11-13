"""Exp27: spectral-aligned foreground-boundary consistency, training only.

GT specifies the trusted foreground-boundary neighborhood. Raw 5-band input
supplies a per-image spectral discontinuity signal, detached from optimization.
The prediction supplies differentiable foreground boundary strength.
All losses use GT on TRAIN; evaluation uses 5-band images only.
"""
import torch
import torch.nn.functional as F


def _grad_magnitude(z):
    # z: B,C,H,W; inexpensive isotropic finite differences
    dx = F.pad(z[..., :, 1:] - z[..., :, :-1], (0,1,0,0))
    dy = F.pad(z[..., 1:, :] - z[..., :-1, :], (0,0,0,1))
    return (dx.square() + dy.square() + 1e-6).sqrt()


def spectral_boundary_evidence(x, eps=1e-6):
    """Robust 5-band evidence [B,1,H,W] normalized to [0,1] per image.

    IMPORTANT: expects raw input used by backbone, not reflectance. Handles
    negative normalized values; this is gradient strength, not NDVI.
    """
    if x.ndim != 4 or x.shape[1] != 5:
        raise ValueError('Expected 5-band input [B,5,H,W]')
    with torch.no_grad():
        v = x.detach().float()
        # per-band robust range to avoid any one band dominating, preserve spectral diversity
        flat = v.flatten(2)
        lo = torch.quantile(flat, .05, dim=-1, keepdim=True).unsqueeze(-1)
        hi = torch.quantile(flat, .95, dim=-1, keepdim=True).unsqueeze(-1)
        v = ((v-lo)/(hi-lo).clamp_min(eps)).clamp(0.,1.)
        g = _grad_magnitude(F.avg_pool2d(v, 3, 1, 1)).mean(dim=1, keepdim=True)
        # Normalize gradient by per-image 95th percentile; avoid per-image thresholds
        q = torch.quantile(g.flatten(2), .95, dim=-1, keepdim=True).unsqueeze(-1)
        return (g/q.clamp_min(eps)).clamp(0,1)


def foreground_edge_gt(y, radius=3):
    if y.ndim != 3:
        raise ValueError('Expected targets [B,H,W]')
    valid = (y >= 0) & (y < 3)
    fg = ((y>0)&valid).float()[:,None]
    k=2*radius+1
    dil=F.max_pool2d(fg,k,1,radius)
    ero=-F.max_pool2d(-fg,k,1,radius)
    return (dil-ero).clamp(0,1)*valid[:,None]


def spectral_boundary_consistency(logits, y, x, radius=3, min_evidence=.25):
    """Spectral-gated GT-localized soft edge agreement.

    We target *relative spectral edge evidence* within GT boundary regions.
    Penalty is zero if no spectral reliable GT edge, preventing global smoothing.
    No auxiliary network, no change to inference cost.
    """
    if logits.shape[:2] != (y.shape[0],3) or logits.shape[-2:] != y.shape[-2:]:
        raise ValueError('Logits [B,3,H,W] and labels [B,H,W] required')
    if x.shape[0] != y.shape[0] or x.shape[1] != 5 or x.shape[-2:] != y.shape[-2:]:
        raise ValueError('Image [B,5,H,W] must match logits/labels')
    evidence=spectral_boundary_evidence(x)
    valid=((y>=0)&(y<3))[:,None]
    gt_band=foreground_edge_gt(y, radius)
    # mask high-evidence, GT-confirmed edges; no gradients through evidence or mask
    gate=(gt_band*valid*torch.clamp((evidence-min_evidence)/(1-min_evidence),0,1)).detach()
    p=logits.float().softmax(dim=1)[:,1:].sum(dim=1,keepdim=True)
    edge_pred=_grad_magnitude(F.avg_pool2d(p,3,1,1))
    # Bound predictive edge strength to [0,1] with fixed multiplier (no learnable scaling)
    edge_pred=(edge_pred*3.).clamp(0,1)
    # evidence soft target; only in GT boundary neighborhood
    numerator=(gate*(edge_pred-evidence).square()).sum()
    return numerator / gate.sum().clamp_min(1.), {
        'gate_fraction':float((gate>0).float().mean().detach()),
        'mean_evidence_on_gate':float((evidence*gate).sum().detach()/gate.sum().clamp_min(1.)),
    }
