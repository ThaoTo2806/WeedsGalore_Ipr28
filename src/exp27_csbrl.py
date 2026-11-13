"""Counterfactual spectral reliability for 5-band crop/weed segmentation.

TRAIN only: GT defines a narrow foreground boundary band; a frozen teacher
estimates how perturbing RGB (0:3) and NIR/RE (3:5) changes vegetation
probabilities. No extra modules or forwards are used at inference.

This is an experimental hypothesis, not an established novelty claim.
"""
import torch
import torch.nn.functional as F
from exp27_sbcl import foreground_edge_gt, spectral_boundary_evidence, _grad_magnitude

ARMS = ('control', 'sbcl', 'csbrl', 'stage', 'random', 'rgb', 'ms')


def perturb_groups(x, attenuation=0.35):
    """Return two plausible band-group attenuations, without zeroing channels.

    x may be normalized; per-image per-channel spatial mean is valid either way.
    x_rgb: RGB preserved, NIR/RE attenuated toward their own image mean.
    x_ms: NIR/RE preserved, RGB attenuated toward their own image mean.
    """
    if x.ndim != 4 or x.shape[1] != 5 or not 0 < attenuation <= 1:
        raise ValueError('Expected [B,5,H,W] and attenuation in (0,1]')
    with torch.no_grad():
        a = x.detach().float()
        means = a.mean(dim=(-2, -1), keepdim=True)
        rgb = a.clone(); ms = a.clone()
        rgb[:, 3:] = (1-attenuation)*a[:, 3:] + attenuation*means[:, 3:]
        ms[:, :3] = (1-attenuation)*a[:, :3] + attenuation*means[:, :3]
        return rgb, ms


def teacher_reliability(teacher, x, alpha=5.0, attenuation=0.35):
    """Return reliability maps from frozen teacher; GT never used here.

    A low sensitivity to group perturbations is *hypothesized* to imply robust
    boundary evidence. This assumption must be tested against controls.
    """
    with torch.no_grad():
        xr, xm = perturb_groups(x, attenuation)
        def pv(t):
            z = teacher(t).float()
            if z.shape[-2:] != x.shape[-2:]:
                z = F.interpolate(z, size=x.shape[-2:], mode='bilinear', align_corners=False)
            return z.softmax(1)[:, 1:].sum(1, keepdim=True)
        p = pv(x.float())
        p_rgb = pv(xr)
        p_ms = pv(xm)
        d_rgb = (p - p_rgb).abs()
        d_ms = (p - p_ms).abs()
        return {
            'joint': torch.exp(-alpha * (d_rgb + d_ms)).clamp(0, 1),
            'rgb': torch.exp(-alpha * d_rgb).clamp(0, 1),
            'ms': torch.exp(-alpha * d_ms).clamp(0, 1),
            'sensitivity_rgb': d_rgb,
            'sensitivity_ms': d_ms,
        }


def boundary_loss(logits, target, evidence, reliability, radius=3, min_evidence=0.25):
    """Reliability-weighted spectral boundary consistency loss.

    Target labels are used ONLY on train; reward/proxy never selects checkpoints
    on TEST. The spectral gradient target is detached.
    """
    if logits.ndim != 4 or logits.shape[1] != 3 or logits.shape[0] != target.shape[0]:
        raise ValueError('Expected logits [B,3,H,W], target [B,H,W]')
    if logits.shape[-2:] != target.shape[-2:]:
        raise ValueError('Logits/target spatial mismatch')
    if evidence.shape != reliability.shape or evidence.shape != logits[:, :1].shape:
        raise ValueError('Reliability and evidence must be [B,1,H,W]')
    gt_band = foreground_edge_gt(target, radius)
    valid = ((target >= 0) & (target < 3))[:, None]
    spectral_gate = ((evidence-min_evidence)/(1-min_evidence)).clamp(0, 1)
    gate = (gt_band * valid * spectral_gate * reliability.clamp(0,1)).detach()
    p = logits.float().softmax(1)[:, 1:].sum(1, keepdim=True)
    pred_edge = (_grad_magnitude(F.avg_pool2d(p, 3, 1, 1))*3).clamp(0, 1)
    loss = (gate * (pred_edge-evidence.detach()).square()).sum() / gate.sum().clamp_min(1)
    return loss, {'gate_fraction': float((gate > 0).float().mean().item()),
                  'gate_mass': float(gate.mean().item())}


def csbrl_loss(logits, target, x, reliability_maps, arm, stage_ids=None, radius=3):
    if arm not in ARMS:
        raise ValueError(f'Unknown arm: {arm}')
    if arm == 'control':
        return logits.sum()*0, {'gate_fraction': 0.0, 'gate_mass': 0.0}
    evidence = spectral_boundary_evidence(x)
    if arm == 'sbcl':
        rel = torch.ones_like(evidence)
    elif arm == 'random':
        # Spatial shuffle of the SAME reliability values: matched gate mass
        # and distribution, but no alignment to each boundary pixel.
        if reliability_maps is None: raise ValueError('Random control needs teacher maps')
        rel = reliability_maps['joint'].flatten(2)
        rel = rel.gather(2, torch.rand(rel.shape,device=rel.device).argsort(dim=2))
        rel = rel.reshape_as(evidence)
    else:
        if reliability_maps is None:
            raise ValueError('Teacher reliability maps required for this arm')
        key = 'joint' if arm in ('csbrl', 'stage') else arm
        rel = reliability_maps[key]
        if arm == 'stage':
            if stage_ids is None or len(stage_ids) != target.shape[0]:
                raise ValueError('Stage ids required for stage arm')
            # Prespecified monotone schedule, NOT inferred location or tracking.
            # Later growth: downweight less of the uncertainty penalty.
            factor = torch.tensor([0.85, 0.95, 1.05, 1.15],
                                  dtype=rel.dtype, device=rel.device)
            rel = (rel * factor[stage_ids.long()].view(-1,1,1,1)).clamp(0,1)
    return boundary_loss(logits, target, evidence, rel, radius=radius)
