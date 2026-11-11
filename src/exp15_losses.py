"""Exp15 training-only objectives. No inference alteration and no validation/test labels."""
import torch
import torch.nn.functional as F


def _gradient(gt_sorted):
    n = gt_sorted.numel()
    inter = gt_sorted.sum() - gt_sorted.cumsum(0)
    union = gt_sorted.sum() + torch.arange(1, n + 1, device=gt_sorted.device, dtype=gt_sorted.dtype) - gt_sorted.cumsum(0)
    jaccard = 1.0 - inter / union.clamp_min(1e-6)
    if n > 1:
        jaccard[1:] = jaccard[1:] - jaccard[:-1].clone()
    return jaccard


def cw_lovasz_loss(logits, labels, ignore_index=-1):
    """Class-wise Lovasz-Softmax for crop and weed; aggregation over batch pixels."""
    if logits.ndim != 4 or logits.shape[1] != 3 or labels.shape != (logits.shape[0], *logits.shape[-2:]):
        raise ValueError('Expected logits [B,3,H,W] and labels [B,H,W]')
    valid = (labels != ignore_index).reshape(-1)
    if not bool(valid.any()):
        return logits.float().sum() * 0
    probs = logits.float().softmax(dim=1).permute(0,2,3,1).reshape(-1, 3)[valid]
    gt = labels.reshape(-1)[valid]
    values = []
    for cl in (1,2):
        y = (gt == cl).to(probs.dtype)
        if not bool(y.any()):
            continue
        error = (y - probs[:,cl]).abs()
        sorted_error, order = torch.sort(error, descending=True)
        values.append(torch.dot(sorted_error, _gradient(y[order])))
    return torch.stack(values).mean() if values else logits.float().sum() * 0


def spectral_spatial_cw_loss(logits, labels, features, ignore_index=-1, scale=4, temperature=0.8, radius=2):
    """Train-only spectral ambiguity weighted CW discrimination at Crop/Weed interfaces.

    Per-image 5-band crop/weed class centroids (GT train-only), ambiguity weight
    based on relative centroid distances, interface focus, and binary CW CE.
    Downsampled to save memory. No label/centroid used at evaluation.
    """
    if logits.ndim != 4 or logits.size(1) != 3 or features.ndim != 4 or features.size(1) != 5:
        raise ValueError('Expected logits [B,3,H,W], features [B,5,H,W]')
    if labels.shape != (logits.size(0), *logits.shape[-2:]) or features.shape[0] != logits.shape[0]:
        raise ValueError('Incompatible feature/label/logit shapes')
    if scale < 1 or temperature <= 0 or radius < 1:
        raise ValueError('Invalid parameters')
    H, W = logits.shape[-2:]
    size = (max(1, H//scale), max(1, W//scale))
    z = F.interpolate(logits.float(), size=size, mode='bilinear', align_corners=False)
    x = F.interpolate(features.float(), size=size, mode='area').detach()
    y = F.interpolate(labels.unsqueeze(1).float(), size=size, mode='nearest').squeeze(1).long()
    losses = []
    for i in range(z.shape[0]):
        mask_c, mask_w = y[i] == 1, y[i] == 2
        if not (bool(mask_c.any()) and bool(mask_w.any())):
            continue
        # Normalize per-band within training image, limit degeneracy.
        xi = x[i]
        mu = xi.mean(dim=(1,2), keepdim=True)
        sd = xi.std(dim=(1,2), keepdim=True).clamp_min(1e-3)
        xn = ((xi - mu)/sd).clamp(-6,6)
        pc = xn[:, mask_c].mean(dim=1)[:,None,None]
        pw = xn[:, mask_w].mean(dim=1)[:,None,None]
        dc = (xn - pc).square().mean(0)
        dw = (xn - pw).square().mean(0)
        # Ambiguity high when distance to correct and opposite centroid is similar.
        dif = (dc - dw).abs()
        ambiguity = torch.exp(-dif / temperature).detach()
        k = 2*radius + 1
        cnear = F.max_pool2d(mask_c.float()[None,None], k, 1, radius)[0,0] > 0
        wnear = F.max_pool2d(mask_w.float()[None,None], k, 1, radius)[0,0] > 0
        near_interface = cnear & wnear
        valid = (mask_c | mask_w) & near_interface
        if not bool(valid.any()):
            continue
        # CW-specific log odds, background excluded to avoid conflicting vegetation/BG loss.
        binary_logits = z[i,1:3].permute(1,2,0)[valid]
        targets = (y[i][valid] - 1).long()
        loss_pixels = F.cross_entropy(binary_logits, targets, reduction='none')
        weights = (0.5 + ambiguity[valid]).clamp(0.5, 1.5)
        losses.append((loss_pixels * weights).sum() / weights.sum().clamp_min(1))
    return torch.stack(losses).mean() if losses else logits.float().sum() * 0
