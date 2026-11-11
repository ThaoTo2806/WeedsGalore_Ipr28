"""Exp14 ASCD: train-only asymmetrical spectral confusion loss.

No GT labels at inference. No new inference head, network, or learnable parameters.
All hyperparameters are pre-registered, not optimized against validation/test.
"""
import torch
import torch.nn.functional as F


def _standardize(x, valid, eps=1e-4):
    # Normalize 5 channels within the full training tile, no validation information.
    with torch.no_grad():
        x = x.float()
        mu = x.mean(dim=(-2, -1), keepdim=True)
        sd = x.std(dim=(-2, -1), keepdim=True, unbiased=False).clamp_min(eps)
        return ((x - mu) / sd).clamp(-6, 6)


def ascd_loss(logits, labels, features, *, hard_alpha=1.0,
              spectral_alpha=1.0, spectral_temperature=1.0,
              interface_radius=3, ignore_index=-1):
    """Conditional CW loss with direction, interface/interior, spectral ambiguity.

    - Crop and weed are balanced by per-class conditional CE (two-class CW softmax).
    - Pixels close to *opposite plant class* are pooled separately from interiors.
    - Wrong-plant-class confidence and spectral confusion (per-image normalized
      five-band GT *training* prototypes) are detached, capped weight multipliers.
    - If no paired class is present in tile, spectral component falls back to 1.
    """
    if logits.ndim != 4 or logits.shape[1] != 3:
        raise ValueError('logits must be [B,3,H,W]')
    if labels.shape != logits.shape[:1] + logits.shape[2:]:
        raise ValueError('label/logits spatial mismatch')
    if features.shape != (logits.shape[0], 5, logits.shape[2], logits.shape[3]):
        raise ValueError('features must be matching [B,5,H,W]')
    if hard_alpha < 0 or spectral_alpha < 0 or spectral_temperature <= 0 or interface_radius < 0:
        raise ValueError('invalid ASCD settings')
    if ignore_index in (1, 2):
        raise ValueError('ignore_index cannot be a plant class')
    plant = (labels == 1) | (labels == 2)
    if not bool(plant.any()):
        return logits.float().sum() * 0.

    # Force stable float32 computation independent of model AMP.
    cw_logits = logits[:, 1:3].float()
    target = torch.where(plant, labels - 1, torch.zeros_like(labels)).long()
    ce = F.cross_entropy(cw_logits, target, reduction='none')

    with torch.no_grad():
        # Opposite-class conditional confusion, separated by GT direction.
        probs = F.softmax(cw_logits, dim=1)
        wrong = 1.0 - probs.gather(1, target.unsqueeze(1)).squeeze(1)
        weight = 1.0 + hard_alpha * wrong

        # Spectral ambiguity, per-image prototypes from TRAIN labels only.
        if spectral_alpha > 0:
            z = _standardize(features, plant)
            for b in range(logits.shape[0]):
                crop = labels[b] == 1
                weed = labels[b] == 2
                # Avoid unreliable estimates when a class is almost absent.
                if crop.sum().item() < 16 or weed.sum().item() < 16:
                    continue
                mu_c = z[b, :, crop].mean(dim=1)
                mu_w = z[b, :, weed].mean(dim=1)
                # Distances computed for all pixels but only used on plant pixels.
                d_c = ((z[b] - mu_c[:, None, None]) ** 2).mean(dim=0)
                d_w = ((z[b] - mu_w[:, None, None]) ** 2).mean(dim=0)
                d_own = torch.where(crop, d_c, d_w)
                d_other = torch.where(crop, d_w, d_c)
                spectral_confusion = torch.sigmoid(
                    ((d_own - d_other) / spectral_temperature).clamp(-20., 20.))
                weight[b] *= 1.0 + spectral_alpha * spectral_confusion

        # Distinct crop-weed interface only (not soil/vegetation boundary).
        if interface_radius:
            k = 2 * interface_radius + 1
            crop_zone = F.max_pool2d((labels == 1).float().unsqueeze(1),
                                     k, stride=1, padding=interface_radius).squeeze(1) > 0
            weed_zone = F.max_pool2d((labels == 2).float().unsqueeze(1),
                                     k, stride=1, padding=interface_radius).squeeze(1) > 0
            interface = ((labels == 1) & weed_zone) | ((labels == 2) & crop_zone)
        else:
            interface = torch.zeros_like(plant)

    # Balance class direction AND interface/interior where present.
    pieces = []
    for class_id in (1, 2):
        own = labels == class_id
        for region in (interface, ~interface):
            mask = own & region
            if bool(mask.any()):
                pieces.append((ce[mask] * weight[mask]).mean())
    return torch.stack(pieces).mean()
