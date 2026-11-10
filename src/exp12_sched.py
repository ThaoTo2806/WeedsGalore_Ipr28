"""Torch-free schedule helpers for Exp12 (Fixed-Schedule Training Protocol, FSP)."""
import math


def lr_factor(step, total_steps, warmup_steps, schedule='cosine', min_ratio=0.0):
    """Multiplier of the base lr at optimizer step `step` (0-based)."""
    if schedule == 'const':
        return 1.0
    if schedule != 'cosine':
        raise ValueError("schedule must be 'const' or 'cosine'")
    if warmup_steps > 0 and step < warmup_steps:
        return (step + 1) / warmup_steps
    denom = max(1, total_steps - warmup_steps)
    progress = min(1.0, max(0.0, (step - warmup_steps) / denom))
    return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))


def ema_decay_at(step, decay):
    """EMA decay with the usual warm-up (1+n)/(10+n); 0 disables EMA."""
    if decay <= 0:
        return 0.0
    return min(decay, (1.0 + step) / (10.0 + step))
