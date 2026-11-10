"""Exp12 helpers: input rescaling, EMA, test-time augmentation (TTA) and weed-bias decision rule.

TTA here is a flip/rotation average of the probabilities of ONE network (not an ensemble of models).
"""
import torch
from torch.nn import functional as F

TTA_VIEWS = {
    'none': [(0, 0)],
    'hflip': [(0, 0), (0, 1)],
    'flips': [(0, 0), (0, 1), (2, 1), (2, 0)],               # identity, h-flip, v-flip, 180 deg
    'd4': [(k, f) for k in range(4) for f in (0, 1)],        # the 8 symmetries of the square
}


def scaled_size(hw, scale):
    """Round to a multiple of 4 (decoder grid) so 600 * 1.5 = 900 stays valid."""
    return tuple(max(4, int(round(v * scale / 4.0)) * 4) for v in hw)


def rescale_batch(features, labels, binary_labels, scale):
    """Resize images (bilinear) and labels (nearest) for training at another input scale."""
    if scale == 1.0:
        return features, labels, binary_labels
    size = scaled_size(features.shape[-2:], scale)
    features = F.interpolate(features, size=size, mode='bilinear', align_corners=False)

    def nearest(t):
        return F.interpolate(t[:, None].float(), size=size, mode='nearest')[:, 0].to(t.dtype)
    return features, nearest(labels), nearest(binary_labels)


@torch.no_grad()
def ema_update(ema_model, model, decay):
    """In-place EMA of the whole state dict; non-float buffers are copied."""
    source = model.state_dict()
    for key, value in ema_model.state_dict().items():
        if value.dtype.is_floating_point:
            value.mul_(decay).add_(source[key].detach(), alpha=1.0 - decay)
        else:
            value.copy_(source[key])


def clean_state(model):
    """state_dict without THOP profiling buffers (keeps checkpoints loadable)."""
    return {k: v for k, v in model.state_dict().items()
            if not (k.endswith('total_ops') or k.endswith('total_params'))}


def _transform(x, k, flip):
    x = torch.rot90(x, k, dims=(2, 3))
    return torch.flip(x, dims=(3,)) if flip else x


def _inverse(x, k, flip):
    x = torch.flip(x, dims=(3,)) if flip else x
    return torch.rot90(x, -k, dims=(2, 3))


@torch.no_grad()
def predict_probs(forward_fn, x, scale=1.0, tta='none'):
    """Average class probabilities over the TTA views. forward_fn(x) -> (N, C, H, W) logits or
    log-probabilities (softmax of normalised log-probabilities returns the probabilities)."""
    if tta not in TTA_VIEWS:
        raise ValueError(f'tta must be one of {sorted(TTA_VIEWS)}')
    total, views = None, TTA_VIEWS[tta]
    for k, flip in views:
        view = _transform(x, k, flip)
        inp, _, _ = rescale_batch(view, view[:, 0], view[:, 0], scale)
        probs = F.softmax(forward_fn(inp).float(), dim=1)
        if probs.shape[-2:] != view.shape[-2:]:
            probs = F.interpolate(probs, size=view.shape[-2:], mode='bilinear', align_corners=False)
        probs = _inverse(probs, k, flip)
        total = probs if total is None else total + probs
    return total / len(views)


@torch.no_grad()
def predict_labels(forward_fn, x, scale=1.0, tta='none', weed_bias=0.0):
    """Predicted class map. weed_bias adds a constant to the Weed log-probability (class 2); it is
    used only for the registered recall-precision control, never to tune a result."""
    probs = predict_probs(forward_fn, x, scale, tta)
    if weed_bias != 0.0:
        logp = torch.log(probs.clamp_min(1e-8))
        logp[:, 2] += weed_bias
        return logp.argmax(1)
    return probs.argmax(1)
