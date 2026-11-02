"""CCMC-SR: Class-Conditioned Mass-Conserving Spatial Reduction for SegFormer MiT-B0.

Plan reference: paper/README_CCMC_SR_v2_3_oracle_leakage_reviewed.md (sections 3, 6.2-6.4, 8.4).

The K/V spatial-reduction convolution (``attention.self.sr``) of the selected MiT stages
receives a re-weighted input. Inside every *full* SR window (kernel = stride = R):

    w      = eps + (1 - eps) * p                (p = class prior in [0, 1])
    w_norm = w / mean_window(w)                 (mean of w_norm per window == 1)
    X_mod  = X * (1 + lambda * (w_norm - 1)),   lambda = sigmoid(a) in (0, 1)

Positions outside full SR windows (dropped by the un-padded SR conv) keep factor 1.
Q and the residual path are untouched because only the SR-conv input is modified
(HF ``SegformerEfficientSelfAttention`` computes Q from the un-reshaped hidden states).

Controls implemented with the *same* prior head / aux loss / lambda:
  injection='none'          -> K1 (head + aux loss, prior not connected to SR)
  injection='pre_sr_gate'   -> K5 (gate before SR, image-level normalisation, NO window norm)
  injection='post_sr_gate'  -> K6 (gate on reduced K/V tokens AFTER the SR LayerNorm, before
                               the key/value projections; K/V-grid norm). A gate placed before
                               the per-token LayerNorm would be cancelled by it (scale
                               invariance), so it is applied after.
  prior_perturb='window_shuffle' -> K2 (prior permuted inside every SR window)
  prior_perturb='global_shuffle' -> K2b / placebo (prior permuted over the whole map)
  prior_mode='learned_free' -> K3 (head trained only through the segmentation loss)

Prior modes:
  none          -> L0 (no prior, encoder identical to HF MiT-B0)
  ndvi          -> L1 (fixed NDVI prior, no head)
  learned_veg   -> L2 (head + aux loss on Crop U Weed occupancy)
  learned_weed  -> L3 (head + aux loss on Weed occupancy)
  learned_free  -> K3 (head, no aux loss)
  gt_veg/gt_weed -> C2/C3 (oracle ceiling, GT LEAKAGE, diagnostic only)
"""

import math

import torch
from torch import nn
from torch.nn import functional as F

from .segformer import PretrainedMiTEncoder, SegFormer5Band


PRIOR_MODES = ('none', 'ndvi', 'learned_veg', 'learned_weed', 'learned_free', 'gt_veg', 'gt_weed')
LEARNED_MODES = ('learned_veg', 'learned_weed', 'learned_free')
GT_MODES = ('gt_veg', 'gt_weed')
OVERRIDE_MODES = ('uniform', 'gt_veg', 'gt_weed', 'ndvi')
INJECTIONS = ('ccmc_sr', 'none', 'pre_sr_gate', 'post_sr_gate')
PERTURBS = ('none', 'window_shuffle', 'global_shuffle')

WEED_LABEL = 2  # binary labels: 0 background, 1 crop, 2 weed


# ----------------------------------------------------------------------------------------
# Pure tensor operators (unit-tested on synthetic tensors, no HF dependency)
# ----------------------------------------------------------------------------------------

def full_window_extent(size, ratio):
    """Extent covered by full SR windows (SR conv: kernel=stride=ratio, padding=0)."""
    return (size // ratio) * ratio


def window_normalized_factor(prior, ratio, lam, eps):
    """CCMC-SR factor. prior: (B,1,H,W) in [0,1]. Returns fp32 factor (B,1,H,W).

    Inside full windows: 1 + lam * (w / mean_window(w) - 1); outside: exactly 1.
    """
    prior = prior.float()
    height, width = prior.shape[-2:]
    hc, wc = full_window_extent(height, ratio), full_window_extent(width, ratio)
    if hc == 0 or wc == 0:
        return torch.ones_like(prior)
    w = eps + (1.0 - eps) * prior[..., :hc, :wc]
    mu = F.avg_pool2d(w, kernel_size=ratio, stride=ratio, padding=0, ceil_mode=False)
    mu_up = mu.repeat_interleave(ratio, dim=2).repeat_interleave(ratio, dim=3)
    w_norm = w / mu_up
    delta = lam * (w_norm - 1.0)
    delta = F.pad(delta, (0, width - wc, 0, height - hc), value=0.0)
    return 1.0 + delta


def image_normalized_factor(prior, lam, eps):
    """Spatial gate WITHOUT window normalisation (K5/K6). Mean over the image is 1.

    Image-level normalisation keeps identity at a uniform prior (same as CCMC-SR) and
    matches the gate magnitude, so the only difference to CCMC-SR is window locality.
    """
    prior = prior.float()
    w = eps + (1.0 - eps) * prior
    w_norm = w / w.mean(dim=(2, 3), keepdim=True)
    return 1.0 + lam * (w_norm - 1.0)


def window_shuffle(prior, ratio, generator=None):
    """Permute prior values inside every full SR window (K2). Edges unchanged."""
    batch, channels, height, width = prior.shape
    hc, wc = full_window_extent(height, ratio), full_window_extent(width, ratio)
    if hc == 0 or wc == 0:
        return prior
    inner = prior[..., :hc, :wc]
    windows = F.unfold(inner, kernel_size=ratio, stride=ratio)  # (B, C*R*R, L)
    windows = windows.view(batch, channels, ratio * ratio, -1)
    noise = torch.rand(windows.shape, generator=generator, device='cpu').to(windows.device)
    order = noise.argsort(dim=2)
    windows = torch.gather(windows, 2, order).view(batch, channels * ratio * ratio, -1)
    inner = F.fold(windows, output_size=(hc, wc), kernel_size=ratio, stride=ratio)
    top = torch.cat([inner, prior[..., :hc, wc:]], dim=3)
    return torch.cat([top, prior[..., hc:, :]], dim=2)


def global_shuffle(prior, generator=None):
    """Permute prior values over the whole map, per image (K2b placebo)."""
    batch, channels, height, width = prior.shape
    flat = prior.reshape(batch, channels, -1)
    noise = torch.rand(flat.shape, generator=generator, device='cpu').to(flat.device)
    order = noise.argsort(dim=2)
    return torch.gather(flat, 2, order).view(batch, channels, height, width)


def class_mask_from_labels(labels, kind):
    """labels: (B,H,W) binary labels (0 bg, 1 crop, 2 weed). kind: 'weed' | 'veg'."""
    if kind == 'weed':
        mask = labels == WEED_LABEL
    elif kind == 'veg':
        mask = labels >= 1
    else:
        raise ValueError(f'unknown mask kind {kind}')
    return mask.float().unsqueeze(1)


def occupancy(labels, kind, size):
    """Soft occupancy target/prior: average pooling of the class mask to `size`."""
    return F.adaptive_avg_pool2d(class_mask_from_labels(labels, kind), size)


def ndvi_prior(x, size, red_idx=0, nir_idx=3):
    """Fixed NDVI prior clipped to [0,1] (L1). Input band order: R, G, B, NIR, RE."""
    red, nir = x[:, red_idx:red_idx + 1].float(), x[:, nir_idx:nir_idx + 1].float()
    ndvi = (nir - red) / (nir + red).abs().clamp_min(1e-6)
    return F.adaptive_avg_pool2d(ndvi.clamp(0.0, 1.0), size)


def aux_target(labels, prior_mode, size):
    if prior_mode == 'learned_weed':
        return occupancy(labels, 'weed', size)
    if prior_mode == 'learned_veg':
        return occupancy(labels, 'veg', size)
    return None


def corrupt_weed_labels(labels, spec, seed=0):
    """Corrupt the WEED part of a binary label map, for GT-Weed+noise oracle (N6).

    spec examples: 'dilate:5', 'erode:3', 'shift:8', 'drop:0.3' (drop fraction of weed
    connected components), 'none'. Returns a new label map; crop pixels are kept unless
    overwritten by dilated weed. Only used to build the PRIOR, never the evaluation labels.
    """
    if spec in (None, '', 'none'):
        return labels
    kind, _, value = spec.partition(':')
    weed = (labels == WEED_LABEL).float().unsqueeze(1)
    if kind == 'dilate':
        k = int(value)
        weed = F.max_pool2d(weed, 2 * k + 1, stride=1, padding=k)
    elif kind == 'erode':
        k = int(value)
        weed = 1.0 - F.max_pool2d(1.0 - weed, 2 * k + 1, stride=1, padding=k)
    elif kind == 'shift':
        k = int(value)
        weed = torch.roll(weed, shifts=(k, k), dims=(2, 3))
    elif kind == 'drop':
        import cv2
        import numpy as np
        frac = float(value)
        rng = np.random.default_rng(seed)
        out = []
        for b in range(weed.shape[0]):
            mask = weed[b, 0].cpu().numpy().astype(np.uint8)
            n, comp = cv2.connectedComponents(mask, connectivity=8)
            keep = np.ones(n, dtype=bool)
            if n > 1:
                keep[1:] = rng.random(n - 1) >= frac
            out.append(torch.from_numpy((keep[comp] & (mask > 0)).astype(np.float32)))
        weed = torch.stack(out).unsqueeze(1).to(labels.device)
    else:
        raise ValueError(f'unknown corruption spec {spec}')
    weed = weed.squeeze(1) > 0.5
    corrupted = torch.where(labels == WEED_LABEL, torch.zeros_like(labels), labels)
    return torch.where(weed, torch.full_like(labels, WEED_LABEL), corrupted)


# ----------------------------------------------------------------------------------------
# Modules
# ----------------------------------------------------------------------------------------

class PriorHead(nn.Module):
    """Weed/vegetation prior head on F1 (stage-1 output, stride 4).

    The output conv is zero-initialised so p == 0.5 everywhere at init -> w_norm == 1 ->
    the modulated model is exactly the baseline at step 0, while lambda > 0 keeps a
    non-zero gradient from the segmentation loss into the head.
    """

    def __init__(self, in_channels=32, hidden_channels=32):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
        )
        self.out = nn.Conv2d(hidden_channels, 1, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, feature1):
        return self.out(self.body(feature1))


class CCMCMiTEncoder(PretrainedMiTEncoder):
    """HF MiT-B0 (5-band) with prior-modulated K/V spatial reduction in chosen stages.

    State-dict keys of the HF encoder are unchanged (modulation is done with forward
    hooks on ``attention.self.sr``), so an L0 checkpoint loads into this encoder.
    """

    def __init__(self, in_channels=5, prior_mode='none', injection='ccmc_sr',
                 prior_perturb='none', stages=(2, 3), eps=0.1, lam_init=0.05,
                 learn_lambda=True, model_name='nvidia/mit-b0'):
        super().__init__(model_name=model_name, in_channels=in_channels)
        if prior_mode not in PRIOR_MODES:
            raise ValueError(f'prior_mode must be one of {PRIOR_MODES}')
        if injection not in INJECTIONS:
            raise ValueError(f'injection must be one of {INJECTIONS}')
        if prior_perturb not in PERTURBS:
            raise ValueError(f'prior_perturb must be one of {PERTURBS}')
        stages = tuple(int(s) for s in stages)
        if 1 in stages:
            raise ValueError('Prior is taken from the stage-1 OUTPUT: it cannot modulate '
                             'stage 1 (circular dependency). Use stages from {2,3,4}.')
        if not 0.0 < eps <= 1.0:
            raise ValueError('eps must be in (0, 1]')
        self.prior_mode = prior_mode
        self.injection = injection
        self.prior_perturb = prior_perturb
        self.stages = stages
        self.stage_ids = [s - 1 for s in stages]
        self.eps = float(eps)
        self.sr_ratios = list(self.config.sr_ratios)
        self.hidden_sizes = list(self.config.hidden_sizes)
        for sid in self.stage_ids:
            if self.sr_ratios[sid] <= 1 and injection != 'none' and prior_mode != 'none':
                raise ValueError(f'stage {sid + 1} has sr_ratio 1 (no SR conv)')
        self.uses_prior = prior_mode != 'none'
        self.injects = self.uses_prior and injection != 'none'
        self.lam_logit = None
        if self.uses_prior:
            logit = math.log(lam_init / (1.0 - lam_init))
            self.lam_logit = nn.Parameter(torch.full((len(self.stage_ids),), logit),
                                          requires_grad=learn_lambda)
        self._ctx = None
        self._hook_handles = []
        if self.injects:
            self._register_hooks()

    # -- hooks ---------------------------------------------------------------------------
    def _sr_modules(self, stage_id):
        modules = []
        for layer in self.encoder.block[stage_id]:
            attention = layer.attention.self
            sr = getattr(attention, 'sr', None)
            if sr is None:
                sr = getattr(attention, 'sequence_reduction', None)
            if not isinstance(sr, nn.Conv2d):
                raise RuntimeError(f'Cannot find SR conv in stage {stage_id + 1}; check the '
                                   'pinned transformers version (expected attention.self.sr).')
            modules.append(sr)
        return modules

    def _register_hooks(self):
        for k, sid in enumerate(self.stage_ids):
            for j, sr in enumerate(self._sr_modules(sid)):
                if self.injection in ('ccmc_sr', 'pre_sr_gate'):
                    handle = sr.register_forward_pre_hook(self._make_pre_hook(k, sid))
                    self._hook_handles.append(handle)
                else:
                    norm = self.encoder.block[sid][j].attention.self.layer_norm
                    self._hook_handles.append(
                        sr.register_forward_hook(self._make_sr_record_hook(k, sid)))
                    self._hook_handles.append(
                        norm.register_forward_hook(self._make_post_norm_hook(k, sid)))

    def lambdas(self):
        if self.lam_logit is None:
            return []
        return torch.sigmoid(self.lam_logit.detach()).cpu().tolist()

    def _stage_prior(self, sid, size):
        prior = self._ctx['prior']
        if tuple(prior.shape[-2:]) != tuple(size):
            prior = F.adaptive_avg_pool2d(prior, size)
        if self.prior_perturb == 'window_shuffle':
            prior = window_shuffle(prior, self.sr_ratios[sid], generator=self._ctx['generator'])
        return prior

    def _factor(self, k, sid, in_size):
        cache = self._ctx['cache']
        if k in cache:
            return cache[k]
        ratio = self.sr_ratios[sid]
        lam = torch.sigmoid(self.lam_logit[k]).float()
        prior = self._stage_prior(sid, in_size)
        if self.injection == 'ccmc_sr':
            factor = window_normalized_factor(prior, ratio, lam, self.eps)
        elif self.injection == 'pre_sr_gate':
            factor = image_normalized_factor(prior, lam, self.eps)
        else:  # post_sr_gate: K/V grid of the SR conv (floor division, no padding)
            kv_prior = F.avg_pool2d(prior, kernel_size=ratio, stride=ratio, padding=0)
            factor = image_normalized_factor(kv_prior, lam, self.eps)
        cache[k] = factor
        return factor

    def _make_pre_hook(self, k, sid):
        def hook(module, args):
            if self._ctx is None:
                return None
            x = args[0]
            if x.dim() != 4 or x.shape[1] != self.hidden_sizes[sid]:
                raise RuntimeError(f'Unexpected SR input shape {tuple(x.shape)} at stage {sid + 1}')
            factor = self._factor(k, sid, x.shape[-2:])
            return (x * factor.to(x.dtype),) + tuple(args[1:])
        return hook

    def _make_sr_record_hook(self, k, sid):
        """K6 step 1: compute the K/V-grid factor from the SR input size (output untouched)."""
        def hook(module, args, output):
            if self._ctx is None:
                return None
            factor = self._factor(k, sid, args[0].shape[-2:])
            if tuple(factor.shape[-2:]) != tuple(output.shape[-2:]):
                raise RuntimeError(f'K/V grid mismatch {tuple(factor.shape)} vs {tuple(output.shape)}')
            return None
        return hook

    def _make_post_norm_hook(self, k, sid):
        """K6 step 2: scale the LayerNorm-ed K/V tokens (B, h*w, C) before key/value."""
        def hook(module, args, output):
            if self._ctx is None:
                return None
            factor = self._ctx['cache'][k]
            tokens = factor.flatten(2).transpose(1, 2)  # (B, h*w, 1)
            if tokens.shape[1] != output.shape[1]:
                raise RuntimeError(f'K/V token mismatch {tokens.shape[1]} vs {output.shape[1]}')
            return output * tokens.to(output.dtype)
        return hook

    # -- forward -------------------------------------------------------------------------
    def forward(self, x, prior_fn=None):
        """Mirror of HF SegformerEncoder.forward (transformers 4.45.x), plus prior capture."""
        encoder = self.encoder
        batch_size = x.shape[0]
        hidden_states = x
        features, extras = [], {}
        self._ctx = None
        try:
            for idx, (embedding, blocks, norm) in enumerate(
                    zip(encoder.patch_embeddings, encoder.block, encoder.layer_norm)):
                hidden_states, height, width = embedding(hidden_states)
                for block in blocks:
                    outputs = block(hidden_states, height, width)
                    hidden_states = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
                hidden_states = norm(hidden_states)
                hidden_states = hidden_states.reshape(batch_size, height, width, -1)
                hidden_states = hidden_states.permute(0, 3, 1, 2).contiguous()
                features.append(hidden_states)
                if idx == 0 and self.uses_prior and prior_fn is not None:
                    prior, extras = prior_fn(hidden_states)
                    generator = None
                    if not self.training:
                        generator = torch.Generator().manual_seed(0)  # deterministic eval
                    if self.prior_perturb == 'global_shuffle':
                        prior = global_shuffle(prior, generator=generator)
                    if self.injects:
                        self._ctx = {'prior': prior.float(), 'cache': {}, 'generator': generator}
        finally:
            self._ctx = None
        return tuple(features), extras


class CCMCSegFormer(SegFormer5Band):
    """SegFormer-B0 5-band + (optional) CCMC-SR. Decoder identical to SegFormer5Band.

    forward(x)                                     -> logits (B, K, H, W)
    forward(x, prior_mask=labels, return_aux=True) -> (logits, extras)
        extras: 'prior' (B,1,h1,w1) used prior, 'prior_logits' (if head exists).
    prior_override in OVERRIDE_MODES swaps the prior at inference on the SAME checkpoint
    (oracle intervention, section 6.3). GT overrides leak labels: diagnostic only.
    """

    def __init__(self, num_classes=3, in_channels=5, use_low_level_refine=False,
                 use_multiscale_fusion=True, prior_mode='learned_weed', injection='ccmc_sr',
                 prior_perturb='none', stages=(2, 3), eps=0.1, lam_init=0.05,
                 learn_lambda=True, prior_hidden=32):
        encoder = CCMCMiTEncoder(in_channels=in_channels, prior_mode=prior_mode,
                                 injection=injection, prior_perturb=prior_perturb,
                                 stages=stages, eps=eps, lam_init=lam_init,
                                 learn_lambda=learn_lambda)
        super().__init__(num_classes=num_classes, pretrained_backbone=True,
                         in_channels=in_channels, use_low_level_refine=use_low_level_refine,
                         use_multiscale_fusion=use_multiscale_fusion, encoder=encoder)
        self.prior_mode = prior_mode
        self.in_channels = in_channels
        # Created AFTER the decoder so decoder init is identical to L0 for the same seed.
        self.prior_head = None
        if prior_mode in LEARNED_MODES:
            self.prior_head = PriorHead(encoder.hidden_sizes[0], prior_hidden)

    def _build_prior(self, x, feature1, prior_mask, prior_override):
        mode = prior_override or self.prior_mode
        size = feature1.shape[-2:]
        extras = {}
        if self.prior_head is not None:
            extras['prior_logits'] = self.prior_head(feature1)
        if mode in LEARNED_MODES:
            if self.prior_head is None:
                raise RuntimeError(f'{mode} requires a prior head')
            prior = torch.sigmoid(extras['prior_logits'].float())
        elif mode == 'ndvi':
            if self.in_channels < 4:
                raise ValueError('NDVI prior needs R and NIR bands')
            prior = ndvi_prior(x, size)
        elif mode in GT_MODES:
            if prior_mask is None:
                raise ValueError(f'{mode} (oracle) needs prior_mask=labels')
            prior = occupancy(prior_mask, 'weed' if mode == 'gt_weed' else 'veg', size)
        elif mode == 'uniform':
            prior = torch.full((x.shape[0], 1) + tuple(size), 0.5, device=x.device)
        else:
            raise ValueError(f'unsupported prior mode {mode}')
        extras['prior'] = prior
        return prior, extras

    def forward(self, x, prior_mask=None, prior_override=None, return_aux=False):
        if prior_override is not None and prior_override not in OVERRIDE_MODES:
            raise ValueError(f'prior_override must be one of {OVERRIDE_MODES}')

        def prior_fn(feature1):
            return self._build_prior(x, feature1, prior_mask, prior_override)

        features, extras = self.encoder(x, prior_fn=prior_fn)
        logits = self.decode(x, features)
        if return_aux:
            return logits, extras
        return logits
