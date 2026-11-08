"""SC-LLR experimental ablation: local aggregation on the existing raw LLR feature.

Specification: doc/README_EXP11_ERROR_BUDGET_SC_LLR.md sections 4.1-4.2.
Not a claim of a novel operator or of a guaranteed IoU improvement (the operator belongs
to the pixel-adaptive / guided-filtering family). No ensemble; one SegFormer encoder and
one classifier.

Arms (they differ ONLY in which logit terms enter the neighbourhood softmax; all three
instantiate identical parameters, so the shared initialisation is bit-identical):
    sc_llr_static : learned per-offset scalars only (content independent)       -> B1
    sc_llr_sem    : semantic cosine term only                                   -> M2
    sc_llr_spec   : spectral term AND semantic cosine term                      -> M1
Parameters not used by an arm receive no gradient (Adam skips them).

Fixed constants (never swept on validation): SIGMA_INIT, TAU_INIT. sigma and tau are
positive through softplus and remain learnable.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F

from .segformer import SegFormer5Band

SC_LLR_HEADS = ('sc_llr_static', 'sc_llr_sem', 'sc_llr_spec')

SIGMA_INIT = 1.0
TAU_INIT = 0.5          # cosine in [-1, 1]; tau=1 would make the weights almost uniform
_EPS = 1e-3
_MASKED = -1e4


def _inv_softplus(value):
    return math.log(math.expm1(value))


class SpectralContrastOperator(nn.Module):
    """Pixel-adaptive neighbourhood mixing on H/2 features, with valid-neighbour mask."""

    def __init__(self, channels, in_channels=5, radius=1, guide_dim=8,
                 embed_dim=16, alpha_init=0.1):
        super().__init__()
        if radius < 1 or radius > 3:
            raise ValueError('radius must be 1..3')
        if not 0 < alpha_init < 1:
            raise ValueError('alpha_init must be strictly between 0 and 1')
        self.radius = int(radius)
        self.spectral_projection = nn.Conv2d(in_channels, guide_dim, 1)    # Phi
        self.semantic_projection = nn.Conv2d(channels, embed_dim, 1)       # Psi
        self.offset_logits = nn.Parameter(torch.zeros((2 * radius + 1) ** 2))
        self.sigma_raw = nn.Parameter(torch.tensor(_inv_softplus(SIGMA_INIT - _EPS)))
        self.tau_raw = nn.Parameter(torch.tensor(_inv_softplus(TAU_INIT - _EPS)))
        self.alpha_logit = nn.Parameter(torch.tensor(math.log(alpha_init / (1 - alpha_init))))
        # inference-only controls, deliberately not checkpoint state
        self._alpha_override = None
        self._branch_scale = 1.0

    @property
    def sigma(self):
        return F.softplus(self.sigma_raw) + _EPS

    @property
    def tau(self):
        return F.softplus(self.tau_raw) + _EPS

    def set_alpha_override(self, value=None):
        if value is not None and not (0.0 <= float(value) <= 1.0):
            raise ValueError('alpha override must be in [0,1]')
        self._alpha_override = None if value is None else float(value)

    def set_branch_scale(self, value):
        if float(value) < 0:
            raise ValueError('branch scale must be >= 0')
        self._branch_scale = float(value)

    def effective_alpha(self, like):
        if self._alpha_override is not None:
            return like.new_tensor(self._alpha_override)
        return torch.sigmoid(self.alpha_logit) * self._branch_scale

    def _shift(self, tensor, dy, dx):
        """Out[i] = tensor[i + (dy,dx)], zero at out-of-bounds."""
        r = self.radius
        _, _, h, w = tensor.shape
        padded = F.pad(tensor, (r, r, r, r))
        return padded[:, :, r + dy:r + dy + h, r + dx:r + dx + w]

    def _offsets(self):
        r = self.radius
        return [(dy, dx) for dy in range(-r, r + 1) for dx in range(-r, r + 1)]

    def weights(self, features, image, mode):
        """Normalised B x K x H x W affinities (usable for synthetic tests)."""
        if mode not in SC_LLR_HEADS:
            raise ValueError(f'Unsupported mode {mode}')
        batch, _, h, w = features.shape
        if image.shape[0] != batch:
            raise ValueError('batch dimensions differ')
        if mode == 'sc_llr_spec':
            guides = F.interpolate(image, size=(h, w), mode='area')
            spectral = self.spectral_projection(guides).float()
            two_sigma_sq = 2.0 * self.sigma.float() ** 2
        if mode in ('sc_llr_sem', 'sc_llr_spec'):
            semantic = F.normalize(self.semantic_projection(features).float(),
                                   p=2, dim=1, eps=1e-6)
            tau = self.tau.float()
        valid = torch.ones((1, 1, h, w), device=features.device, dtype=torch.float32)
        scores = []
        for index, (dy, dx) in enumerate(self._offsets()):
            score = torch.zeros((batch, 1, h, w), device=features.device, dtype=torch.float32)
            if mode == 'sc_llr_static':
                score = score + self.offset_logits[index].float()
            if mode == 'sc_llr_spec':
                diff = spectral - self._shift(spectral, dy, dx)
                score = score - diff.square().sum(1, keepdim=True) / two_sigma_sq
            if mode in ('sc_llr_sem', 'sc_llr_spec'):
                score = score + (semantic * self._shift(semantic, dy, dx)).sum(1, keepdim=True) / tau
            score = score.masked_fill(self._shift(valid, dy, dx) <= 0, _MASKED)
            scores.append(score)
        return torch.softmax(torch.cat(scores, dim=1), dim=1)

    def forward(self, features, image, mode, return_aux=False):
        alpha = self.effective_alpha(features)
        if self._alpha_override is not None:
            off = self._alpha_override == 0.0
        else:
            off = self._branch_scale == 0.0
        if off:                               # operator off: exactly the plain LLR feature
            return (features, {'sc_alpha': alpha.detach()}) if return_aux else features
        affinity = self.weights(features, image, mode)
        filtered = 0.0
        for k, (dy, dx) in enumerate(self._offsets()):
            filtered = filtered + affinity[:, k:k + 1].to(features.dtype) * self._shift(features, dy, dx)
        out = features + alpha * (filtered - features)
        if not return_aux:
            return out
        with torch.no_grad():
            entropy = -(affinity.clamp_min(1e-12).log() * affinity).sum(1).mean()
        return out, {'sc_alpha': alpha.detach(), 'sc_weight_entropy': entropy}


class SCLLRSegFormer(SegFormer5Band):
    """One SegFormer5Band + raw LLR + a small local operator before the classifier."""

    def __init__(self, num_classes=3, in_channels=5, use_multiscale_fusion=True,
                 pretrained_backbone=True, decoder_dim=128, encoder=None,
                 head='sc_llr_spec', sc_radius=1, sc_guide_dim=8,
                 sc_embed_dim=16, sc_alpha_init=0.1):
        if head not in SC_LLR_HEADS:
            raise ValueError(f'head must be one of {SC_LLR_HEADS}')
        super().__init__(
            num_classes=num_classes, in_channels=in_channels,
            use_low_level_refine=True, llr_source='raw',
            use_multiscale_fusion=use_multiscale_fusion,
            pretrained_backbone=pretrained_backbone,
            decoder_dim=decoder_dim, encoder=encoder,
        )
        self.head = head
        # created AFTER the base model: base initialisation equals the plain LLR model
        self.sc_llr = SpectralContrastOperator(
            decoder_dim, in_channels, sc_radius, sc_guide_dim, sc_embed_dim, sc_alpha_init)

    def set_alpha_override(self, value=None):
        self.sc_llr.set_alpha_override(value)

    def set_branch_scale(self, value):
        """Activity audit hook: 0.0 removes the operator at inference."""
        self.sc_llr.set_branch_scale(value)

    def forward(self, x, return_aux=False):
        features = self.decode_features(x, self.encoder(x))
        expected = tuple((v + 1) // 2 for v in x.shape[-2:])
        if tuple(features.shape[-2:]) != expected:
            raise ValueError('Expected raw LLR at H/2. Check input dimensions and decoder.')
        if return_aux:
            refined, extras = self.sc_llr(features, x, self.head, return_aux=True)
        else:
            refined, extras = self.sc_llr(features, x, self.head), None
        logits = self.classifier(refined)
        logits = F.interpolate(logits, size=x.shape[-2:], mode='bilinear', align_corners=False)
        return (logits, extras) if return_aux else logits