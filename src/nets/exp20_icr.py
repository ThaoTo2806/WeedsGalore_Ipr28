"""Exp20: single-network intra-image crop reference with reliability-based abstention.
No GT is accessed in forward; no ensembles, no external teacher.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from nets.segformer import SegFormer5Band


class ICRSegFormer(SegFormer5Band):
    def __init__(self, num_classes=3, in_channels=5, **kwargs):
        if num_classes != 3 or in_channels != 5:
            raise ValueError('ICR requires 3 classes and 5 bands')
        if not kwargs.get('use_low_level_refine', True):
            raise ValueError('ICR requires raw LLR at H/2')
        if kwargs.get('llr_source', 'raw') != 'raw':
            raise ValueError('ICR requires llr_source=raw')
        super().__init__(num_classes=num_classes, in_channels=in_channels, **kwargs)
        # Semantic similarity and spectral similarity both use trainable calibration,
        # not a prototype bank or second encoder.
        self.crop_project = nn.Conv2d(128, 32, kernel_size=1, bias=False)
        self.refine_delta = nn.Sequential(
            nn.Conv2d(4, 24, 1), nn.ReLU(inplace=True),
            nn.Conv2d(24, 2, 1)
        )
        # Zero-init residual: at step 0, exactly matches baseline predictions.
        nn.init.zeros_(self.refine_delta[-1].weight)
        nn.init.zeros_(self.refine_delta[-1].bias)
        self.max_delta = 1.5
        self.min_crop_fraction = 0.0005
        self.crop_confidence = 0.75

    def forward(self, x):
        feats = self.decode_features(x, self.encoder(x))
        coarse = self.classifier(feats)
        b, c, h, w = coarse.shape
        # Derive crop reference exclusively from this model's current predictions.
        with torch.no_grad():
            probs = torch.softmax(coarse.float(), dim=1)
            crop_prob = probs[:, 1:2]
            weed_prob = probs[:, 2:3]
            confident = (crop_prob >= self.crop_confidence).float()
            weight = crop_prob * confident
            support = confident.flatten(1).mean(1).view(b, 1, 1, 1)
            reliability = (support / self.min_crop_fraction).clamp(0, 1)
            # A low-confidence image contributes no reference-based correction.
            weight_sum = weight.sum((2,3), keepdim=True).clamp_min(1e-6)
        sem = F.normalize(self.crop_project(feats).float(), dim=1, eps=1e-6)
        prototype = (sem * weight).sum((2,3), keepdim=True) / weight_sum
        sem_sim = (sem * F.normalize(prototype, dim=1, eps=1e-6)).sum(1, keepdim=True)
        # Physical spectral comparison in 5 original input bands.
        bands = F.interpolate(x.float(), size=(h,w), mode='area')
        spec_mu = (bands * weight).sum((2,3),keepdim=True) / weight_sum
        spec_sim = F.cosine_similarity(bands, spec_mu, dim=1, eps=1e-6).unsqueeze(1)
        margin = crop_prob - weed_prob
        vegetation = crop_prob + weed_prob
        evidence = torch.cat((sem_sim, spec_sim, margin, vegetation), dim=1)
        delta = torch.tanh(self.refine_delta(evidence.to(feats.dtype))).float()
        # Crop vs Weed contrast only. Do not directly alter background logit.
        # Opposite crop/weed residual conserves a class-specific discrimination axis.
        crop_change = (delta[:, 0:1] - delta[:, 1:2]) * self.max_delta * reliability
        correction = torch.cat((torch.zeros_like(crop_change), crop_change, -crop_change), dim=1)
        logits = coarse.float() + correction
        return F.interpolate(logits, size=x.shape[-2:], mode='bilinear', align_corners=False)
