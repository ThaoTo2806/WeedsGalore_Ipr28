"""Train-only multispectral masked reconstruction with spectral-ratio consistency.

One MiT-B2 encoder; the disposable reconstructor is discarded before segmentation.
No semantic masks, validation pixels, or test files are read during SSL.
"""
import torch
from torch import nn
from torch.nn import functional as F


def make_masks(x, patch=16, spatial_p=0.55, band_p=0.25, generator=None):
    """Return corrupted image and Bx5xHxW Boolean supervision mask."""
    b, c, h, w = x.shape
    assert c == 5 and h % patch == 0 and w % patch == 0
    grid = torch.rand((b, 1, h // patch, w // patch), device=x.device, generator=generator)
    spatial = F.interpolate((grid < spatial_p).float(), size=(h, w), mode='nearest').bool()
    bands = torch.rand((b, c, 1, 1), device=x.device, generator=generator) < band_p
    # At least one masked spatial patch is overwhelmingly likely; band masks act globally.
    supervised = spatial | bands
    corrupted = x.masked_fill(supervised, 0.0)
    return corrupted, supervised


def spectral_ratio(x, numerator, denominator, eps=0.05):
    a, b = x[:, numerator], x[:, denominator]
    return (a - b) / (a + b + eps)


def ssl_loss(pred, target, mask, spectral_weight=0.20):
    """Masked full-band L1 + masked NDVI/NDRE relational consistency.

    Mask upsample/downsample must match prediction resolution. NDVI uses NIR/R;
    NDRE uses NIR/RE. All target values come from unlabeled training images.
    """
    assert pred.shape == target.shape == mask.shape
    m = mask.float()
    l1 = ((pred.float() - target.float()).abs() * m).sum() / m.sum().clamp_min(1)
    rel = 0.0
    for num, den in ((3, 0), (3, 4)):
        relevant = (mask[:, num] | mask[:, den]).float()
        a = spectral_ratio(pred.float(), num, den)
        b = spectral_ratio(target.float(), num, den)
        rel = rel + ((a - b).abs().clamp(max=5.0) * relevant).sum() / relevant.sum().clamp_min(1)
    return l1 + spectral_weight * rel / 2


class MaskedSpectralLearner(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        dim = encoder.dims[0]
        self.reconstructor = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1), nn.GELU(),
            nn.Conv2d(dim, 5, 1))

    def forward(self, x):
        feature1 = self.encoder(x)[0]
        return self.reconstructor(feature1)


def assert_train_only_dataset(dataset):
    assert dataset.split == 'train', 'SSL must use training images only'
    assert len(dataset) == 104, 'Expected sealed 104 train images'
