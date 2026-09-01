# The code in this file was originally taken from https://github.com/VainF/DeepLabV3Plus-Pytorch and is licensed under the MIT License.

import torch
from torch import nn
from torch.nn import functional as F

from .utils import _SimpleSegmentationModel

__all__ = ["DeepLabV3"]


class DeepLabV3(_SimpleSegmentationModel):
  """
    Implements DeepLabV3 model from
    `"Rethinking Atrous Convolution for Semantic Image Segmentation"
    <https://arxiv.org/abs/1706.05587>`_.
    Arguments:
        backbone (nn.Module): the network used to compute the features for the model.
            The backbone should return an OrderedDict[Tensor], with the key being
            "out" for the last feature map used, and "aux" if an auxiliary classifier
            is used.
        classifier (nn.Module): module that takes the "out" element returned from
            the backbone and returns a dense prediction.
        aux_classifier (nn.Module, optional): auxiliary classifier used during training
    """
  pass


class DeepLabHeadV3Plus(nn.Module):

  def __init__(self, in_channels, low_level_channels, num_classes, aspp_dilate=[12, 24, 36]):
    super(DeepLabHeadV3Plus, self).__init__()
    self.project = nn.Sequential(
        nn.Conv2d(low_level_channels, 48, 1, bias=False),
        nn.BatchNorm2d(48),
        nn.ReLU(inplace=True),
    )

    self.aspp = ASPP(in_channels, aspp_dilate)

    self.classifier = nn.Sequential(
        nn.Conv2d(304, 256, 3, padding=1, bias=False), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
        nn.Conv2d(256, num_classes, 1))
    self._init_weight()

  def forward(self, feature):
    low_level_feature = self.project(feature['low_level'])
    output_feature = self.aspp(feature['out'])

    output_feature = F.interpolate(
        output_feature, size=low_level_feature.shape[2:], mode='bilinear', align_corners=False)
    return self.classifier(torch.cat([low_level_feature, output_feature], dim=1))

  def _init_weight(self):
    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight)
      elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


class DeepLabHead(nn.Module):

  def __init__(self, in_channels, num_classes, aspp_dilate=[12, 24, 36]):
    super(DeepLabHead, self).__init__()

    self.classifier = nn.Sequential(
        ASPP(in_channels, aspp_dilate), nn.Conv2d(256, 256, 3, padding=1, bias=False), nn.BatchNorm2d(256),
        nn.ReLU(inplace=True), nn.Conv2d(256, num_classes, 1))
    self._init_weight()

  def forward(self, feature):
    return self.classifier(feature['out'])

  def _init_weight(self):
    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight)
      elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

class ChannelAttention(nn.Module):
  """Channel attention (as used in CBAM): learns 'what' is important along the channel axis."""

  def __init__(self, in_channels, reduction=16):
    super(ChannelAttention, self).__init__()
    self.avg_pool = nn.AdaptiveAvgPool2d(1)
    self.max_pool = nn.AdaptiveMaxPool2d(1)
    hidden_channels = max(in_channels // reduction, 8)
    self.mlp = nn.Sequential(
        nn.Conv2d(in_channels, hidden_channels, 1, bias=False),
        nn.ReLU(inplace=True),
        nn.Conv2d(hidden_channels, in_channels, 1, bias=False),
    )
    self.sigmoid = nn.Sigmoid()

  def forward(self, x):
    avg_out = self.mlp(self.avg_pool(x))
    max_out = self.mlp(self.max_pool(x))
    attn = self.sigmoid(avg_out + max_out)
    return x * attn

class SpatialAttention(nn.Module):
  """Spatial attention (as used in CBAM): learns 'where' is important along the spatial axes."""

  def __init__(self, kernel_size=7):
    super(SpatialAttention, self).__init__()
    self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
    self.sigmoid = nn.Sigmoid()

  def forward(self, x):
    avg_out = torch.mean(x, dim=1, keepdim=True)
    max_out, _ = torch.max(x, dim=1, keepdim=True)
    attn = self.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))
    return x * attn

class CBAM(nn.Module):
  """Convolutional Block Attention Module: sequential channel then spatial attention.
    Ref: "CBAM: Convolutional Block Attention Module" (Woo et al., 2018)."""

  def __init__(self, in_channels, reduction=16, spatial_kernel_size=7):
    super(CBAM, self).__init__()
    self.channel_attention = ChannelAttention(in_channels, reduction)
    self.spatial_attention = SpatialAttention(spatial_kernel_size)

  def forward(self, x):
    x = self.channel_attention(x)
    x = self.spatial_attention(x)
    return x

class DeepLabHeadV3PlusAttention(nn.Module):
  """DeepLabV3+ decoder head with a CBAM attention module inserted right after ASPP
    (i.e. between the multi-scale ASPP features and the decoder), so that the model
    can select the most informative channels/regions before fusing with the
    low-level features and reconstructing the segmentation mask.
    
    NOTE: CBAM is currently DISABLED (commented out below). The spectral-guided
    architecture (see TinySpectralEncoder / SpectralGate / SpectralGuidedBackbone below)
    replaces CBAM's role with a NIR/RE-driven Spectral Gate applied directly on the
    backbone features, per the project README. Re-enable by uncommenting
    `self.attention = CBAM(256)` and the call in forward()."""

  def __init__(self, in_channels, low_level_channels, num_classes, aspp_dilate=[12, 24, 36]):
    super(DeepLabHeadV3PlusAttention, self).__init__()
    self.project = nn.Sequential(
        nn.Conv2d(low_level_channels, 48, 1, bias=False),
        nn.BatchNorm2d(48),
        nn.ReLU(inplace=True),
    )

    self.aspp = ASPP(in_channels, aspp_dilate)
    #self.attention = CBAM(256)  # 256 == ASPP output channels

    self.classifier = nn.Sequential(
        nn.Conv2d(304, 256, 3, padding=1, bias=False), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
        nn.Conv2d(256, num_classes, 1))
    self._init_weight()

  def forward(self, feature):
    low_level_feature = self.project(feature['low_level'])
    output_feature = self.aspp(feature['out'])
    #output_feature = self.attention(output_feature)  # <- attention applied between ASPP and decoder

    output_feature = F.interpolate(
        output_feature, size=low_level_feature.shape[2:], mode='bilinear', align_corners=False)
    return self.classifier(torch.cat([low_level_feature, output_feature], dim=1))

  def _init_weight(self):
    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight)
      elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
class TinySpectralEncoder(nn.Module):
  """Extremely lightweight encoder for the NIR + Red-Edge bands.
    A depthwise conv (spatial context per band) followed by a 1x1 pointwise conv
    (cheap cross-band mixing, e.g. NDVI/NDRE-like contrasts) extracts local
    spectral-spatial patterns,  Returns a full-resolution spectral feature map
    (NOT globally pooled) so that downstream gating can use *where* the
    vegetation-relevant signal is, not just a single whole-image summary.
    Cost: O(H*W*embed_channels) instead of routing NIR/RE through a full ResNet branch."""

  def __init__(self, in_channels=2, embed_channels=32):
    super(TinySpectralEncoder, self).__init__()
    self.dwconv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels, bias=False)
    self.pointwise = nn.Conv2d(in_channels, embed_channels, kernel_size=1, bias=False)
    self.bn = nn.BatchNorm2d(embed_channels)
    self.act = nn.ReLU(inplace=True)
    #self.pool = nn.AdaptiveAvgPool2d(1)
  def forward(self, x):
    x = self.dwconv(x)
    x = self.pointwise(x)
    x = self.bn(x)
    #x = self.act(x)
    #return self.pool(x)  # (B, embed_channels, 1, 1) spectral descriptor
    return self.act(x)  # (B, embed_channels, H, W) full-resolution spectral feature map
class SpectralGate(nn.Module):
  """Cross-modal channel + spatial gate with explicit alpha/beta balancing.

    This keeps the original spectral-guided idea:
      - channel gate: which RGB channels should be emphasized ("what")
      - spatial gate: where the vegetation-like response is located ("where")

    We expose two learnable parameters, alpha and beta, and normalize them so that
    alpha + beta = 1. This allows the model to decide how much emphasis to give
    to the per-channel gate versus the per-pixel gate during training. The gate is
    fused as a weighted geometric mean, which is smoother and more stable than a
    hard product of two gates.
    """

  def __init__(self, descriptor_channels, feature_channels):
    super(SpectralGate, self).__init__()
    self.channel_fc = nn.Conv2d(descriptor_channels, feature_channels, kernel_size=1, bias=True)
    self.spatial_conv = nn.Conv2d(descriptor_channels, 1, kernel_size=1, bias=True)
    self.sigmoid = nn.Sigmoid()

    # Explicit per-branch weights. They are normalized to sum to 1 before use.
    self.alpha = nn.Parameter(torch.tensor(0.5))
    self.beta = nn.Parameter(torch.tensor(0.5))

  def get_balanced_weights(self):
    denom = self.alpha + self.beta + 1e-8
    alpha = self.alpha / denom
    beta = self.beta / denom
    return alpha, beta

  def forward(self, spectral_map, feature):
    # Match the spectral feature map to the current feature map resolution.
    spectral_map = F.adaptive_avg_pool2d(spectral_map, feature.shape[-2:])

    # Channel gate: global summary across spatial positions -> emphasize useful channels.
    channel_gate = self.sigmoid(self.channel_fc(F.adaptive_avg_pool2d(spectral_map, 1)))

    # Spatial gate: per-pixel modulation -> emphasize vegetation-like regions.
    spatial_gate = self.sigmoid(self.spatial_conv(spectral_map))

    alpha, beta = self.get_balanced_weights()
    alpha = alpha.to(device=channel_gate.device, dtype=channel_gate.dtype)
    beta = beta.to(device=channel_gate.device, dtype=channel_gate.dtype)

    # Weighted geometric fusion keeps both branches while letting the model adapt their
    # relative importance. This is usually more stable than a hard product with equal
    # weighting of the two gates.
    gate = torch.pow(channel_gate.clamp_min(1e-6), alpha) * torch.pow(spatial_gate.clamp_min(1e-6), beta)
    gate = gate.clamp_min(1e-6)

    return feature * gate
class SpectralGuidedBackbone(nn.Module):
  """Two-branch backbone described in the README:
      RGB (3 bands)      -> standard ResNet (IntermediateLayerGetter -> 'out'/'low_level')
      NIR + RE (2 bands) -> TinySpectralEncoder -> spatial spectral feature map -> SpectralGate
    The spectral feature map gates both the 'out' (ASPP input) and 'low_level' (decoder
    skip connection) RGB feature maps -- per-channel AND per-location -- so NIR/RE
    information reaches every stage of the decoder without ever passing through the
    (expensive) RGB backbone itself."""
  def __init__(self, rgb_backbone, out_channels, low_level_channels, spectral_embed=32):
    super(SpectralGuidedBackbone, self).__init__()
    self.rgb_backbone = rgb_backbone  # IntermediateLayerGetter -> {'out': ..., 'low_level': ...}
    self.spectral_encoder = TinySpectralEncoder(in_channels=2, embed_channels=spectral_embed)
    self.gate_out = SpectralGate(spectral_embed, out_channels)
    self.gate_low_level = SpectralGate(spectral_embed, low_level_channels)
  def forward(self, x):
    rgb, nir_re = x[:, :3], x[:, 3:5]
    features = self.rgb_backbone(rgb)
    spectral_map = self.spectral_encoder(nir_re)  # (B, spectral_embed, H, W), full input resolution
    features['out'] = self.gate_out(spectral_map, features['out'])
    features['low_level'] = self.gate_low_level(spectral_map, features['low_level'])
    return features
class LiteASPPConv(nn.Sequential):
  """Depthwise-separable atrous conv branch: same receptive field as a standard 3x3
    atrous conv but far cheaper (depthwise 3x3 + pointwise 1x1 instead of a full dense
    3x3), used to build the lighter ("Lite") ASPP."""
  def __init__(self, in_channels, out_channels, dilation):
    modules = [
        nn.Conv2d(in_channels, in_channels, 3, padding=dilation, dilation=dilation, groups=in_channels, bias=False),
        nn.Conv2d(in_channels, out_channels, 1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    ]
    super(LiteASPPConv, self).__init__(*modules)
class LiteASPP(nn.Module):
  """Lightweight ASPP: depthwise-separable atrous branches + fewer output channels
    (128 by default instead of 256) to cut params/FLOPs relative to the standard ASPP,
    while keeping the same multi-scale-context design."""
  def __init__(self, in_channels, atrous_rates, out_channels=128):
    super(LiteASPP, self).__init__()
    modules = []
    modules.append(
        nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)))
    rate1, rate2, rate3 = tuple(atrous_rates)
    modules.append(LiteASPPConv(in_channels, out_channels, rate1))
    modules.append(LiteASPPConv(in_channels, out_channels, rate2))
    modules.append(LiteASPPConv(in_channels, out_channels, rate3))
    modules.append(ASPPPooling(in_channels, out_channels))
    self.convs = nn.ModuleList(modules)
    self.project = nn.Sequential(
        nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Dropout(0.1),
    )
  def forward(self, x):
    res = []
    for conv in self.convs:
      res.append(conv(x))
    res = torch.cat(res, dim=1)
    return self.project(res)
class DeepLabHeadV3PlusLite(nn.Module):
  """Same decoder structure as DeepLabHeadV3Plus but built from LiteASPP (depthwise-
    separable, `aspp_channels` output channels) instead of the standard ASPP (256
    channels) -- the reduced-compute decoder half of the spectral-guided architecture."""
  def __init__(self, in_channels, low_level_channels, num_classes, aspp_dilate=[12, 24, 36], aspp_channels=128):
    super(DeepLabHeadV3PlusLite, self).__init__()
    self.project = nn.Sequential(
        nn.Conv2d(low_level_channels, 48, 1, bias=False),
        nn.BatchNorm2d(48),
        nn.ReLU(inplace=True),
    )
    self.aspp = LiteASPP(in_channels, aspp_dilate, out_channels=aspp_channels)
    self.classifier = nn.Sequential(
        nn.Conv2d(aspp_channels + 48, aspp_channels, 3, padding=1, bias=False), nn.BatchNorm2d(aspp_channels),
        nn.ReLU(inplace=True), nn.Conv2d(aspp_channels, num_classes, 1))
    self._init_weight()
  def forward(self, feature):
    low_level_feature = self.project(feature['low_level'])
    output_feature = self.aspp(feature['out'])

    output_feature = F.interpolate(
        output_feature, size=low_level_feature.shape[2:], mode='bilinear', align_corners=False)
    return self.classifier(torch.cat([low_level_feature, output_feature], dim=1))

  def _init_weight(self):
    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight)
      elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


class AtrousSeparableConvolution(nn.Module):
  """ Atrous Separable Convolution
    """

  def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True):
    super(AtrousSeparableConvolution, self).__init__()
    self.body = nn.Sequential(
        # Separable Conv
        nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
            groups=in_channels),
        # PointWise Conv
        nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=bias),
    )

    self._init_weight()

  def forward(self, x):
    return self.body(x)

  def _init_weight(self):
    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight)
      elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


class ASPPConv(nn.Sequential):

  def __init__(self, in_channels, out_channels, dilation):
    modules = [
        nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    ]
    super(ASPPConv, self).__init__(*modules)


class ASPPPooling(nn.Sequential):

  def __init__(self, in_channels, out_channels):
    super(ASPPPooling, self).__init__(
        nn.AdaptiveAvgPool2d(1), nn.Conv2d(in_channels, out_channels, 1, bias=False), nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True))

  def forward(self, x):
    size = x.shape[-2:]
    x = super(ASPPPooling, self).forward(x)
    return F.interpolate(x, size=size, mode='bilinear', align_corners=False)


class ASPP(nn.Module):

  def __init__(self, in_channels, atrous_rates):
    super(ASPP, self).__init__()
    out_channels = 256
    modules = []
    modules.append(
        nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)))

    rate1, rate2, rate3 = tuple(atrous_rates)
    modules.append(ASPPConv(in_channels, out_channels, rate1))
    modules.append(ASPPConv(in_channels, out_channels, rate2))
    modules.append(ASPPConv(in_channels, out_channels, rate3))
    modules.append(ASPPPooling(in_channels, out_channels))

    self.convs = nn.ModuleList(modules)

    self.project = nn.Sequential(
        nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Dropout(0.1),
    )

  def forward(self, x):
    res = []
    for conv in self.convs:
      res.append(conv(x))
    res = torch.cat(res, dim=1)
    return self.project(res)


def convert_to_separable_conv(module):
  new_module = module
  if isinstance(module, nn.Conv2d) and module.kernel_size[0] > 1:
    new_module = AtrousSeparableConvolution(module.in_channels, module.out_channels, module.kernel_size, module.stride,
                                            module.padding, module.dilation, module.bias)
  for name, child in module.named_children():
    new_module.add_module(name, convert_to_separable_conv(child))
  return new_module
