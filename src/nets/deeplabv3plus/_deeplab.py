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

# ============================================================================
# SR-CBAM (Spectral-Relation CBAM)
# extends CBAM with a Spectral branch
# that conditions both the channel and the spatial attention gates on the raw
# 5-band input (R, G, B, NIR, RE), in addition to the ASPP feature map.
# ============================================================================

class SpectralRelation(nn.Module):
  """Cong doan 3 - Spectral Relation.
    Hoc quan he (band <-> band) giua 5 kenh pho, thay vi chi tinh do quan trong
    cua tung band rieng le:
    - Band Embedding: moi band duoc tom tat bang gia tri trung binh toan anh
      roi nhung (embed) thanh vector.
    - Spectral Relation: ma tran tuong quan 5x5 hoc duoc bang self-attention
      giua cac band embedding, dung de tron (mix) lai 5 band goc theo quan he
      pho da hoc."""

  def __init__(self, num_bands=5, embed_dim=16):
    super(SpectralRelation, self).__init__()
    self.band_embed = nn.Linear(1, embed_dim)  # Band Embedding
    self.query = nn.Linear(embed_dim, embed_dim)
    self.key = nn.Linear(embed_dim, embed_dim)
    self.scale = embed_dim**-0.5

  def forward(self, x):
    n, c, h, w = x.shape  # c == num_bands
    band_stat = x.mean(dim=(2, 3)).unsqueeze(-1)  # (N, 5, 1) global stat per band
    band_embed = self.band_embed(band_stat)  # (N, 5, embed_dim) -- Band Embedding
    q = self.query(band_embed)
    k = self.key(band_embed)
    relation = torch.softmax(torch.bmm(q, k.transpose(1, 2)) * self.scale, dim=-1)  # (N, 5, 5) -- A_spec

    x_flat = x.view(n, c, h * w)
    x_mixed = torch.bmm(relation, x_flat).view(n, c, h, w)  # bands re-mixed according to A_spec
    return x_mixed, relation
class SpectralEncoder(nn.Module):
  """Cong doan 2 - Spectral Encoder.
    E = Conv_1x1(X); S = Downsample(E).
    Conv 1x1 hoc cach ket hop 5 gia tri pho tai cung mot vi tri pixel; sau do
    downsample (adaptive avg pool) ve dung do phan giai cua F_ASPP."""

  def __init__(self, num_bands=5, spec_channels=32):
    super(SpectralEncoder, self).__init__()
    self.conv1x1 = nn.Conv2d(num_bands, spec_channels, kernel_size=1, bias=False)
    self.bn = nn.BatchNorm2d(spec_channels)
    self.act = nn.ReLU(inplace=True)

  def forward(self, x, output_size):
    e = self.act(self.bn(self.conv1x1(x)))  # E = Conv_1x1(X)
    s = F.adaptive_avg_pool2d(e, output_size)  # S = Downsample(E)
    return s
class SpectralBranch(nn.Module):
  """Step 2 + 3 are merged: Given the input 5-band X, generate a 'Spectral Descriptor'
    S_spec which has the same spatial resolution as F_ASPP, is conditioned on the relationships among the spectral bands  (Spectral Relation)."""

  def __init__(self, num_bands=5, spec_channels=32, relation_embed_dim=16):
    super(SpectralBranch, self).__init__()
    self.relation = SpectralRelation(num_bands, relation_embed_dim)
    self.encoder = SpectralEncoder(num_bands, spec_channels)

  def forward(self, x, output_size):
    x_mixed, relation = self.relation(x)  # Step 3
    descriptor = self.encoder(x_mixed, output_size)  # Step 2
    return descriptor, relation

class SRCBAM(nn.Module):
  """SR-CBAM (Spectral-Relation CBAM).
    CBAM is extended with an additional Spectral branch, conditioning both Channel Attention and Spatial Attention 
    on the original 5-band input (R, G, B, NIR, RE), rather than relying solely on
    F_ASPP as in the original CBAM. The module takes two inputs: f_aspp (feature ASPP) and x (the original 5-band input)."""

  def __init__(self, in_channels, num_bands=5, spec_channels=32, reduction=16, spatial_kernel_size=7):
    super(SRCBAM, self).__init__()
    # Step 2 + 3: Spectral Encoder + Spectral Relation -> Spectral Descriptor
    self.spectral_branch = SpectralBranch(num_bands=num_bands, spec_channels=spec_channels)

    # Step 4: Spectral-conditioned Channel Attention
    self.avg_pool = nn.AdaptiveAvgPool2d(1)
    self.max_pool = nn.AdaptiveMaxPool2d(1)
    hidden_channels = max(in_channels // reduction, 8)
    self.channel_mlp = nn.Sequential(
        nn.Conv2d(in_channels, hidden_channels, 1, bias=False),
        nn.ReLU(inplace=True),
        nn.Conv2d(hidden_channels, in_channels, 1, bias=False),
    )
    self.spec_channel_proj = nn.Conv2d(spec_channels, in_channels, kernel_size=1, bias=False)  # P_c (global) -> M_c

    # Step 5: Feature Fusion (spectral-guided gating): F_fusion = F_c (.) (1 + G_c)
    self.spec_fusion_proj = nn.Conv2d(spec_channels, in_channels, kernel_size=1, bias=False)  # P_c (spatial) -> G_c

    # Step 6: Spectral-conditioned Spatial Attention
    self.spec_spatial_proj = nn.Conv2d(spec_channels, 1, kernel_size=1, bias=False)  # P_s
    self.spatial_conv = nn.Conv2d(3, 1, spatial_kernel_size, padding=spatial_kernel_size // 2, bias=False)

    self.sigmoid = nn.Sigmoid()
    self._init_weight()

  def forward(self, f_aspp, x):
    # Step 2 + 3: encode raw 5-band input -> Spectral Descriptor (cung resolution voi F_ASPP)
    spec_descriptor, _relation = self.spectral_branch(x, output_size=f_aspp.shape[2:])

    # Step 4: Spectral-conditioned Channel Attention
    # M_c = sigmoid( MLP(AvgPool(F_ASPP)) + MLP(MaxPool(F_ASPP)) + P_c(S_spec) )
    avg_out = self.channel_mlp(self.avg_pool(f_aspp))
    max_out = self.channel_mlp(self.max_pool(f_aspp))
    spec_global = self.spec_channel_proj(self.avg_pool(spec_descriptor))
    m_c = self.sigmoid(avg_out + max_out + spec_global)
    f_c = f_aspp * m_c  # F_c = F_ASPP (.) M_c

    # Step 5: Feature Fusion. G_c = sigmoid(P_c(S)); F_fusion = F_c (.) (1 + G_c)
    g_c = self.sigmoid(self.spec_fusion_proj(spec_descriptor))
    f_fusion = f_c * (1 + g_c)

    # Step 6: Spectral-conditioned Spatial Attention
    # M_s = sigmoid( Conv_7x7[ Avg_C(F_fusion), Max_C(F_fusion), P_s(S_spec) ] )
    avg_s = torch.mean(f_fusion, dim=1, keepdim=True)
    max_s, _ = torch.max(f_fusion, dim=1, keepdim=True)
    spec_s = self.spec_spatial_proj(spec_descriptor)
    m_s = self.sigmoid(self.spatial_conv(torch.cat([avg_s, max_s, spec_s], dim=1)))
    f_attn = f_fusion * m_s  # F_attn = F_fusion (.) M_s


    return f_attn

  def _init_weight(self):
    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight)
      elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

class DeepLabHeadV3PlusAttention(nn.Module):
  """DeepLabV3+ decoder head with an attention module inserted right after ASPP
    (i.e. between the multi-scale ASPP features and the decoder), so that the model
    can select the most informative channels/regions before fusing with the
    low-level features and reconstructing the segmentation mask.

    attention_type:
      - 'cbam': original CBAM (channel + spatial attention on F_ASPP only).
      - 'sr_cbam': SR-CBAM (see README), CBAM conditioned additionally on the raw
        multispectral input (R, G, B, NIR, RE)."""


  def __init__(self, in_channels, low_level_channels, num_classes, aspp_dilate=[12, 24, 36], attention_type='cbam', num_bands=5):
    super(DeepLabHeadV3PlusAttention, self).__init__()
    self.project = nn.Sequential(
        nn.Conv2d(low_level_channels, 48, 1, bias=False),
        nn.BatchNorm2d(48),
        nn.ReLU(inplace=True),
    )

    self.aspp = ASPP(in_channels, aspp_dilate)
    self.attention_type = attention_type.lower()
    if self.attention_type == 'cbam':
      self.attention = CBAM(256)  # 256 == ASPP output channels
    elif self.attention_type == 'sr_cbam':
      self.attention = SRCBAM(256, num_bands=num_bands)  # 256 == ASPP output channels
    else:
      raise ValueError(f"Unknown attention_type '{attention_type}'. Expected one of: 'cbam', 'sr_cbam'.")

    self.classifier = nn.Sequential(
        nn.Conv2d(304, 256, 3, padding=1, bias=False), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
        nn.Conv2d(256, num_classes, 1))
    self._init_weight()

  def forward(self, feature):
    low_level_feature = self.project(feature['low_level'])
    output_feature = self.aspp(feature['out'])
    if self.attention_type == 'sr_cbam':
      output_feature = self.attention(output_feature, feature['input'])  # SR-CBAM: F_ASPP + raw multispectral input
    else:
      output_feature = self.attention(output_feature)  # CBAM: F_ASPP only

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
