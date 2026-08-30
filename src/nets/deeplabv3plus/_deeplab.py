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
    low-level features and reconstructing the segmentation mask."""

  def __init__(self, in_channels, low_level_channels, num_classes, aspp_dilate=[12, 24, 36]):
    super(DeepLabHeadV3PlusAttention, self).__init__()
    self.project = nn.Sequential(
        nn.Conv2d(low_level_channels, 48, 1, bias=False),
        nn.BatchNorm2d(48),
        nn.ReLU(inplace=True),
    )

    self.aspp = ASPP(in_channels, aspp_dilate)
    self.attention = CBAM(256)  # 256 == ASPP output channels

    self.classifier = nn.Sequential(
        nn.Conv2d(304, 256, 3, padding=1, bias=False), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
        nn.Conv2d(256, num_classes, 1))
    self._init_weight()

  def forward(self, feature):
    low_level_feature = self.project(feature['low_level'])
    output_feature = self.aspp(feature['out'])
    output_feature = self.attention(output_feature)  # <- attention applied between ASPP and decoder

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
