# The code in this file was originally taken from https://github.com/VainF/DeepLabV3Plus-Pytorch and is licensed under the MIT License.
# props to: https://github.com/VainF/DeepLabV3Plus-Pytorch

from .utils import IntermediateLayerGetter
from ._deeplab import (DeepLabHead, DeepLabHeadV3Plus, DeepLabHeadV3PlusAttention, DeepLabHeadV3PlusLite, DeepLabV3, SpectralGuidedBackbone)
from .backbones import resnet


def _segm_resnet(name, backbone_name, num_classes, output_stride, pretrained_backbone, use_attention=False):

  if backbone_name in ('resnet18', 'resnet34'):
    # BasicBlock (ResNet18/34) does not support dilated convolutions (raises
    # NotImplementedError for dilation > 1 in BasicBlock.__init__), unlike Bottleneck
    # (ResNet50/101). Keep standard strides (true output stride = 32) and use milder
    # ASPP dilation rates sized for the resulting smaller feature map.
    replace_stride_with_dilation = [False, False, False]
    aspp_dilate = [6, 12, 18]
  elif output_stride == 8:
    replace_stride_with_dilation = [False, True, True]
    aspp_dilate = [12, 24, 36]
  else:
    replace_stride_with_dilation = [False, False, True]
    aspp_dilate = [6, 12, 18]

  backbone = resnet.__dict__[backbone_name](
      pretrained=pretrained_backbone, replace_stride_with_dilation=replace_stride_with_dilation)

  # ResNet18/34 use BasicBlock (expansion=1) -> layer4=512, layer1=64.
  # ResNet50/101 use Bottleneck (expansion=4) -> layer4=2048, layer1=256.
  if backbone_name in ('resnet18', 'resnet34'):
    inplanes = 512
    low_level_planes = 64
  else:
    inplanes = 2048
    low_level_planes = 256

  if name == 'deeplabv3plus':
    return_layers = {'layer4': 'out', 'layer1': 'low_level'}
    if use_attention:
      classifier = DeepLabHeadV3PlusAttention(inplanes, low_level_planes, num_classes, aspp_dilate)
    else:
      classifier = DeepLabHeadV3Plus(inplanes, low_level_planes, num_classes, aspp_dilate)
  elif name == 'deeplabv3':
    return_layers = {'layer4': 'out'}
    classifier = DeepLabHead(inplanes, num_classes, aspp_dilate)
  backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

  model = DeepLabV3(backbone, classifier)
  return model

def _segm_resnet_spectral(backbone_name, num_classes, output_stride, pretrained_backbone, spectral_embed=32,
                          aspp_channels=128, use_spectral_indices=True):
  """Builds the spectral-guided model from the README: RGB (3ch) through a standard
    ResNet backbone, NIR+RE (2ch, or NIR+RE+NDVI-like+NDRE-like (4ch) if
    use_spectral_indices=True) through a TinySpectralEncoder whose descriptor gates
    the RGB backbone features (SpectralGuidedBackbone), then DeepLabHeadV3PlusLite
    (LiteASPP + decoder) for the segmentation head. Expects 5-band (R,G,B,NIR,RE) input.
    use_spectral_indices: see SpectralGuidedBackbone docstring."""
  if backbone_name in ('resnet18', 'resnet34'):
    # BasicBlock now supports dilation; keep layer3/layer4 at stride 1 for OS=8.
    replace_stride_with_dilation = [False, True, True]
    aspp_dilate = [12, 24, 36]
  elif output_stride == 8:
    replace_stride_with_dilation = [False, True, True]
    aspp_dilate = [12, 24, 36]
  else:
    replace_stride_with_dilation = [False, False, True]
    aspp_dilate = [6, 12, 18]
  # RGB-only backbone: keeps the standard 3-channel conv1, so ImageNet-pretrained
  # weights are used unmodified (no channel-count surgery needed).
  rgb_backbone = resnet.__dict__[backbone_name](
      pretrained=pretrained_backbone, replace_stride_with_dilation=replace_stride_with_dilation)
  if backbone_name in ('resnet18', 'resnet34'):
    inplanes = 512
    low_level_planes = 64
  else:
    inplanes = 2048
    low_level_planes = 256
  return_layers = {'layer4': 'out', 'layer1': 'low_level'}
  rgb_backbone = IntermediateLayerGetter(rgb_backbone, return_layers=return_layers)
  backbone = SpectralGuidedBackbone(rgb_backbone, out_channels=inplanes, low_level_channels=low_level_planes,
                                     spectral_embed=spectral_embed, use_spectral_indices=use_spectral_indices)
  classifier = DeepLabHeadV3PlusLite(inplanes, low_level_planes, num_classes, aspp_dilate, aspp_channels=aspp_channels)
  model = DeepLabV3(backbone, classifier)
  return model

def _load_model(arch_type, backbone, num_classes, output_stride, pretrained_backbone, use_attention=False):

  if backbone == 'mobilenetv2':
    model = _segm_mobilenet(
        arch_type, backbone, num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone)
  elif backbone.startswith('resnet'):
    model = _segm_resnet(
        arch_type, backbone, num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone, use_attention=use_attention)
  elif backbone.startswith('hrnetv2'):
    model = _segm_hrnet(arch_type, backbone, num_classes, pretrained_backbone=pretrained_backbone)
  else:
    raise NotImplementedError
  return model


# Deeplab v3
def deeplabv3_resnet50(num_classes=21, output_stride=8, pretrained_backbone=False):
  """Constructs a DeepLabV3 model with a ResNet-50 backbone.
    Args:
        num_classes (int): number of classes.
        output_stride (int): output stride for deeplab.
        pretrained_backbone (bool): If True, use the pretrained backbone.
    """
  return _load_model(
      'deeplabv3', 'resnet50', num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone)


def deeplabv3_resnet101(num_classes=21, output_stride=8, pretrained_backbone=False):
  """Constructs a DeepLabV3 model with a ResNet-101 backbone.
    Args:
        num_classes (int): number of classes.
        output_stride (int): output stride for deeplab.
        pretrained_backbone (bool): If True, use the pretrained backbone.
    """
  return _load_model(
      'deeplabv3', 'resnet101', num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone)


# Deeplab v3+
def deeplabv3plus_resnet50(num_classes=21, output_stride=8, pretrained_backbone=False):
  """Constructs a DeepLabV3 model with a ResNet-50 backbone.
    Args:
        num_classes (int): number of classes.
        output_stride (int): output stride for deeplab.
        pretrained_backbone (bool): If True, use the pretrained backbone.
    """
  return _load_model(
      'deeplabv3plus', 'resnet50', num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone)


def deeplabv3plus_resnet101(num_classes=21, output_stride=8, pretrained_backbone=False):
  """Constructs a DeepLabV3+ model with a ResNet-101 backbone.
    Args:
        num_classes (int): number of classes.
        output_stride (int): output stride for deeplab.
        pretrained_backbone (bool): If True, use the pretrained backbone.
    """
  return _load_model(
      'deeplabv3plus', 'resnet101', num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone)

def deeplabv3plus_resnet50_attn(num_classes=21, output_stride=8, pretrained_backbone=False):
  """Constructs a DeepLabV3+ model with a ResNet-50 backbone and a CBAM attention
    module inserted between the ASPP module and the decoder
    (backbone -> ASPP -> attention -> decoder -> segmentation mask).
    Args:
        num_classes (int): number of classes.
        output_stride (int): output stride for deeplab.
        pretrained_backbone (bool): If True, use the pretrained backbone.
    """
  return _load_model(
      'deeplabv3plus', 'resnet50', num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone,
      use_attention=True)

def deeplabv3plus_resnet34_spectral(num_classes=21, output_stride=8, pretrained_backbone=False, spectral_embed=32,
                                    aspp_channels=128, use_spectral_indices=True):
  """Constructs the spectral-guided DeepLabV3+ model described in the project README:
    ResNet-34 RGB backbone + a tiny NIR/RE spectral branch (TinySpectralEncoder) whose
    descriptor gates the RGB features (SpectralGuidedBackbone, replacing CBAM) +
    LiteASPP (depthwise-separable, `aspp_channels` output channels) + decoder.
    Expects 5-band input ordered (R, G, B, NIR, RE).
    Args:
        num_classes (int): number of classes.
        output_stride (int): output stride for deeplab.
        pretrained_backbone (bool): If True, use the ImageNet-pretrained ResNet-34 backbone
            (only applied to the RGB branch, which keeps a standard 3-channel conv1).
        spectral_embed (int): channel width of the NIR/RE spectral descriptor.
        aspp_channels (int): output channels of LiteASPP (lower = fewer params/FLOPs).
        use_spectral_indices (bool): if True (default), feed the spectral encoder
            [NIR, RE, NDVI-like, NDRE-like] (4ch) instead of raw [NIR, RE] (2ch).
            Set False to reproduce the earlier raw-band-only behavior.
    """
  return _segm_resnet_spectral(
      'resnet34', num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone,
      spectral_embed=spectral_embed, aspp_channels=aspp_channels, use_spectral_indices=use_spectral_indices)