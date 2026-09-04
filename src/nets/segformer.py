import torch
from torch import nn
from torch.nn import functional as F


class OverlapPatchEmbed(nn.Module):
  def __init__(self, in_channels, embed_dim, kernel_size=3, stride=2):
    super(OverlapPatchEmbed, self).__init__()
    self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size, stride,
                          padding=kernel_size // 2, bias=False)
    self.norm = nn.BatchNorm2d(embed_dim)

  def forward(self, x):
    return F.gelu(self.norm(self.proj(x)))


class EfficientSelfAttention(nn.Module):
  def __init__(self, dim, num_heads, sr_ratio):
    super(EfficientSelfAttention, self).__init__()
    if dim % num_heads != 0:
      raise ValueError('dim must be divisible by num_heads')
    self.num_heads = num_heads
    self.head_dim = dim // num_heads
    self.scale = self.head_dim ** -0.5
    self.q = nn.Conv2d(dim, dim, 1, bias=False)
    self.kv = nn.Conv2d(dim, dim * 2, 1, bias=False)
    self.sr_ratio = sr_ratio
    if sr_ratio > 1:
      self.sr = nn.Conv2d(dim, dim, sr_ratio, sr_ratio, bias=False)
      self.sr_norm = nn.BatchNorm2d(dim)
    self.proj = nn.Conv2d(dim, dim, 1, bias=False)

  def forward(self, x):
    batch_size, channels, height, width = x.shape
    query = self.q(x).flatten(2).transpose(1, 2)
    if self.sr_ratio > 1:
      reduced = F.gelu(self.sr_norm(self.sr(x)))
    else:
      reduced = x
    key_value = self.kv(reduced).flatten(2).transpose(1, 2)
    key, value = key_value.chunk(2, dim=-1)

    query = query.reshape(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
    key = key.reshape(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
    value = value.reshape(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
    attention = torch.matmul(query, key.transpose(-2, -1)) * self.scale
    attention = attention.softmax(dim=-1)
    output = torch.matmul(attention, value).transpose(1, 2).reshape(batch_size, -1, channels)
    output = output.transpose(1, 2).reshape(batch_size, channels, height, width)
    return self.proj(output)


class MixFFN(nn.Module):
  def __init__(self, dim, expansion=4):
    super(MixFFN, self).__init__()
    hidden_dim = dim * expansion
    self.fc1 = nn.Conv2d(dim, hidden_dim, 1, bias=False)
    self.dwconv = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1,
                             groups=hidden_dim, bias=False)
    self.fc2 = nn.Conv2d(hidden_dim, dim, 1, bias=False)
    self.norm = nn.BatchNorm2d(hidden_dim)

  def forward(self, x):
    x = F.gelu(self.norm(self.dwconv(self.fc1(x))))
    return self.fc2(x)


class TransformerBlock(nn.Module):
  def __init__(self, dim, num_heads, sr_ratio):
    super(TransformerBlock, self).__init__()
    self.norm1 = nn.BatchNorm2d(dim)
    self.attention = EfficientSelfAttention(dim, num_heads, sr_ratio)
    self.norm2 = nn.BatchNorm2d(dim)
    self.ffn = MixFFN(dim)

  def forward(self, x):
    x = x + self.attention(self.norm1(x))
    return x + self.ffn(self.norm2(x))


class MiTEncoder(nn.Module):
  def __init__(self):
    super(MiTEncoder, self).__init__()
    self.stem = OverlapPatchEmbed(5, 32, kernel_size=7, stride=2)
    self.embed1 = OverlapPatchEmbed(32, 32, stride=2)
    self.embed2 = OverlapPatchEmbed(32, 64, stride=2)
    self.embed3 = OverlapPatchEmbed(64, 160, stride=2)
    self.embed4 = OverlapPatchEmbed(160, 256, stride=2)
    self.stage1 = nn.Sequential(TransformerBlock(32, 1, 8))
    self.stage2 = nn.Sequential(*[TransformerBlock(64, 2, 4) for _ in range(2)])
    self.stage3 = nn.Sequential(*[TransformerBlock(160, 5, 2) for _ in range(2)])
    self.stage4 = nn.Sequential(*[TransformerBlock(256, 8, 1) for _ in range(2)])

  def forward(self, x):
    x = self.stem(x)
    feature1 = self.stage1(self.embed1(x))
    feature2 = self.stage2(self.embed2(feature1))
    feature3 = self.stage3(self.embed3(feature2))
    feature4 = self.stage4(self.embed4(feature3))
    return feature1, feature2, feature3, feature4


class PretrainedMiTEncoder(nn.Module):
  """Official SegFormer MiT-B0 encoder adapted from 3 to 5 input bands.

  The first patch projection keeps the pretrained RGB filters. NIR and RE are
  initialized with the mean RGB filter, then all weights can be fine-tuned.
  """

  def __init__(self, model_name='nvidia/mit-b0'):
    super(PretrainedMiTEncoder, self).__init__()
    try:
      from transformers import SegformerModel
    except ImportError as error:
      raise ImportError(
          'Pretrained SegFormer requires transformers. Install it with '
          '`pip install transformers==4.45.2`.') from error

    pretrained = SegformerModel.from_pretrained(model_name)
    patch_projection = pretrained.encoder.patch_embeddings[0].proj
    adapted_projection = nn.Conv2d(
        5, patch_projection.out_channels, patch_projection.kernel_size,
        patch_projection.stride, patch_projection.padding, bias=False)
    with torch.no_grad():
      adapted_projection.weight[:, :3].copy_(patch_projection.weight)
      rgb_mean = patch_projection.weight.mean(dim=1, keepdim=True)
      adapted_projection.weight[:, 3:5].copy_(rgb_mean.repeat(1, 2, 1, 1))
    pretrained.encoder.patch_embeddings[0].proj = adapted_projection
    self.encoder = pretrained.encoder
    self.config = pretrained.config

  def forward(self, x):
    outputs = self.encoder(x, output_hidden_states=True, return_dict=True)
    hidden_states = outputs.hidden_states
    features = []
    for hidden_state, (height, width) in zip(hidden_states, ((x.shape[-2] // 4, x.shape[-1] // 4),
                                                              (x.shape[-2] // 8, x.shape[-1] // 8),
                                                              (x.shape[-2] // 16, x.shape[-1] // 16),
                                                              (x.shape[-2] // 32, x.shape[-1] // 32))):
      features.append(hidden_state.transpose(1, 2).reshape(x.shape[0], hidden_state.shape[-1], height, width))
    return tuple(features)


class SegFormer5Band(nn.Module):
  def __init__(self, num_classes=3, decoder_dim=128, pretrained_backbone=False):
    super(SegFormer5Band, self).__init__()
    self.encoder = PretrainedMiTEncoder() if pretrained_backbone else MiTEncoder()
    self.linear1 = nn.Conv2d(32, decoder_dim, 1, bias=False)
    self.linear2 = nn.Conv2d(64, decoder_dim, 1, bias=False)
    self.linear3 = nn.Conv2d(160, decoder_dim, 1, bias=False)
    self.linear4 = nn.Conv2d(256, decoder_dim, 1, bias=False)
    self.fuse = nn.Sequential(
        nn.Conv2d(decoder_dim * 4, decoder_dim, 1, bias=False),
        nn.BatchNorm2d(decoder_dim),
        nn.ReLU(inplace=True),
        nn.Conv2d(decoder_dim, decoder_dim, 3, padding=1, groups=decoder_dim, bias=False),
        nn.BatchNorm2d(decoder_dim),
        nn.ReLU(inplace=True),
    )
    self.classifier = nn.Conv2d(decoder_dim, num_classes, 1)

  def forward(self, x):
    input_size = x.shape[-2:]
    feature1, feature2, feature3, feature4 = self.encoder(x)
    target_size = feature1.shape[-2:]
    features = [self.linear1(feature1), self.linear2(feature2),
                self.linear3(feature3), self.linear4(feature4)]
    features = [F.interpolate(feature, size=target_size, mode='bilinear', align_corners=False)
                for feature in features]
    output = self.classifier(self.fuse(torch.cat(features, dim=1)))
    return F.interpolate(output, size=input_size, mode='bilinear', align_corners=False)
