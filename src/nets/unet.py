import torch
from torch import nn
from torch.nn import functional as F


class ConvBlock(nn.Module):
  def __init__(self, in_channels, out_channels):
    super(ConvBlock, self).__init__()
    self.block = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )

  def forward(self, x):
    return self.block(x)


class UNet(nn.Module):
  """Compact U-Net baseline supporting RGB and 5-band inputs."""

  def __init__(self, in_channels=5, num_classes=3, base_channels=32):
    super(UNet, self).__init__()
    channels = [base_channels, base_channels * 2, base_channels * 4, base_channels * 8]
    self.enc1 = ConvBlock(in_channels, channels[0])
    self.enc2 = ConvBlock(channels[0], channels[1])
    self.enc3 = ConvBlock(channels[1], channels[2])
    self.bottleneck = ConvBlock(channels[2], channels[3])
    self.pool = nn.MaxPool2d(2)
    self.up3 = nn.ConvTranspose2d(channels[3], channels[2], 2, stride=2)
    self.dec3 = ConvBlock(channels[2] * 2, channels[2])
    self.up2 = nn.ConvTranspose2d(channels[2], channels[1], 2, stride=2)
    self.dec2 = ConvBlock(channels[1] * 2, channels[1])
    self.up1 = nn.ConvTranspose2d(channels[1], channels[0], 2, stride=2)
    self.dec1 = ConvBlock(channels[0] * 2, channels[0])
    self.classifier = nn.Conv2d(channels[0], num_classes, 1)

  def forward(self, x):
    input_size = x.shape[-2:]
    feature1 = self.enc1(x)
    feature2 = self.enc2(self.pool(feature1))
    feature3 = self.enc3(self.pool(feature2))
    bottleneck = self.bottleneck(self.pool(feature3))
    decoded3 = self.dec3(torch.cat([self.up3(bottleneck), feature3], dim=1))
    decoded2 = self.dec2(torch.cat([self.up2(decoded3), feature2], dim=1))
    decoded1 = self.dec1(torch.cat([self.up1(decoded2), feature1], dim=1))
    output = self.classifier(decoded1)
    return F.interpolate(output, size=input_size, mode='bilinear', align_corners=False)
