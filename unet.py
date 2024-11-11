import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class ConvBlock(nn.Module):
  def __init__(self, in_channels, out_channels):
    super().__init__()
    self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=2, padding=1)
    self.norm = nn.BatchNorm2d(out_channels)
    self.relu = nn.ReLU(in_place=True)
  def forward(self, x):
    return self.relu(self.norm(self.conv1(x)))

class UNet(nn.Module):
  def __init__(self, in_channels, out_channels):
    super().__init__()
    self.in = in_channels
    self.out = out_channels
    self.enc1 = ConvBlock(in_channels, 64)
    self.enc2 = ConvBlock(64, 128)
    self.enc3 = ConvBlock(128, 256)
    self.bottleneck = ConvBlock(256, 512)
    
    self.pool = nn.MaxPool2d(2)

    self.scale_t = nn.Linear(64, 512)

    self.dec1 = ConvBlock(512, 256)
    self.dec2 = ConvBlock(256, 128)
    self.dec3 = ConvBlock(128, 64)

    self.up1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
    self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
    self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)

    self.out = nn.Conv2d(64, out_channels, kernel_size=1)

  def pos_embedding(t, embedding_dim):
    half = embedding_dim // 2
    exp = -1 * math.log(10000) / (half - 1)
    ind = torch.arange(half, dtype=torch.float32)
    exponents = torch.exp(exp * ind)
    t_ret = t * exponents
    return torch.cat([torch.sin(t_ret), torch.cos(t_ret)], dim=-1)

  def forward(self, x, t):
    embedded_t = self.pos_embedding(t, 64).to(x.device)
    mlp_t = self.scale_t(embedded_t).unsqueeze(-1).unsqueeze(-1)

    x1 = self.enc1(x)
    x2 = self.enc2(self.pool(x1))
    x3 = self.enc3(self.pool(x2))
    x4 = self.bottleneck(x3 + mlp_t)

    o3 = self.dec1(torch.cat([self.up1(x4), x3]) + mlp_t)
    o2 = self.dec2(torch.cat([self.up2(o3), x2]) + mlp_t)
    o1 = self.dec3(torch.cat([self.up3(o2), x1]) + mlp_t)

    return self.out(o1)