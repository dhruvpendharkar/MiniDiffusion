{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "gpuType": "T4",
      "authorship_tag": "ABX9TyPqWQ1Rd5dft7xngoTLOknh"
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    },
    "accelerator": "GPU"
  },
  "cells": [
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "RSNPhvF1w5Dt"
      },
      "outputs": [],
      "source": [
        "import torch\n",
        "import torch.nn as nn\n",
        "import torch.nn.functional as F\n",
        "import torchvision"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "class ConvBlock(nn.Module):\n",
        "  def __init__(self, in_channels, out_channels):\n",
        "    super().__init__()\n",
        "    self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=2, padding=1)\n",
        "    self.norm = nn.BatchNorm2d(out_channels)\n",
        "    self.relu = nn.ReLU(in_place=True)\n",
        "  def forward(self, x):\n",
        "    return self.relu(self.norm(self.conv1(x)))\n",
        "\n",
        "class UNet(nn.Module):\n",
        "  def __init__(self, in_channels, out_channels):\n",
        "    super().__init__()\n",
        "    self.in = in_channels\n",
        "    self.out = out_channels\n",
        "    self.enc1 = ConvBlock(in_channels, 64)\n",
        "    self.enc2 = ConvBlock(64, 128)\n",
        "    self.enc3 = ConvBlock(128, 256)\n",
        "    self.bottleneck = ConvBlock(256, 512)\n",
        "\n",
        "    self.pool = nn.MaxPool2d(2)\n",
        "\n",
        "    self.scale_t = nn.Linear(64, 512)\n",
        "\n",
        "    self.dec1 = ConvBlock(512, 256)\n",
        "    self.dec2 = ConvBlock(256, 128)\n",
        "    self.dec3 = ConvBlock(128, 64)\n",
        "\n",
        "    self.up1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)\n",
        "    self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)\n",
        "    self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)\n",
        "\n",
        "    self.out = nn.Conv2d(64, out_channels, kernel_size=1)\n",
        "\n",
        "  def pos_embedding(t, embedding_dim):\n",
        "    half = embedding_dim // 2\n",
        "    exp = -1 * math.log(10000) / (half - 1)\n",
        "    ind = torch.arange(half, dtype=torch.float32)\n",
        "    exponents = torch.exp(exp * ind)\n",
        "    t_ret = t * exponents\n",
        "    return torch.cat([torch.sin(t_ret), torch.cos(t_ret)], dim=-1)\n",
        "\n",
        "  def forward(self, x, t):\n",
        "    embedded_t = self.pos_embedding(t, 64).to(x.device)\n",
        "    mlp_t = self.scale_t(embedded_t).unsqueeze(-1).unsqueeze(-1)\n",
        "\n",
        "    x1 = self.enc1(x)\n",
        "    x2 = self.enc2(self.pool(x1))\n",
        "    x3 = self.enc3(self.pool(x2))\n",
        "    x4 = self.bottleneck(x3 + mlp_t)\n",
        "\n",
        "    o3 = self.dec1(torch.cat([self.up1(x4), x3]) + mlp_t)\n",
        "    o2 = self.dec2(torch.cat([self.up2(o3), x2]) + mlp_t)\n",
        "    o1 = self.dec3(torch.cat([self.up3(o2), x1]) + mlp_t)\n",
        "\n",
        "    return self.out(o1)\n",
        "\n",
        "\n",
        "\n",
        "\n",
        "\n"
      ],
      "metadata": {
        "id": "ldjtWVQuxh60"
      },
      "execution_count": null,
      "outputs": []
    }
  ]
}