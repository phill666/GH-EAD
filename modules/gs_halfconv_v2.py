from __future__ import annotations

import torch
from torch import Tensor, nn


def channel_shuffle(x: Tensor, groups: int) -> Tensor:
    batch_size, channels, height, width = x.shape
    if groups <= 1 or channels % groups != 0:
        return x
    channels_per_group = channels // groups
    x = x.view(batch_size, groups, channels_per_group, height, width)
    x = x.transpose(1, 2).contiguous()
    return x.view(batch_size, channels, height, width)


class PartialSpatialConv(nn.Module):
    def __init__(self, channels: int, n_div: int = 2, kernel_size: int = 3):
        super().__init__()
        if channels <= 0:
            raise ValueError(f'channels must be positive, got {channels}.')
        if n_div < 2:
            raise ValueError(f'n_div must be at least 2, got {n_div}.')

        self.conv_channels = max(1, min(channels, channels // n_div))
        self.skip_channels = channels - self.conv_channels
        self.partial_conv = nn.Conv2d(
            self.conv_channels,
            self.conv_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            bias=False,
        )

    def forward(self, x: Tensor) -> Tensor:
        x_conv = self.partial_conv(x[:, :self.conv_channels])
        if self.skip_channels == 0:
            return x_conv
        x_skip = x[:, self.conv_channels:]
        return torch.cat((x_conv, x_skip), dim=1)


class ChannelGate(nn.Module):
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        hidden = max(8, channels // reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, hidden, kernel_size=1, bias=True)
        self.act = nn.SiLU(inplace=True)
        self.fc2 = nn.Conv2d(hidden, channels, kernel_size=1, bias=True)
        self.gate = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        weights = self.pool(x)
        weights = self.fc1(weights)
        weights = self.act(weights)
        weights = self.fc2(weights)
        weights = self.gate(weights)
        return x * weights


class DepthwisePointwiseMixer(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3):
        super().__init__()
        self.depthwise = nn.Conv2d(
            channels,
            channels,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            groups=channels,
            bias=False,
        )
        self.pointwise = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.act = nn.GELU()

    def forward(self, x: Tensor) -> Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        return self.act(x)


class GSHalfConvV2(nn.Module):
    """Lightweight GS-HalfConv block used only in the v2 variant branches."""

    def __init__(
        self,
        channels: int,
        groups: int = 2,
        n_div: int = 2,
        reduction: int = 4,
        mix_kernel_size: int = 3,
    ):
        super().__init__()
        if channels <= 0:
            raise ValueError(f'channels must be positive, got {channels}.')
        if groups < 1:
            raise ValueError(f'groups must be positive, got {groups}.')

        base_group_channels = channels // groups
        last_group_channels = channels - base_group_channels * (groups - 1)
        self.group_channels = [base_group_channels] * (groups - 1) + [last_group_channels]
        self.partial_blocks = nn.ModuleList(
            PartialSpatialConv(group_channels, n_div=n_div, kernel_size=3)
            for group_channels in self.group_channels
        )
        self.shuffle_groups = groups if channels % groups == 0 else 1
        self.mixer = DepthwisePointwiseMixer(channels, kernel_size=mix_kernel_size)
        self.channel_gate = ChannelGate(channels, reduction=reduction)
        self.norm = nn.BatchNorm2d(channels)
        self.res_scale = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        grouped_inputs = torch.split(x, self.group_channels, dim=1)
        grouped_outputs = [
            block(current)
            for current, block in zip(grouped_inputs, self.partial_blocks)
        ]
        out = torch.cat(grouped_outputs, dim=1)
        out = channel_shuffle(out, self.shuffle_groups)
        out = self.mixer(out)
        out = self.channel_gate(out)
        out = self.norm(out)
        return identity + self.res_scale * out
