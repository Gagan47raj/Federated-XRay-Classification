import torch
import torch.nn as nn


class SpatialAttention(nn.Module):

    def __init__(
        self,
        kernel_size=7
    ):

        super().__init__()

        self.conv = nn.Conv2d(
            in_channels=2,
            out_channels=1,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=False
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        avg_out = torch.mean(
            x,
            dim=1,
            keepdim=True
        )

        max_out, _ = torch.max(
            x,
            dim=1,
            keepdim=True
        )

        attention = torch.cat(
            [avg_out, max_out],
            dim=1
        )

        attention = self.conv(
            attention
        )

        attention = self.sigmoid(
            attention
        )

        print(
            attention.min().item(),
            attention.max().item()
)

        return x * attention