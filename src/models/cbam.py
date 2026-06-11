import torch.nn as nn

from src.models.channel_attention import (
    ChannelAttention
)

from src.models.spatial_attention import (
    SpatialAttention
)


class CBAM(nn.Module):

    def __init__(
        self,
        in_channels,
        reduction_ratio=16,
        kernel_size=7
    ):

        super().__init__()

        self.channel_attention = ChannelAttention(
            in_channels=in_channels,
            reduction_ratio=reduction_ratio
        )

        self.spatial_attention = SpatialAttention(
            kernel_size=kernel_size
        )

    def forward(self, x):

        x = self.channel_attention(x)

        x = self.spatial_attention(x)

        return x