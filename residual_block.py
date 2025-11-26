import torch
from torch import nn

from typing import Optional


class ConditionedSequential(nn.Module):
    """
    A utility module behaving similar to nn.Sequential but which passes conditioning to modules that expects it.
    """

    def __init__(self, *layers: nn.Module):
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor, cond: Optional[torch.Tensor]) -> torch.Tensor:
        for module in self.layers:
            if isinstance(module, ResidualBlock):
                x = module(x, cond)
            else:
                x = module(x)
        return x


class ResidualBlock(nn.Module):
    """
    A residual block that takes in an input of shape (batch, channels, height, width)
    and returns an output of shape (batch, out_channels, height, width)
    """

    def __init__(self, channels: int, in_channels: Optional[int] = None, out_channels: Optional[int] = None,
                 cond_features: Optional[int] = None,
                 dropout_rate: float = 0.2):
        super().__init__()

        out_channels = out_channels if out_channels is not None else channels
        in_channels = in_channels if in_channels is not None else channels
        do_project = in_channels != out_channels

        self.skip_connection = nn.Identity() if not do_project else nn.Conv2d(in_channels, out_channels, kernel_size=1)

        self.input_stage = nn.Sequential(
            nn.GroupNorm(8, in_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(in_channels, channels, kernel_size=3, padding="same")
        )  # (batch, in_channels, height, width) -> (batch, channels, height, width)

        if cond_features is not None:
            self.cond_stage = nn.Linear(cond_features, channels)  # (batch, cond_features) -> (batch, channels)
        else:
            self.cond_stage = None

        self.output_stage = nn.Sequential(
            nn.GroupNorm(8, channels),
            nn.SiLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Conv2d(channels, out_channels, kernel_size=3, padding="same")
        )
        with torch.no_grad():
            last_conv = self.output_stage[-1]
            last_conv.weight.detach().mul_(0.001)  # pyright: ignore  # Check if this makes sense: The official implementation zeros this layer, but that causes gradients to vanish
            last_conv.bias.detach().mul_(0.001) # pyright: ignore 

    def forward(self, x: torch.Tensor, cond: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Takes in a spatial tensor x and conditioning features and returns output
        :argument x: Tensor of shape (batch, in_channels, height, width)
        :argument conditioning: Tensor of shape (batch, cond_features)
        :returns output: Tensor of shape (batch, out_channels, height, width)
        """

        hidden = self.input_stage(x)
        if self.cond_stage is not None:
            assert cond is not None
            cond_hidden = self.cond_stage(cond)  # shape (batch, channels)
            hidden += cond_hidden.view(*cond_hidden.shape, *(1 for _ in range(hidden.ndim - cond_hidden.ndim)))
        else:
            assert cond is None
        return self.output_stage(hidden) + self.skip_connection(x)


def _test_residual_block():
    batch = 10
    cond_features = 64
    resolution = 256
    in_channels = 16
    channels = 32
    out_channels = 2

    # Test case where in_channels != out_channels
    block = ResidualBlock(channels, in_channels=in_channels, out_channels=out_channels, cond_features=cond_features)

    conditioning = torch.randn(batch, cond_features)
    data = torch.randn(batch, in_channels, resolution, resolution)

    output = block(data, conditioning)
    assert output.shape == (batch, out_channels, resolution, resolution)

    # Test case where in_channels = out_channels
    out_channels = 3
    block = ResidualBlock(channels, in_channels=in_channels, out_channels=out_channels, cond_features=cond_features)

    conditioning = torch.randn(batch, cond_features)
    data = torch.randn(batch, in_channels, resolution, resolution)

    output = block.forward(data, conditioning)
    assert output.shape == (batch, out_channels, resolution, resolution)


if __name__ == '__main__':
    _test_residual_block()

