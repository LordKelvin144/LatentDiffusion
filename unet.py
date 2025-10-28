import torch
from torch import nn

from typing import Tuple, Optional
import itertools

from residual_block import ResidualBlock


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


class UNet(nn.Module):
    def __init__(self, in_channels: int, base_channels: int, multipliers: Tuple[int, ...] = (1, 2, 3, 4), 
                 down_sample: Tuple[bool, ...] = (False, False, False, True),
                 out_channels: Optional[int] = None,
                 cond_features: Optional[int] = None,
                 n_blocks: int = 2,
                 dropout_rate: float = 0.2):
        super().__init__()
        self.in_channels = in_channels
        self.base_channels = base_channels
        assert multipliers[0] == 1
        self.multipliers = multipliers
        self.down_sample = down_sample
        self.cond_features = cond_features
        self.n_blocks = n_blocks
        self.dropout_rate = dropout_rate

        self.down_sampling_factor = 2**sum(down_sample)
        
        self.input_projection = nn.Conv2d(in_channels, base_channels, 1)

        layer_edge_channels = multipliers + (multipliers[-1],)

        # Construct the contracting stage
        self.contracting_stage = nn.ModuleList()
        skip_channels = []
        for layer_i, (in_multiplier, out_multiplier) in enumerate(itertools.pairwise(layer_edge_channels)):
            blocks = []
            for block_i in range(n_blocks):
                block_channels = base_channels*in_multiplier
                block_out = block_channels if block_i < n_blocks-1 else base_channels*out_multiplier

                blocks.append(
                    ResidualBlock(channels=block_channels, out_channels=block_out, 
                                   dropout_rate=dropout_rate, cond_features=cond_features)
                )
            skip_channels.append(block_out)  # pyright: ignore

            if down_sample[layer_i]:
                blocks.append(nn.MaxPool2d(2, 2))  # Downsample 2x

            self.contracting_stage.append(ConditionedSequential(*blocks))

        # Construct the middle stage
        self.middle_stage = ConditionedSequential(
            ResidualBlock(channels=base_channels*multipliers[-1], cond_features=cond_features, dropout_rate=dropout_rate), 
            # TODO: Add attention here
            ResidualBlock(channels=base_channels*multipliers[-1], cond_features=cond_features, dropout_rate=dropout_rate)
        )

        # Construct the expanding stage
        self.expanding_stage = nn.ModuleList()
        for layer_i, (out_multiplier, in_multiplier) in reversed(list(enumerate(itertools.pairwise(layer_edge_channels)))):
            blocks = []
            if down_sample[layer_i]:
                blocks.append(nn.Upsample(scale_factor=2))

            for block_i in range(n_blocks):
                block_channels = base_channels*out_multiplier
                block_in = block_channels if block_i > 0 else base_channels*in_multiplier + skip_channels.pop()

                blocks.append(
                    ResidualBlock(channels=block_channels, in_channels=block_in, 
                                   dropout_rate=dropout_rate, cond_features=cond_features)
                )

            self.expanding_stage.append(ConditionedSequential(*blocks))

        self.out_channels = in_channels if out_channels is None else out_channels
        self.final = nn.Sequential(
            nn.GroupNorm(4, base_channels),
            nn.SiLU(),
            nn.Conv2d(base_channels, self.out_channels, kernel_size=1)
        )

    def _pad(self, x: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        pad_height = pad_width = 0
        chunk_size = self.down_sampling_factor
        if x.shape[2] % chunk_size != 0:
            pad_height = chunk_size - x.shape[2] % chunk_size
        if x.shape[3] % chunk_size != 0:
            pad_width = chunk_size - x.shape[3] % chunk_size
        if pad_height or pad_width:
            import warnings
            warnings.warn(f"Input shape {x.shape} is not cleanly divisible using downsampling factor {self.down_sampling_factor}, applying padding ...")
        x = nn.functional.pad(x, pad=(0, pad_width, 0, pad_height))
        return x, pad_height, pad_width

    def _depad(self, x: torch.Tensor, pad_height: int, pad_width: int) -> torch.Tensor:
        return x[:, :, :x.shape[2]-pad_height, :x.shape[3]-pad_width]

    def forward(self, x: torch.Tensor, cond: Optional[torch.Tensor] = None):
        """
        Processes an input image of shape (batch, in_channels, height, width) with an optional conditioning of shape (batch, cond_features)
        and returns a prediction of shape (batch, out_channels, heigh, width)
        """
        in_shape = x.shape
        x, pad_height, pad_width = self._pad(x)
        h = self.input_projection(x)

        hiddens = []
        for block in self.contracting_stage:
            h = block(h, cond=cond)
            hiddens.append(h)

        h = self.middle_stage(h, cond=cond)

        for block in self.expanding_stage:
            h = block(torch.cat([h, hiddens.pop()], dim=1), cond=cond)

        out = self.final(h)
        out = self._depad(out, pad_height, pad_width)
        assert out.shape == in_shape
        return out


def _test_unet():
    import numpy as np
    import matplotlib.pyplot as plt

    batch = 1
    cond_features = 64
    resolution = 32
    in_channels = 3
    channels = 32

    # Test case where in_channels != out_channels
    net = UNet(in_channels=in_channels, base_channels=channels, dropout_rate=0.05)

    print(net)
    print("")

    data = torch.randn(batch, in_channels, resolution, resolution)

    output = net(data)

    print("\n")
    print(output[0])
    assert output.shape == (batch, in_channels, resolution, resolution)

    # Try training on noise
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001)

    losses = []
    for it in range(1000):
        output = net(data)
        
        true_target = torch.randn_like(data)

        output = net(true_target)
        loss = torch.square(output - true_target).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        if it % 10 == 0:
            print(np.mean(losses[-10:]))
    plt.plot(losses)
    plt.xlabel("iterations")
    plt.ylabel("loss")
    plt.show()


if __name__ == '__main__':
    #_test_residual_block()
    _test_unet()
    #_debug_resblock()

