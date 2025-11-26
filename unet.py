import torch
from torch import nn

from typing import Tuple, Optional
import itertools

from data_utils import chunk_pad, chunk_depad

from residual_block import ResidualBlock, ConditionedSequential


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
            nn.SiLU(inplace=True),
            nn.Conv2d(base_channels, self.out_channels, kernel_size=1)
        )

    def forward(self, x: torch.Tensor, cond: Optional[torch.Tensor] = None):
        """
        Processes an input image of shape (batch, in_channels, height, width) with an optional conditioning of shape (batch, cond_features)
        and returns a prediction of shape (batch, out_channels, heigh, width)
        """
        in_shape = x.shape
        x, pad_height, pad_width = chunk_pad(x, self.down_sampling_factor)
        h = self.input_projection(x)

        hiddens = []
        for block in self.contracting_stage:
            h = block(h, cond=cond)
            hiddens.append(h)

        h = self.middle_stage(h, cond=cond)

        for block in self.expanding_stage:
            h = block(torch.cat([h, hiddens.pop()], dim=1), cond=cond)

        out = self.final(h)
        out = chunk_depad(out, pad_height, pad_width)
        assert out.shape == in_shape
        return out


def _test_unet():
    import numpy as np
    import matplotlib.pyplot as plt
    import time

    batch = 16
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
    t_start = time.perf_counter()
    for it in range(100):
        output = net(data)
        
        true_target = torch.randn_like(data)

        output = net(true_target)
        loss = torch.square(output - true_target).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        if it % 100 == 0:
            print(np.mean(losses[-10:]))
    t_end = time.perf_counter()
    print(f"Time per step: {(t_end - t_start) / 100 :.3g} s")
    plt.plot(losses)
    plt.xlabel("iterations")
    plt.ylabel("loss")
    plt.show()


if __name__ == '__main__':
    #_test_residual_block()
    _test_unet()
    #_debug_resblock()

