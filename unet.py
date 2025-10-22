import torch
from torch import nn

from typing import Tuple, Optional


class _UnetContractingBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, bottom: bool = False):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding="same")
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding="same")
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding="same")
        if not bottom:
            self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        else:
            self.maxpool = None

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        :argument x: Tensor of shape (batch, in_channels, height, width)
        :returns skip_output: Tensor of shape (batch, out_channels, height, width) corresponding to the outputs for skip connections
        :returns down_sampled: Tensor of shape (batch, out_channels, height / 2, width / 2) corresponding to the down-sampled outputs
        """
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        skip_output = x
        if self.maxpool is not None:
            down_sampled = self.maxpool(x)
        else:
            down_sampled = None
        return skip_output, down_sampled


class _UnetExpandingBlock(nn.Module):
    def __init__(self, channels: int, include_upsample: bool = True, skip_connections: bool = True):
        super().__init__()

        self.channels = channels
        in_channels = 2*channels if skip_connections else channels
        self.conv1 = nn.Conv2d(in_channels, channels, kernel_size=3, padding="same")
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding="same")
        if include_upsample:
            self.upsample = nn.ConvTranspose2d(channels, channels // 2, kernel_size=2, stride=2)
        else:
            self.upsample = None

    def forward(self, skip_input: Optional[torch.Tensor], expanded_input: Optional[torch.Tensor]):
        """
        :argument skip_input: Tensor of shape (batch, channels, height, width); input from skip connections
        :argument expanded_input: Tensor of shape (batch, channels, height, width); input from previous expansion block
        :returns output: Tensor of shape (batch, channels/2, 2*height, 2*width) if self has upsample, othersie (batch, channels, height, width)
        """
        if expanded_input is not None and skip_input is not None:
            input_ = torch.cat((skip_input, expanded_input), dim=-3)
        else:
            input_ = skip_input if skip_input is not None else expanded_input
        assert input_ is not None
        x = self.relu(self.conv1(input_))
        x = self.relu(self.conv2(x))
        if self.upsample is not None:
            output = self.upsample(x)
        else:
            output = x
        return output


class UNet(nn.Module):
    def __init__(self, in_channels: int, base_channels: int, n_blocks: int = 5, out_channels: Optional[int] = None):
        super().__init__()
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.n_blocks = n_blocks
        self.contracting_blocks = nn.ModuleList(
            [_UnetContractingBlock(in_channels, base_channels)] + 
            [_UnetContractingBlock(base_channels*2**i, base_channels*2**(i+1)) for i in range(n_blocks-2)] +
            [_UnetContractingBlock(base_channels*2**(n_blocks-2), base_channels*2**(n_blocks-1), bottom=True)]
        )
        self.expanding_blocks = nn.ModuleList(
            [_UnetExpandingBlock(channels=base_channels*2**(n_blocks-1), skip_connections=False)] +
            [_UnetExpandingBlock(channels=base_channels*2**i) for i in range(n_blocks-2, 0, -1)] + 
            [_UnetExpandingBlock(channels=base_channels, include_upsample=False)]
        )
        self.out_channels = in_channels if out_channels is None else out_channels
        self.final = torch.nn.Conv2d(base_channels, self.out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor):
        x_: Optional[torch.Tensor] = x
        assert x_.shape[1] == self.in_channels

        skip_values = []
        for block in self.contracting_blocks:
            skip_value, x_ = block(x_)
            skip_values.append(skip_value)
        assert x_ is None

        for block in self.expanding_blocks:
            skip_input = skip_values.pop()
            x_ = block(skip_input=skip_input, expanded_input=x_)
        output = self.final(x_)
        assert output.shape[0] == x.shape[0]
        assert output.shape[1] == self.out_channels
        assert output.shape[2:] == x.shape[2:]
        return output


def _test_unet():
    data = torch.randn(10, 5, 256, 256)
    model = UNet(in_channels=5, base_channels=16, n_blocks=4)
    data_prime = model(data)
    assert data_prime.shape == data.shape


if __name__ == '__main__':
    _test_unet()
