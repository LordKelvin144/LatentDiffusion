import torch
from torch import nn

from typing import Tuple, Optional


class _ContractingBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, bottom: bool = False):
        super().__init__()
        assert out_channels > in_channels

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


class _ExpandingBlock(nn.Module):
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
    def __init__(self, in_channels: int, start_channels: int, n_blocks: int = 5):
        super().__init__()
        self.contracting_blocks = nn.ModuleList(
            [_ContractingBlock(in_channels, start_channels)] + 
            [_ContractingBlock(start_channels*2**i, start_channels*2**(i+1)) for i in range(n_blocks-2)] +
            [_ContractingBlock(start_channels*2**(n_blocks-2), start_channels*2**(n_blocks-1), bottom=True)]
        )
        self.expanding_blocks = nn.ModuleList(
            [_ExpandingBlock(channels=start_channels*2**(n_blocks-1), skip_connections=False)] +
            [_ExpandingBlock(channels=start_channels*2**i) for i in range(n_blocks-2, 0, -1)] + 
            [_ExpandingBlock(channels=start_channels, include_upsample=False)]
        )
        self.final = torch.nn.Conv2d(start_channels, in_channels, kernel_size=1)

    def forward(self, x: torch.Tensor):
        x_: Optional[torch.Tensor] = x
        skip_values = []
        for block in self.contracting_blocks:
            skip_value, x_ = block(x_)
            skip_values.append(skip_value)
        assert x_ is None

        for block in self.expanding_blocks:
            skip_input = skip_values.pop()
            x_ = block(skip_input=skip_input, expanded_input=x_)
        output = self.final(x_)
        assert output.shape == x.shape
        return output


class AutoEncoder(nn.Module):
    def __init__(self, in_channels: int, start_channels: int, n_blocks: int = 5):
        super().__init__()
        self.contracting_blocks = nn.ModuleList(
            [_ContractingBlock(in_channels, start_channels)] + 
            [_ContractingBlock(start_channels*2**i, start_channels*2**(i+1)) for i in range(n_blocks-2)] +
            [_ContractingBlock(start_channels*2**(n_blocks-2), start_channels*2**(n_blocks-1), bottom=True)]
        )
        self.expanding_blocks = nn.ModuleList(
            [_ExpandingBlock(channels=start_channels*2**(n_blocks-1), skip_connections=False)] +
            [_ExpandingBlock(channels=start_channels*2**i, skip_connections=False) for i in range(n_blocks-2, 0, -1)] + 
            [_ExpandingBlock(channels=start_channels, include_upsample=False, skip_connections=False)]
        )
        self.final = torch.nn.Conv2d(start_channels, in_channels, kernel_size=1)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        :parameter x: Image data of shape (batch, in_channels, height, width)
        :returns z: Encoded data of shape (batch, z_channels, z_height, z_width)
        """
        to_contract: Optional[torch.Tensor] = x
        for block in self.contracting_blocks:
            skip_value, to_contract = block(to_contract)
        assert to_contract is None

        return skip_value  # pyright: ignore

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        :parameter z: Encoded data of shape (batch, z_channels, z_height, z_width)
        :returns x: Recovered image data of shape (batch, in_channels, height, width)
        """
        skip_input = z
        expanded_input: Optional[torch.Tensor] = None
        for i, block in enumerate(self.expanding_blocks):
            if i == 0:
                skip_input = z
            else:
                skip_input = None
            expanded_input = block(skip_input=skip_input, expanded_input=expanded_input)
        output = self.final(expanded_input)
        return output


def _test_unet():
    data = torch.randn(10, 5, 256, 256)
    model = UNet(in_channels=5, start_channels=16, n_blocks=4)
    data_prime = model(data)
    assert data_prime.shape == data.shape


def _test_autoencoder():
    data = torch.randn(10, 5, 256, 256)
    model = AutoEncoder(in_channels=5, start_channels=16, n_blocks=4)
    z = model.encode(data)
    x = model.decode(z)
    assert x.shape == data.shape


if __name__ == '__main__':
    _test_unet()
    _test_autoencoder()

