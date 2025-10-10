import torch
from torch import nn

from typing import Tuple, Optional


class _ConvBlock(nn.Module):
    def __init__(self, channels: int, in_channels: Optional[int] = None, out_channels: Optional[int] = None, n_conv: int = 2):
        super().__init__()

        layers = []
        for i in range(n_conv):
            this_in_channels = in_channels if in_channels is not None and i == 0 else channels
            this_out_channels = out_channels if out_channels is not None and i == n_conv-1 else channels

            layers += [nn.Conv2d(this_in_channels, this_out_channels, 3, padding="same"), nn.ReLU()]

        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class Encoder(nn.Module):
    def __init__(self, data_channels: int, base_channels: int, n_blocks: int = 3):
        super().__init__()
        
        self.data_channels = data_channels
        self.compression_factor = 2**(n_blocks - 1)

        layers = []
        channels = base_channels
        for i in range(n_blocks):
            if i == 0:
                layers.append(_ConvBlock(in_channels=data_channels, channels=channels))
            else:
                layers.append(_ConvBlock(in_channels=channels // 2, channels=channels))
            
            if i != n_blocks - 1:
                layers.append(nn.MaxPool2d(2, 2))
                channels = 2*channels
    
        self.latent_channels = channels
        self.compression_stage = nn.Sequential(*layers)
        self.final = nn.Conv2d(self.latent_channels, self.latent_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.ndim == 4
        assert x.shape[1] == self.data_channels
        assert x.shape[2] % self.compression_factor == 0 and x.shape[3] % self.compression_factor == 0
        return self.final(self.compression_stage(x))


class Decoder(nn.Module):
    def __init__(self, data_channels: int, base_channels: int, n_blocks: int = 3):
        super().__init__()

        self.data_channels = data_channels
        self.compression_factor = 2**(n_blocks - 1)
        self.latent_channels = base_channels * self.compression_factor

        layers = []
        channels = self.latent_channels
        for i in range(n_blocks):

            layers.append(_ConvBlock(channels=channels))
            
            if i != n_blocks - 1:
                layers += [nn.ConvTranspose2d(channels, channels // 2, kernel_size=2, stride=2), nn.ReLU()]
                channels = channels // 2

        self.expansion_stage = nn.Sequential(*layers)
        self.final = nn.Conv2d(channels, data_channels, kernel_size=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        assert z.ndim == 4
        assert z.shape[1] == self.latent_channels
        return self.final(self.expansion_stage(z))


class AutoEncoder(nn.Module):
    def __init__(self, data_channels: int, base_channels: int, n_blocks: int = 3):
        super().__init__()
        self.encoder = Encoder(data_channels=data_channels, base_channels=base_channels, n_blocks=n_blocks)
        self.decoder = Decoder(data_channels=data_channels, base_channels=base_channels, n_blocks=n_blocks)

        self.data_channels = data_channels
        self.compression_factor = self.encoder.compression_factor
        self.latent_channels = base_channels * self.compression_factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        :parameter z: Encoded data of shape (batch, z_channels, z_height, z_width)
        :returns x: Recovered image data of shape (batch, in_channels, height, width)
        """
        return self.decoder(z)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        :parameter x: Image data of shape (batch, in_channels, height, width)
        :returns z: Encoded data of shape (batch, z_channels, z_height, z_width)
        """
        return self.encoder(x)


def _test_autoencoder():
    data = torch.randn(10, 3, 216, 176)
    model = AutoEncoder(data_channels=3, base_channels=32, n_blocks=3)

    for name, param in model.named_parameters():
        print(name, param.shape)
    z = model.encode(data)
    assert z.shape[1] == model.latent_channels
    x = model.decode(z)
    assert x.shape == data.shape


if __name__ == '__main__':
    _test_autoencoder()

