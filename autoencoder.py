import torch
from torch import nn

import pathlib

from typing import Tuple, Optional, Dict, Any


class NormalDistribution:
    def __init__(self, mean: torch.Tensor, log_std: torch.Tensor):
        self.mean = mean
        self.log_std = log_std

        self._kl_mean = None

    def kl(self, reduction: str):
        """The KL divergence to N(0,I)"""
        per_axis = -0.5 - self.log_std + 0.5*torch.square(self.mean) + 0.5*torch.exp(2*self.log_std)
        self._kl_mean = per_axis.mean((1, 2, 3))

        if reduction == "mean":
            return self._kl_mean
        elif reduction == "sum":
            return self._kl_mean * per_axis.numel()
        else:
            raise NotImplementedError(f"Invalid reduction {reduction}")

    def log_likelihood(self, x, reduction: str):
        s2 = torch.exp(2*self.log_std)
        log_l2 = -0.5*torch.square(x - self.mean) / s2
        log_norm = -self.log_std  # - torch.log(torch.sqrt(2.0*torch.pi))

        if reduction == "sum":
            return log_l2.sum((1,2,3)) + log_norm.sum((1,2,3))
        elif reduction == "mean":
            return log_l2.mean((1,2,3)) + log_norm.mean((1,2,3))
        else:
            raise NotImplementedError(f"Invalid reduction {reduction}")

    def rsample(self) -> torch.Tensor:
        epsilon = torch.randn(size=self.mean.shape, device=self.mean.device, dtype=self.mean.dtype)
        x = self.mean + torch.exp(self.log_std) * epsilon
        return x

    @property
    def shape(self) -> torch.Size:
        return self.mean.shape



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
    def __init__(self, data_channels: int, base_channels: int, n_blocks: int = 3, latent_channels: Optional[int] = None, stochastic: bool = True):
        super().__init__()
        
        self.data_channels = data_channels
        self.compression_factor = 2**(n_blocks - 1)
        self.stochastic = stochastic

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
    
        self.latent_channels = latent_channels if latent_channels is not None else channels
        self.compression_stage = nn.Sequential(*layers)

        output_channels = self.latent_channels if not stochastic else 2*self.latent_channels
        self.final = nn.Conv2d(channels, output_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean, _log_std = self._forward_computation(x)
        return mean

    def _forward_computation(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        assert x.ndim == 4
        assert x.shape[1] == self.data_channels
        assert x.shape[2] % self.compression_factor == 0 and x.shape[3] % self.compression_factor == 0

        # If we have tried to encode this value previously, retrieve cached value
        output = self.final(self.compression_stage(x))
        if self.stochastic:
            mean = output[:, :self.latent_channels, :, :]
            log_std = output[:, self.latent_channels:, :, :]

            return mean, log_std
        else:
            mean = output
            return mean, None

    def stochastic_encode(self, x: torch.Tensor) -> NormalDistribution:
        """
        Computes a distribution version of the encoding of x.
        Returns 3 tensors of shape (batch, self.latent_channels, z_height, z_width)
        :returns distribution: distribution p(z|x)
        """
        if not self.stochastic:
            raise NotImplementedError("Cannot call 'sample' on an encoder which is not stochastic.")
        mean, log_std = self._forward_computation(x)
        assert log_std is not None
        return NormalDistribution(mean=mean, log_std=log_std)


class Decoder(nn.Module):
    def __init__(self, data_channels: int, base_channels: int, latent_channels: int, n_blocks: int = 3, stochastic: bool = True):
        super().__init__()

        self.stochastic = stochastic
        self.data_channels = data_channels
        self.compression_factor = 2**(n_blocks - 1)
        self.latent_channels = latent_channels
        self.base_channels = base_channels

        layers = []
        channels = self.base_channels * 2**(n_blocks-1)
        for i in range(n_blocks):
            in_channels = latent_channels if i == 0 else None

            layers.append(_ConvBlock(channels=channels, in_channels=in_channels))
            
            if i != n_blocks - 1:
                layers += [nn.ConvTranspose2d(channels, channels // 2, kernel_size=2, stride=2), nn.ReLU()]
                channels = channels // 2

        self.expansion_stage = nn.Sequential(*layers)
        output_channels = 2*data_channels if stochastic else data_channels
        self.final = nn.Conv2d(channels, output_channels, kernel_size=1)

    def _forward_computation(self, z: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        assert z.ndim == 4
        assert z.shape[1] == self.latent_channels

        output = self.final(self.expansion_stage(z))

        if self.stochastic:
            mean = output[:, :self.data_channels, :, :]
            log_std = output[:, self.data_channels:, :, :]

            log_std = -3*torch.ones_like(log_std)

            self._z_cache = z.detach()
            self._x_mean_cache = mean
            self._x_log_std_cache = log_std

            return mean, log_std
        else:
            mean = output
            return mean, None

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        mean, log_std = self._forward_computation(z)
        return mean

    def stochastic_decode(self, z: torch.Tensor) -> NormalDistribution:
        """
        Computes a distribution version of the encoding of z.
        :returns distribution: distribution of shape (batch, data_channels, height, width)
        """
        if not self.stochastic:
            raise NotImplementedError("Cannot sample on a decoder which is not stochastic")
        mean, log_std = self._forward_computation(z)
        assert log_std is not None
        return NormalDistribution(mean=mean, log_std=log_std)


class AutoEncoder(nn.Module):
    def __init__(self, data_channels: int, base_channels: int, n_blocks: int = 3, latent_channels: Optional[int] = None, stochastic: bool = True):
        super().__init__()
        self.encoder = Encoder(data_channels=data_channels, base_channels=base_channels, n_blocks=n_blocks, 
                               latent_channels=latent_channels, stochastic=stochastic)
        self.latent_channels = self.encoder.latent_channels
        self.decoder = Decoder(data_channels=data_channels, base_channels=base_channels,
                               latent_channels=self.latent_channels, n_blocks=n_blocks, stochastic=stochastic)

        self.stochastic = stochastic
        self.data_channels = data_channels
        self.base_channels = base_channels
        self.n_blocks = n_blocks
        self.compression_factor = self.encoder.compression_factor
        self.latent_channels = self.encoder.latent_channels

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

    def stochastic_encode(self, x: torch.Tensor) -> NormalDistribution:
        return self.encoder.stochastic_encode(x)

    def stochastic_decode(self, z: torch.Tensor) -> NormalDistribution:
        return self.decoder.stochastic_decode(z)

    def save(self, path: pathlib.Path, metadata: Optional[Dict[str, Any]] = None):
        from safetensors.torch import save_file
        if metadata is None:
            metadata = dict()
        constructor_metadata = {
            "data_channels": str(self.data_channels),
            "base_channels": str(self.base_channels),
            "n_blocks": str(self.n_blocks),
            "latent_channels": str(self.latent_channels),
            "stochastic": str(self.stochastic)
        }
        all_metadata = {key: str(value) for key, value in metadata.items()} | constructor_metadata
        save_file(self.state_dict(), path, metadata=all_metadata)

    @classmethod
    def load(cls, path: pathlib.Path) -> Tuple['AutoEncoder', Dict[str, Any]]:
        from safetensors import safe_open

        state_dict = {}
        with safe_open(path, framework="pt") as f:
            for key in f.keys():
                state_dict[key] = f.get_tensor(key)
    
            metadata = f.metadata()

        model = AutoEncoder(data_channels=int(metadata["data_channels"]),
                            base_channels=int(metadata["base_channels"]),
                            n_blocks=int(metadata["n_blocks"]),
                            latent_channels=int(metadata["latent_channels"]),
                            stochastic=bool(metadata["stochastic"]))
        model.load_state_dict(state_dict)
        return model, metadata


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

