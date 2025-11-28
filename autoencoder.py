import torch
from torch import nn
from torch.utils.data import Dataset
from torchvision.io import decode_image

import pathlib
import itertools

from typing import Tuple, Optional, Dict, Any

from residual_block import ResidualBlock, ConditionedSequential


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
    def __init__(self, 
                 data_channels: int, 
                 base_channels: int, 
                 down_sample: Tuple[bool, ...],
                 multipliers: Tuple[int, ...], 
                 n_blocks: int = 2, 
                 latent_channels: Optional[int] = None,
                 stochastic: bool = True,
                 dropout_rate=0.2):
        super().__init__()
        
        self.data_channels = data_channels
        self.stochastic = stochastic
        self.down_sampling_factor = 2**sum(down_sample)
        
        self.input_projection = nn.Conv2d(data_channels, base_channels, 1)

        layer_edge_channels = multipliers + (multipliers[-1],)

        # Construct the contracting stage
        layers = [] 
        for layer_i, (in_multiplier, out_multiplier) in enumerate(itertools.pairwise(layer_edge_channels)):
            for block_i in range(n_blocks):
                block_channels = base_channels*in_multiplier
                block_out = block_channels if block_i < n_blocks-1 else base_channels*out_multiplier

                layers.append(
                    ResidualBlock(channels=block_channels, out_channels=block_out, 
                                   dropout_rate=dropout_rate, cond_features=None)
                )

            if down_sample[layer_i]:
                layers.append(nn.MaxPool2d(2, 2))  # Downsample 2x

        self.contracting_stage = ConditionedSequential(*layers)
    
        self.latent_channels = latent_channels if latent_channels is not None else base_channels*multipliers[-1]

        output_channels = self.latent_channels if not stochastic else 2*self.latent_channels
        self.final = nn.Sequential(
            nn.GroupNorm(8, base_channels*multipliers[-1]),
            nn.SiLU(),
            nn.Conv2d(base_channels*multipliers[-1], output_channels, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean, _log_std = self._forward_computation(x)
        return mean

    def _forward_computation(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        assert x.ndim == 4
        assert x.shape[1] == self.data_channels
        assert x.shape[2] % self.down_sampling_factor == 0 and x.shape[3] % self.down_sampling_factor == 0

        h = self.input_projection(x)
        h = self.contracting_stage(h, cond=None)
        output = self.final(h)
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
    def __init__(self, 
                 data_channels: int, 
                 base_channels: int, 
                 latent_channels: int,
                 down_sample: Tuple[bool, ...],
                 multipliers: Tuple[int, ...], 
                 n_blocks: int = 2, 
                 stochastic: bool = True,
                 dropout_rate: float = 0.2):
        super().__init__()

        self.stochastic = stochastic
        self.data_channels = data_channels
        self.down_sampling_factor = 2**sum(down_sample)
        self.latent_channels = latent_channels
        self.base_channels = base_channels

        layer_edge_channels = multipliers + (multipliers[-1],)

        self.input_projection = nn.Conv2d(latent_channels, self.base_channels*multipliers[-1], 1)

        layers = []
        for layer_i, (out_multiplier, in_multiplier) in reversed(list(enumerate(itertools.pairwise(layer_edge_channels)))):
            if down_sample[layer_i]:
                layers.append(nn.Upsample(scale_factor=2))

            for block_i in range(n_blocks):
                block_channels = base_channels*out_multiplier
                block_in = block_channels if block_i > 0 else base_channels*in_multiplier

                layers.append(
                    ResidualBlock(channels=block_channels, in_channels=block_in, 
                                  dropout_rate=dropout_rate, cond_features=None)
                )

        self.expanding_stage = ConditionedSequential(*layers)

        output_channels = 2*data_channels if stochastic else data_channels
        self.final = nn.Sequential(
            nn.GroupNorm(8, base_channels),
            nn.SiLU(),
            nn.Conv2d(base_channels, output_channels, kernel_size=1)
        )

    def _forward_computation(self, z: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        assert z.ndim == 4
        assert z.shape[1] == self.latent_channels

        h = self.input_projection(z)
        h = self.expanding_stage(h, cond=None)
        output = self.final(h)

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
    def __init__(self, 
                 data_channels: int,
                 base_channels: int,
                 down_sample: Tuple[bool, ...],
                 multipliers: Tuple[int, ...], 
                 n_blocks: int = 2,
                 latent_channels: Optional[int] = None,
                 stochastic: bool = True,
                 dropout_rate: float = 0.2):
        super().__init__()
        self.encoder = Encoder(data_channels=data_channels, 
                               base_channels=base_channels, 
                               down_sample=down_sample,
                               multipliers=multipliers,
                               n_blocks=n_blocks, 
                               latent_channels=latent_channels,
                               stochastic=stochastic,
                               dropout_rate=dropout_rate)
        self.latent_channels = self.encoder.latent_channels
        self.decoder = Decoder(data_channels=data_channels,
                               base_channels=base_channels,
                               down_sample=down_sample,
                               multipliers=multipliers,
                               n_blocks=n_blocks,
                               latent_channels=self.latent_channels,
                               stochastic=stochastic,
                               dropout_rate=dropout_rate)

        self.stochastic = stochastic
        self.data_channels = data_channels
        self.base_channels = base_channels
        self.n_blocks = n_blocks
        self.multipliers = multipliers
        self.down_sample = down_sample
        self.down_sampling_factor = self.encoder.down_sampling_factor
        self.latent_channels = self.encoder.latent_channels
        self.dropout_rate = dropout_rate

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
            "down_sample": str(self.down_sample),
            "multipliers": str(self.multipliers),
            "n_blocks": str(self.n_blocks),
            "latent_channels": str(self.latent_channels),
            "stochastic": str(self.stochastic),
            "dropout_rate": str(self.dropout_rate)
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

        multipliers = tuple(int(el.strip()) for el in metadata["multipliers"][1:-1].split(","))
        down_sample = tuple(el.strip() == "True" for el in metadata["down_sample"][1:-1].split(","))
        model = AutoEncoder(data_channels=int(metadata["data_channels"]),
                            base_channels=int(metadata["base_channels"]),
                            down_sample=down_sample,
                            multipliers=multipliers,
                            n_blocks=int(metadata["n_blocks"]),
                            latent_channels=int(metadata["latent_channels"]),
                            stochastic=bool(metadata["stochastic"]),
                            dropout_rate=float(metadata["dropout_rate"]))
        model.load_state_dict(state_dict)
        return model, metadata

    def sha256_digest(self) -> str:
        import hashlib
        hasher = hashlib.sha256()
        for name, tensor in self.state_dict().items():
            hasher.update(name.encode('utf-8'))
            hasher.update(tensor.cpu().numpy().tobytes(order='C'))
    
        return hasher.hexdigest()


class EncodedImgDataset(Dataset):
    """
    A dataset which wraps an autoencoder applied to a set of images.
    Fetching at an index returns the encoded version of the image at the index.
    Optionally, the encoded images can be cached (though beware that this uses significant disk space).
    """
    
    def __init__(self, dataset: Dataset, autoencoder: AutoEncoder, directory: pathlib.Path | str = pathlib.Path("./encoded_img")):
        print(f"Creating EncodedImgDataset with autoencoder {autoencoder.sha256_digest()}")
        super().__init__()
        self._base_dataset = dataset
        self._autoencoder = autoencoder
        assert not self._autoencoder.training

        self._device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        self._len = len(dataset)  # pyright: ignore
        self._directory = pathlib.Path(directory) / autoencoder.sha256_digest()
        if not self._directory.exists():
            print(f"Created directory for encoded images at {self._directory}")
            self._directory.mkdir()

        self._chunk_size = 16
        with torch.no_grad():
            img, label = dataset[0]
            img = img.to(self._device)
            self._latent_shape = autoencoder.encode(img[None, ...]).shape[1:]
            self._label_shape = label.shape
        self._filled_idx = set()

    def _get_chunk(self, i: int) -> Tuple[torch.Tensor, torch.Tensor]:
        path = self._directory / (str(i).zfill(6)+".pt")
        label_path = self._directory / (str(i).zfill(6) + ".label.pt")

        if path.exists():
            return torch.load(path), torch.load(label_path)

        img = torch.zeros(self._chunk_size, *self._latent_shape, dtype=torch.float16)
        label = torch.zeros(self._chunk_size, *self._label_shape, dtype=torch.float16)
        return img, label

    def _save_chunk(self, i: int, img: torch.Tensor, label: torch.Tensor):
        path = self._directory / (str(i).zfill(6)+".pt")
        label_path = self._directory / (str(i).zfill(6) + ".label.pt")
        torch.save(img, path)
        torch.save(label, label_path)

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        chunk_i = idx // self._chunk_size
        sub_i = idx % self._chunk_size
        batch_img, batch_label = self._get_chunk(chunk_i)

        img, label = batch_img[sub_i], batch_label[sub_i]

        if idx in self._filled_idx:
            # We know this index has been filled 
            return img, label

        if not torch.all(img == 0.0):
            # This index has been filled on some different run of the program, since it holds non-zero data
            self._filled_idx.add(idx)
            return img, label
        else:
            # This index is empty.
            # Get the relevant encoding and save the data
            img, label = self._base_dataset[idx]
            with torch.no_grad():
                batch_img[sub_i] = self._autoencoder.encode(img.to(self._device)[None, ...]).cpu().squeeze(0)
            batch_label[sub_i] = label
            self._filled_idx.add(idx)
            self._save_chunk(chunk_i, batch_img, batch_label)

            return batch_img[sub_i], batch_label[sub_i]


def _test_autoencoder():
    data = torch.randn(10, 3, 216, 176)
    model = AutoEncoder(data_channels=3, base_channels=32, down_sample=(True, True, True, False), multipliers=(1, 2, 4, 8), latent_channels=5)

    z = model.encode(data)
    assert z.shape[1] == model.latent_channels
    x = model.decode(z)
    assert x.shape == data.shape


def _test_save_load():
    model = AutoEncoder(data_channels=3, base_channels=32, down_sample=(True, True), multipliers=(1, 2), latent_channels=1)
    digest = model.sha256_digest()

    path = pathlib.Path("_temp_autoencoder.safetensors")

    try:
        model.save(path, metadata={"foo": "bar"})
        loaded_model, metadata = AutoEncoder.load(path)
    finally:
        if path.exists():
            path.unlink()

    assert loaded_model.sha256_digest() == digest
    assert metadata["foo"] == "bar"


if __name__ == '__main__':
    _test_autoencoder()
    _test_save_load()
    print("Tests successful!")

