import torch
from torch import nn

import random
import pathlib

from autoencoder import Encoder


from typing import Optional, Dict, Any, Tuple


class PatchDiscriminator(nn.Module):
    def __init__(self, data_channels: int, base_channels: int = 32, n_blocks: int = 4, patch_size: int = 32):
        super().__init__()

        self.patch_size = patch_size
        self.data_channels = data_channels
        self.base_channels = base_channels
        self.n_blocks = n_blocks

        self.encoder = Encoder(data_channels=data_channels, base_channels=base_channels, n_blocks=n_blocks)
        self.output_pipeline = nn.Sequential(nn.ReLU(), nn.Flatten(), nn.LazyLinear(1))

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """
        :argument img: Input image [batch, channels, height, width] where (height, width) must be tiled by self.patch_size
        """
        h_margin = img.shape[2] % self.patch_size
        w_margin = img.shape[3] % self.patch_size
        h_offset = random.randint(0, h_margin-1) if h_margin > 0 else 0
        w_offset = random.randint(0, w_margin-1) if w_margin > 0 else 0

        sub_img = img[:, :, h_offset:, w_offset:]
        patches = sub_img \
            .unfold(2, self.patch_size, self.patch_size) \
            .unfold(3, self.patch_size, self.patch_size)  # [batch, channels, h_patches, w_patches, patch_size, patch_size]
        patches = patches.permute((0, 2, 3, 1, 4, 5))  # [batch, h_patches, w_patches, patch_size, patch_size]
        batch, h_patches, w_patches, channels = patches.shape[:4]
        patches = patches.reshape((batch*h_patches*w_patches, channels, self.patch_size, self.patch_size))

        z = self.encoder(patches)  # Shape [batch', encoder_channels, z_height, z_width]
        patch_logits = self.output_pipeline(z)  # shape [batch', 1]
        logits = patch_logits.reshape(batch, h_patches, w_patches)

        return logits

    def save(self, path: pathlib.Path, metadata: Optional[Dict[str, Any]] = None):
        from safetensors.torch import save_file
        if metadata is None:
            metadata = dict()
        constructor_metadata = {
            "data_channels": str(self.data_channels),
            "base_channels": str(self.base_channels),
            "n_blocks": str(self.n_blocks),
            "patch_size": str(self.patch_size)
        }
        all_metadata = {key: str(value) for key, value in metadata.items()} | constructor_metadata
        save_file(self.state_dict(), path, metadata=all_metadata)

    @classmethod
    def load(cls, path: pathlib.Path) -> Tuple['PatchDiscriminator', Dict[str, Any]]:
        from safetensors import safe_open

        state_dict = {}
        kwargs = {}
        kwarg_keys = {"data_channels", "base_channels", "n_blocks", "patch_size"}
        with safe_open(path, framework="pt") as f:
            for key in f.keys():
                state_dict[key] = f.get_tensor(key)

            metadata = f.metadata()
            for key in kwarg_keys:
                kwargs[key] = int(metadata[key])
    
        model = cls(**kwargs)
        model.load_state_dict(state_dict)
        return model, metadata


class Discriminator(nn.Module):
    def __init__(self, data_channels: int, base_channels: int = 32, n_blocks: int = 3):
        super().__init__()
        self.encoder = Encoder(data_channels=data_channels, base_channels=base_channels, n_blocks=n_blocks)

        self.collect = nn.Sequential(nn.ReLU(), nn.Conv2d(self.encoder.latent_channels, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        collected = self.collect(self.encoder(x))  # [batch, 1, latent_height, latent_width]
        return collected.squeeze(1)  # [batch, latent_height, latent_width]


def _test_patch_discriminator():
    data = torch.randn(10, 3, 216, 176)

    discriminator = PatchDiscriminator(data_channels=3)
    discriminator(data)


if __name__ == '__main__':
    _test_patch_discriminator()

