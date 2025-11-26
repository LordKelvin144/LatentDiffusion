import torch
from torch import nn

import pathlib

from autoencoder import Encoder
from data_utils import chunk_pad, chunk_depad


from typing import Optional, Dict, Any, Tuple


class PatchDiscriminator(nn.Module):

    _STRING_CONSTRUCTOR = {
        "data_channels": int,
        "base_channels": int,
        "patch_size": int,
        "dropout_rate": float,
        "n_blocks": int,
        "multipliers": lambda multiplier_str: tuple(int(el.strip()) for el in multiplier_str[1:-1].split(",")),
        "down_sample": lambda down_str: tuple(el.strip() == "True" for el in down_str[1:-1].split(","))
    }

    def __init__(self, 
                 data_channels: int,
                 base_channels: int,
                 down_sample: Tuple[bool, ...],
                 multipliers: Tuple[int, ...],
                 n_blocks: int = 2,
                 patch_size: int = 32, 
                 dropout_rate: float = 0.2):
        super().__init__()

        self.patch_size = patch_size
        self.data_channels = data_channels
        self.base_channels = base_channels

        self._constructor = {
            "data_channels": data_channels,
            "base_channels": base_channels,
            "patch_size": patch_size,
            "down_sample": down_sample,
            "multipliers": multipliers,
            "n_blocks": n_blocks,
            "dropout_rate": dropout_rate,
        }

        self.encoder = Encoder(data_channels=data_channels,
                               base_channels=base_channels,
                               down_sample=down_sample,
                               multipliers=multipliers,
                               n_blocks=n_blocks,
                               dropout_rate=dropout_rate)
        self.output_pipeline = nn.Sequential(
            nn.GroupNorm(8, self.encoder.latent_channels),
            nn.SiLU(),
            nn.Flatten(),
            nn.LazyLinear(1)
        )

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """
        :argument img: Input image [batch, channels, height, width] where (height, width) must be tiled by self.patch_size
        """
        padded_img, _pad_height, _pad_width = chunk_pad(img, self.patch_size)

        patches = padded_img \
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
        constructor_metadata = {key: str(val) for key, val in self._constructor.items()}
        all_metadata = {key: str(value) for key, value in metadata.items()} | constructor_metadata
        save_file(self.state_dict(), path, metadata=all_metadata)

    @classmethod
    def load(cls, path: pathlib.Path) -> Tuple['PatchDiscriminator', Dict[str, Any]]:
        from safetensors import safe_open

        state_dict = {}
        kwargs = {}
        with safe_open(path, framework="pt") as f:
            for key in f.keys():
                state_dict[key] = f.get_tensor(key)

            metadata = f.metadata()

        for key, value in metadata.items():
            if key in cls._STRING_CONSTRUCTOR:
                kwargs[key] = cls._STRING_CONSTRUCTOR[key](value)

        model = cls(**kwargs)
        model.load_state_dict(state_dict)
        return model, metadata


def _test_patch_discriminator():
    data = torch.randn(10, 3, 216, 176)

    discriminator = PatchDiscriminator(data_channels=3, base_channels = 16, down_sample=(True, True, True, False), multipliers=(1, 2, 4, 8))
    discriminator(data)


def _test_save_load():
    data = torch.randn(10, 3, 216, 176)

    discriminator = PatchDiscriminator(data_channels=3, base_channels = 16, down_sample=(True, True, True, False), multipliers=(1, 2, 4, 8))
    original_output = discriminator(data)

    path = pathlib.Path("_temp_autoencoder.safetensors")

    try:
        discriminator.save(path, metadata={"foo": "bar"})
        loaded_model, metadata = discriminator.load(path)
    finally:
        if path.exists():
            path.unlink()

    loaded_output = loaded_model(data)

    assert metadata["foo"] == "bar"
    assert torch.allclose(original_output, loaded_output, atol=1e-2)



if __name__ == '__main__':
    _test_patch_discriminator()
    _test_save_load()
    print("Tests successful!")

