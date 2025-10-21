import torch
from torch import nn

import math
import pathlib

from unet import UNet
from schedule import Schedule


from typing import Optional, Dict, Any


class TimeEncoder(nn.Module):
    def __init__(self, n_steps: int):
        super().__init__()

        wavelengths = [4**i for i in range(1, int(math.ceil(math.log(n_steps, 4))) + 1)]
        self._omega = nn.Parameter(2*math.pi / torch.tensor(wavelengths), requires_grad=False)

    @property
    def n_features(self) -> int:
        return 2*self._omega.numel()

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        :argument t: Batch of time points shape (batch,)
        :returns encoding: Batch of encodings, shape (batch, n_features)
        """
        assert not self._omega.requires_grad

        phase = t[:, None]*self._omega[None, :] # Shape (batch, n_features // 2)
        encoding = torch.cat((torch.sin(phase), torch.cos(phase)), dim=1)  # Shape (batch, n_feature)
        assert encoding.shape[1] == self.n_features

        return encoding


class Denoiser(nn.Module):
    def __init__(self, schedule: Schedule, data_channels: int, base_channels: int):
        super().__init__()

        self.schedule = schedule
        self.data_channels = data_channels
        self.time_encoder = TimeEncoder(n_steps=schedule.n_steps)
        self.unet = UNet(data_channels + self.time_encoder.n_features, 
                         base_channels,
                         out_channels=data_channels,
                         n_blocks=2)

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        """
        Takes in a batch of data x and batch of times t and predicts the noise epsilon that each data point corresponds to in the forward diffusion process.
        :argument x: A batch of data of shape (batch, img_channels, height, width)
        :arument t: A batch of time steps, shape (batch,)
        :returns epsilon: The predicted epsilon noise to recover x0, shape (batch, img_channels, height, width)
        """
        time_encoding = self.time_encoder(t)  # Shape (batch, time_features)
        batch, time_features = time_encoding.shape

        chunk_size = 2**(self.unet.n_blocks - 1)
        
        pad_height = pad_width = 0
        if x.shape[2] % chunk_size != 0:
            pad_height = chunk_size - x.shape[2] % chunk_size
        if x.shape[3] % chunk_size != 0:
            pad_width = chunk_size - x.shape[3] % chunk_size

        x_padded = nn.functional.pad(x, pad=(0, pad_width, 0, pad_height))

        time_channels = time_encoding[..., None, None].broadcast_to((batch, time_features, x_padded.shape[2], x_padded.shape[3]))

        unet_input = torch.cat((x_padded, time_channels), dim=1)
        epsilon_prediction = self.unet(unet_input)[:, :, :x.shape[2], :x.shape[3]]
        assert epsilon_prediction.shape == x.shape
        return epsilon_prediction

    def save(self, path: pathlib.Path, metadata: Optional[Dict[str, Any]] = None):
        from safetensors.torch import save_file
        if metadata is None:
            metadata = dict()
        
        constructor_metadata = {
            "n_steps": str(self.schedule.n_steps),
            "schedule_name": str(self.schedule.name()),
            "data_channels": str(self.data_channels),
            "base_channels": str(self.unet.base_channels),
            "n_blocks": str(self.unet.n_blocks),
        }
        all_metadata = {key: str(value) for key, value in metadata.items()} | constructor_metadata
        save_file(self.state_dict(), path, metadata=all_metadata)

    @classmethod
    def load(cls, path: pathlib.Path, schedule: Schedule) -> 'Denoiser':
        from safetensors import safe_open

        state_dict = {}
        with safe_open(path, framework="pt") as f:
            for key in f.keys():
                state_dict[key] = f.get_tensor(key)
    
            metadata = f.metadata()

        if schedule.name() != metadata["schedule_name"]:
            raise ValueError("Tried loading a diffuser model with an incompatible schedule. "
                             f"Loaded schedule has name {metadata['schedule_name']}, "
                             f"but reference schedule has name {schedule.name()}.")
        if schedule.n_steps != int(metadata["n_steps"]):
            raise ValueError("Tried loading a diffuser model with an incompatible schedule. "
                             f"Loaded schedule has n_steps={metadata['n_steps']}, "
                             f"but reference schedule has name {schedule.n_steps}.")
        model = Denoiser(schedule=schedule, 
                         data_channels=int(metadata["data_channels"]),
                         base_channels=int(metadata["base_channels"]))
        model.load_state_dict(state_dict)
        return model
        

def _test_time_encoding():
    import matplotlib.pyplot as plt
    times = torch.arange(16)
    encoder = TimeEncoder(n_steps=16)
    time_encoded = encoder(times)

    fig, axs = plt.subplots(int(time_encoded.shape[1]), 1, sharex=True)
    for i in range(time_encoded.shape[1]):
        axs[i].plot(times, time_encoded[:, i])
    plt.show()


if __name__ == '__main__':
    _test_time_encoding()

