import torch
from torch import nn

import math
import pathlib

from unet import UNet
from schedule import Schedule


from typing import Optional, Dict, Any, Tuple


class TimeEncoder(nn.Module):
    def __init__(self, n_steps: int, n_features: int = 6):
        super().__init__()

        wavelengths = torch.logspace(math.log(8), math.log(2*n_steps), n_features // 2, base=math.e)

        self._omega = nn.Parameter(2*math.pi / wavelengths, requires_grad=False)

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
    def __init__(self, schedule: Schedule, data_channels: int, base_channels: int,
                 down_sample: Tuple[bool, ...] = (True, False, True, False),
                 multipliers: Tuple[int, ...] = (1, 2, 3, 4),
                 dropout_rate=0.05):
        super().__init__()

        self.schedule = schedule
        self.data_channels = data_channels
        self.time_encoder = TimeEncoder(n_steps=schedule.n_steps, n_features=12)
        #self.unet = UNet(data_channels + self.time_encoder.n_features, 
                         #base_channels,
                         #out_channels=data_channels,
                         #n_blocks=2)
        self.unet = UNet(in_channels=data_channels, base_channels=base_channels,
                         down_sample=down_sample,
                         multipliers=multipliers,
                         dropout_rate=dropout_rate, cond_features=self.time_encoder.n_features)

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        """
        Takes in a batch of data x and batch of times t and predicts the noise epsilon that each data point corresponds to in the forward diffusion process.
        :argument x: A batch of data of shape (batch, img_channels, height, width)
        :arument t: A batch of time steps, shape (batch,)
        :returns epsilon: The predicted epsilon noise to recover x0, shape (batch, img_channels, height, width)
        """
        time_encoding = self.time_encoder(t)  # Shape (batch, time_features)

        epsilon_prediction = self.unet.forward(x, time_encoding)
        assert epsilon_prediction.shape == x.shape
        return epsilon_prediction

    @torch.no_grad()
    def generate(self, img_shape: Tuple[int, int], batch_size: int) -> torch.Tensor:
        """
        Generates random batch_size samples from the distribution.
        :argument img_shape: Shape of images
        :argument batch_size: Number of samples
        :returns samples: Tensor of shape (batch, img_channels, *img_shape)
        """
        from diffusion import reverse_step

        x = torch.randn(batch_size, self.data_channels, *img_shape, device=self.schedule.beta.device)
        for t in range(self.schedule.n_steps, 0, -1):
            ts = torch.tensor([t for _ in range(batch_size)], dtype=torch.int, device=self.schedule.beta.device)

            noise_prediction = self(x, ts)
            x = reverse_step(self.schedule, xt=x, t=ts, epsilon=noise_prediction)
        return x

    def save(self, path: pathlib.Path, metadata: Optional[Dict[str, Any]] = None):
        from safetensors.torch import save_file
        if metadata is None:
            metadata = dict()
        
        constructor_metadata = {
            "schedule.n_steps": str(self.schedule.n_steps),
            "schedule.name": str(self.schedule.name()),
            "data_channels": str(self.data_channels),
            "base_channels": str(self.unet.base_channels),
            "unet.multipliers": str(self.unet.multipliers),
            "unet.down_sample": str(self.unet.down_sample),
            "unet.n_blocks": str(self.unet.n_blocks),
            "unet.dropout_rate": str(self.unet.dropout_rate)
        }

        all_metadata = {key: str(value) for key, value in metadata.items()} | constructor_metadata
        save_file(self.state_dict(), path, metadata=all_metadata)

    @classmethod
    def load(cls, path: pathlib.Path, schedule: Schedule) -> Tuple['Denoiser', Dict[str, str]]:
        from safetensors import safe_open

        state_dict = {}
        with safe_open(path, framework="pt") as f:
            for key in f.keys():
                state_dict[key] = f.get_tensor(key)
    
            metadata = f.metadata()

        if schedule.name() != metadata["schedule.name"]:
            raise ValueError("Tried loading a diffuser model with an incompatible schedule. "
                             f"Loaded schedule has name {metadata['schedule_name']}, "
                             f"but reference schedule has name {schedule.name()}.")
        if schedule.n_steps != int(metadata["schedule.n_steps"]):
            raise ValueError("Tried loading a diffuser model with an incompatible schedule. "
                             f"Loaded schedule has n_steps={metadata['n_steps']}, "
                             f"but reference schedule has name {schedule.n_steps}.")

        multipliers = tuple(int(el.strip()) for el in metadata["unet.multipliers"][1:-1].split(","))
        down_sample = tuple(el.strip() == "True" for el in metadata["unet.down_sample"][1:-1].split(","))
        print(f"{multipliers = }, {down_sample = }")
        model = Denoiser(schedule=schedule, 
                         data_channels=int(metadata["data_channels"]),
                         base_channels=int(metadata["base_channels"]),
                         down_sample=down_sample,
                         multipliers=multipliers,
                         dropout_rate=float(metadata["unet.dropout_rate"]))
        model.load_state_dict(state_dict)
        return model, metadata
        

def _test_time_encoding():
    import matplotlib.pyplot as plt
    times = torch.arange(1000)
    encoder = TimeEncoder(n_steps=1000, n_features=12)
    time_encoded = encoder(times)

    fig, axs = plt.subplots(int(time_encoded.shape[1]), 1, sharex=True)
    for i in range(time_encoded.shape[1]):
        axs[i].plot(times, time_encoded[:, i])
    plt.show()


def _test_toy_problem():
    data_channels = 3
    resolution = 64
    batch = 16
    corrupt = False  # Try substituting true time with random corrupted time to see drop in performance -> show model actually *uses* time conditioning

    from schedule import LinearSchedule
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    schedule = LinearSchedule(beta_start=1e-4, beta_end=0.02, n_steps=1000, device=device)

    denoiser = Denoiser(schedule, data_channels=data_channels, base_channels=64).to(device)

    data = torch.randn(batch, data_channels, resolution, resolution, device=device)
    stationary_target = torch.randn(batch, data_channels, resolution, resolution, device=device)

    # Create a toy objective where a time step is sampled and it is used as a linear parameter controlling a mix between target and 0
    optimizer = torch.optim.Adam(denoiser.parameters(), lr=0.0001)

    print(f"{corrupt = }")

    for iteration in range(2000):
        t_true = torch.randint(low=0, high=schedule.n_steps+1, size=(batch,), device=device)
        if corrupt:
            t = torch.randint(low=0, high=schedule.n_steps+1, size=(batch,), device=device)
        else:
            t = t_true
        mix_param = t_true/schedule.n_steps

        target = data + (stationary_target - data)*mix_param[:, None, None, None]
        prediction = denoiser(data, t)
        loss = torch.square(prediction - target).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if iteration % 100 == 0:
            print(loss.item())


if __name__ == '__main__':
    #_test_time_encoding()
    _test_toy_problem()

