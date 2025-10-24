import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, Subset
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, ChainedScheduler

import numpy as np
import matplotlib.pyplot as plt

from autoencoder import AutoEncoder
from denoiser import Denoiser
from data_utils import make_celeba, denormalize
from diffusion import sample_training_examples, reverse_step, forward_process, estimate_initial
from schedule import Schedule, LinearSchedule

import pathlib

from typing import List, Dict, Optional, Union

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class TrainingConfig:
    batch: int
    lr: float
    epochs: int
    warmup_steps: int = 1000
    warmup_factor: float = 0.1
    cosine_annealing_steps: int = 1_000_000
    cosine_annealing_factor: float = 0.1


class TrainLogger:
    def __init__(self, window: int, log_vars: Dict[str, str]):
        self._log_vars = log_vars
        self._logs: Dict[str, List[float]] = {key: [] for key in log_vars.keys()}
        self._cum_logs: Dict[str, List[float]] = {key: [] for key in log_vars.keys()}
        self.window = window
        self._iteration = 0

    def log(self, **kwargs):
        self._iteration += 1

        for key, value in kwargs.items():
            self._logs[key].append(value)
            if self._cum_logs[key]:
                self._cum_logs[key].append(self._cum_logs[key][-1] + value)
            else:
                self._cum_logs[key].append(value)

    def moving_avg(self, key: str, n: Optional[int] = None) -> np.ndarray:
        if n is None:
            n = self.window
        a = np.array(self._cum_logs[key])
        windowed_sum = a.copy()
        windowed_sum[n:] -= windowed_sum[:-n]
        windowed_avg = windowed_sum / np.minimum(np.arange(a.size) + 1.0, n)
        return windowed_avg

    def plot(self, include_keys: Optional[List[str]] = None):
        fig, ax = plt.subplots()

        if include_keys is None:
            include_keys = list(self._log_vars.keys())

        for i, key in enumerate(include_keys):
            ax.plot(self._logs[key], color=f"C{i}", alpha=0.1)
            ax.plot(self.moving_avg(key), label=self._log_vars[key], color=f"C{i}")
            ax.set_xlabel("Iteration")
            ax.set_ylabel("Loss")

    def current_repr(self, window: Optional[int] = None) -> str:
        if window is None:
            window = self.window
        return f"Step {self._iteration - 1}; " + "; ".join(f"{key} = {np.mean(self._logs[key][-window:]):.4g}" for key in self._log_vars.keys())

    @property
    def iteration(self) -> int:
        return self._iteration


class TrainingCallback(ABC):
    def __init__(self, epoch_interval: int = 1, step_interval: Optional[int] = None):
        self.epoch_interval = epoch_interval
        self.step_interval = step_interval
        
    def check(self, epoch: int, step: int, batches_per_epoch: int):
        if epoch % self.epoch_interval != 0:
            return

        if self.step_interval is not None and step % self.step_interval == 0 and step != 0:
            self(epoch, step)

        elif self.step_interval is None and step == batches_per_epoch - 1:
            self(epoch, step)

    @abstractmethod
    def __call__(self, epoch: int, step: int) -> None:
        pass


class PrintLossCallback(TrainingCallback):
    def __init__(self, logger: TrainLogger, epoch_interval: int = 1, step_interval: Optional[int] = None):
        super().__init__(epoch_interval=epoch_interval, step_interval=step_interval)
        self._logger = logger

    def __call__(self, epoch: int, step: int) -> None:
        print(self._logger.current_repr())


class PlotLossCallback(TrainingCallback):
    def __init__(self, logger: TrainLogger, epoch_interval: int = 1, step_interval: Optional[int] = None, filename: Optional[str] = None):
        super().__init__(epoch_interval=epoch_interval, step_interval=step_interval)
        self._logger = logger
        self._filename = filename

    def __call__(self, epoch: int, step: int) -> None:
        self._logger.plot()
        plt.xscale("log")
        if self._filename is not None:
            plt.savefig(self._filename)
            plt.close()
        else:
            plt.show()


class PlotPredictionsCallback(TrainingCallback):
    def __init__(self, image_source: Union[torch.Tensor, DataLoader], autoencoder: AutoEncoder, denoiser: Denoiser, schedule: Schedule, 
                 epoch_interval: int = 1, step_interval: Optional[int] = None, filename: Optional[str] = None):
        super().__init__(epoch_interval=epoch_interval, step_interval=step_interval)
        self._filename = filename
        self._autoencoder = autoencoder
        self._denoiser = denoiser
        self._schedule = schedule
        if isinstance(image_source, DataLoader):
            self._image_source = iter(image_source)
        else:
            self._image_source = image_source

    def __call__(self, epoch: int, step: int) -> None:
        if isinstance(self._image_source, torch.Tensor):
            x0 = self._image_source
        else:
            x0 = next(self._image_source)[0].to(self._schedule.beta.device)  # TODO: Make this cleaner
        plot_denoiser_predictions(x0=x0, schedule=self._schedule, autoencoder=self._autoencoder, denoiser=self._denoiser)

        if self._filename is not None:
            plt.savefig(self._filename, dpi=300)
            plt.close()
        else:
            plt.show()


class SaveModelCallback(TrainingCallback):
    def __init__(self, denoiser: Denoiser, file_prefix: Union[str, pathlib.Path], epoch_interval: int = 1, step_interval: Optional[int] = None):
        super().__init__(epoch_interval=epoch_interval, step_interval=step_interval)
        self._file_prefix = pathlib.Path(file_prefix)
        self._denoiser = denoiser

    def __call__(self, epoch: int, step: int) -> None:
        model_path = self._file_prefix.parent / (self._file_prefix.name + f"_epoch{epoch}.safetensors")
        print(f"Saving denoiser checkpoint to {model_path} ...")
        self._denoiser.save(model_path, metadata={"epoch": epoch, "step": step})


@torch.no_grad()
def plot_denoiser_predictions(x0: torch.Tensor, schedule: Schedule, denoiser: Denoiser, autoencoder: AutoEncoder):

    n = x0.shape[0]
    z0 = autoencoder.encode(x0)
    t = torch.linspace(1, schedule.n_steps // 2, n, dtype=torch.int, device=z0.device)
    zt, _ = forward_process(schedule, z0, t=t)

    xvt = autoencoder.decode(zt)

    epsilon_prediction = denoiser(zt, t)
    z0_pred = estimate_initial(schedule, zt, t=t, epsilon=epsilon_prediction)

    x0_pred = autoencoder.decode(z0_pred)

    # Full reconstruction
    z_recon = torch.zeros_like(zt[:n])
    for i in range(n):
        steps = int(t[i])
        this_z = zt.detach()[i, None]
        this_t = t.detach()[i, None]
        for _ in range(steps):
            this_epsilon = denoiser(this_z, this_t)
            this_z = reverse_step(schedule, this_z, this_t, this_epsilon)
            this_t = this_t - 1

        z_recon[i] = this_z.squeeze(0)
    xv_recon = autoencoder.decode(z_recon)

    x0_denorm = denormalize(x0.cpu())
    xvt_denorm = denormalize(xvt.cpu())
    x0_pred_denorm = denormalize(x0_pred.cpu())
    xv_recon = denormalize(xv_recon.cpu())

    fig, axs = plt.subplots(n, 4, figsize=(6, 8))
    for i in range(n):
        axs[i,0].imshow(x0_denorm[i])
        axs[i,1].imshow(xvt_denorm[i])
        axs[i,2].imshow(x0_pred_denorm[i])
        axs[i,3].imshow(xv_recon[i])


def train_latent_denoiser(denoiser: Denoiser, 
                          autoencoder: AutoEncoder,
                          schedule: Schedule,
                          train_set: Dataset,
                          val_set: Dataset,
                          config: TrainingConfig,
                          logger: TrainLogger,
                          callbacks: List[TrainingCallback]):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_loader = DataLoader(train_set, batch_size=config.batch, shuffle=True, drop_last=True)

    optimizer = torch.optim.Adam(params=denoiser.parameters(), lr=config.lr)
    lr_schedule = ChainedScheduler([LinearLR(optimizer, start_factor=config.warmup_factor, total_iters=config.warmup_steps),
                                    CosineAnnealingLR(optimizer, T_max=config.cosine_annealing_steps, eta_min=config.lr*config.cosine_annealing_factor)],
                                   optimizer=optimizer)

    for epoch in range(config.epochs):
        step_i = 0
        for step_i, (x0, _) in enumerate(train_loader):
            # Get latent coding for the images
            x0 = x0.to(device)
            with torch.no_grad():
                z0 = autoencoder.encode(x0)  # Shape (batch, z_channels, z_height, z_width)

            # Get noised examples and their corresponding times and epsilons
            zt, t, epsilon = sample_training_examples(schedule, z0)

            # Get model prediction
            epsilon_prediction = denoiser(zt, t)  # Shape (batch, z_channels, z_height, z_width)

            loss = torch.square(epsilon_prediction - epsilon).mean()

            optimizer.zero_grad()
            loss.backward()

            nn.utils.clip_grad_norm_(denoiser.parameters(), 1.0)

            optimizer.step()
            lr_schedule.step()

            logger.log(loss=loss.item())

            for callback in callbacks:
                callback.check(epoch, step_i, len(train_loader))


def main():
    train_set, val_set = make_celeba(savedir="/ml/data")
    #train_set = Subset(train_set, indices=list(range(128)))

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    schedule = LinearSchedule(beta_start=1e-4, beta_end=0.02, n_steps=1000, device=device)
    autoencoder, _meatadata = AutoEncoder.load(pathlib.Path("checkpoints/autoencoder/v3.safetensors"))
    autoencoder = autoencoder.to(device)

    denoiser = Denoiser(schedule=schedule, 
                        data_channels=autoencoder.latent_channels, 
                        base_channels=128).to(device)

    logger = TrainLogger(window=100, log_vars={"loss": "loss"})
    callbacks: List[TrainingCallback] = [
        PrintLossCallback(logger, step_interval=100),
        PlotLossCallback(logger, step_interval=500, filename="fig/diffusion/losses.png"),
        PlotPredictionsCallback(image_source=DataLoader(val_set, batch_size=5, shuffle=False), autoencoder=autoencoder, denoiser=denoiser, schedule=schedule, filename="fig/diffusion/predictions.png", 
                                step_interval=1000),
        SaveModelCallback(denoiser, file_prefix="checkpoints/denoiser/temp", epoch_interval=1)
    ]

    training_config = TrainingConfig(batch=64, lr=0.0001, epochs=10000)
    train_latent_denoiser(denoiser=denoiser, 
                          autoencoder=autoencoder,
                          schedule=schedule,
                          train_set=train_set,
                          val_set=val_set,
                          config=training_config,
                          logger=logger,
                          callbacks=callbacks)


if __name__ == '__main__':
    main()

