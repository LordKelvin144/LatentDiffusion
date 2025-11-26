import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, Subset
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, ChainedScheduler, StepLR

from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt

from autoencoder import AutoEncoder, EncodedImgDataset
from denoiser import Denoiser
from data_utils import make_celeba, denormalize
from train_utils import TrainLogger, TrainingCallback, PrintLossCallback, PlotLossCallback
from diffusion import reverse_step, forward_process, estimate_initial, DiffusionLossTracker
from schedule import Schedule, LinearSchedule

import pathlib

from typing import List, Optional, Union, Dict

from dataclasses import dataclass


@dataclass
class TrainingConfig:
    batch: int
    lr: float
    epochs: int
    denoiser_channels: int
    warmup_steps: int = 200
    warmup_factor: float = 0.1
    cosine_annealing_steps: int = 1_000_000
    cosine_annealing_factor: float = 0.1


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

    def __call__(self, epoch: int, step: int, batches_per_epoch: int) -> None:
        if isinstance(self._image_source, torch.Tensor):
            x0 = self._image_source
        else:
            x0 = next(self._image_source)[0].to(self._schedule.beta.device)  # TODO: Make this cleaner
        was_training = self._denoiser.training
        self._denoiser.eval()
        plot_denoiser_predictions(x0=x0, schedule=self._schedule, autoencoder=self._autoencoder, denoiser=self._denoiser)
        self._denoiser.train(was_training)

        if self._filename is not None:
            plt.savefig(self._filename, dpi=300)
            plt.close()
        else:
            plt.show()


class PlotBinnedLossCallback(TrainingCallback):
    def __init__(self, loss_tracker: DiffusionLossTracker, filename: Optional[str] = None, xlog: bool = True, ylog: bool = True, epoch_interval: int = 1, step_interval: Optional[int] = None):
        super().__init__(epoch_interval=epoch_interval, step_interval=step_interval)
        self._xlog = xlog
        self._ylog = ylog
        self._loss_tracker = loss_tracker
        self._filename = filename

    def __call__(self, epoch: int, step: int, batches_per_epoch: int) -> None:
        self._loss_tracker.bin_loss_over_time()
        if self._xlog:
            plt.xscale("log")
        if self._ylog:
            plt.yscale("log")

        if self._filename is not None:
            plt.savefig(self._filename, dpi=200)
            plt.close()
        else:
            plt.show()


class SaveDenoiserCallback(TrainingCallback):
    def __init__(self, denoiser: Denoiser, file_prefix: Union[str, pathlib.Path], autoencoder_hash: str, epoch_interval: int = 1, step_interval: Optional[int] = None):
        super().__init__(epoch_interval=epoch_interval, step_interval=step_interval)
        self._file_prefix = pathlib.Path(file_prefix)
        self._denoiser = denoiser
        self._autoencoder_hash = autoencoder_hash

    def __call__(self, epoch: int, step: int, batches_per_epoch: int) -> None:
        model_path = self._file_prefix.parent / (self._file_prefix.name + f"_epoch{epoch}.safetensors")
        print(f"Saving denoiser checkpoint to {model_path} ...")
        self._denoiser.save(model_path, metadata={"epoch": epoch, "step": step, "autoencoder": self._autoencoder_hash})


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


@torch.no_grad()
def illustrate_generation(denoiser: Denoiser, autoencoder: AutoEncoder):
    import matplotlib.pyplot as plt
    z0 = denoiser.generate(img_shape=(54, 44), batch_size=15)
    x0 = autoencoder.decode(z0)
    x_denorm = denormalize(x0.detach().cpu())
    
    fig, axs = plt.subplots(5, 3, figsize=(6, 8))
    for i in range(15):
        axs.flatten()[i].imshow(x_denorm[i])
        axs.flatten()[i].axis("off")


def train_latent_denoiser(denoiser: Denoiser, 
                          autoencoder: AutoEncoder,
                          schedule: Schedule,
                          train_set: Dataset,
                          config: TrainingConfig,
                          logger: TrainLogger,
                          diffusion_loss_tracker: Optional[DiffusionLossTracker] = None,
                          callbacks: Optional[List[TrainingCallback]] = None):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create a dataset which loads images and encodes them,
    # optionally with caching
    encoded_train_set = EncodedImgDataset(train_set, autoencoder, directory=pathlib.Path("./encoded_img"))
    train_loader = DataLoader(encoded_train_set, batch_size=config.batch, shuffle=True, drop_last=True, pin_memory=True)

    optimizer = torch.optim.Adam(params=denoiser.parameters(), lr=config.lr)
    lr_schedule = ChainedScheduler([LinearLR(optimizer, start_factor=config.warmup_factor, total_iters=config.warmup_steps),
                                    #StepLR(optimizer, step_size=10000, gamma=0.5),
                                    CosineAnnealingLR(optimizer, T_max=config.cosine_annealing_steps, eta_min=config.lr*config.cosine_annealing_factor)
                                    ],
                                   optimizer=optimizer)

    if callbacks is None:
        callbacks = []

    denoiser.train()
    for epoch in range(config.epochs):
        step_i = 0
        for step_i, (z0, _) in enumerate(tqdm(train_loader)):
            z0 = z0.to(device)

            # Get noised examples and their corresponding times and epsilons
            #p = torch.sqrt(schedule.alpha_bar[1:])
            #p = p / p.sum()
            #p = 0.5*p + 0.5*torch.ones_like(p) / p.numel()
            #categorical = torch.distributions.categorical.Categorical(probs=p)
            #t = categorical.sample((z0.shape[0],)) + 1

            #import matplotlib.pyplot as plt
            #plt.plot(p.cpu().numpy())
            #plt.show()

            t = torch.randint(low=1, high=schedule.n_steps+1, size=(z0.shape[0],), device=schedule.beta.device)
            zt, epsilon = forward_process(schedule, z0, t)

            # Get model prediction
            #epsilon_prediction = denoiser(zt, t)  # Shape (batch, z_channels, z_height, z_width)
            epsilon_prediction = denoiser(zt, t)

            loss_per_item = torch.square(epsilon_prediction - epsilon).mean((1, 2, 3))
            loss = loss_per_item.mean()

            loss.backward()

            nn.utils.clip_grad_norm_(denoiser.parameters(), 1.0)

            optimizer.step()
            lr_schedule.step()

            logger.log(loss=loss.detach())
            if diffusion_loss_tracker is not None:
                diffusion_loss_tracker.add(t=t, losses=loss_per_item.detach())

            for callback in callbacks:
                callback.check(epoch, step_i, len(train_loader))


def train():
    train_set, val_set = make_celeba(savedir="/ml/data")
    #train_set = Subset(train_set, indices=list(range(2048*256)))

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    training_config = TrainingConfig(batch=64, lr=0.0001, epochs=10000, denoiser_channels=32)

    schedule = LinearSchedule(beta_start=1e-4, beta_end=0.02, n_steps=1000, device=device)
    autoencoder, _meatadata = AutoEncoder.load(pathlib.Path("checkpoints/autoencoder/bigger_6.safetensors"))
    autoencoder = autoencoder.to(device)
    autoencoder.eval()

    denoiser = Denoiser(schedule=schedule, 
                        data_channels=autoencoder.latent_channels, 
                        base_channels=training_config.denoiser_channels,
                        multipliers=(1, 2, 3, 4,),
                        down_sample=(False, False, False, True,),
                        dropout_rate=0.2)
    #denoiser, denoiser_metadata = Denoiser.load(path=pathlib.Path("checkpoints/denoiser/temp_epoch0.safetensors"), schedule=schedule)
    #assert denoiser_metadata["autoencoder"] == autoencoder.sha256_digest()
    denoiser = denoiser.to(device)

    logger = TrainLogger(window=500, log_vars={"loss": "loss"})
    diffusion_loss_tracker = DiffusionLossTracker(schedule)

    callbacks: List[TrainingCallback] = [
        PrintLossCallback(logger, step_interval=250),
        PlotLossCallback(logger, step_interval=250, filename="fig/diffusion/losses.jpg"),
        PlotPredictionsCallback(image_source=DataLoader(val_set, batch_size=5, shuffle=False), autoencoder=autoencoder, denoiser=denoiser, schedule=schedule, filename="fig/diffusion/predictions.jpg", 
                                step_interval=500),
        PlotBinnedLossCallback(diffusion_loss_tracker, step_interval=50, filename="fig/diffusion/bin_losses.jpg"),
        SaveDenoiserCallback(denoiser, file_prefix="checkpoints/denoiser/temp", autoencoder_hash=autoencoder.sha256_digest(), epoch_interval=1)
    ]

    train_latent_denoiser(denoiser=denoiser, 
                          autoencoder=autoencoder,
                          schedule=schedule,
                          train_set=train_set,
                          config=training_config,
                          logger=logger,
                          diffusion_loss_tracker=diffusion_loss_tracker,
                          callbacks=callbacks)


def showcase_trained():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    schedule = LinearSchedule(beta_start=1e-4, beta_end=0.02, n_steps=1000, device=device)
    autoencoder, _meatadata = AutoEncoder.load(pathlib.Path("checkpoints/autoencoder/v5.safetensors"))
    autoencoder = autoencoder.to(device)
    autoencoder.eval()

    #denoiser_path = pathlib.Path("checkpoints/denoiser/v5.safetensors")
    denoiser_path = pathlib.Path("checkpoints/denoiser/epoch4.safetensors")
    #denoiser_path = pathlib.Path("checkpoints/denoiser/new_epoch14_loss_0_034.safetensors")
    denoiser, denoiser_metadata = Denoiser.load(denoiser_path, schedule)
    assert denoiser_metadata["autoencoder"] == autoencoder.sha256_digest()
    denoiser.to(device)
    denoiser.eval()

    illustrate_generation(denoiser, autoencoder)
    plt.savefig("fig/diffusion/generated_new.png", dpi=300)
    plt.show()


if __name__ == '__main__':
    train()
    #showcase_trained()

