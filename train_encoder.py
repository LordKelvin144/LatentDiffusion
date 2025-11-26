import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision.transforms.functional import resize as img_resize

from tqdm import tqdm

import numpy as np

import pathlib

from data_utils import make_celeba, denormalize, TrainingConfig
from autoencoder import AutoEncoder
from discriminator import PatchDiscriminator
from train_utils import TrainLogger, TrainingCallback, PrintLossCallback, PlotLossCallback

from typing import Tuple, Optional, List, Union
import matplotlib.pyplot as plt


class ShowcaseAutoencoderCallback(TrainingCallback):
    def __init__(self, image_source: Union[torch.Tensor, DataLoader], 
                 autoencoder: AutoEncoder,
                 discriminator: PatchDiscriminator,
                 epoch_interval: int = 1, step_interval: Optional[int] = None, filename: Optional[str] = None,
                 device: Optional[torch.device] = None):
        super().__init__(epoch_interval=epoch_interval, step_interval=step_interval)
        self._filename = filename
        self._autoencoder = autoencoder
        self._discriminator = discriminator
        self._device = device if device is not None else torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        if isinstance(image_source, DataLoader):
            self._image_source = iter(image_source)
        else:
            self._image_source = image_source

    @torch.no_grad()
    def _showcase_autoencoder(self, x: torch.Tensor):
        n = 5

        z = self._autoencoder.encode(x.detach())
        reconstruction = self._autoencoder.decode(z)

        logits_on_fake = self._discriminator(reconstruction)  # Shape (batch, h_patches, w_patches)
        logits_on_real = self._discriminator(x)
        batch, h_patches, w_patches = logits_on_fake.shape

        fake_here = -logits_on_fake.reshape(batch, 1, h_patches, w_patches)  # Shape (batch, 1, h_patches, w_patches)
        fake_here = torch.clip(fake_here, min=0.0)
        fake_here = img_resize(fake_here, size=reconstruction.shape[-2:])  # Shape (batch, 1, height, width)

        real_here = logits_on_fake.reshape(batch, 1, h_patches, w_patches)  # Shape (batch, 1, h_patches, w_patches)
        real_here = torch.clip(real_here, min=0.0)
        real_here = img_resize(real_here, size=reconstruction.shape[-2:])  # Shape (batch, 1, height, width)

        annotated_reconstruction = 0.5*reconstruction.detach().cpu()
        annotated_reconstruction[:, 0, :, :] += fake_here.cpu()[:, 0, :, :]
        annotated_reconstruction[:, 1, :, :] += real_here.cpu()[:, 0, :, :]

        fake_here = -logits_on_real.reshape(batch, 1, h_patches, w_patches)  # Shape (batch, 1, h_patches, w_patches)
        fake_here = torch.clip(fake_here, min=0.0)
        fake_here = img_resize(fake_here, size=reconstruction.shape[-2:])  # Shape (batch, 1, height, width)

        real_here = logits_on_real.reshape(batch, 1, h_patches, w_patches)  # Shape (batch, 1, h_patches, w_patches)
        real_here = torch.clip(real_here, min=0.0)
        real_here = img_resize(real_here, size=reconstruction.shape[-2:])  # Shape (batch, 1, height, width)

        annotated_real = 0.5*x.detach().cpu()
        annotated_real[:, 0, :, :] += fake_here.cpu()[:, 0, :, :]
        annotated_real[:, 1, :, :] += real_here.cpu()[:, 0, :, :]

        images_denorm = denormalize(x.detach().cpu())
        reconstruction_denorm = denormalize(reconstruction.detach().cpu())
        annotated_fake_denorm = denormalize(annotated_reconstruction)
        annotated_real_denorm = denormalize(annotated_real)

        fig, axs = plt.subplots(n, 4, figsize=(8, 8))
        for i in range(n):
            axs[i,0].imshow(images_denorm[i])
            axs[i,0].axis("off")
            axs[i,1].imshow(reconstruction_denorm[i])
            axs[i,1].axis("off")
            axs[i,2].imshow(annotated_fake_denorm[i])
            axs[i,2].axis("off")
            axs[i,3].imshow(annotated_real_denorm[i])
            axs[i,3].axis("off")

        axs[0,0].set_title("Original")
        axs[0,1].set_title("Recon")
        axs[0,2].set_title("Recon")
        axs[0,3].set_title("Original")

    def __call__(self, epoch: int, step: int, batches_per_epoch: int) -> None:
        if isinstance(self._image_source, torch.Tensor):
            x = self._image_source
        else:
            x = next(self._image_source)[0].to(self._device)
        was_training = self._autoencoder.training
        self._autoencoder.eval()
        self._showcase_autoencoder(x)
        self._autoencoder.train(was_training)

        if self._filename is not None:
            plt.savefig(self._filename, dpi=300)
            plt.close()
        else:
            plt.show()


class SaveModelsCallback(TrainingCallback):
    def __init__(self, autoencoder: AutoEncoder, discriminator: PatchDiscriminator, 
                 autoencoder_file_prefix: Union[str, pathlib.Path], 
                 discriminator_file_prefix: Union[str, pathlib.Path], 
                 epoch_interval: int = 1, step_interval: Optional[int] = None):
        super().__init__(epoch_interval=epoch_interval, step_interval=step_interval)
        self._autoencoder_file_prefix = pathlib.Path(autoencoder_file_prefix)
        self._discriminator_file_prefix = pathlib.Path(discriminator_file_prefix)
        self._discriminator = discriminator
        self._autoencoder = autoencoder

    def __call__(self, epoch: int, step: int, batches_per_epoch: int) -> None:
        model_path = self._autoencoder_file_prefix.parent / (self._autoencoder_file_prefix.name + f"_epoch{epoch}.safetensors")
        print(f"Saving autoencoder checkpoint to {model_path} ...")
        self._autoencoder.save(model_path, metadata={"epoch": epoch, "step": step})
        discriminator_path = self._discriminator_file_prefix.parent / (self._discriminator_file_prefix.name + f"_epoch{epoch}.safetensors")
        print(f"Saving discriminator checkpoint to {discriminator_path} ...")
        self._discriminator.save(discriminator_path, metadata={"epoch": epoch, "step": step})


def train(train_set: Dataset,
          training_config: TrainingConfig,
          autoencoder: AutoEncoder,
          discriminator: PatchDiscriminator,
          logger: TrainLogger,
          callbacks: List[TrainingCallback]):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_loader = DataLoader(train_set, batch_size=training_config.batch, shuffle=True, drop_last=True)

    model_optimizer = torch.optim.Adam(params=autoencoder.parameters(), lr=training_config.lr)
    discriminator_optimizer = torch.optim.Adam(params=discriminator.parameters(), lr=training_config.lr, weight_decay=1e-4)
    logit_bce = nn.BCEWithLogitsLoss()

    discriminator_losses = []

    for epoch in range(training_config.start_epoch, training_config.epochs):
        for step_i, (img, _) in enumerate(tqdm(train_loader)):
            model_optimizer.zero_grad()

            img = img.to(device)

            update_auto_encoder = True  # (step_i & 4 == 0)
            update_discriminator = True  # (step_i & 4 != 0)

            if update_auto_encoder:
                # Freeze discriminator
                for param in discriminator.parameters():
                    param.requires_grad = False
                for param in autoencoder.parameters():
                    param.requires_grad = True

                # Get auto-encoding
                z_distribution = autoencoder.stochastic_encode(img)
                reconstruction = autoencoder.decode(z_distribution.rsample())

                # Plain L2 reconstruction loss
                reconstruction_loss = torch.square(reconstruction-img).mean()

                # Discriminator-based realism loss
                discriminator_on_fake = discriminator(reconstruction)
                # Make loss where generator tries to set class=1 on reconstructed data (i.e. think it is real)
                gan_realism_loss = logit_bce(discriminator_on_fake, torch.ones_like(discriminator_on_fake)).mean()

                # KL divergence loss
                kl_loss = z_distribution.kl(reduction="mean").mean(0)
                
                # Combine to form overall loss
                loss = reconstruction_loss + 0.01*kl_loss + 0.025*gan_realism_loss

                # Save losses to logger
                logger.log(loss=loss.detach(), l2=reconstruction_loss.detach(), kl=kl_loss.detach(), gan=gan_realism_loss.detach())

                loss.backward()
                nn.utils.clip_grad_norm_(autoencoder.parameters(), 1.0)
                model_optimizer.step()

            if update_discriminator:

                discriminator_optimizer.zero_grad()

                for param in autoencoder.parameters():
                    param.requires_grad = False
                for param in discriminator.parameters():
                    param.requires_grad = True

                with torch.no_grad():
                    z_distribution = autoencoder.stochastic_encode(img)
                    reconstruction = autoencoder.decode(z_distribution.rsample())

                discriminator_on_real = discriminator(img)
                discriminator_on_fake = discriminator(reconstruction)

                # Train discriminator to predict class=1 on real and class=0 on reconstructed
                loss = logit_bce(discriminator_on_real, torch.ones_like(discriminator_on_real)).mean() + \
                        logit_bce(discriminator_on_fake, torch.zeros_like(discriminator_on_fake)).mean()

                discriminator_losses.append(loss.item())

                loss.backward()
                discriminator_optimizer.step()

            for callback in callbacks:
                callback.check(epoch, step_i, len(train_loader))


def main():
    train_set, val_set = make_celeba(savedir="/ml/data")
    #train_set = Subset(train_set, indices=list(range(1024)))
    training_config = TrainingConfig(batch=16, lr=0.00005, epochs=15)

    logger = TrainLogger(window=250, log_vars={"loss": "loss", "l2": "l2", "kl": "kl", "gan": "gan"})
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    autoencoder = AutoEncoder(data_channels=3,
                              base_channels=64, 
                              multipliers=(1, 2, 4), 
                              down_sample=(True, True, False),
                              latent_channels=6, stochastic=True).to(device)
    discriminator = PatchDiscriminator(data_channels=3, base_channels=16, multipliers=(1, 2, 4), down_sample=(True, True, True,)).to(device)

    callbacks: List[TrainingCallback] = [
        PrintLossCallback(logger, step_interval=50),
        PlotLossCallback(logger, include_keys=["loss", "l2"], step_interval=50, filename="fig/autoencoder/losses.jpg", logy=True),
        PlotLossCallback(logger, include_keys=["kl", "gan"], step_interval=50, filename="fig/autoencoder/regularization.jpg"),
        ShowcaseAutoencoderCallback(image_source=DataLoader(val_set, batch_size=5),
                                    autoencoder=autoencoder, discriminator=discriminator, filename="fig/autoencoder/predictions.jpg", device=device,
                                    step_interval=50),  # pyright: ignore
        SaveModelsCallback(autoencoder, discriminator, "checkpoints/autoencoder/temp", "checkpoints/discriminator/temp", step_interval=500)
    ]
    train(train_set, 
          autoencoder=autoencoder, discriminator=discriminator, 
          training_config=training_config, logger=logger, 
          callbacks=callbacks
    )
    #train_denoising(train_set, training_config=training_config, 
                    #inherit_autoencoder=pathlib.Path("checkpoints/autoencoder/epoch0.safetensors"),
                    #inherit_discriminator=pathlib.Path("checkpoints/discriminator/epoch0.safetensors"))


if __name__ == '__main__':
    main()

