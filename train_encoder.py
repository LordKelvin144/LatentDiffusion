import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, Subset

import numpy as np

import pathlib

from data_utils import make_celeba, denormalize, TrainingConfig
from autoencoder import AutoEncoder
from discriminator import PatchDiscriminator

from typing import Tuple, Optional
import matplotlib.pyplot as plt


def train_denoising(train_set: Dataset,
                    training_config: TrainingConfig,
                    inherit_autoencoder: Optional[pathlib.Path] = None,
                    inherit_discriminator: Optional[pathlib.Path] = None):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_loader = DataLoader(train_set, batch_size=training_config.batch, shuffle=True, drop_last=True)

    if inherit_autoencoder is not None:
        model, metadata = AutoEncoder.load(inherit_autoencoder)
        model = model.to(device)
        start_epoch = int(metadata["epoch"]) + 1
    else:
        model = AutoEncoder(data_channels=3, base_channels=64, n_blocks=4, latent_channels=16, stochastic=True).to(device)
        start_epoch = 0

    if inherit_discriminator is not None:
        discriminator, metadata = PatchDiscriminator.load(inherit_discriminator)
        distriminator = discriminator.to(device)
    else:
        discriminator = PatchDiscriminator(data_channels=3).to(device)

    model_optimizer = torch.optim.Adam(params=model.parameters(), lr=training_config.lr)
    discriminator_optimizer = torch.optim.Adam(params=discriminator.parameters(), lr=training_config.lr)
    logit_bce = nn.BCEWithLogitsLoss()

    model_losses = []
    l2_losses = []
    kl_losses = []
    gan_realism_losses = []
    discriminator_losses = []

    for epoch in range(start_epoch, training_config.epochs):
        for step_i, (img, _) in enumerate(train_loader):
            # Create a noisy version of original image
            img = img.to(device)
            noisy_image = torch.zeros_like(img)
            #noisy_image[:, :, :, :] += torch.mean(img, dim=1)[:, None, :, :] # + 0.2*torch.randn(*img.shape, device=device)
            noisy_image[:, :, :, :] += img.detach()

            update_auto_encoder = True  # (step_i & 4 == 0)
            update_discriminator = True  # (step_i & 4 != 0)
            #assert update_auto_encoder + update_discriminator == 1

            if update_auto_encoder:
                # Freeze discriminator
                for param in discriminator.parameters():
                    param.requires_grad = False
                for param in model.parameters():
                    param.requires_grad = True

                # Get auto-encoding
                z_distribution = model.stochastic_encode(noisy_image)
                reconstruction = model.decode(z_distribution.rsample())

                # Plain L2 reconstruction loss
                reconstruction_loss = torch.square(img - reconstruction).mean()

                # Discriminator-based realism loss
                discriminator_on_fake = discriminator(reconstruction).mean((-1, -2))
                # Make loss where generator tries to set class=1 on reconstructed data (i.e. think it is real)
                gan_realism_loss = logit_bce(discriminator_on_fake, torch.ones(training_config.batch, device=device))

                # KL divergence loss
                kl_loss = z_distribution.kl(reduction="mean").mean(0)
                
                loss = reconstruction_loss + 0.01*kl_loss + 0.1*gan_realism_loss

                l2_losses.append(reconstruction_loss.item())
                kl_losses.append(kl_loss.item())
                gan_realism_losses.append(gan_realism_loss.item())

                model_losses.append(loss.item())

                model_optimizer.zero_grad()
                loss.backward()
                model_optimizer.step()

            if update_discriminator:
                for param in model.parameters():
                    param.requires_grad = False
                for param in discriminator.parameters():
                    param.requires_grad = True

                with torch.no_grad():
                    z = model.encode(noisy_image)
                    reconstruction = model.decode(z)

                discriminator_on_real = discriminator(img).mean((-1, -2))
                discriminator_on_fake = discriminator(reconstruction).mean((-1, -2))

                #print(f"{discriminator_on_real = }")
                #print(f"{discriminator_on_fake = }")

                # Train discriminator to predict class=1 on real and class=0 on reconstructed
                loss = logit_bce(discriminator_on_real, torch.ones(training_config.batch, device=device)) + \
                        logit_bce(discriminator_on_fake, torch.zeros(training_config.batch, device=device))

                discriminator_losses.append(loss.item())

                discriminator_optimizer.zero_grad()
                loss.backward()
                discriminator_optimizer.step()


            if step_i % 50 == 0:
                print(f"Step {step_i}; model={np.mean(model_losses[-50:]):.4g}; lr={np.mean(l2_losses[-50:]):.4g}; kl={np.mean(kl_losses[-50:]):.4g}; gan.={np.mean(gan_realism_losses[-50:]):.4g}")

            if step_i % 100 == 0:
                n = 5
                with torch.no_grad():
                    z = model.encode(noisy_image.detach())
                    reconstruction = model.decode(z)
                    images_denorm = denormalize(img.detach().cpu())
                    reconstruction_denorm = denormalize(reconstruction.detach().cpu())
                    noisy_images_denorm = denormalize(noisy_image.detach().cpu())

                    z_start = z[0].detach()
                    z_end = z[n-1].detach()

                    z_interpolation = torch.linspace(0, 1, n, device=device)[:, None, None, None]*(z_end-z_start)[None, ...] + z_start[None, ...]
                    img_interpolation = denormalize(model.decode(z_interpolation).cpu())

                fig, axs = plt.subplots(n, 3, figsize=(8, 8))
                for i in range(n):
                    axs[i,0].imshow(noisy_images_denorm[i])
                    axs[i,1].imshow(reconstruction_denorm[i])
                    axs[i,2].imshow(images_denorm[i])
                plt.savefig(f"fig/samples_epoch{epoch}.png", dpi=300)
                fig.clear()
                plt.close()

                fig, axs = plt.subplots(n, 1, figsize=(8,6))
                for i in range(n):
                    axs[i].imshow(img_interpolation[i])
                plt.savefig(f"fig/interpolation_epoch{epoch}.png", dpi=300)
                fig.clear()
                plt.close()

        fig, ax = plt.subplots(figsize=(5,5))
        ax.plot(model_losses, label="Model loss")
        ax.plot(l2_losses, label="L2 loss")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Loss")
        ax.set_xscale("log")
        ax.set_ylim(-0.1, 1.1)
        fig.savefig(f"fig/autoencoder/loss_epoch{epoch}.png", dpi=200)
        fig.clear()
        plt.close()

        checkpoint_dir = pathlib.Path("checkpoints")
        model_path = checkpoint_dir / f"autoencoder/epoch{epoch}.safetensors"
        discriminator_path = checkpoint_dir / f"discriminator/epoch{epoch}.safetensors"

        print(f"Saving model checkpoint to {model_path} ...")
        model.save(model_path, metadata={"epoch": epoch})

        print(f"Saving discriminator checkpoint to {discriminator_path} ...")
        discriminator.save(discriminator_path, metadata={"epoch": epoch})


def main():
    train_set, val_set = make_celeba(savedir="/ml/data")
    #train_set = Subset(train_set, indices=list(range(1024)))
    training_config = TrainingConfig(batch=32, lr=0.0001, epochs=15)

    train_denoising(train_set, training_config=training_config)
    #train_denoising(train_set, training_config=training_config, 
                    #inherit_autoencoder=pathlib.Path("checkpoints/autoencoder/epoch2.safetensors"),
                    #inherit_discriminator=pathlib.Path("checkpoints/discriminator/epoch2.safetensors"))


if __name__ == '__main__':
    main()

