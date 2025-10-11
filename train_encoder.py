import torch
from torch import nn
from torchvision.datasets import CelebA
from torchvision.transforms.v2 import Normalize, ToImage, ToDtype, Compose, RandomHorizontalFlip, RandomCrop
from torch.utils.data import Dataset, DataLoader, Subset
#from torch.optim.lr_scheduler import CyclicLR

import numpy as np

from dataclasses import dataclass
import pathlib

from autoencoder import AutoEncoder
#from autoencoder_old import AutoEncoder
from discriminator import PatchDiscriminator

from typing import Tuple
import matplotlib.pyplot as plt

norm_mean = torch.tensor([0.485, 0.456, 0.406])
norm_std = torch.tensor([0.229, 0.224, 0.225])


def make_celeba(savedir, ) -> Tuple[Dataset, Dataset]:
    transform = Compose([
        RandomCrop(size=(216, 176)),
        RandomHorizontalFlip(p=0.3),
        ToImage(),
        ToDtype(torch.float32, scale=True),
        Normalize(mean=norm_mean.numpy().tolist(), std=norm_std.numpy().tolist())
    ])
    train_set = CelebA(root=savedir, split="train", download=True, transform=transform)
    val_set = CelebA(root=savedir, split="valid", download=True, transform=transform)

    return train_set, val_set


@torch.no_grad()
def denormalize(img_tensor: torch.Tensor) -> torch.Tensor:
    img_01 = img_tensor * norm_std[None, :, None, None] + norm_mean[None, :, None, None]
    img_01 = img_01.clamp(min=0.0, max=1.0).permute(0, 2, 3, 1)
    return img_01


@dataclass
class TrainingConfig:
    batch: int
    lr: float
    epochs: int


def train_denoising(train_set: Dataset, val_set: Dataset, training_config: TrainingConfig):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_loader = DataLoader(train_set, batch_size=training_config.batch, shuffle=True, drop_last=True)

    model = AutoEncoder(data_channels=3, base_channels=32, n_blocks=3).to(device)
    discriminator = PatchDiscriminator(data_channels=3).to(device)

    model_optimizer = torch.optim.Adam(params=model.parameters(), lr=training_config.lr)
    discriminator_optimizer = torch.optim.Adam(params=discriminator.parameters(), lr=training_config.lr)
    logit_bce = nn.BCEWithLogitsLoss()
    model_losses = []
    discriminator_losses = []

    for epoch in range(training_config.epochs):
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
                z = model.encode(noisy_image)
                reconstruction = model.decode(z)

                # Plain L2 reconstruction loss
                reconstruction_loss = torch.square(reconstruction - img).mean()

                discriminator_on_fake = discriminator(reconstruction).mean((-1, -2))
                # Make loss where generator tries to set class=1 on reconstructed data (i.e. think it is real)
                discriminator_loss = logit_bce(discriminator_on_fake, torch.ones(training_config.batch, device=device))
                
                loss = reconstruction_loss + 0.05*discriminator_loss
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
                print(f"Step {step_i}; model_loss={np.mean(model_losses[-50:]):.4g}; discriminator_loss={np.mean(discriminator_losses[-50:]):.4g}")

            if step_i % 500 == 0:
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

                if step_i > 0:
                    fig, ax = plt.subplots(figsize=(5,5))
                    ax.plot(model_losses, label="Model loss")
                    ax.plot(discriminator_losses, label="Discriminator loss")
                    ax.set_xlabel("Iteration")
                    ax.set_ylabel("Loss")
                    ax.set_ylim(None, 3)
                    fig.savefig(f"fig/loss_epoch{epoch}.png", dpi=200)
                    fig.clear()
                    plt.close()

                    checkpoint_dir = pathlib.Path("checkpoints")
                    model_path = checkpoint_dir / f"autoencoder/epoch{epoch}_temp.safetensors"
                    discriminator_path = checkpoint_dir / f"discriminator/epoch{epoch}_temp.safetensors"

                    print(f"Saving model checkpoint to {model_path} ...")
                    model.save(model_path, metadata={"epoch": epoch, "step": step_i})

                    print(f"Saving discriminator checkpoint to {discriminator_path} ...")
                    discriminator.save(discriminator_path, metadata={"epoch": epoch, "step": step_i})


def main():
    train_set, val_set = make_celeba(savedir="/ml/data")
    #train_set = Subset(train_set, list(range(1000)))
    training_config = TrainingConfig(batch=32, lr=0.001, epochs=15)
    train_denoising(train_set, val_set, training_config=training_config)


if __name__ == '__main__':
    main()

