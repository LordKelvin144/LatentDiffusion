import torch
from torchvision.datasets import CelebA
from torchvision.transforms.v2 import Normalize, ToImage, ToDtype, Compose, RandomHorizontalFlip, RandomCrop
from torch.utils.data import Dataset, DataLoader, Subset
#from torch.optim.lr_scheduler import CyclicLR

import numpy as np

from dataclasses import dataclass

from autoencoder import AutoEncoder
#from autoencoder_old import AutoEncoder

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

    train_loader = DataLoader(train_set, batch_size=training_config.batch, shuffle=True)

    model = AutoEncoder(data_channels=3, base_channels=32, n_blocks=3).to(device)

    optimizer = torch.optim.Adam(params=model.parameters(), lr=training_config.lr)
    losses = []

    for epoch in range(training_config.epochs):
        for step_i, (img, _) in enumerate(train_loader):
            img = img.to(device)
            noisy_image = img + 1.0*torch.randn(*img.shape, device=device)

            z = model.encode(noisy_image)
            prediction = model.decode(z)
            loss = torch.square(prediction - img).mean()
            losses.append(loss.item())

            if step_i % 50 == 0:
                print(f"Step {step_i}; loss={np.mean(losses[-50:]):.4g}")

            if step_i % 500 == 0:
                n = 5
                fig, axs = plt.subplots(n, 3)
                images_denorm = denormalize(img.detach().cpu())
                prediction_denorm = denormalize(prediction.detach().cpu())
                noisy_images_denorm = denormalize(noisy_image.detach().cpu())
                for i in range(n):
                    axs[i,0].imshow(noisy_images_denorm[i])
                    axs[i,1].imshow(prediction_denorm[i])
                    axs[i,2].imshow(images_denorm[i])
                plt.show()

                with torch.no_grad():
                    z_start = z[0].detach()
                    z_end = z[n-1].detach()

                    z_interpolation = torch.linspace(0,1,n, device=device)[:, None, None, None]*(z_end-z_start)[None, ...] + z_start[None, ...]
                    img_interpolation = denormalize(model.decode(z_interpolation).cpu())

                fig, axs = plt.subplots(n, 1)
                for i in range(n):
                    axs[i].imshow(img_interpolation[i])
                plt.show()




            optimizer.zero_grad()
            loss.backward()
            optimizer.step()




def main():
    train_set, val_set = make_celeba(savedir="/ml/data")
    #train_set = Subset(train_set, list(range(1000)))
    training_config = TrainingConfig(batch=32, lr=0.001, epochs=15)
    train_denoising(train_set, val_set, training_config=training_config)


if __name__ == '__main__':
    main()

