import torch
from torchvision.transforms.v2 import Compose, RandomCrop, RandomHorizontalFlip, ToImage, ToDtype, Normalize

from torchvision.datasets import CelebA, MNIST
from torch.utils.data import Dataset

from dataclasses import dataclass

from typing import Tuple


norm_mean = torch.tensor([0.485, 0.456, 0.406])
norm_std = torch.tensor([0.229, 0.224, 0.225])


def make_celeba(savedir) -> Tuple[Dataset, Dataset]:
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


def make_mnist(savedir) -> Tuple[MNIST, MNIST]:
    transform = Compose([
        ToImage(),
        ToDtype(torch.float32, scale=True),
        Normalize(mean=(0.5,), std=(0.5,))
    ])
    train_set = MNIST(root=savedir, train=True, download=True, transform=transform)
    val_set = MNIST(root=savedir, train=False, download=True, transform=transform)

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

