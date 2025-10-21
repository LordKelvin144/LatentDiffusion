import torch
from torch import nn
from torchvision.datasets import MNIST
from torch.utils.data import Dataset, DataLoader, Subset
from torch.optim.lr_scheduler import CosineAnnealingLR

import numpy as np

from autoencoder import AutoEncoder
from denoiser import Denoiser
from data_utils import TrainingConfig, make_mnist
from diffusion import sample_training_examples, reverse_step
from schedule import Schedule, LinearSchedule


import pathlib


def illustrate_model_on_batch(schedule: Schedule, denoiser: Denoiser, x0: torch.Tensor, xt: torch.Tensor, t: torch.Tensor):

    import matplotlib.pyplot as plt
    n = 5
    with torch.no_grad():
        x0_denorm = x0.detach().cpu().squeeze(1)
        xt_denorm = xt.detach().cpu().squeeze(1)

        # Try to reconstruct image
        x_recon = torch.zeros_like(xt[:n])
        for i in range(n):
            steps = int(t[i])
            this_x = xt.detach()[i, None]
            this_t = t.detach()[i, None]
            for _ in range(steps):
                epsilon_prediction = denoiser(this_x, this_t)
                
                this_x = reverse_step(schedule, this_x, this_t, epsilon_prediction)
                this_t = this_t - 1

            x_recon[i] = this_x.squeeze(0)

        x_recon_denorm = x_recon.detach().cpu().squeeze(1)

    fig, axs = plt.subplots(n, 3, figsize=(6, 8))
    for i in range(n):
        axs[i,0].imshow(x0_denorm[i])
        axs[i,1].imshow(xt_denorm[i])
        axs[i,2].imshow(x_recon_denorm[i])


def train_latent_denoiser(schedule: Schedule, train_set: MNIST, val_set: MNIST, training_config: TrainingConfig):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_loader = DataLoader(train_set, batch_size=training_config.batch, shuffle=True, drop_last=True)

    denoiser = Denoiser(schedule=schedule, 
                        data_channels=1, 
                        base_channels=64).to(device)

    optimizer = torch.optim.Adam(params=denoiser.parameters(), lr=training_config.lr)

    losses = []

    for epoch in range(training_config.epochs):
        for step_i, (x0, _) in enumerate(train_loader):
            x0 = x0.to(device)
            xt, t, epsilon = sample_training_examples(schedule, x0)

            # Get model prediction
            epsilon_prediction = denoiser(xt, t)  # Shape (batch, z_channels, z_height, z_width)

            loss = torch.square(epsilon_prediction - epsilon).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        print(f"Epoch {epoch}; loss={np.mean(losses[-len(train_loader):]):.4g}")

        if epoch % 10 == 0:
            import matplotlib.pyplot as plt

            illustrate_model_on_batch(schedule, denoiser, x0=x0, xt=xt, t=t)
            plt.show()
            plt.close()

            fig, ax = plt.subplots(figsize=(6, 6))
            ax.plot(losses)
            ax.set_xscale("log")
            ax.set_xlabel("iteration")
            ax.set_ylabel("loss")
            #fig.savefig("fig/diffusion/losses.png")
            plt.show()
            plt.close()

        #model_path = pathlib.Path(f"checkpoints/denoiser/epoch{epoch}.safetensors")
        #print(f"Saving denoiser checkpoint to {model_path} ...")
        #denoiser.save(model_path, metadata={"epoch": epoch, "step": step_i})


def main():
    train_set, val_set = make_mnist(savedir="/ml/data")
    training_config = TrainingConfig(batch=64, lr=0.0001, epochs=10000)
    schedule = LinearSchedule(beta_start=1e-4, beta_end=0.02, n_steps=1000,
                              device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    #train_set = Subset(train_set, indices=list(range(1024)))
    train_latent_denoiser(schedule, train_set, val_set, training_config=training_config)


if __name__ == '__main__':
    main()

