import torch
from torch import nn
from torchvision.datasets import MNIST
from torch.utils.data import Dataset, DataLoader, Subset
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR

import numpy as np

from denoiser import Denoiser
from data_utils import TrainingConfig, make_mnist
from diffusion import forward_process, reverse_step, DiffusionLossTracker
from schedule import Schedule, LinearSchedule

import pathlib


@torch.no_grad()
def illustrate_model_on_batch(denoiser: Denoiser, x0: torch.Tensor, xt: torch.Tensor, t: torch.Tensor):
    import matplotlib.pyplot as plt
    n = 5
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
            
            this_x = reverse_step(denoiser.schedule, this_x, this_t, epsilon_prediction)
            this_t = this_t - 1

        x_recon[i] = this_x.squeeze(0)

    x_recon_denorm = x_recon.detach().cpu().squeeze(1)

    fig, axs = plt.subplots(n, 3, figsize=(6, 8))
    for i in range(n):
        axs[i,0].imshow(x0_denorm[i])
        axs[i,1].imshow(xt_denorm[i])
        axs[i,2].imshow(x_recon_denorm[i])


def illustrate_generation(denoiser: Denoiser):
    import matplotlib.pyplot as plt
    x = denoiser.generate(img_shape=(28, 28), batch_size=15).cpu()
    x_denorm = x.detach().cpu().squeeze(1)
    
    fig, axs = plt.subplots(5, 3, figsize=(6, 8))
    for i in range(15):
        axs.flatten()[i].imshow(x_denorm[i])


def train_denoiser(schedule: Schedule, train_set: MNIST, training_config: TrainingConfig):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_loader = DataLoader(train_set, batch_size=training_config.batch, shuffle=True, drop_last=True)

    denoiser = Denoiser(schedule=schedule, 
                        data_channels=1, 
                        base_channels=64).to(device)
    #denoiser, _ = Denoiser.load(pathlib.Path("checkpoints/mnist/v1.safetensors"), schedule=schedule)
    denoiser = denoiser.to(device)

    optimizer = torch.optim.Adam(params=denoiser.parameters(), lr=training_config.lr)
    scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=50)

    losses = []
    loss_tracker = DiffusionLossTracker(schedule)

    denoiser.train()
    for epoch in range(training_config.epochs):
        for step_i, (x0, _) in enumerate(train_loader):
            x0 = x0.to(device)

            # Get diffusion sample
            t = torch.randint(low=1, high=schedule.n_steps+1, size=(x0.shape[0],), device=device)
            xt, epsilon = forward_process(schedule, x0, t)

            # Get model prediction
            epsilon_prediction = denoiser(xt, t)  # Shape (batch, z_channels, z_height, z_width)

            loss_by_batch = torch.square(epsilon_prediction - epsilon).mean((1, 2, 3))  # Shape batch
            loss = loss_by_batch.mean()

            optimizer.zero_grad()
            loss.backward()

            nn.utils.clip_grad_norm_(denoiser.parameters(), 1.0)
                
            optimizer.step()
            scheduler.step()

            loss_tracker.add(t=t, losses=loss_by_batch.detach())

            if step_i % 100 == 0:
                import matplotlib.pyplot as plt
                # loss_tracker.current_errorbar(nbins=10)
                loss_tracker.bin_loss_over_time()
                plt.xscale("log")
                plt.yscale("log")
                plt.savefig("fig/mnist/bin_losses.jpg", dpi=200)
                plt.close()
                print(f"Step {step_i}; loss={np.mean(losses[-100:]):.4g}")

            losses.append(loss.item())


        print(f"Epoch {epoch}; loss={np.mean(losses[-len(train_loader):]):.4g}")

        # Plot diagnoistics
        denoiser.eval()
        import matplotlib.pyplot as plt

        illustrate_model_on_batch(denoiser, x0=x0, xt=xt, t=t)
        plt.savefig("fig/mnist/predictions.png")
        plt.close()

        illustrate_generation(denoiser)
        plt.savefig("fig/mnist/generated.png")
        plt.close()

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(losses)
        ax.set_xlabel("iteration")
        ax.set_ylabel("loss")
        ax.set_yscale("log")
        fig.savefig("fig/mnist/losses.png")
        #plt.show()
        plt.close()
        denoiser.train()

        # Save model
        model_path = pathlib.Path(f"checkpoints/mnist/epoch{epoch}.safetensors")
        print(f"Saving denoiser checkpoint to {model_path} ...")
        denoiser.save(model_path, metadata={"epoch": epoch, "step": step_i})


def showcase_trained():
    import matplotlib.pyplot as plt
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    schedule = LinearSchedule(beta_start=1e-4, beta_end=0.02, n_steps=1000,
                              device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    model_path = pathlib.Path("checkpoints/mnist/v1.safetensors")

    denoiser, _ = Denoiser.load(model_path, schedule)
    denoiser = denoiser.to(device)
    denoiser.eval()
    illustrate_generation(denoiser)
    plt.show()


def train():
    train_set, val_set = make_mnist(savedir="/ml/data")
    training_config = TrainingConfig(batch=128, lr=0.0001, epochs=10000)
    schedule = LinearSchedule(beta_start=1e-4, beta_end=0.02, n_steps=1000,
                              device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    #train_set = Subset(train_set, indices=list(range(1024)))
    train_denoiser(schedule, train_set, training_config=training_config)


if __name__ == '__main__':
    #showcase_trained()
    train()

