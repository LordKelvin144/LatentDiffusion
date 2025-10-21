import torch

from schedule import Schedule, LinearSchedule


from typing import Tuple


@torch.no_grad()
def forward_process(schedule: Schedule, x0: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes the forward diffusion from an initial point x0, until point t. Acts on a batch of data and timepoints.
    :argument schedule: The diffusion noise schedule to use
    :argument x0: A batch of noise-free data at time 0, shape (batch, *data_shape) where data_shape is 0 or more data dimensions
    :argument t: A batch of times to use for applying noise, shape (batch,)
    :returns xt: A batch of noised versions of x0, shape (batch, *data_shape)
    :returns epsilon: A batch of ground-truth epsilon noise of shape (batch, *data_shape) used to get xt from x0
    """

    batch = x0.shape[0]
    data_shape = x0.shape[1:]

    epsilon = torch.randn(x0.shape, device=x0.device, dtype=x0.dtype)

    # Extract which means and variances are relevant at the given time points
    alpha_bar = schedule.alpha_bar[t]  # Shape (batch,)
    alpha_bar = alpha_bar.view(batch, *[1 for _ in range(len(data_shape))])  # Shape (batch, 1, ..., 1); broadcastable with x0
    xt = torch.sqrt(alpha_bar)*x0 + torch.sqrt(1.0-alpha_bar)*epsilon  # Shape (batch, *data_shape)
    return xt, epsilon


@torch.no_grad()
def sample_training_examples(schedule: Schedule, x0: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Takes a batch of noise-free samples x0 of shape (batch, *data_shape) and computes the regression data (xt, t, epsilon) for training a predictor for reverse diffusion.
    :argument schedule: The diffusion noise schedule to use
    :argument x0: A batch of noise-free data at time 0, shape (batch, *data_shape) where data_shape is 0 or more data dimensions
    :returns xt: A batch of noised versions of x0, shape (batch, *data_shape)
    :returns t: A batch of times to use for applying noise, shape (batch,)
    :returns epsilon: The correct noise value used for generating xt, shape (batch, *data_shape)
    """

    batch = x0.shape[0]
    t = torch.randint(low=1, high=schedule.n_steps, size=(batch,), device=x0.device)
    xt, epsilon = forward_process(schedule, x0, t)
    return xt, t, epsilon


@torch.no_grad()
def reverse_step(schedule: Schedule, xt: torch.Tensor, t: torch.Tensor, epsilon: torch.Tensor) -> torch.Tensor:
    """
    Takes a noise estimate and computes the estimated previous x_{t-1} from it. Operates on a batch.
    :argument schedule: The diffusion noise schedule to use
    :argument xt: A batch of noised data, shape (batch, *data_shape)
    :argument t: A batch of current times at each image in xt, shape (batch,)
    :argument epsilon: The (estimated) noise epsilon which has been applied from x0 to xt, shape (batch, *data_shape)
    :returns xprev: Sample of x_{t-1} using approximate epsilon
    """
    batch = xt.shape[0]
    data_shape = xt.shape[1:]

    inject_noise = torch.randn(xt.shape, device=xt.device, dtype=xt.dtype)

    beta = schedule.beta[t]  # Shape (batch,)
    alpha_bar = schedule.alpha_bar[t]  # Shape (batch,)
    pre_epsilon_factor = beta / torch.sqrt(1-alpha_bar)
    inv_sqrt_alpha = 1.0 / torch.sqrt(1.0-beta)
    sigma = torch.sqrt(beta)

    broadcast_shape = [batch] + [1 for _ in range(len(data_shape))]
    pre_epsilon_factor = pre_epsilon_factor.view(broadcast_shape)
    inv_sqrt_alpha = inv_sqrt_alpha.view(broadcast_shape)
    sigma = sigma.view(broadcast_shape)

    x_prev = inv_sqrt_alpha * (xt - pre_epsilon_factor*epsilon) + sigma * inject_noise
    return x_prev


def analytic_epsilon(schedule: Schedule, x0: torch.Tensor, xt: torch.Tensor, t: torch.Tensor):
    batch = x0.shape[0]
    data_shape = xt.shape[1:]
    broadcast_shape = [batch] + [1 for _ in range(len(data_shape))]

    alpha_bar = schedule.alpha_bar[t]  # Shape (batch,)
    alpha_bar = alpha_bar.view(broadcast_shape)

    true_epsilon = (xt - torch.sqrt(alpha_bar)*x0) / torch.sqrt(1.0-alpha_bar)
    assert true_epsilon.shape == xt.shape
    return true_epsilon


def _illustrate_forward():
    from sklearn.datasets import make_swiss_roll
    import matplotlib.pyplot as plt
    x, _ = make_swiss_roll(n_samples=1000)
    x = torch.from_numpy(x).cpu()

    schedule = LinearSchedule(beta_start=1e-4, beta_end=0.02, n_steps=1000, device=torch.device("cpu"))

    ts = [0, 50, 100, 200, 300, 400, 500, 750, 1000]
    fig, axs = plt.subplots(3, 3)
    for i, t in enumerate(ts):
        xt, _ = forward_process(schedule, x0=x, t=torch.tensor([t]).cpu().tile((1000,)))
        axs.flatten()[i].scatter(xt[:, 0], xt[:, 2], label=f"t={t}")
        axs.flatten()[i].legend()
    plt.show()


@torch.no_grad()
def _illustrate_backward():
    from sklearn.datasets import make_swiss_roll
    import matplotlib.pyplot as plt

    batch = 1000

    x_initial, _ = make_swiss_roll(n_samples=batch)
    x_initial = torch.from_numpy(x_initial).cpu()
    schedule = LinearSchedule(beta_start=1e-4, beta_end=0.02, n_steps=1000, device=torch.device("cpu"))

    data_shape = x_initial.shape[1:]

    t_final = torch.ones(batch, dtype=torch.int64, device=x_initial.device)*schedule.n_steps

    x_final, epsilon = forward_process(schedule, x0=x_initial, t=t_final)
    x = x_final
    x_recon = torch.zeros((schedule.n_steps+1, batch, *data_shape), dtype=x.dtype, device=x.device)
    x_recon[schedule.n_steps, :, :] = x_final

    for t in range(schedule.n_steps, 0, -1):
        t_batched = torch.ones(batch, dtype=torch.int64, device=x.device) * t
        true_epsilon = analytic_epsilon(schedule, x0=x_initial, xt=x, t=t_batched)
        if t == schedule.n_steps:
            assert torch.allclose(true_epsilon, epsilon)

        x = reverse_step(schedule, xt=x, t=t_batched, epsilon=true_epsilon)
        x_recon[t-1] = x.detach()

    ts = [0, 50, 100, 200, 300, 400, 500, 750, 1000]
    fig, axs = plt.subplots(3, 3)
    for i, t in enumerate(ts):
        axs.flatten()[i].scatter(x_recon[t, :, 0], x_recon[t, :, 2], label=f"t={t}")
        axs.flatten()[i].legend()
    plt.show()


if __name__ == '__main__':
    _illustrate_forward()
    _illustrate_backward()

