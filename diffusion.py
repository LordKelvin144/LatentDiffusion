import torch
import numpy as np

from schedule import Schedule, LinearSchedule
import itertools

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


@torch.no_grad()
def estimate_initial(schedule: Schedule, xt: torch.Tensor, t: torch.Tensor, epsilon: torch.Tensor) -> torch.Tensor:
    """
    Takes a noise estimate and computes the estimated previous x_0 from it. Operates on a batch.
    :argument schedule: The diffusion noise schedule to use
    :argument xt: A batch of noised data, shape (batch, *data_shape)
    :argument t: A batch of current times at each image in xt, shape (batch,)
    :argument epsilon: The (estimated) noise epsilon which has been applied from x0 to xt, shape (batch, *data_shape)
    :returns x0: Approximation of x0 according to epsilon
    """
    batch = xt.shape[0]
    data_shape = xt.shape[1:]

    # Extract which means and variances are relevant at the given time points
    alpha_bar = schedule.alpha_bar[t]  # Shape (batch,)
    alpha_bar = alpha_bar.view(batch, *[1 for _ in range(len(data_shape))])  # Shape (batch, 1, ..., 1); broadcastable with x0
    x0 = (xt - torch.sqrt(1.0-alpha_bar)*epsilon) / torch.sqrt(alpha_bar)
    return x0


def analytic_epsilon(schedule: Schedule, x0: torch.Tensor, xt: torch.Tensor, t: torch.Tensor):
    batch = x0.shape[0]
    data_shape = xt.shape[1:]
    broadcast_shape = [batch] + [1 for _ in range(len(data_shape))]

    alpha_bar = schedule.alpha_bar[t]  # Shape (batch,)
    alpha_bar = alpha_bar.view(broadcast_shape)

    true_epsilon = (xt - torch.sqrt(alpha_bar)*x0) / torch.sqrt(1.0-alpha_bar)
    assert true_epsilon.shape == xt.shape
    return true_epsilon


class DiffusionLossTracker:
    """
    A class for tracking losses when training diffusion models. Facilitates useful operations like plotting loss vs schedule time steps.
    """
    def __init__(self, schedule: Schedule):
        self._schedule = schedule

        self._losses = np.zeros(0)
        self._t = np.zeros(0, dtype=np.int64)
        self._iterations = 0

        self._ema_per_timestep = np.ones(schedule.n_steps)

    @torch.no_grad()
    def add(self, t: torch.Tensor, losses: torch.Tensor):
        """Takes in a batch of time steps t and per-batch losses 'losses' and registers them with the tracker."""
        assert t.shape == losses.shape
        t_cpu = t.cpu().numpy()
        losses_cpu = losses.cpu().numpy()
        self._iterations += 1
        self._t = np.hstack((self._t, t_cpu))
        self._losses = np.hstack((self._losses, losses_cpu))

        for i, sub_t in enumerate(t):
            self._ema_per_timestep[sub_t-1] = self._ema_per_timestep[sub_t-1]*0.99 + 0.01*losses_cpu[i]

    def current_errorbar(self, nbins: int = 5, window: int = 10):
        import matplotlib.pyplot as plt
        bins = np.linspace(0, self._schedule.n_steps, nbins+1)[None, :]

        idx = self._t[-window:, None] <= bins[None, :]  # Shape (window, bins+1), t such that t <= bin[j]
        n_per_bin = np.diff(np.sum(idx, axis=0))  # Shape (bins,); number of elements where bin[j] <= t < bin[j+1]

        selected_losses = np.where(idx, self._losses[-window:, None], 0.0)  # Shape (window, bins+1); keep elements where t < bin[j]
        cumsum = np.sum(selected_losses, axis=0)  # Shape (bins+1); sum of elements smaller than bin[j]
        mean = np.diff(cumsum, axis=0) / n_per_bin  # Shape (bins,); sum of elements between bin[i] and bin[i+1]

        cumsum_loss2 = np.sum(selected_losses**2, axis=0)
        e_loss2 = np.diff(cumsum_loss2, axis=0) / n_per_bin

        std = np.sqrt(e_loss2 - mean**2) / np.sqrt(n_per_bin + 1)

        plt.errorbar(bins[:-1] + 0.5*bins[1] - 0.5*bins[0], mean, std)
        plt.xlabel("Schedule time step")
        plt.ylabel("Loss")

    def bin_loss_over_time(self, nbins: int = 5, window: int = 10):
        import matplotlib.pyplot as plt
        bins = np.linspace(0, self._schedule.n_steps, nbins+1)

        iterations = np.linspace(0, self._iterations, self._t.size) + 1
        window = int(window * (self._t.size / self._iterations))

        idx = self._t[:, None] <= bins[None, :]  # Shape (iterations, bins+1), t such that t <= bin[j]
        n_per_bin_cumsum = np.diff(np.cumsum(idx, axis=0), axis=1)  # Shape (iterations, bins,); number of elements where bin[j] <= t < bin[j+1]

        selected_losses = np.where(idx, self._losses[:, None], 0.0)  # Shape (iterations, bins+1); keep elements where t < bin[j]
        cumsum = np.cumsum(selected_losses, axis=0)  # Shape (iterations, bins+1); sum of elements smaller than bin[j]

        sum_in_bin = np.diff(cumsum, axis=1)  # Shape (iterations, bins); [i,j] holds sum over elements up to iteration i such that bin[j] <= t < bin[j+1]
        windowed_sum_in_bin = sum_in_bin.copy()
        if windowed_sum_in_bin.shape[0] >= window:
            windowed_sum_in_bin[window:] -= sum_in_bin[:-window]  # Shape (n_windows, bins)
        windowed_n_per_bin = n_per_bin_cumsum.copy()
        if windowed_n_per_bin.shape[0] >= window:
            windowed_n_per_bin[window:] -= n_per_bin_cumsum[:-window]  # Shape (n_windows, n_bins)

        with np.errstate(divide="ignore", invalid="ignore"):
            mean = windowed_sum_in_bin / windowed_n_per_bin  # Shape (iterations, n_bins)

        for i, (bin_start, bin_end) in enumerate(itertools.pairwise(bins)):
            plt.plot(iterations, mean[:, i], label=f"{int(bin_start)}-{int(bin_end)}")
            plt.xlabel("Iteration")
            plt.ylabel("Loss")
        plt.legend()

    @property
    def ema_per_timestep(self) -> np.ndarray:
        return self._ema_per_timestep




def _illustrate_forward():
    from sklearn.datasets import make_swiss_roll
    import matplotlib.pyplot as plt
    x, _ = make_swiss_roll(n_samples=1000)
    x = torch.from_numpy(x).cpu()

    schedule = LinearSchedule(beta_start=1e-4, beta_end=0.02, n_steps=1000, device=torch.device("cpu"))
    print(schedule.beta[-5:], schedule.alpha_bar[-5:])

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

