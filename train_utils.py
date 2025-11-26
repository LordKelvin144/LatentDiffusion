import torch
import numpy as np
import matplotlib.pyplot as plt

from abc import ABC, abstractmethod

from typing import List, Dict, Optional


class TrainLogger:
    def __init__(self, window: int, log_vars: Dict[str, str]):
        self._log_vars = log_vars
        self._logs: Dict[str, np.ndarray] = {key: np.zeros(0) for key in log_vars.keys()}
        self._cum_logs: Dict[str, np.ndarray] = {key: np.zeros(0) for key in log_vars.keys()}
        self._new_logs: Dict[str, List[torch.Tensor]] = {key: [] for key in log_vars.keys()}

        self.window = window
        self._iteration = 0
        self._unhandled_logs = False

    def log(self, **kwargs: torch.Tensor):
        self._iteration += 1
        self._unhandled_logs = True

        for key, value in kwargs.items():
            self._new_logs[key].append(value)
    
    def _update_logs(self):
        if not self._unhandled_logs:
            return

        for key in self._logs.keys():
            new_logs = torch.stack(self._new_logs[key]).cpu().numpy()
            cum_logs = self._cum_logs[key]
            self._logs[key] = np.hstack((self._logs[key], new_logs))
            self._cum_logs[key] = np.hstack((cum_logs, np.cumsum(new_logs) + (cum_logs[-1] if cum_logs.size > 0 else 0.0)))
        self._new_logs = {key: [] for key in self._log_vars.keys()}
        self._unhandled_logs = False

    def moving_avg(self, key: str, n: Optional[int] = None) -> np.ndarray:
        self._update_logs()
        if n is None:
            n = self.window
        a = np.array(self._cum_logs[key])
        windowed_sum = a.copy()
        windowed_sum[n:] -= windowed_sum[:-n]
        windowed_avg = windowed_sum / np.minimum(np.arange(a.size) + 1.0, n)
        return windowed_avg

    def plot(self, include_keys: Optional[List[str]] = None, logx: bool = False, logy: bool = False):
        fig, ax = plt.subplots()

        if include_keys is None:
            include_keys = list(self._log_vars.keys())

        self._update_logs()
        for i, key in enumerate(include_keys):
            ax.plot(self._logs[key], color=f"C{i}", alpha=0.1)
            ax.plot(self.moving_avg(key), label=self._log_vars[key], color=f"C{i}")
            ax.set_xlabel("Iteration")
            ax.set_ylabel("Loss")
        if logx:
            ax.set_xscale("log")
        if logy:
            ax.set_yscale("log")
        fig.legend()

    def current_repr(self, window: Optional[int] = None, step: Optional[int] = None, epoch: Optional[int] = None) -> str:
        self._update_logs()
        if window is None:
            window = self.window
        if epoch is not None:
            header = f"Epoch {epoch}; "
            if step is not None:
                header += f"Step {step}; "
        else:
            header = f"Step {step if step is not None else self._iteration - 1}; "
        return header + "; ".join(f"{key} = {np.mean(self._logs[key][-window:]):.4g}" for key in self._log_vars.keys())

    @property
    def iteration(self) -> int:
        return self._iteration


class TrainingCallback(ABC):
    def __init__(self, epoch_interval: int = 1, step_interval: Optional[int] = None):
        self.epoch_interval = epoch_interval
        self.step_interval = step_interval
        
    def check(self, epoch: int, step: int, batches_per_epoch: int):
        if epoch % self.epoch_interval != 0:
            return

        if self.step_interval is not None and step % self.step_interval == 0 and step != 0:
            self(epoch, step, batches_per_epoch)

        elif self.step_interval is None and step == batches_per_epoch - 1:
            self(epoch, step, batches_per_epoch)

    @abstractmethod
    def __call__(self, epoch: int, step: int, batches_per_epoch: int) -> None:
        pass


class PrintLossCallback(TrainingCallback):
    def __init__(self, logger: TrainLogger, epoch_interval: int = 1, step_interval: Optional[int] = None):
        super().__init__(epoch_interval=epoch_interval, step_interval=step_interval)
        self._logger = logger

    def __call__(self, epoch: int, step: int, batches_per_epoch: int) -> None:
        window = self.step_interval if self.step_interval is not None else batches_per_epoch*self.epoch_interval
        print(self._logger.current_repr(window))


class PlotLossCallback(TrainingCallback):
    def __init__(self, logger: TrainLogger, include_keys: Optional[List[str]] = None, epoch_interval: int = 1, step_interval: Optional[int] = None, filename: Optional[str] = None,
                 logx: bool = False, logy: bool = False):
        super().__init__(epoch_interval=epoch_interval, step_interval=step_interval)
        self._logger = logger
        self._filename = filename
        self._include_keys = include_keys
        self._logx = logx
        self._logy = logy

    def __call__(self, epoch: int, step: int, batches_per_epoch: int) -> None:
        self._logger.plot(self._include_keys, logx=self._logx, logy=self._logy)
        if self._filename is not None:
            plt.savefig(self._filename)
            plt.close()
        else:
            plt.show()



if __name__ == '__main__':
    import time
    logger = TrainLogger(window=100, log_vars={"loss": "loss"})
    t_start = time.perf_counter()
    for _ in range(1000):
        logger.log(loss=0.5)
    t_end = time.perf_counter()
    print("Average time:", (t_end - t_start) / 1000)
