import torch

from abc import ABC, abstractmethod

from typing import Optional, Dict, Type


SCHEDULE_TYPE_BY_NAME: Dict[str, Type['Schedule']] = {}


class Schedule(ABC):
    """An object representing a diffusion schedule."""

    def __init__(self, n_steps: int):
        self.n_steps = n_steps
        self._beta: Optional[torch.Tensor] = None
        self._alpha_bar = None

    @property
    def beta(self) -> torch.Tensor:
        assert self._beta is not None
        return self._beta

    @property
    def alpha_bar(self) -> torch.Tensor:
        if self._alpha_bar is not None:
            return self._alpha_bar
        else:
            self._alpha_bar = torch.cumprod(1.0-self.beta, dim=0)
            return self._alpha_bar

    @classmethod
    @abstractmethod
    def name(cls) -> str:
        pass

    @property
    def __len__(self) -> int:
        return self.beta.numel()


class LinearSchedule(Schedule):
    def __init__(self, beta_start: float, beta_end: float, n_steps: int, device: Optional[torch.device] = None):
        super().__init__(n_steps=n_steps)
        if device is None:
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        self._beta = torch.linspace(beta_start, beta_end, n_steps, device=device)
        self._beta = torch.cat((torch.zeros(1, device=device), self._beta), dim=0)

    @classmethod
    def name(cls) -> str:
        return "linear"


for type_ in [LinearSchedule]:
    SCHEDULE_TYPE_BY_NAME[type_.name()] = type_

