from abc import ABC, abstractmethod

from ev_at.core.pydantic.validator import validate_epoch

__all__ = [
    "EvidentialLossRegScheduler",
    "ConstantRegScheduler",
    "LinearRegScheduler",
    "ExponentialRegScheduler",
]


class EvidentialLossRegScheduler(ABC):
    """
    Template class to implement regularization factor schedulers for evidential
    losses.

    Each scheduler must implement the `step` method, which updates the
    regularization factor.

    Args:
        initial_reg_factor: The initial regularization factor.
        max_reg_factor: The maximum regularization factor. If None, no maximum is
            applied.

    Attributes:
        initial_reg_factor: The initial regularization factor.
        max_reg_factor: The maximum regularization factor.
        reg_factor: The current regularization factor.
        epoch: The current epoch.
    """

    def __init__(
        self, initial_reg_factor: float = 0.0, max_reg_factor: float | None = None
    ):
        self.initial_reg_factor = initial_reg_factor
        self.max_reg_factor = (
            max_reg_factor if max_reg_factor is not None else float("inf")
        )
        self._current_reg_factor = initial_reg_factor
        self._epoch: int | None = None

    @abstractmethod
    def step(self):
        pass

    @property
    def reg_factor(self):
        return self._current_reg_factor

    @reg_factor.setter
    def reg_factor(self, value: float):
        self._current_reg_factor = min(self.max_reg_factor, value)

    @property
    def epoch(self):
        return self._epoch

    def update_epoch(self, value: int) -> None:
        try:
            validate_epoch(value)
        except Exception as e:
            raise ValueError("Epoch must be a positive integer.") from e
        if self._epoch is not None and value < self._epoch:
            raise ValueError("Epoch value cannot be decreased.")

        self._epoch = value


class ConstantRegScheduler(EvidentialLossRegScheduler):
    """Scheduler that keeps the regularization factor constant.

    Args:
        reg_factor: The regularization factor that will be kept constant.
    """

    def __init__(self, reg_factor: float):
        super().__init__(reg_factor, reg_factor)

    def step(self):
        pass


class LinearRegScheduler(EvidentialLossRegScheduler):
    r"""Scheduler that increases the regularization factor linearly.

    Args:
        initial_reg_factor: The initial regularization factor.
        max_reg_factor: The maximum regularization factor.
        gamma: The linear increase factor.

    Denoting :math:`\lambda_t` the regularization factor at epoch :math:`t`, the
    update rule is given by:

    .. math::

        \lambda_t = \lambda_0 + \gamma~t
    """

    def __init__(
        self,
        initial_reg_factor: float,
        max_reg_factor: float | None,
        gamma: float,
    ):
        super().__init__(initial_reg_factor, max_reg_factor)
        self.gamma = gamma

    def step(self):
        new_reg_factor = self.initial_reg_factor + self.gamma * self.epoch
        self.reg_factor = new_reg_factor


class ExponentialRegScheduler(EvidentialLossRegScheduler):
    r"""Scheduler that increases the regularization factor exponentially.

    Args:
        initial_reg_factor: The initial regularization factor.
        max_reg_factor: The maximum regularization factor.
        gamma: The exponential increase factor.

    Denoting :math:`\lambda_t` the regularization factor at epoch :math:`t`, the
    update rule is given by:

    .. math::

        \lambda_t = \lambda_0 \cdot \gamma^t
    """

    def __init__(
        self,
        initial_reg_factor: float,
        max_reg_factor: float | None,
        gamma: float,
    ):
        super().__init__(initial_reg_factor, max_reg_factor)
        self.gamma = gamma

    def step(self):
        new_reg_factor = self.initial_reg_factor * (self.gamma**self.epoch)
        self.reg_factor = min(new_reg_factor, self.max_reg_factor)
