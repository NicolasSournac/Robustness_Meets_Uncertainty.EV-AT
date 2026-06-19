from abc import abstractmethod

import torch
from torch import Tensor
from torch.nn.modules.loss import _Loss

from ev_at import EVAT_CONSTANTS
from ev_at.nn.loss.scheduler import EvidentialLossRegScheduler

__all__ = [
    "EvidentialLoss",
    "CategoricalEvidentialLoss",
    "ExpectedSquaredErrorLoss",
    "ExpectedNLLLoss",
    "Type2NLLoss",
]


class EvidentialLoss(_Loss):
    r"""Template class for evidential losses with optional regularization

    Child classes must implement the `objective` and `regularization` methods.
    The :func:`objective` method computes the main loss term, while the
    :func:`regularization` method computes an optional regularization term that
    can be weighted by a factor controlled by a scheduler.

    Each evidential loss must be used in the following way:

    .. code-block:: python
        loss_fn = SomeEvidentialLoss(reg_scheduler=SomeScheduler(...))
        for epoch in range(num_epochs):
            loss_fn.step_regularizer(epoch) # Update epoch and regularization factor

            for batch in data_loader:
                ...
                loss = loss_fn(preds, targets)

    Args:
        reg_scheduler: A scheduler to update the regularization factor over epochs.
        reduction: Specifies the reduction to apply along the batch dimension.
            Should be one of 'none', 'mean' or 'sum'.

    The loss is called with two arguments (`input` and `target`).
    """

    def __init__(
        self,
        reg_scheduler: EvidentialLossRegScheduler | None = None,
        reduction: str = "mean",
    ):
        super().__init__(reduction=reduction)
        self.reg_scheduler = reg_scheduler

    def forward(self, input: Tensor, target: Tensor):
        objective = self.objective(input, target)
        regularization = self.regularization(input, target)

        if self.reg_scheduler is not None:
            reg_factor = self.reg_factor
            loss = objective + reg_factor * regularization
        else:
            loss = objective

        return self.reduce(loss)

    def step_regularizer(self, epoch: int):
        if self.reg_scheduler is not None:
            self.reg_scheduler.update_epoch(epoch)
            self.reg_scheduler.step()

    @property
    def reg_factor(self):
        if self.reg_scheduler is not None:
            return self.reg_scheduler.reg_factor
        return

    @abstractmethod
    def objective(self, input: Tensor, target: Tensor) -> Tensor:
        pass

    @abstractmethod
    def regularization(self, input: Tensor, target: Tensor) -> Tensor:
        pass

    def reduce(self, loss: Tensor) -> Tensor:
        match self.reduction:
            case "mean":
                return loss.mean()
            case "sum":
                return loss.sum()
            case "none":
                return loss
            case _:
                raise ValueError(f"Invalid reduction mode: {self.reduction}")


class CategoricalEvidentialLoss(EvidentialLoss):
    r"""Abstract base class for evidential losses with categorical outputs.

    This class defines a common regularization term for all categorical
    evidential losses. The regularization corresponds to the entropy of the posterior
    Dirichlet distribution. The goal is to push the model to predict low evidence
    for incorrect classes, while allowing high evidence for the correct class.

    Denoting :math:`\alpha = (\alpha_1, \ldots, \alpha_C)` the parameters of the
    posterior Dirichlet distribution, the regularization term is given by:

    .. math::

        \text{Reg}(\alpha) = \sum_{i=1}^C \left( \log \Gamma(\alpha_i)
            - \log \Gamma(\alpha_0) \right) + (\alpha_0 - C) \psi(\alpha_0)
            - \sum_{i=1}^C (\alpha_i - 1) \psi(\alpha_i)

    where :math:`\alpha_0 = \sum_{i=1}^C \alpha_i` and :math:`\psi` is the
    digamma function.

    Args:
        reg_scheduler: A scheduler to update the regularization factor over epochs.
        reduction: Specifies the reduction to apply along the batch dimension.
            Should be one of 'none', 'mean' or 'sum'.

    The loss is called with two arguments:
        - input: Predicted evidence for each class. Shape: :math:`(N, C, *)`.
        - target: Ground truth class indices. Shape: :math:`(N, *)`.

    Shape:
        - input: :math:`(N, C, *)` where :math:`N` is the batch size, :math:`C`
          is the number of classes, and :math:`*` means any number of additional
          dimensions.
        - target: :math:`(N, *)` where :math:`N` is the batch size and :math:`*`
          means any number of additional dimensions.
        - output: scalar. If reduction is 'none', then :math:`N`.
    """

    def _target_one_hot(self, input: Tensor, target: Tensor) -> Tensor:
        num_classes = input.size(1)
        if target.dim() == input.dim() and target.size(1) == num_classes:
            return target.float().permute(0, -1, *range(1, input.dim() - 1))
        return (
            torch.nn.functional.one_hot(target, num_classes=num_classes)
            .float()
            .permute(0, -1, *range(1, input.dim() - 1))
        )

    def regularization(self, input: Tensor, target: Tensor) -> Tensor:
        num_classes = input.size(1)

        target_one_hot = self._target_one_hot(input, target)

        input = target_one_hot + (1 - target_one_hot) * input
        alpha_0 = input.sum(dim=1, keepdim=True)

        return -(
            torch.lgamma(input).sum(dim=1, keepdim=True)
            - torch.lgamma(alpha_0)
            + (alpha_0 - num_classes) * torch.digamma(alpha_0)
            - ((input - 1) * torch.digamma(input)).sum(dim=1, keepdim=True)
        ).squeeze(1)


class ExpectedSquaredErrorLoss(CategoricalEvidentialLoss):
    r"""Expected squared error loss for categorical evidential outputs.

    This loss corresponds to the expected squared error under the predicted
    Dirichlet distribution. It is given by:

    .. math::

        \text{EBS}(\alpha, y) = \mathbb{E}_{\mathbf{p} \sim \text{Dir}(\alpha)}
            \left[ \left( y - \frac{\alpha}{\alpha_0} \right)^2 \right]

    where :math:`\alpha` are the parameters of the Dirichlet distribution,
    :math:`\alpha_0 = \sum_{i=1}^C \alpha_i`, and :math:`y` is the one-hot
    encoded target vector.

    Args:
        reg_scheduler: A scheduler to update the regularization factor over epochs.
        reduction: Specifies the reduction to apply along the batch dimension.
            Should be one of 'none', 'mean' or 'sum'.

    The loss is called with two arguments:
        - input: Predicted evidence for each class. Shape: :math:`(N, C, *)`.
        - target: Ground truth class indices. Shape: :math:`(N, *)`.

    Shape:
        - input: :math:`(N, C, *)` where :math:`N` is the batch size, :math:`C`
          is the number of classes, and :math:`*` means any number of additional
          dimensions.
        - target: :math:`(N, *)` where :math:`N` is the batch size and :math:`*`
          means any number of additional dimensions.
        - output: scalar. If reduction is 'none', then :math:`N`.
    """

    def objective(self, input: Tensor, target: Tensor) -> Tensor:
        alpha_0 = input.sum(dim=1, keepdim=True)
        one_hot_target = self._target_one_hot(input, target)

        return (
            (one_hot_target - (input / alpha_0)).pow(2)
            + (input * (alpha_0 - input) / (alpha_0.pow(2) * (alpha_0 + 1)))
        ).mean(dim=1)


class ExpectedNLLLoss(CategoricalEvidentialLoss):
    r"""Expected negative log categorical likelihood evidential loss.

    This loss corresponds to the expected negative log likelihood under the
    predicted Dirichlet distribution. It is given by:

    .. math::

        \text{ENLL}(\alpha, y) = \mathbb{E}_{\mathbf{p} \sim \text{Dir}(\alpha)}
            \left[ -\log p_{y} \right]

    where :math:`\alpha` are the parameters of the Dirichlet distribution,
    :math:`p` is a categorical distribution drawn from the Dirichlet, and
    :math:`y` is the target class index.

    Args:
        reg_scheduler: A scheduler to update the regularization factor over epochs.
        reduction: Specifies the reduction to apply along the batch dimension.
            Should be one of 'none', 'mean' or 'sum'.

    The loss is called with two arguments:
        - input: Predicted evidence for each class. Shape: :math:`(N, C, *)`.
        - target: Ground truth class indices. Shape: :math:`(N, *)`.

    Shape:
        - input: :math:`(N, C, *)` where :math:`N` is the batch size, :math:`C`
          is the number of classes, and :math:`*` means any number of additional
          dimensions.
        - target: :math:`(N, *)` where :math:`N` is the batch size and :math:`*`
          means any number of additional dimensions.
        - output: scalar. If reduction is 'none', then :math:`N`.
    """

    def objective(self, input: Tensor, target: Tensor) -> Tensor:
        alpha_0 = input.sum(dim=1, keepdim=True)
        one_hot_target = self._target_one_hot(input, target)

        return (
            one_hot_target
            * (torch.digamma(alpha_0) - torch.digamma(input + EVAT_CONSTANTS.EPS_F32))
        ).sum(dim=1)


class Type2NLLoss(CategoricalEvidentialLoss):
    r"""Type-2 negative log likelihood evidential loss.

    This loss corresponds to the negative log likelihood of the target class
    under the expected categorical distribution induced by the predicted Dirichlet
    distribution. It is the log likelihood of a Categorical distribution with
    parameters given by the mean of the Dirichlet distribution.

    Denoting :math:`\alpha = (\alpha_1, \ldots, \alpha_C)` the parameters of the
    posterior Dirichlet distribution, the loss is given by:

    .. math::

        \text{NLL}(\alpha, y) = -\log \left( \frac{\alpha_y}{\alpha_0} \right)

    where :math:`\alpha_0 = \sum_{i=1}^C \alpha_i`.

    Args:
        reg_scheduler: A scheduler to update the regularization factor over epochs.
        reduction: Specifies the reduction to apply along the batch dimension.
            Should be one of 'none', 'mean' or 'sum'.

    The loss is called with two arguments:
        - input: Predicted evidence for each class. Shape: :math:`(N, C, *)`.
        - target: Ground truth class indices. Shape: :math:`(N, *)`.

    Shape:
        - input: :math:`(N, C, *)` where :math:`N` is the batch size, :math:`C`
          is the number of classes, and :math:`*` means any number of additional
          dimensions.
        - target: :math:`(N, *)` where :math:`N` is the batch size and :math:`*`
          means any number of additional dimensions.
        - output: scalar. If reduction is 'none', then :math:`N`.
    """

    def objective(self, input: Tensor, target: Tensor) -> Tensor:
        probs = input / input.sum(dim=1, keepdim=True)
        one_hot_target = self._target_one_hot(input, target)

        return -(
            one_hot_target * torch.log(probs.clamp(min=EVAT_CONSTANTS.EPS_F32))
        ).sum(dim=1)
