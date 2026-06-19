import lightning as pl
import torch
import torch.nn as nn

from ev_at.core.logging import get_logger
from ev_at.modules.emff import EMFFModule
from ev_at.modules.evidential import EvidentialModule

logger = get_logger("attack_utils")

__all__ = [
    "compute_linf_dist",
    "sample_linf_epsilon_ball",
    "pgd_setup",
    "pgd_update",
    "EvidentialModuleWrapper",
]


class EvidentialModuleWrapper(nn.Module):
    """
    Wrapper for EvidentialModule.
    """

    def __init__(self, model: EvidentialModule):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        concentrations, preds, uncertainty = self.model(x)
        pseudo_logits = torch.log(concentrations)
        return pseudo_logits, preds, uncertainty


class EMFFModuleWrapper(nn.Module):
    """
    Wrapper for EMFFModule.
    """

    def __init__(self, model: EMFFModule):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits, preds, uncertainty, _ = self.model(x, None, None, is_eval=True)
        return logits, preds, uncertainty


def compute_linf_dist(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Compute the L-infinity distance between two tensors.

    Args:
        a (torch.Tensor): First tensor.
        b (torch.Tensor): Second tensor.

    Returns:
        torch.Tensor: The L-infinity distance between the two tensors.
    """
    return torch.max(torch.abs(a - b))


def compute_l2_dist(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Compute the L2 distance between two tensors.

    Args:
        a (torch.Tensor): First tensor.
        b (torch.Tensor): Second tensor.

    Returns:
        torch.Tensor: The L2 distance between the two tensors.
    """
    delta = a - b
    l2_norm = delta.view(delta.size(0), -1).norm(p=2, dim=1)
    return torch.max(l2_norm)


def sample_linf_epsilon_ball(inputs: torch.Tensor, epsilon: float) -> torch.Tensor:
    """
    Sample a random point within an L-inf epsilon ball around the given inputs.

    Args:
        inputs (torch.Tensor): The input tensor.
        epsilon (float): The radius of the epsilon ball.

    Returns:
        torch.Tensor: A random point within the L-inf epsilon ball.
    """
    adversarial_inputs = inputs + torch.empty_like(inputs).uniform_(-epsilon, epsilon)
    return torch.clamp(adversarial_inputs, 0, 1).detach().requires_grad_(True)


def pgd_setup(
    model: pl.LightningModule,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    loss_fn: nn.Module,
):
    """
    Prepare the model and inputs for PGD attack.
    """
    model.zero_grad()
    if inputs.grad is not None:
        inputs.grad.zero_()

    inputs.requires_grad = True
    if isinstance(model, EMFFModule):
        logits = model(inputs, y=None, epoch=None, is_eval=True)[
            0
        ]  # Assuming model returns (logits, ...)
    else:
        logits = model(inputs)[0]  # Assuming model returns (logits, ...)
    loss = loss_fn(logits, targets)

    return logits, loss


def pgd_update(
    natural_inputs: torch.Tensor,
    current_inputs: torch.Tensor,
    gradient: torch.Tensor,
    alpha: float,
    epsilon: float,
) -> torch.Tensor:
    """
    Perform a single PGD update step.
    """
    current_inputs = current_inputs + (alpha * gradient.sign())
    perturbation = torch.clamp(
        current_inputs - natural_inputs, min=-epsilon, max=epsilon
    )
    return torch.clamp(natural_inputs + perturbation, 0, 1).detach()
