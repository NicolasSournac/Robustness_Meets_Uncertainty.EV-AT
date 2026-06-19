from typing import Literal

import lightning as pl
import torch
import torch.nn as nn

from ev_at.core.logging import get_logger
from ev_at.modules import EvidentialModule
from ev_at.modules.emff import EMFFModule

from .base import AdversarialAttack
from .utils import (
    EvidentialModuleWrapper,
    pgd_setup,
    pgd_update,
    sample_linf_epsilon_ball,
)

logger = get_logger("pgd_attack")


class PGD(AdversarialAttack):
    """
    Implementation of the Projected Gradient Descent (PGD) adversarial attack.
    This attack iteratively perturbs the input data to maximize the loss,
    while ensuring that the perturbations remain within a specified bound.
    """

    def __init__(
        self,
        loss_fn: nn.Module,
        epsilon: float = 0.1,
        num_iterations: int = 40,
        alpha: float | None = None,
        norm: Literal["Linf", "L2"] = "Linf",
    ):
        """
        Initialize the PGD attack with a specified perturbation magnitude.

        Args:
            loss_fn (nn.Module): The loss function to use for calculating gradients.
            epsilon (float): The magnitude of the perturbation to apply to the input
                data.
            num_iterations (int): The number of iterations to perform the attack.
            alpha (float | None): The step size for each iteration.
                If None, it is set to epsilon / num_iterations.
            norm (str): The norm to use for the attack. Can be "Linf" or "L2".
        """
        super().__init__()
        self.epsilon = epsilon
        self.loss_fn = loss_fn
        self.num_iterations = num_iterations
        self.alpha = alpha if alpha is not None else epsilon / num_iterations
        self.norm = norm

    def _prepare_model(self, model):
        if isinstance(model, EvidentialModule) and isinstance(
            self.loss_fn, nn.CrossEntropyLoss
        ):
            return EvidentialModuleWrapper(model)
        return model

    def _forward(  # type: ignore
        self, model: pl.LightningModule, inputs: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Perform the PGD attack on the given model and data.

        Args:
            model (pl.LightningModule): The model to attack.
            inputs (torch.Tensor): The input data batch to attack.
            targets (torch.Tensor): The target labels for the input data.

        Returns:
            torch.Tensor: The adversarially perturbed batch.
        """
        inputs = inputs.clone().detach()
        if self.norm == "Linf":
            x_adv = sample_linf_epsilon_ball(inputs, self.epsilon)

            for _ in range(self.num_iterations):
                _, loss = pgd_setup(model, x_adv, targets, self.loss_fn)

                grad = torch.autograd.grad(
                    loss,
                    x_adv,
                    retain_graph=False,
                    create_graph=False,
                )[0]
                x_adv = pgd_update(inputs, x_adv, grad, self.alpha, self.epsilon)
        elif self.norm == "L2":
            bs = inputs.size(0)
            delta = torch.randn_like(inputs)
            delta_flat = delta.view(bs, -1)
            delta_norm = delta_flat.norm(p=2, dim=1).clamp_min(1e-12)
            delta = delta / delta_norm.view(-1, 1, 1, 1)

            # rayon aléatoire dans [0, epsilon]
            r = torch.rand(bs, device=inputs.device).view(-1, 1, 1, 1)
            delta = delta * (r * self.epsilon)
            x_adv = (inputs + delta).clamp(0.0, 1.0)

            for _ in range(self.num_iterations):
                x_adv = x_adv.detach().requires_grad_(True)

                if isinstance(model, EMFFModule):
                    student = model(x_adv, y=None, epoch=None, is_eval=True)[
                        0
                    ]  # Assuming model returns (logits, ...)
                else:
                    student = model(x_adv)[0]  # Assuming model returns (logits, ...)

                if isinstance(model, EvidentialModule):
                    student = torch.log(student.clamp_min(1e-12))

                # On maximise la divergence => gradient ascent sur x_adv
                cost = self.loss_fn(student, targets)

                grad = torch.autograd.grad(cost, x_adv, only_inputs=True)[0]
                grad = torch.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)

                # normalisation L2 du gradient
                grad_flat = grad.view(bs, -1)
                grad_norm = grad_flat.norm(p=2, dim=1).clamp_min(1e-12)
                grad = grad / grad_norm.view(-1, 1, 1, 1)

                # step
                x_adv = x_adv.detach() + self.alpha * grad

                # projection sur boule L2 de rayon epsilon autour de x
                delta = x_adv - inputs
                delta = torch.nan_to_num(delta, nan=0.0, posinf=0.0, neginf=0.0)

                delta_flat = delta.view(bs, -1)
                delta_norm = delta_flat.norm(p=2, dim=1).clamp_min(1e-12)
                factor = torch.clamp(self.epsilon / delta_norm, max=1.0)
                delta = delta * factor.view(-1, 1, 1, 1)

                # clamp image
                x_adv = (inputs + delta).clamp(0.0, 1.0)
        return x_adv
