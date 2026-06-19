from typing import Literal

from torch import Tensor, nn
from torch.distributions import Dirichlet
from torchmetrics import Metric

from ev_at.core.modules import Module
from ev_at.nn.functional import evidence_relu, evidence_softplus
from ev_at.nn.loss.evidential import EvidentialLoss

__all__ = ["EvidentialModule"]


class EvidentialModule(Module):
    """
    Module for evidential deep learning

    This module converts the model's output logits
    into concentration parameters of a Dirichlet distribution
    hence allowing evidential deep learning.
    """

    def __init__(
        self,
        model: nn.Module,
        loss_fn: EvidentialLoss,
        activation: Literal["softplus", "relu"],
        uncertainty_score: Metric,
    ):
        super().__init__(model, loss_fn, uncertainty_score)
        match activation:
            case "softplus":
                self.activation = evidence_softplus
            case "relu":
                self.activation = evidence_relu
            case _:
                raise ValueError(f"Invalid activation function: {activation}")

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        self.uncertainty_score = self.uncertainty_score.to(x.device)

        logits = self.model(x)
        concentrations = self.activation(logits)
        dist = Dirichlet(concentrations)
        preds = dist.mean.argmax(dim=1)

        uncertainty = self.uncertainty_score(concentrations)
        self.uncertainty_score.reset()

        return concentrations, preds, uncertainty

    def on_train_epoch_start(self):
        super().on_train_epoch_start()
        self.loss_fn.step_regularizer(epoch=self.current_epoch)
