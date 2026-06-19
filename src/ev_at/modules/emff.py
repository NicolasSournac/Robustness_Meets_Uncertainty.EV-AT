import torch
from torch import Tensor

from ev_at.core.modules import Module

__all__ = ["EMFFModule"]


class EMFFModule(Module):
    """EMFF module for classification.

    This module encapsulates the logic
    for computing the forward pass of the EMFF model.

    .. note::
        The forward pass returns logits, label predictions and uncertainty.
    """

    def forward(
        self, x: Tensor, y: Tensor, epoch: int, is_eval: bool = False
    ) -> tuple[Tensor, Tensor, Tensor]:
        self.uncertainty_score = self.uncertainty_score.to(x.device)

        logits, r_outputs, nr_outputs, rec_outputs, evidence_b, tmc_loss = self.model(
            x, y, epoch, is_eval
        )
        probabilities = torch.softmax(logits, dim=1)
        preds = torch.argmax(logits, dim=1)

        uncertainty = self.uncertainty_score(probabilities)
        self.uncertainty_score.reset()

        return logits, preds, uncertainty, tmc_loss
