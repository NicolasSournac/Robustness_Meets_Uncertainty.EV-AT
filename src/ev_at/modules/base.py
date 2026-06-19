import torch
from torch import Tensor

from ev_at.core.modules import Module

__all__ = ["BaseModule"]


class BaseModule(Module):
    """Simple training module for classification.

    .. note::
        The forward pass returns logits, label predictions and uncertainty.
    """

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        self.uncertainty_score = self.uncertainty_score.to(x.device)

        logits = self.model(x)
        probabilities = torch.softmax(logits, dim=1)
        preds = torch.argmax(logits, dim=1)

        uncertainty = self.uncertainty_score(probabilities)
        self.uncertainty_score.reset()

        return logits, preds, uncertainty
