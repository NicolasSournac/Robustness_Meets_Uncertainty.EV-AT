from abc import ABC, abstractmethod

import lightning as pl
import torch
import torch.nn as nn
from torchmetrics import Metric


class Module(ABC, pl.LightningModule):
    """
    Base class for all modules.

    A module is a wrapper around a PyTorch model
    that also includes the loss function
    and the uncertainty score metric.

    It defines the forward pass for the model.
    """

    def __init__(self, model: nn.Module, loss_fn: nn.Module, uncertainty_score: Metric):
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.uncertainty_score = uncertainty_score

    @abstractmethod
    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x (torch.Tensor): Input tensor representing the images.

        Returns:
            logits (torch.Tensor): Raw model outputs.
            preds (torch.Tensor): Label predictions.
            uncertainty (torch.Tensor): Uncertainty scores.

        Shapes:
            - x: (B, 3, H, W)
            - logits: (B, K)
            - preds: (B, 1)
            - uncertainty: (B, 1)
        """
        ...
        pass

    def on_train_epoch_start(self):
        super().on_train_epoch_start()
        self.uncertainty_score = self.uncertainty_score.to(self.device)

    def on_train_batch_end(
        self,
        outputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        batch: tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
    ):
        super().on_train_batch_end(outputs, batch, batch_idx)
        self.uncertainty_score.reset()

    def on_train_epoch_end(self):
        self.uncertainty_score = self.uncertainty_score.to(self.device)
