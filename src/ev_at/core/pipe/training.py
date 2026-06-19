import time
from abc import ABC, abstractmethod

import lightning as pl
import torch
from torch import Tensor
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer
from torchmetrics import MetricCollection
from torchmetrics.classification import Accuracy, F1Score, Precision, Recall

from ev_at.core.modules import Module

__all__ = ["TrainingModule"]


class TrainingModule(ABC, pl.LightningModule):
    """Default training Module for classification.

    This abstract module serves as a base class for specific training modules.
    It defines the essential structure and methods that any derived training module
    must implement.

    .. note::
        The model's forward pass returns logits, label predictions and uncertainty.
    """

    def __init__(
        self,
        model: Module,
        optimizer: Optimizer,
        scheduler: LRScheduler | None = None,
        num_classes: int = 10,
    ):
        """Initialize the standard training module."""
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.num_classes = num_classes

        # == Metrics Setup == #
        standard_metrics = MetricCollection(
            {
                "precision": Precision(
                    task="multiclass", num_classes=self.num_classes, average="macro"
                ),
                "recall": Recall(
                    task="multiclass", num_classes=self.num_classes, average="macro"
                ),
                "f1_score": F1Score(
                    task="multiclass", num_classes=self.num_classes, average="macro"
                ),
                "accuracy": Accuracy(task="multiclass", num_classes=self.num_classes),
            }
        )
        self.train_metrics = standard_metrics.clone(prefix="train_")
        self.val_metrics = standard_metrics.clone(prefix="val_")

    def on_train_epoch_start(self):
        super().on_train_epoch_start()
        self.model.on_train_epoch_start()
        self.epoch_start_time = time.time()

    @abstractmethod
    def training_step(self, batch, batch_idx):
        """
        Training step for the model.
        """
        ...

    def on_train_batch_end(
        self,
        outputs: tuple[Tensor, Tensor, Tensor],
        batch: list[Tensor, Tensor],
        batch_idx: int,
    ):
        super().on_train_batch_end(outputs, batch, batch_idx)
        self.model.on_train_batch_end(outputs, batch, batch_idx)

    def on_train_epoch_end(self):
        train_metrics = self.train_metrics.compute()
        self.log_dict(train_metrics)
        self.train_metrics.reset()
        self.model.on_train_epoch_end()
        if self.epoch_start_time is not None:
            epoch_duration = time.time() - self.epoch_start_time
            self.log(
                "epoch_time_seconds",
                epoch_duration,
                on_epoch=True,
                prog_bar=True,
            )

    @abstractmethod
    def validation_step(self, batch, batch_idx):
        """
        Validation step for the model.
        """
        ...

    def on_validation_epoch_end(self):
        val_metrics = self.val_metrics.compute()
        self.log_dict(val_metrics)
        for i, param_group in enumerate(self.optimizer.param_groups):
            self.log(f"lr_{i}", param_group["lr"], on_epoch=True, prog_bar=False)
        self.val_metrics.reset()

    def on_before_optimizer_step(self, optimizer):
        """Monitor gradient magnitudes before optimizer step"""
        # Log overall gradient norm
        total_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), float("inf")
        )
        self.log(
            "grad_norm/total",
            total_norm,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )

    def configure_optimizers(self):
        optimization_config = {
            "optimizer": self.optimizer,
        }
        if self.scheduler is not None:
            optimization_config["lr_scheduler"] = {
                "scheduler": self.scheduler,
                "interval": "epoch",
            }
        return optimization_config
