from torch import Tensor

from ev_at.core.pipe import TrainingModule

__all__ = ["StandardTrainingModule"]


class StandardTrainingModule(TrainingModule):
    """Simple training module for classification.

    The module is designed for "classical" training without adversarial attack.

    .. note::
        The forward pass returns logits, label predictions and uncertainty.
    """

    def training_step(self, batch: list[Tensor], batch_idx: int) -> Tensor:
        """
        Training step for the model.
        """
        logits, preds, _ = self.model.forward(batch[0])
        loss = self.model.loss_fn(logits, batch[1])
        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        self.train_metrics.update(preds.long(), batch[1].long())
        return loss

    def validation_step(self, batch: list[Tensor], batch_idx: int) -> Tensor:
        """
        Validation step for the model.
        """
        logits, preds, _ = self.model.forward(batch[0])
        loss = self.model.loss_fn(logits, batch[1])
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.val_metrics.update(preds.long(), batch[1].long())
        return loss
