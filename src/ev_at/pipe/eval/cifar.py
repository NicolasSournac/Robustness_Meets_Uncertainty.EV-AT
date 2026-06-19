from pydantic import BaseModel, field_validator
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    Accuracy,
    BinaryAUROC,
    F1Score,
    Precision,
    Recall,
)

from ev_at.core.logging import get_logger
from ev_at.core.modules import Module
from ev_at.core.pipe.eval import EvaluationModule
from ev_at.core.pydantic.model_output import ModelOutput
from ev_at.metrics import AUGRiskCoverage, AURiskCoverage

logger = get_logger("eval")

__all__ = ["CIFARClassificationEvaluationModule"]


class Cifar10ModelOutput(BaseModel):
    """
    Specific Pydantic model for the Cifar10 dataset.
    Checks that the model outputs have the expected shape.
    The expected shape is (B, 10) for logits.
    The expected shape is (B) for preds and uncertainty.
    This is a subclass of ModelOutput to ensure that the basic validation
    is also applied.
    """

    output: ModelOutput

    @field_validator("output")
    @classmethod
    def check_shapes(cls, v: ModelOutput) -> ModelOutput:
        if not v.logits.ndim == 2 or v.logits.shape[1:] != (10,):
            msg = f"{cls.__name__}.logits must have shape (B, 10))."
            logger.error(msg)
            raise ValueError(msg)
        if not v.preds.ndim == 1:
            msg = f"{cls.__name__}.preds must have shape (B)."
            logger.error(msg)
            raise ValueError(msg)
        if not v.uncertainty.ndim == 1:
            msg = f"{cls.__name__}.uncertainty must have shape (B)."
            logger.error(msg)
            raise ValueError(msg)
        return v


class Cifar100ModelOutput(BaseModel):
    """
    Specific Pydantic model for the Cifar10 dataset.
    Checks that the model outputs have the expected shape.
    The expected shape is (B, 100) for logits.
    The expected shape is (B) for preds and uncertainty.
    This is a subclass of ModelOutput to ensure that the basic validation
    is also applied.
    """

    output: ModelOutput

    @field_validator("output")
    @classmethod
    def check_shapes(cls, v: ModelOutput) -> ModelOutput:
        if not v.logits.ndim == 2 or v.logits.shape[1:] != (100,):
            msg = f"{cls.__name__}.logits must have shape (B, 100))."
            logger.error(msg)
            raise ValueError(msg)
        if not v.preds.ndim == 1:
            msg = f"{cls.__name__}.preds must have shape (B)."
            logger.error(msg)
            raise ValueError(msg)
        if not v.uncertainty.ndim == 1:
            msg = f"{cls.__name__}.uncertainty must have shape (B)."
            logger.error(msg)
            raise ValueError(msg)
        return v


class CIFARClassificationEvaluationModule(EvaluationModule):
    """
    A LightningModule handling evaluation logic for training pipelines.
    This class is designed to allow the user to implement custom model forward logic
    that matches their specific evaluation needs.

    Args:
        attacks (list): List of attack functions to apply during evaluation.
        model (nn.Module | pl.LightningModule): The model to evaluate.
            It should implement a forward method providing logits, preds and uncertainty
            given a batch.
        ckpt_path (str, optional): Path to the model checkpoint. Defaults to None.
    """

    def __init__(
        self,
        attacks: list,
        model: Module,
        num_classes: int = 10,
    ):
        self.num_classes = num_classes
        super().__init__(
            attacks=attacks,
            model=model,
            output_validation_model=Cifar10ModelOutput
            if num_classes == 10
            else Cifar100ModelOutput,
        )

    @property
    def standard_metrics(self) -> MetricCollection:
        return MetricCollection(
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

    @property
    def uncertainty_metrics(self) -> MetricCollection:
        return MetricCollection({"auroc": BinaryAUROC()})

    @property
    def selective_classification_metrics(self) -> MetricCollection:
        return MetricCollection(
            {
                "aurc": AURiskCoverage(),
                "augrc": AUGRiskCoverage(),
            }
        )
