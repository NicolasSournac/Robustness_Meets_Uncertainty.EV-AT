import torch
from pydantic import BaseModel, ConfigDict, field_validator

from ev_at.core.logging import get_logger

logger = get_logger("output-validation")

__all__ = ["ModelOutput"]


class ModelOutput(BaseModel):
    """
    Pydantic model for basic validation of model outputs.
    It checks that the model outputs three tensors:
    - logits: Raw model outputs (logits).
    - preds: Predicted class labels.
    - uncertainty: Uncertainty scores.
    """

    logits: torch.Tensor
    preds: torch.Tensor
    uncertainty: torch.Tensor

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @field_validator("preds", "logits", "uncertainty")
    @classmethod
    def check_tensor(cls, v: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(v):
            msg = f"{cls.__name__} must be a torch.Tensor."
            logger.error(msg)
            raise TypeError(msg)
        return v
