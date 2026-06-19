import inspect
from abc import ABC, abstractmethod

import lightning as pl
import torch
import torch.nn as nn

__all__ = ["AdversarialAttack"]


class AdversarialAttack(nn.Module, ABC):
    """
    Base class for adversarial attacks.
    All adversarial attacks should inherit from this class.
    """

    @abstractmethod
    def _forward(
        self,
        model: pl.LightningModule | nn.Module,
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Perform the adversarial attack on the given model and data.

        Args:
            model (pl.LightningModule): The model to attack.
            inputs (torch.Tensor): The input data batch to attack.
            targets (torch.Tensor): The target labels for the input data.

        Returns:
            torch.Tensor: The adversarially perturbed batch.
        """
        pass

    @abstractmethod
    def _prepare_model(self, model: pl.LightningModule | nn.Module) -> nn.Module:
        """
        Prepare the model for the attack.

        Args:
            model (pl.LightningModule | nn.Module): The model to prepare.
        Returns:
            nn.Module: The prepared model.
        """
        pass

    def forward(
        self,
        model: pl.LightningModule | nn.Module,
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Perform the adversarial attack on the given model and data.

        Wraps the model using `_prepare_model` before calling `_forward`.

        Args:
            model (pl.LightningModule | nn.Module): The model to attack.
            inputs (torch.Tensor): The input data batch to attack.
            targets (torch.Tensor): The target labels for the input data.

        Returns:
            torch.Tensor: The adversarially perturbed batch.
        """
        model = self._prepare_model(model)
        return self._forward(model, inputs, targets)

    def __str__(self) -> str:
        """
        Return a string representation of the attack, with __init__ args.
        """
        sig = inspect.signature(self.__init__)
        params = []
        for name, param in sig.parameters.items():
            if name == "self":
                continue
            if hasattr(self, name):
                value = getattr(self, name)
                params.append(f"{name}={value!r}")
            else:
                if param.default is not inspect.Parameter.empty:
                    params.append(f"{name}={param.default!r}")
                else:
                    params.append(f"{name}=<?>")
        return f"{self.__class__.__name__}({', '.join(params)})"
