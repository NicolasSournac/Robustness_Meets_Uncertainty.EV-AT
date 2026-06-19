import lightning as pl
import torch
from autoattack import AutoAttack

from ev_at.core.logging import get_logger
from ev_at.modules import EMFFModule, EvidentialModule

from .base import AdversarialAttack
from .utils import EMFFModuleWrapper, EvidentialModuleWrapper

logger = get_logger("fgsm_attack")


class AutoAttackWrapper(AdversarialAttack):
    """
    Wrapper of the AutoAttack (Croce et al., 2020) adversarial attack.
    """

    def __init__(self, epsilon: float = 0.1, batch_size: int = 128, norm: str = "Linf"):
        """
        Initialize the FGSM attack with a specified perturbation magnitude.

        Args:
            loss_fn (nn.Module): The loss function to use for calculating gradients.
            epsilon (float): The magnitude of the perturbation to apply to the input
                data.
            batch_size (int): The batch size to use for the attack.
            norm (str): The norm to use for the attack. Can be "Linf" or "L2".

        """
        super().__init__()
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.norm = norm

    def _prepare_model(self, model):
        if isinstance(model, EvidentialModule):
            return EvidentialModuleWrapper(model)
        if isinstance(model, EMFFModule):
            return EMFFModuleWrapper(model)
        return model

    def _forward(  # type: ignore
        self, model: pl.LightningModule, inputs: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Perform the AutoAttack on the given model and data.

        Args:
            model (pl.LightningModule): The model to attack.
            inputs (torch.Tensor): The input data batch to attack.
            targets (torch.Tensor): The target labels for the input data.

        Returns:
            torch.Tensor: The adversarially perturbed batch.
        """
        inputs = inputs.clone().detach().requires_grad_(True)

        model.zero_grad()

        def model_wrapper(x):
            return model(x)[0]

        adversary = AutoAttack(
            model_wrapper,
            norm=self.norm,
            eps=self.epsilon,
            version="standard",
            verbose=False,
            seed=0,
        )
        # AutoAttack returns, for each original input,
        # the first adversarial example that caused the model
        # to misclassify that input (if any)
        adversarial_inputs = adversary.run_standard_evaluation(
            inputs, targets, bs=self.batch_size
        )
        return adversarial_inputs.detach()
