import os
from abc import ABC, abstractmethod
from collections.abc import Callable

import lightning as pl
import matplotlib.pyplot as plt
import torch
from pydantic import BaseModel
from sklearn.metrics import auc, roc_curve
from torchmetrics import MetricCollection

from ev_at.attacks import compute_l2_dist, compute_linf_dist
from ev_at.core.logging import get_logger
from ev_at.core.modules import Module
from ev_at.core.pydantic.model_output import ModelOutput
from ev_at.modules import EMFFModule, EvidentialModule

logger = get_logger("evaluation")

__all__ = ["EvaluationModule"]


class EvaluationModule(ABC, pl.LightningModule):
    """
    Abstract evaluation module for pipelines.

    All models are evaluated using this evaluation logic.
    """

    def __init__(
        self,
        attacks: list,
        model: Module,
        output_validation_model: Callable[[ModelOutput], BaseModel] | None = None,
    ):
        """
        Args:
            attacks (list): List of adversarial attacks to evaluate.
            model (nn.Module | pl.LightningModule): The model to evaluate.
            output_validation_model (Callable[[ModelOutput], BaseModel] | None):
                Optional callable to validate model outputs. It only checks that
                the model output is three tensors:
                - logits: Raw model outputs (logits).
                - preds: Predicted class labels.
                - uncertainty: Uncertainty map.
        """
        super().__init__()
        self.attacks = attacks
        self.model = model
        if output_validation_model is None:

            def identity(x: ModelOutput) -> ModelOutput:
                return x

            output_validation_model = identity
        else:
            self.output_validator = output_validation_model

        standard_metrics = self.standard_metrics
        self.nat_metrics = standard_metrics.clone(prefix="nat_")
        uncertainty_metrics = self.uncertainty_metrics
        self.nat_uncertainty_metrics = uncertainty_metrics.clone(prefix="nat_uncert_")
        selective_classification_metrics = self.selective_classification_metrics
        self.nat_selective_classification_metrics = (
            selective_classification_metrics.clone(prefix="nat_")
        )
        self.adv_metrics = []
        self.adv_uncertainty_metrics = []
        self.adv_selective_classification_metrics = []
        for attack in self.attacks:
            attack_name = str(attack).split("(")[0]
            if "loss_fn" in str(attack):
                loss_fn_part = str(attack).split("loss_fn=")[1].split(",")[0]
                if "epsilon=" in str(attack):
                    epsilon_part = str(attack).split("epsilon=")[1].split(",")[0]
                    attack_name += f"_{epsilon_part}"
                if "num_iterations=" in str(attack):
                    num_iterations_part = (
                        str(attack).split("num_iterations=")[1].split(",")[0]
                    )
                    attack_name += f"_{num_iterations_part}"
                attack_name += f"_{loss_fn_part}"
                attack_name = attack_name.replace("CrossEntropyLoss()", "CE")
                attack_name = attack_name.replace("Type2NLLoss()", "T2NLL")
                attack_name = attack_name.replace("ExpectedNLLLoss()", "ENLL")
                attack_name = attack_name.replace("ExpectedSquaredErrorLoss()", "ESE")

            self.adv_metrics.append(standard_metrics.clone(prefix=f"{attack_name}_"))
            self.adv_uncertainty_metrics.append(
                uncertainty_metrics.clone(prefix=f"{attack_name}_uncert_")
            )
            self.adv_selective_classification_metrics.append(
                selective_classification_metrics.clone(prefix=f"{attack_name}_")
            )

        # Storage for later analysis (ROC curves, uncertainty distributions, etc.)
        self._storage = {
            "nat": {"scores": [], "targets": [], "probs": [], "errors": []},
        }
        for attack in self.attacks:
            self._storage[str(attack)] = {
                "scores": [],
                "targets": [],
                "probs": [],
                "errors": [],
            }

    @property
    @abstractmethod
    def standard_metrics(self) -> MetricCollection: ...

    @property
    @abstractmethod
    def uncertainty_metrics(self) -> MetricCollection: ...

    @property
    @abstractmethod
    def selective_classification_metrics(self) -> MetricCollection: ...

    def training_step(self, batch, batch_idx):
        """
        Training step is not used in evaluation module.
        """
        msg = "Training step is not applicable for evaluation module."
        logger.error(msg)
        raise NotImplementedError(msg)

    def validation_step(self, batch, batch_idx):
        """
        Validation step is not used in evaluation module.
        """
        msg = "Validation step is not applicable for evaluation module."
        logger.error(msg)
        raise NotImplementedError(msg)

    def on_test_start(self):
        """
        Called at the start of the test phase to move metrics to the appropriate device.
        """
        self.nat_metrics.to(self.device)
        for adv_metric in self.adv_metrics:
            adv_metric.to(self.device)
        self.nat_uncertainty_metrics.to(self.device)
        for adv_uncertainty_metric in self.adv_uncertainty_metrics:
            adv_uncertainty_metric.to(self.device)
        self.nat_selective_classification_metrics.to(self.device)
        for (
            adv_selective_classification_metric
        ) in self.adv_selective_classification_metrics:
            adv_selective_classification_metric.to(self.device)

    def test_step(self, batch: list[torch.Tensor, torch.Tensor], batch_idx):
        """
        Test step for evaluation.
        This method should be overridden by subclasses to implement custom evaluation
            logic.

        Args:
            batch: The input batch of data.
            batch_idx: The index of the batch.

        Returns:
        """
        inputs, targets = batch[0], batch[1]
        if isinstance(self.model, EMFFModule):
            logits, preds, uncertainty, _ = self.model(inputs, None, None, is_eval=True)
        else:
            logits, preds, uncertainty = self.model(inputs)
        nat_output = ModelOutput(logits=logits, preds=preds, uncertainty=uncertainty)
        self.output_validator(output=nat_output)
        # Natural metrics
        self.nat_metrics.update(nat_output.preds.long(), targets.long())

        # Natural uncertainty metrics
        error = nat_output.preds.long() != targets.long()
        # Store for ROC curves & uncertainty distribution plots
        self._storage["nat"]["scores"].append(nat_output.uncertainty.detach().cpu())
        self._storage["nat"]["errors"].append(error.long().detach().cpu())
        self._storage["nat"]["targets"].append(targets.long().detach().cpu())
        self.nat_uncertainty_metrics.update(
            nat_output.uncertainty.float(), error.long()
        )

        if isinstance(self.model, EvidentialModule):  # Logits are concentrations
            probs = logits / logits.sum(dim=-1, keepdim=True)
        else:  # Base module
            probs = torch.softmax(logits, dim=-1)
        self._storage["nat"]["probs"].append(probs.detach().cpu())

        # Natural selective classification metrics
        self.nat_selective_classification_metrics.update(
            probs, targets.long(), nat_output.uncertainty
        )
        for (
            attack,
            adv_metric,
            adv_uncertainty_metric,
            adv_selective_classification_metric,
        ) in zip(
            self.attacks,
            self.adv_metrics,
            self.adv_uncertainty_metrics,
            self.adv_selective_classification_metrics,
            strict=True,
        ):
            with torch.enable_grad():
                adv_inputs = attack(self.model, inputs, targets)
            if attack.norm == "L2":
                dist = compute_l2_dist(inputs, adv_inputs)
            elif attack.norm == "Linf":
                dist = compute_linf_dist(inputs, adv_inputs)
            assert dist <= attack.epsilon + 1e-5, (
                f"epsilon {dist.item()} exceeds epsilon {attack.epsilon}"
            )
            if isinstance(self.model, EMFFModule):
                logits, preds, uncertainty, _ = self.model(
                    adv_inputs, None, 0, is_eval=True
                )
            else:
                logits, preds, uncertainty = self.model(adv_inputs)
            adv_output = ModelOutput(
                logits=logits, preds=preds, uncertainty=uncertainty
            )
            # Adversarial metrics
            adv_metric.update(adv_output.preds.long(), targets.long())

            # Adversarial uncertainty metrics
            error = adv_output.preds.long() != targets.long()
            key = str(attack)
            self._storage[key]["scores"].append(adv_output.uncertainty.detach().cpu())
            self._storage[key]["errors"].append(error.long().detach().cpu())
            self._storage[key]["targets"].append(targets.long().detach().cpu())
            adv_uncertainty_metric.update(adv_output.uncertainty.float(), error.long())

            if isinstance(self.model, EvidentialModule):  # Logits are concentrations
                probs = logits / logits.sum(dim=-1, keepdim=True)
            else:  # Base module
                probs = torch.softmax(logits, dim=-1)
            self._storage[key]["probs"].append(probs.detach().cpu())

            # Adversarial selective classification metrics
            adv_selective_classification_metric.update(
                probs, targets.long(), adv_output.uncertainty
            )

    def on_test_epoch_end(self):
        """
        Called at the end of the test epoch to compute and log metrics.
        """
        nat_metrics = self.nat_metrics.compute()
        nat_uncertainty_metrics = self.nat_uncertainty_metrics.compute()
        nat_selective_classification_metrics = (
            self.nat_selective_classification_metrics.compute()
        )
        self.log_dict(nat_metrics)
        self.log_dict(nat_uncertainty_metrics)
        self.log_dict(nat_selective_classification_metrics)
        self.nat_metrics.reset()
        self.nat_uncertainty_metrics.reset()
        self.nat_selective_classification_metrics.reset()
        for (
            adv_metric,
            adv_uncertainty_metric,
            adv_selective_classification_metric,
        ) in zip(
            self.adv_metrics,
            self.adv_uncertainty_metrics,
            self.adv_selective_classification_metrics,
            strict=True,
        ):
            adv_metrics = adv_metric.compute()
            adv_uncertainty_metrics = adv_uncertainty_metric.compute()
            adv_selective_classification_metrics = (
                adv_selective_classification_metric.compute()
            )
            self.log_dict(adv_metrics)
            self.log_dict(adv_uncertainty_metrics)
            self.log_dict(adv_selective_classification_metrics)
            adv_metric.reset()
            adv_uncertainty_metric.reset()
            adv_selective_classification_metric.reset()

        # --- ROC curves ---
        self._plot_and_save_roc(
            self._storage["nat"]["scores"],
            self._storage["nat"]["errors"],
            name="nat",
        )

        for attack in self.attacks:
            key = str(attack)
            self._plot_and_save_roc(
                self._storage[key]["scores"],
                self._storage[key]["errors"],
                name=key,
            )

        # --- Uncertainty distributions ---
        self._plot_uncertainty_distribution(
            self._storage["nat"]["scores"],
            self._storage["nat"]["errors"],
            name="nat",
        )

        for attack in self.attacks:
            key = str(attack)
            self._plot_uncertainty_distribution(
                self._storage[key]["scores"],
                self._storage[key]["errors"],
                name=key,
            )

        # Save storage to disk for later analysis
        for k in self._storage:
            self._save_storage_to_disk(
                scores=self._storage[k]["scores"],
                targets=self._storage[k]["targets"],
                probs=self._storage[k]["probs"],
                errors=self._storage[k]["errors"],
                name=k,
            )

        # reset storage
        for k in self._storage:
            self._storage[k]["scores"].clear()
            self._storage[k]["errors"].clear()
            self._storage[k]["probs"].clear()
            self._storage[k]["targets"].clear()

    def _plot_and_save_roc(self, scores, targets, name: str):
        """
        scores: list[Tensor]  -> uncertainty
        targets: list[Tensor] -> error (0 = correct, 1 = incorrect)
        """
        scores = torch.cat(scores).numpy()
        targets = torch.cat(targets).numpy()

        fpr, tpr, _ = roc_curve(targets, scores)
        roc_auc = auc(fpr, tpr)

        plt.figure()
        plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"Uncertainty AUROC – {name}")
        plt.legend(loc="lower right")

        log_dir = self.logger.log_dir
        os.makedirs(log_dir, exist_ok=True)
        path = os.path.join(log_dir, f"roc_uncertainty_{name}.png")

        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()

    def _plot_uncertainty_distribution(self, scores, targets, name: str):
        """
        scores: list[Tensor]  -> uncertainty
        targets: list[Tensor] -> error (0 = correct, 1 = incorrect)
        """
        scores = torch.cat(scores)
        targets = torch.cat(targets)

        correct_uncert = scores[targets == 0].numpy()
        incorrect_uncert = scores[targets == 1].numpy()

        plt.figure()
        plt.hist(
            correct_uncert,
            bins=50,
            density=True,
            alpha=0.6,
            label="Correct",
        )
        plt.hist(
            incorrect_uncert,
            bins=50,
            density=True,
            alpha=0.6,
            label="Incorrect",
        )

        plt.xlabel("Uncertainty")
        plt.ylabel("Density")
        plt.title(f"Uncertainty distribution – {name}")
        plt.legend()

        log_dir = self.logger.log_dir
        os.makedirs(log_dir, exist_ok=True)
        path = os.path.join(log_dir, f"uncertainty_distribution_{name}.png")

        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()

    def _save_storage_to_disk(self, scores, targets, probs, errors, name: str):
        """
        Save the ROC storage to disk for later analysis.
        """
        log_dir = self.logger.log_dir
        os.makedirs(log_dir, exist_ok=True)
        path = os.path.join(log_dir, f"roc_storage_{name}.pt")
        torch.save(
            {"scores": scores, "targets": targets, "probs": probs, "errors": errors},
            path,
        )
