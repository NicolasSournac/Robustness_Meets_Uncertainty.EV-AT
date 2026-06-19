import numpy as np
import torch
from sklearn.metrics import auc
from torch import Tensor
from torchmetrics import Metric


class AURiskCoverage(Metric):
    """Area Under the Risk-Coverage Curve (AURC) metric for evaluating
    selective classification performance.
    """

    def __init__(self):
        super().__init__()
        self.probs: list[Tensor]
        self.target: list[Tensor]
        self.scores: list[Tensor]
        self.add_state("probs", default=[], dist_reduce_fx="cat")
        self.add_state("target", default=[], dist_reduce_fx="cat")
        self.add_state("scores", default=[], dist_reduce_fx="cat")

    def update(self, probs: Tensor, targets: Tensor, uncertainty_scores: Tensor):
        self.probs.append(probs.detach().cpu())
        self.target.append(targets.detach().cpu())
        self.scores.append(uncertainty_scores.detach().cpu())

    def compute(self) -> float:
        probs = torch.cat(self.probs)
        targets = torch.cat(self.target)
        scores = torch.cat(self.scores)
        scores = scores.squeeze()

        probs = probs.cpu().numpy()
        targets = targets.cpu().numpy()
        scores = scores.cpu().numpy()

        num_samples = probs.shape[0]

        preds = np.argmax(probs, axis=1).astype(np.int64)
        errors = (preds != targets).astype(np.int64)

        sorted_indices = np.argsort(scores)
        sorted_errors = errors[sorted_indices]

        k_values = np.arange(1, num_samples + 1)
        coverages = k_values / num_samples

        cum_errors = np.cumsum(sorted_errors)
        risks = cum_errors / k_values

        coverages = np.concatenate([[0.0], coverages])
        risks = np.concatenate([[0.0], risks])

        aurc = auc(coverages, risks)

        return aurc


class AUGRiskCoverage(Metric):
    """Area Under the Generalized Risk-Coverage Curve (AURC) metric for evaluating

    selective classification performance.
    Source: https://arxiv.org/abs/2407.01032
    """

    def __init__(self):
        super().__init__()
        self.probs: list[Tensor]
        self.target: list[Tensor]
        self.scores: list[Tensor]
        self.add_state("probs", default=[], dist_reduce_fx="cat")
        self.add_state("target", default=[], dist_reduce_fx="cat")
        self.add_state("scores", default=[], dist_reduce_fx="cat")

    def update(self, probs: Tensor, targets: Tensor, uncertainty_scores: Tensor):
        self.probs.append(probs.detach().cpu())
        self.target.append(targets.detach().cpu())
        self.scores.append(uncertainty_scores.detach().cpu())

    def compute(self) -> float:
        probs = torch.cat(self.probs)
        targets = torch.cat(self.target)
        scores = torch.cat(self.scores)
        scores = scores.squeeze()

        probs = probs.cpu().numpy()
        targets = targets.cpu().numpy()
        scores = scores.cpu().numpy()

        num_samples = probs.shape[0]

        preds = np.argmax(probs, axis=1).astype(np.int64)
        errors = (preds != targets).astype(np.int64)

        sorted_indices = np.argsort(scores)
        sorted_errors = errors[sorted_indices]

        k_values = np.arange(1, num_samples + 1)
        coverages = k_values / num_samples

        cum_errors = np.cumsum(sorted_errors)
        risks = cum_errors / k_values

        coverages = np.concatenate([[0.0], coverages])
        risks = np.concatenate([[0.0], risks])

        aurc = auc(coverages, risks * coverages)

        return aurc
