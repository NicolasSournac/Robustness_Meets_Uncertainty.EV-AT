import torch
from torch import Tensor
from torchmetrics import Metric

from ev_at import EVAT_CONSTANTS

__all__ = [
    "Entropy",
]


class Entropy(Metric):
    """Compute entropy for categorical distribution"""

    def __init__(self, **metric_kwargs):
        super().__init__(**metric_kwargs)
        self.entropy: list[Tensor]
        self.add_state("entropy", default=[], dist_reduce_fx="cat")

    def update(self, probabilities: Tensor):
        self.entropy.append(
            -(
                probabilities
                * torch.clamp(probabilities, min=EVAT_CONSTANTS.EPS_F32).log2()
            ).sum(dim=1, keepdim=True)
        )

    def compute(self):
        if self.entropy:
            return torch.cat(self.entropy, dim=0).squeeze(1)
