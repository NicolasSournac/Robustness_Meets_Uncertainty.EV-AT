import torch
from torch import Tensor
from torch.nn.functional import relu, softplus

__all__ = ["evidence_softplus", "evidence_relu"]


def evidence_softplus(x: Tensor, min_evidence: float = 1.0) -> Tensor:
    return softplus(x) + torch.tensor(min_evidence, device=x.device)


def evidence_relu(x: Tensor, min_evidence: float = 1.0) -> Tensor:
    return relu(x) + torch.tensor(min_evidence, device=x.device)
