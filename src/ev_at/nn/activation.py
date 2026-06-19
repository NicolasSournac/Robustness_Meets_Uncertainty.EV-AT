from torch import Tensor, nn

from .functional import evidence_relu, evidence_softplus

__all__ = ["EvidenceSoftplus", "EvidenceReLU"]


class EvidenceSoftplus(nn.Module):
    def __init__(self, min_evidence: float = 1.0):
        super().__init__()
        self.min_evidence = min_evidence

    def forward(self, x: Tensor) -> Tensor:
        return evidence_softplus(x, self.min_evidence)


class EvidenceReLU(nn.Module):
    def __init__(self, min_evidence: float = 1.0):
        super().__init__()
        self.min_evidence = min_evidence

    def forward(self, x: Tensor) -> Tensor:
        return evidence_relu(x, self.min_evidence)
