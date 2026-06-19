import torch
from torch import Tensor
from torchmetrics import Metric


class Epistemic(Metric):
    """
    Compute the epistemic uncertainty (mutual information)
    from the concentration parameters of a Dirichlet distribution.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("mutual_info", default=[], dist_reduce_fx="cat")

    def update(self, concentration: Tensor) -> None:
        strength = concentration.sum(dim=1, keepdim=True)
        mutual_info = -(
            (
                torch.log(concentration / strength)
                - torch.digamma(concentration + 1)
                + torch.digamma(strength + 1)
            )
            * (concentration / strength)
        ).sum(dim=1)
        self.mutual_info.append(mutual_info)

    def compute(self) -> Tensor:
        mutual_info = torch.cat(self.mutual_info, dim=0)
        return mutual_info


class Aleatoric(Metric):
    """
    Compute the aleatoric uncertainty (expected entropy)
    from the concentration parameters of a Dirichlet distribution.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("uncertainty", default=[], dist_reduce_fx="cat")

    def update(self, concentration: Tensor) -> None:
        strength = concentration.sum(dim=1, keepdim=True)
        uncertainty = -(
            (torch.digamma(concentration + 1) - torch.digamma(strength + 1))
            * (concentration / strength)
        ).sum(dim=1)
        self.uncertainty.append(uncertainty)

    def compute(self) -> Tensor:
        uncertainty = torch.cat(self.uncertainty, dim=0)
        return uncertainty


class TotalUncertainty(Metric):
    """
    Computes the total uncertainty as the sum of epistemic and aleatoric uncertainties.
    Reuses the existing Epistemic and Aleatoric metric classes.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.epistemic = Epistemic(**kwargs)
        self.aleatoric = Aleatoric(**kwargs)
        self.add_state("total_uncertainty", default=[], dist_reduce_fx="cat")

    def update(self, concentration: Tensor) -> None:
        """
        Updates both epistemic and aleatoric sub-metrics
            using the same concentration tensor.
        """
        self.epistemic.update(concentration)
        self.aleatoric.update(concentration)

    def compute(self) -> Tensor:
        """
        Computes total uncertainty = epistemic + aleatoric.
        """
        epistemic_val = self.epistemic.compute()
        aleatoric_val = self.aleatoric.compute()
        total_uncertainty = epistemic_val + aleatoric_val
        self.total_uncertainty = total_uncertainty
        return total_uncertainty

    def reset(self) -> None:
        """
        Resets both epistemic and aleatoric sub-metrics.
        """
        self.epistemic.reset()
        self.aleatoric.reset()
        super().reset()
