"""
This module defines constants used throughout the framework.
"""

from dataclasses import dataclass

import torch

__all__ = ["EVAT_CONSTANTS"]


@dataclass(frozen=True)
class EVATConstants:
    EPS_F32: float = torch.finfo(torch.float32).eps
    EPS_F64: float = torch.finfo(torch.float64).eps


EVAT_CONSTANTS = EVATConstants()
