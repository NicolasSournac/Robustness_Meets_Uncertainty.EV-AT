from ev_at.pipe.eval.cifar import CIFARClassificationEvaluationModule
from ev_at.pipe.training.at import ATTrainingModule
from ev_at.pipe.training.emff import EMFFTradesTrainingModule
from ev_at.pipe.training.ev_at import EVATTrainingModule
from ev_at.pipe.training.ikl import IKLTrainingModule
from ev_at.pipe.training.standard_training import StandardTrainingModule
from ev_at.pipe.training.trades import TradesTrainingModule

__all__ = [
    "ATTrainingModule",
    "EMFFTradesTrainingModule",
    "TradesTrainingModule",
    "StandardTrainingModule",
    "IKLTrainingModule",
    "CIFARClassificationEvaluationModule",
    "EVATTrainingModule",
]
