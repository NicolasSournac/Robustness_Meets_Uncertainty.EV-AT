from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer

from ev_at.attacks.base import AdversarialAttack
from ev_at.core.pipe import TrainingModule
from ev_at.modules import EMFFModule

__all__ = ["EMFFTradesTrainingModule"]


class EMFFTradesTrainingModule(TrainingModule):
    """EMFF-TRADES Adversarial training module
    Code extracted from:
    https://github.com/zissermannn/emff/tree/main
    """

    def __init__(
        self,
        test_attack: AdversarialAttack,
        model: EMFFModule,
        optimizer: Optimizer,
        scheduler: LRScheduler | None,
        beta: float,
        epsilon: float,
        num_steps: int,
        step_size: float,
        num_classes: int,
        norm: Literal["Linf", "L2"],
    ):
        super().__init__(
            model,
            optimizer,
            scheduler,
            num_classes=num_classes,
        )
        self.test_attack = test_attack
        self.beta = beta
        self.epsilon = epsilon
        self.num_steps = num_steps
        self.step_size = step_size
        self.epochs = 200
        self.norm = norm

    def training_step(self, batch, batch_idx):
        image = batch[0]
        labels = batch[1]
        batch_size = len(image)
        criterion_kl = nn.KLDivLoss(size_average=False)

        x_adv = perturb_input(
            self.model,
            image,
            labels,
            step_size=self.step_size,
            epsilon=self.epsilon,
            perturb_steps=self.num_steps,
            epoch=self.current_epoch,
            norm=self.norm,
        )
        self.model.train()
        self.optimizer.zero_grad()

        logits_adv, adv_preds, _, tmc_loss2 = self.model(
            x_adv, labels, self.current_epoch
        )
        logits_nat, _, _, _ = self.model(image, labels, self.current_epoch)

        # Model's loss on natural examples
        loss_natural = self.model.loss_fn(logits_nat, labels)

        loss_robust = (1.0 / batch_size) * criterion_kl(
            F.log_softmax(logits_adv, dim=1), F.softmax(logits_nat, dim=1)
        )

        # Combine losses
        loss = loss_natural + self.beta * loss_robust + 0.25 * tmc_loss2

        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        self.train_metrics.update(adv_preds.long(), labels.long())
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, _ = batch
        batch_size = len(x)
        criterion_kl = nn.KLDivLoss(size_average=False)

        x_adv, y_adv = x.clone().detach(), y.clone().detach()
        with torch.enable_grad():
            x_adv = self.test_attack(self.model, x_adv, y_adv)
        self.model.eval()
        with torch.no_grad():
            logits_adv, adv_preds, _, tmc_loss2 = self.model(
                x_adv, y, self.current_epoch
            )
            logits_nat, _, _, _ = self.model(x, y, self.current_epoch)

        # Model's loss on natural examples
        loss_natural = self.model.loss_fn(logits_nat, y)

        loss_robust = (1.0 / batch_size) * criterion_kl(
            F.log_softmax(logits_adv, dim=1), F.softmax(logits_nat, dim=1)
        )

        # Combine losses
        loss = loss_natural + self.beta * loss_robust + 0.25 * tmc_loss2

        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.val_metrics.update(adv_preds.long(), y.long())
        return loss_robust


def perturb_input(
    model,
    x_natural,
    y,
    step_size=0.003,
    epsilon=0.031,
    perturb_steps=10,
    epoch=0,
    norm="Linf",
):
    # define KL-loss
    criterion_kl = nn.KLDivLoss(size_average=False)
    model.eval()
    if norm == "Linf":
        # generate adversarial example
        x_adv = (
            x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()
        )
        logits = model(x_natural, y, epoch)[0]
        for _ in range(perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                loss_kl = criterion_kl(
                    F.log_softmax(model(x_adv, y, epoch)[0], dim=1),
                    F.softmax(logits, dim=1),
                )
            grad = torch.autograd.grad(loss_kl, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(
                torch.max(x_adv, x_natural - epsilon), x_natural + epsilon
            )
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
    elif norm == "L2":
        bs = x_natural.size(0)

        # Random start dans la boule L2
        delta = torch.randn_like(x_natural)
        delta_flat = delta.view(bs, -1)
        delta_norm = delta_flat.norm(p=2, dim=1).clamp_min(1e-12)
        delta = delta / delta_norm.view(-1, 1, 1, 1)

        # rayon aléatoire dans [0, epsilon]
        r = torch.rand(bs, device=x_natural.device).view(-1, 1, 1, 1)
        delta = delta * (r * epsilon)
        x_adv = (x_natural + delta).clamp(0.0, 1.0)

        logits = model(x_natural, y, epoch)[0]
        for _ in range(perturb_steps):
            x_adv = x_adv.detach().requires_grad_(True)

            # On maximise la divergence => gradient ascent sur x_adv
            cost = criterion_kl(
                F.log_softmax(model(x_adv, y, epoch)[0], dim=1),
                F.softmax(logits, dim=1),
            )

            grad = torch.autograd.grad(cost, x_adv, only_inputs=True)[0]
            grad = torch.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)

            # normalisation L2 du gradient
            grad_flat = grad.view(bs, -1)
            grad_norm = grad_flat.norm(p=2, dim=1).clamp_min(1e-12)
            grad = grad / grad_norm.view(-1, 1, 1, 1)

            # step
            x_adv = x_adv.detach() + step_size * grad

            # projection sur boule L2 de rayon epsilon autour de x
            delta = x_adv - x_natural
            delta = torch.nan_to_num(delta, nan=0.0, posinf=0.0, neginf=0.0)

            delta_flat = delta.view(bs, -1)
            delta_norm = delta_flat.norm(p=2, dim=1).clamp_min(1e-12)
            factor = torch.clamp(epsilon / delta_norm, max=1.0)
            delta = delta * factor.view(-1, 1, 1, 1)

            # clamp image
            x_adv = (x_natural + delta).clamp(0.0, 1.0)

    return x_adv
