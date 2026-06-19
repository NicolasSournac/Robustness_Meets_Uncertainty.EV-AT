from collections import OrderedDict
from typing import Literal

import torch
import torch.nn.functional as F  # noqa
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer

from ev_at.attacks.base import AdversarialAttack
from ev_at.core.modules import Module
from ev_at.core.pipe import TrainingModule
from ev_at.modules import EvidentialModule

__all__ = ["TradesTrainingModule"]


class TradesTrainingModule(TrainingModule):
    """Trades Adversarial training module.
    Allows usage of AWP and allows the usage of EvidentialModule as the model.
    Code extracted from:
    https://github.com/csdongxian/AWP/tree/main
    """

    def __init__(
        self,
        test_attack: AdversarialAttack,
        model: Module,
        proxy: Module,
        optimizer: Optimizer,
        proxy_optimizer: Optimizer,
        scheduler: LRScheduler | None,
        beta: float,
        epsilon: float,
        num_steps: int,
        step_size: float,
        awp_warmup: int,
        awp_gamma: float,
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
        self.awp_warmup = awp_warmup
        self.awp_gamma = awp_gamma
        self.proxy = proxy
        self.proxy_optimizer = proxy_optimizer
        self.epochs = 200
        self.norm = norm
        self.awp_adversary = TradesAWP(
            model=self.model,
            proxy=self.proxy,
            proxy_optim=self.proxy_optimizer,
            gamma=self.awp_gamma,
        )

    def state_dict(self, *args, destination=None, prefix="", keep_vars=False):
        """Override state_dict to exclude proxy parameters."""
        state = super().state_dict(
            *args, destination=destination, prefix=prefix, keep_vars=keep_vars
        )
        # Remove all proxy-related keys
        keys_to_remove = [k for k in state.keys() if "proxy" in k]
        for key in keys_to_remove:
            del state[key]
        return state

    def load_state_dict(self, state_dict, strict=True):
        """Override load_state_dict to handle missing proxy parameters."""
        # Load with strict=False to ignore missing proxy parameters
        return super().load_state_dict(state_dict, strict=False)

    def training_step(self, batch, batch_idx):
        image = batch[0]
        labels = batch[1]

        x_adv = perturb_input(
            self.model,
            image,
            step_size=self.step_size,
            epsilon=self.epsilon,
            perturb_steps=self.num_steps,
            distance=self.norm,
        )
        self.model.train()

        if self.current_epoch >= self.awp_warmup:
            self.awp = self.awp_adversary.calc_awp(
                inputs_adv=x_adv, inputs_clean=image, targets=labels, beta=self.beta
            )
            self.awp_adversary.perturb(self.awp)
        self.optimizer.zero_grad()

        logits_adv, adv_preds, _ = self.model(x_adv)
        logits_nat, _, _ = self.model(image)

        # Model's loss on natural examples
        loss_natural = self.model.loss_fn(logits_nat, labels)

        # KL divergence loss between natural and adversarial logits
        if isinstance(self.model, EvidentialModule):
            # Convert evidential outputs back to pseudo-logits
            logits_nat = torch.log(logits_nat)
            logits_adv = torch.log(logits_adv)
        loss_robust = F.kl_div(
            F.log_softmax(logits_adv, dim=1),
            F.softmax(logits_nat, dim=1),
            reduction="batchmean",
        )

        # Combine losses
        loss = loss_natural + (self.beta * loss_robust)

        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        self.train_metrics.update(adv_preds.long(), labels.long())
        return loss

    def on_train_batch_end(self, outputs, batch, batch_idx):
        if self.current_epoch >= self.awp_warmup:
            self.awp_adversary.restore(self.awp)
        return super().on_train_batch_end(outputs, batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        x, y, _ = batch

        x_adv, y_adv = x.clone().detach(), y.clone().detach()
        with torch.enable_grad():
            x_adv = self.test_attack(self.model, x_adv, y_adv)
        self.model.eval()
        with torch.no_grad():
            adv_logits, adv_preds, _ = self.model.forward(x_adv)
            nat_logits, _, _ = self.model.forward(x)

        # Model's loss on natural examples
        loss_natural = self.model.loss_fn(nat_logits, y)

        # KL divergence loss between natural and adversarial logits
        if isinstance(self.model, EvidentialModule):
            # Convert evidential outputs back to pseudo-logits
            nat_logits = torch.log(nat_logits)
            adv_logits = torch.log(adv_logits)
        loss_robust = F.kl_div(
            F.log_softmax(adv_logits, dim=1),
            F.softmax(nat_logits, dim=1),
            reduction="batchmean",
        )

        # Combine losses
        loss = loss_natural + (self.beta * loss_robust)

        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.val_metrics.update(adv_preds.long(), y.long())
        return loss_robust


def perturb_input(
    model, x_natural, step_size=0.003, epsilon=0.031, perturb_steps=10, distance="Linf"
):
    model.eval()
    if distance == "Linf":
        x_adv = (
            x_natural.detach()
            + 0.001 * torch.randn(x_natural.shape).to(x_natural.device).detach()
        )
        for _ in range(perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                if isinstance(model, EvidentialModule):
                    loss_kl = F.kl_div(
                        F.log_softmax(torch.log(model(x_adv)[0]), dim=1),
                        F.softmax(torch.log(model(x_natural)[0]), dim=1),
                        reduction="sum",
                    )
                else:
                    loss_kl = F.kl_div(
                        F.log_softmax(model(x_adv)[0], dim=1),
                        F.softmax(model(x_natural)[0], dim=1),
                        reduction="sum",
                    )
            grad = torch.autograd.grad(loss_kl, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(
                torch.max(x_adv, x_natural - epsilon), x_natural + epsilon
            )
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
    elif distance == "L2":
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

        for _ in range(perturb_steps):
            x_adv = x_adv.detach().requires_grad_(True)

            # On maximise la divergence => gradient ascent sur x_adv
            if isinstance(model, EvidentialModule):
                cost = F.kl_div(
                    F.log_softmax(torch.log(model(x_adv)[0]), dim=1),
                    F.softmax(torch.log(model(x_natural)[0]), dim=1),
                    reduction="sum",
                )
            else:
                cost = F.kl_div(
                    F.log_softmax(model(x_adv)[0], dim=1),
                    F.softmax(model(x_natural)[0], dim=1),
                    reduction="sum",
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


class TradesAWP:
    def __init__(self, model, proxy, proxy_optim, gamma):
        super().__init__()
        self.model = model
        self.proxy = proxy
        self.proxy_optim = proxy_optim
        self.gamma = gamma

    def calc_awp(self, inputs_adv, inputs_clean, targets, beta):
        self.proxy.load_state_dict(self.model.state_dict())
        self.proxy.train()

        if isinstance(self.proxy, EvidentialModule):
            loss_natural = F.cross_entropy(
                torch.log(self.proxy(inputs_clean)[0]), targets
            )
            loss_robust = F.kl_div(
                F.log_softmax(torch.log(self.proxy(inputs_adv)[0]), dim=1),
                F.softmax(torch.log(self.proxy(inputs_clean)[0]), dim=1),
                reduction="batchmean",
            )
        else:
            loss_natural = F.cross_entropy(self.proxy(inputs_clean)[0], targets)
            loss_robust = F.kl_div(
                F.log_softmax(self.proxy(inputs_adv)[0], dim=1),
                F.softmax(self.proxy(inputs_clean)[0], dim=1),
                reduction="batchmean",
            )
        loss = -1.0 * (loss_natural + beta * loss_robust)

        self.proxy_optim.zero_grad()
        loss.backward()
        self.proxy_optim.step()

        # the adversary weight perturb
        diff = diff_in_weights(self.model, self.proxy)
        return diff

    def perturb(self, diff):
        add_into_weights(self.model, diff, coeff=1.0 * self.gamma)

    def restore(self, diff):
        add_into_weights(self.model, diff, coeff=-1.0 * self.gamma)


EPS = 1e-20


def diff_in_weights(model, proxy):
    diff_dict = OrderedDict()
    model_state_dict = model.state_dict()
    proxy_state_dict = proxy.state_dict()
    for (old_k, old_w), (_, new_w) in zip(
        model_state_dict.items(), proxy_state_dict.items(), strict=True
    ):
        if len(old_w.size()) <= 1:
            continue
        if "weight" in old_k:
            diff_w = new_w - old_w
            diff_dict[old_k] = old_w.norm() / (diff_w.norm() + EPS) * diff_w
    return diff_dict


def add_into_weights(model, diff, coeff=1.0):
    names_in_diff = diff.keys()
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in names_in_diff:
                param.add_(coeff * diff[name])
