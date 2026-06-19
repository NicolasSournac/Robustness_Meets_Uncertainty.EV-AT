from collections import OrderedDict
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer

from ev_at.attacks.base import AdversarialAttack
from ev_at.core.pipe import TrainingModule
from ev_at.modules import EvidentialModule

__all__ = ["EVATTrainingModule"]


class EVATTrainingModule(TrainingModule):
    """EVAT (Ours) Adversarial training module"""

    def __init__(
        self,
        test_attack: AdversarialAttack,
        model: EvidentialModule,
        proxy: EvidentialModule,
        optimizer: Optimizer,
        proxy_optimizer: Optimizer,
        scheduler: LRScheduler | None,
        beta: float,
        epsilon: float,
        num_steps: int,
        step_size: float,
        alpha: float,
        gamma: float,
        awp_warmup: int,
        awp_gamma: float,
        T: float,  # noqa
        train_budget: Literal["low", "high"],
        num_classes: int,
        max_epochs: int,
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
        self.train_budget = train_budget
        self.max_epochs = max_epochs
        self.alpha = alpha
        self.gamma = gamma
        self.awp_warmup = awp_warmup
        self.awp_gamma = awp_gamma
        self.T = T
        self.proxy = proxy
        self.proxy_optimizer = proxy_optimizer
        self.awp_adversary = TradesAWP(
            model=self.model,
            proxy=self.proxy,
            proxy_optim=self.proxy_optimizer,
            gamma=self.awp_gamma,
        )
        self.weight = None
        self.WEIGHT = None
        self.norm = norm

    def on_save_checkpoint(self, checkpoint: dict) -> None:
        """Save the weight and WEIGHT tensors in the checkpoint."""
        checkpoint["weight"] = self.weight.cpu()
        checkpoint["WEIGHT"] = self.WEIGHT.cpu()

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        """Load the weight and WEIGHT tensors from the checkpoint."""
        if "weight" in checkpoint:
            self.weight = checkpoint["weight"].to(self.device)
        if "WEIGHT" in checkpoint:
            self.WEIGHT = checkpoint["WEIGHT"].to(self.device)

    def state_dict(self, *args, destination=None, prefix="", keep_vars=False):
        """Override state_dict to exclude proxy parameters."""
        state = super().state_dict(
            *args, destination=destination, prefix=prefix, keep_vars=keep_vars
        )
        keys_to_remove = [k for k in state.keys() if "proxy" in k]
        for key in keys_to_remove:
            del state[key]
        return state

    def load_state_dict(self, state_dict, strict=True):
        """Override load_state_dict to handle missing proxy parameters."""
        return super().load_state_dict(state_dict, strict=False)

    def on_train_epoch_start(self):
        """Initialize the weight and WEIGHT tensors
        at the start of each training epoch."""
        self.WEIGHT = torch.zeros(
            self.num_classes,
            self.num_classes,
        ).to(self.device)
        self.weight = (
            self.weight
            if self.weight is not None
            else torch.ones(self.num_classes, self.num_classes).to(self.device)
            / self.num_classes
        )
        self.epoch_scale = self.max_epochs / 200.0
        return super().on_train_epoch_start()

    def training_step(self, batch, batch_idx):
        image = batch[0]
        labels = batch[1]

        varepsilon = self.epsilon * (self.current_epoch / 200)
        if self.train_budget == "low":
            step_size = varepsilon
            iters_attack = 2
        elif self.train_budget == "high":
            if self.current_epoch <= int(50 * self.epoch_scale):
                step_size = varepsilon
                iters_attack = 2
            if self.current_epoch <= int(100 * self.epoch_scale):
                step_size = 2 * varepsilon / 3
                iters_attack = 3
            if self.current_epoch <= int(150 * self.epoch_scale):
                step_size = varepsilon / 2
                iters_attack = 4
            if self.current_epoch <= int(200 * self.epoch_scale):
                step_size = varepsilon / 2
                iters_attack = 5
        elif self.train_budget == "full":
            step_size = self.step_size
            iters_attack = self.num_steps
            varepsilon = self.epsilon

        # calculate sample weights
        with torch.no_grad():
            onehot = F.one_hot(labels, self.num_classes).float()
            sample_weight = onehot @ self.weight.to(self.device)

        # craft adversarial examples
        x_adv = perturb_input(
            model=self.model,
            x_natural=image,
            step_size=step_size,
            epsilon=varepsilon,
            perturb_steps=iters_attack,
            distance=self.norm,
            alpha=1.0,
            beta=0.0,
            gamma=1.0,
            CLASS_PRIOR=sample_weight,
        )

        self.model.train()

        # calculate adversarial weight perturbation
        if self.current_epoch >= self.awp_warmup:
            self.awp = self.awp_adversary.calc_awp(
                inputs_adv=x_adv,
                inputs_clean=image,
                targets=labels,
                alpha=self.alpha,
                beta=self.beta,
                gamma=self.gamma,
                CLASS_PRIOR=sample_weight,
            )
            self.awp_adversary.perturb(self.awp)

        self.optimizer.zero_grad()

        logits_adv, adv_preds, _ = self.model(x_adv)
        logits_nat, _, _ = self.model(image)

        bt = labels.size(0)
        with torch.no_grad():
            # update
            self.WEIGHT = self.WEIGHT + (
                onehot[:bt].t()
                @ F.softmax(logits_nat[:bt].clone().detach() / self.T, dim=-1)
            )

        loss_natural = self.model.loss_fn(logits_nat, labels)
        # Compute pseudo-logits for evidential model
        logits_nat = torch.log(logits_nat)
        logits_adv = torch.log(logits_adv)
        loss_robust = dkl_loss(
            logits_adv,
            logits_nat,
            CLASS_PRIOR=sample_weight,
            alpha=self.alpha,
            beta=self.beta,
            gamma=self.gamma,
        )
        loss = loss_natural + loss_robust

        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        self.train_metrics.update(adv_preds.long(), labels.long())
        return loss

    def on_train_batch_end(self, outputs, batch, batch_idx):
        if self.current_epoch >= self.awp_warmup:
            self.awp_adversary.restore(self.awp)
        return super().on_train_batch_end(outputs, batch, batch_idx)

    def on_train_epoch_end(self):
        self.WEIGHT = self.WEIGHT / self.WEIGHT.sum(dim=1, keepdim=True)
        return super().on_train_epoch_end()

    def validation_step(self, batch, batch_idx):
        x, y, _ = batch

        x_adv, y_adv = x.clone().detach(), y.clone().detach()
        with torch.enable_grad():
            x_adv = self.test_attack(self.model, x_adv, y_adv)
        self.model.eval()
        with torch.no_grad():
            adv_logits, adv_preds, _ = self.model.forward(x_adv)
            nat_logits, _, _ = self.model.forward(x)

        nat_logits = torch.log(nat_logits)
        adv_logits = torch.log(adv_logits)

        criterion = nn.CrossEntropyLoss()
        loss = criterion(nat_logits, y)

        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.val_metrics.update(adv_preds.long(), y.long())
        return loss


def dkl_loss(
    logits_student,
    logits_teacher,
    temperature=1.0,
    alpha=1.0,
    beta=1.0,
    gamma=1.0,
    CLASS_PRIOR=None,  # noqa
):
    NUM_CLASSES = logits_teacher.size(1)  # noqa
    delta_n = logits_teacher.view(-1, NUM_CLASSES, 1) - logits_teacher.view(
        -1, 1, NUM_CLASSES
    )
    delta_a = logits_student.view(-1, NUM_CLASSES, 1) - logits_student.view(
        -1, 1, NUM_CLASSES
    )

    assert CLASS_PRIOR is not None, "CLASS PRIOR information should be collected for AT"
    with torch.no_grad():
        CLASS_PRIOR = torch.pow(CLASS_PRIOR, gamma)  # noqa
        p_n = CLASS_PRIOR.view(-1, NUM_CLASSES, 1) @ CLASS_PRIOR.view(
            -1, 1, NUM_CLASSES
        )

    loss_mse = 0.25 * (torch.pow(delta_n - delta_a, 2) * p_n).sum() / p_n.sum()
    loss_sce = (
        -(
            F.softmax(logits_teacher / temperature, dim=1).detach()
            * F.log_softmax(logits_student / temperature, dim=-1)
        )
        .sum(1)
        .mean()
    )
    return beta * loss_mse + alpha * loss_sce


def perturb_input(
    model,
    x_natural,
    step_size=0.003,
    epsilon=0.031,
    perturb_steps=10,
    distance="Linf",
    alpha=1.0,
    beta=1.0,
    gamma=1.0,
    CLASS_PRIOR=None,  # noqa
    eps_for_division=1e-12,
    clamp_min_for_log=1e-12,
    random_start=True,
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
                loss_kl = dkl_loss(
                    torch.log(model(x_adv)[0]),
                    torch.log(model(x_natural)[0]),
                    CLASS_PRIOR=CLASS_PRIOR,
                    alpha=alpha,
                    beta=beta,
                    gamma=gamma,
                )

            grad = torch.autograd.grad(loss_kl, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(
                torch.max(x_adv, x_natural - epsilon), x_natural + epsilon
            )
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
        return x_adv

    elif distance == "L2":
        x = x_natural.detach()

        # Précalc teacher (natural) une seule fois comme torchattacks
        with torch.no_grad():
            teacher = model(x)[0].detach()
            teacher = torch.log(teacher.clamp_min(clamp_min_for_log))
        bs = x.size(0)

        # Random start dans la boule L2
        if random_start:
            delta = torch.randn_like(x)
            delta_flat = delta.view(bs, -1)
            delta_norm = delta_flat.norm(p=2, dim=1).clamp_min(eps_for_division)
            delta = delta / delta_norm.view(-1, 1, 1, 1)

            # rayon aléatoire dans [0, epsilon]
            r = torch.rand(bs, device=x.device).view(-1, 1, 1, 1)
            delta = delta * (r * epsilon)
            x_adv = (x + delta).clamp(0.0, 1.0)
        else:
            x_adv = x.clone()

        for _ in range(perturb_steps):
            x_adv = x_adv.detach().requires_grad_(True)

            student = model(x_adv)[0]
            student = torch.log(student.clamp_min(clamp_min_for_log))

            # On maximise la divergence => gradient ascent sur x_adv
            cost = dkl_loss(
                student,
                teacher,
                CLASS_PRIOR=CLASS_PRIOR,
                alpha=alpha,
                beta=beta,
                gamma=gamma,
            )

            grad = torch.autograd.grad(cost, x_adv, only_inputs=True)[0]
            grad = torch.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)

            # normalisation L2 du gradient
            grad_flat = grad.view(bs, -1)
            grad_norm = grad_flat.norm(p=2, dim=1).clamp_min(eps_for_division)
            grad = grad / grad_norm.view(-1, 1, 1, 1)

            # step
            x_adv = x_adv.detach() + step_size * grad

            # projection sur boule L2 de rayon epsilon autour de x
            delta = x_adv - x
            delta = torch.nan_to_num(delta, nan=0.0, posinf=0.0, neginf=0.0)

            delta_flat = delta.view(bs, -1)
            delta_norm = delta_flat.norm(p=2, dim=1).clamp_min(eps_for_division)
            factor = torch.clamp(epsilon / delta_norm, max=1.0)
            delta = delta * factor.view(-1, 1, 1, 1)

            # clamp image
            x_adv = (x + delta).clamp(0.0, 1.0)

        return x_adv.detach()


class TradesAWP:
    def __init__(self, model, proxy, proxy_optim, gamma):
        super().__init__()
        self.model = model
        self.proxy = proxy
        self.proxy_optim = proxy_optim
        self.gamma = gamma

    def calc_awp(
        self,
        inputs_adv,
        inputs_clean,
        targets,
        alpha=1.0,
        beta=1.0,
        gamma=1.0,
        CLASS_PRIOR=None,  # noqa
    ):
        self.proxy.load_state_dict(self.model.state_dict())
        self.proxy.train()

        logits_nat = self.proxy(inputs_clean)[0]
        logits_adv = self.proxy(inputs_adv)[0]
        logits_nat = torch.log(logits_nat)
        logits_adv = torch.log(logits_adv)
        loss_natural = F.cross_entropy(logits_nat, targets)

        assert CLASS_PRIOR is not None, "CLASS_PRIOR should be collected for AT"
        loss_robust = dkl_loss(
            logits_adv,
            logits_nat,
            CLASS_PRIOR=CLASS_PRIOR,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
        )
        loss = -1.0 * (loss_natural + loss_robust)

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
            diff_dict[old_k] = old_w.norm() / (diff_w.norm() + 1e-20) * diff_w
    return diff_dict


def add_into_weights(model, diff, coeff=1.0):
    names_in_diff = diff.keys()
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in names_in_diff:
                param.add_(coeff * diff[name])
