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

__all__ = ["ATTrainingModule"]


class ATTrainingModule(TrainingModule):
    """AT Adversarial training module.
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
        self.epsilon = epsilon
        self.num_steps = num_steps
        self.step_size = step_size
        self.awp_warmup = awp_warmup
        self.awp_gamma = awp_gamma
        self.proxy = proxy
        self.proxy_optimizer = proxy_optimizer
        self.norm = norm
        self.epochs = 200
        self.awp_adversary = AdvWeightPerturb(
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

        x_adv = attack_pgd(
            self.model,
            image,
            labels,
            epsilon=self.epsilon,
            alpha=self.step_size,
            attack_iters=self.num_steps,
            norm=self.norm,
            restarts=1,
        )
        x_adv = x_adv.detach()
        self.model.train()

        if self.current_epoch >= self.awp_warmup:
            self.awp = self.awp_adversary.calc_awp(inputs_adv=x_adv, targets=labels)
            self.awp_adversary.perturb(self.awp)
        self.optimizer.zero_grad()

        logits_adv, adv_preds, _ = self.model(x_adv)
        loss = self.model.loss_fn(logits_adv, labels)

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

        loss_robust = self.model.loss_fn(adv_logits, y)

        self.log("val_loss", loss_robust, on_epoch=True, prog_bar=True)
        self.val_metrics.update(adv_preds.long(), y.long())
        return loss_robust


def clamp(X, lower_limit, upper_limit):  # noqa
    return torch.max(torch.min(X, upper_limit), lower_limit)


def attack_pgd(
    model,
    X,  # noqa
    y,
    epsilon,
    alpha,
    attack_iters,
    restarts,
    norm,
    early_stop=False,
):
    max_loss = torch.zeros(y.shape[0]).cuda()
    max_delta = torch.zeros_like(X).cuda()
    for _ in range(restarts):
        delta = torch.zeros_like(X).cuda()
        if norm == "Linf":
            delta.uniform_(-epsilon, epsilon)
        elif norm == "L2":
            delta.normal_()
            d_flat = delta.view(delta.size(0), -1)
            n = d_flat.norm(p=2, dim=1).view(delta.size(0), 1, 1, 1)
            r = torch.zeros_like(n).uniform_(0, 1)
            delta *= r / n * epsilon
        delta = clamp(delta, 0 - X, 1 - X)
        delta.requires_grad = True
        for _ in range(attack_iters):
            output = model(X + delta)[0]
            if early_stop:
                index = torch.where(output.max(1)[1] == y)[0]
            else:
                index = slice(None, None, None)
            if not isinstance(index, slice) and len(index) == 0:
                break
            if isinstance(model, EvidentialModule):
                loss = F.cross_entropy(torch.log(output), y)
            else:
                loss = F.cross_entropy(output, y)
            loss.backward()
            grad = delta.grad.detach()
            d = delta[index, :, :, :]
            g = grad[index, :, :, :]
            x = X[index, :, :, :]
            if norm == "Linf":
                d = torch.clamp(d + alpha * torch.sign(g), min=-epsilon, max=epsilon)
            elif norm == "L2":
                g_norm = torch.norm(g.view(g.shape[0], -1), dim=1).view(-1, 1, 1, 1)
                scaled_g = g / (g_norm + 1e-10)
                d = (
                    (d + scaled_g * alpha)
                    .view(d.size(0), -1)
                    .renorm(p=2, dim=0, maxnorm=epsilon)
                    .view_as(d)
                )
            d = clamp(d, 0 - x, 1 - x)
            delta.data[index, :, :, :] = d
            delta.grad.zero_()
            if isinstance(model, EvidentialModule):
                all_loss = F.cross_entropy(
                    torch.log(model(X + delta)[0]), y, reduction="none"
                )
            else:
                all_loss = F.cross_entropy(model(X + delta)[0], y, reduction="none")
        max_delta[all_loss >= max_loss] = delta.detach()[all_loss >= max_loss]
        max_loss = torch.max(max_loss, all_loss)
    return torch.clamp(X + max_delta[: X.size(0)], min=0, max=1)


class AdvWeightPerturb:
    def __init__(self, model, proxy, proxy_optim, gamma):
        super().__init__()
        self.model = model
        self.proxy = proxy
        self.proxy_optim = proxy_optim
        self.gamma = gamma

    def calc_awp(self, inputs_adv, targets):
        self.proxy.load_state_dict(self.model.state_dict())
        self.proxy.train()

        if isinstance(self.proxy, EvidentialModule):
            loss = -F.cross_entropy(torch.log(self.proxy(inputs_adv)[0]), targets)
        else:
            loss = -F.cross_entropy(self.proxy(inputs_adv)[0], targets)

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
