"""
This module provides utility functions and classes for instantiating components
in pipelines, including model, loss function, optimizer, scheduler,
trainer, and callbacks. It also includes a configuration parser for Hydra-based
training and evaluation setups.
"""

from typing import Literal

import lightning
import torch
from hydra.utils import instantiate
from omegaconf import MISSING, DictConfig, OmegaConf
from omegaconf.errors import ConfigAttributeError
from pydantic import BaseModel
from torch import nn

from ev_at.core.environment import environment_loader
from ev_at.core.logging import get_logger

logger = get_logger("pipe-init")

__all__ = [
    "ExperimentConfig",
    "Instantiator",
    "CallbacksHandler",
    "config_parsing",
]


class ExperimentConfig(BaseModel):
    env_file: str
    seed: int
    training_pipe: (
        Literal["standard", "ikl", "trades", "at", "trades-emff", "ev-at"] | None
    ) = None
    model_type: Literal["base", "evidential", "emff"]
    name: str = MISSING


class Instantiator:
    """
    Utility class to instantiate components from the configuration files.

    This class is designed to be used with Hydra training and evaluation configurations.
    It instantiate the following components:
    - Model
    - Loss function
    - Optimizer
    - Scheduler
    - Trainer
    - Callbacks
    - Attacks
    """

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg

    def instantiate_all(self):
        model = self.instantiate_model()
        if hasattr(self.cfg, "optimizer"):
            optimizer = self.instantiate_optimizer(model)

        if hasattr(self.cfg, "proxy"):
            proxy = self.instantiate_proxy()
            proxy_optimizer = self.instantiate_proxy_optimizer(proxy)

        uncertainty_score = self.instantiate_uncertainty_score()
        if hasattr(self.cfg, "scheduler"):
            scheduler = self.instantiate_scheduler(optimizer)
        loss = self.instantiate_loss()
        if hasattr(self.cfg, "callbacks"):
            callbacks = self.instantiate_callbacks()
            if callbacks:
                cb_list = [cb for _, cb in callbacks.items()]
        else:
            cb_list = []
        attacks = self.instantiate_attacks()
        if attacks:
            attacks_list = [attack for _, attack in attacks.items()]
        else:
            attacks_list = []
        trainer = self.instantiate_trainer(cb_list)
        return {
            "model": model,
            "optimizer": optimizer if hasattr(self.cfg, "optimizer") else None,
            "proxy": proxy if hasattr(self.cfg, "proxy") else None,
            "proxy_optimizer": proxy_optimizer if hasattr(self.cfg, "proxy") else None,
            "scheduler": scheduler if hasattr(self.cfg, "scheduler") else None,
            "uncertainty_score": uncertainty_score,
            "loss": loss,
            "trainer": trainer,
            "callbacks": callbacks if hasattr(self.cfg, "callbacks") else None,
            "attacks": attacks_list,
        }

    def get_instatiated_cfg(self) -> DictConfig:
        """
        Returns the instantiated configuration with all components.
        This method resolves the configuration and instantiates all components
        defined in the configuration file.
        """
        cfg = OmegaConf.to_container(self.cfg, resolve=True)
        instances = self.instantiate_all()
        cfg.update(instances)
        return cfg

    def instantiate_model(self) -> nn.Module:
        logger.debug("Instantiating model")
        try:
            model = instantiate(self.cfg.model)
        except ConfigAttributeError as e:
            msg = f"'model' attribute is missing in the configuration file: {e}"
            logger.exception(msg)
            raise e

        model = model.get("model", None)
        if model is None:
            msg = (
                "'model' key is missing in the model configuration file. "
                "Please provide a 'model' configuration."
            )
            logger.error(msg)
            raise ConfigAttributeError(msg)

        return model

    def instantiate_proxy(self) -> nn.Module:
        logger.debug("Instantiating proxy model")
        try:
            proxy = instantiate(self.cfg.proxy)
        except ConfigAttributeError as e:
            msg = f"'proxy' attribute is missing in the configuration file: {e}"
            logger.exception(msg)
            raise e

        proxy = proxy.get("model", None)
        if proxy is None:
            msg = (
                "'proxy' key is missing in the proxy configuration file. "
                "Please provide a 'proxy' configuration."
            )
            logger.error(msg)
            raise ConfigAttributeError(msg)

        return proxy

    def instantiate_loss(self) -> nn.Module:
        logger.debug("Instantiating loss function")
        try:
            loss = instantiate(self.cfg.loss)
        except ConfigAttributeError as e:
            msg = f"'loss' attribute is missing in the configuration file: {e}"
            logger.exception(msg)
            raise e

        loss = loss.get("loss", None)
        if loss is None:
            msg = (
                "'loss' key is missing in the loss configuration file. "
                "Please provide a 'loss' configuration."
            )
            logger.error(msg)
            raise ConfigAttributeError(msg)

        return loss

    def instantiate_optimizer(
        self, model: nn.Module | lightning.LightningModule
    ) -> torch.optim.Optimizer:
        logger.debug("Instantiating optimizer")
        try:
            optimizer = instantiate(self.cfg.optimizer)

        except ConfigAttributeError as e:
            msg = f"'optimizer' attribute is missing in the configuration file: {e}"
            logger.exception(msg)
            raise e

        optimizer = optimizer.get("optimizer", None)
        if optimizer is None:
            msg = (
                "'optimizer' key is missing in the optimizer configuration file. "
                "Please provide an 'optimizer' configuration."
            )
            logger.error(msg)
            raise ConfigAttributeError(msg)

        return optimizer(model.parameters())

    def instantiate_proxy_optimizer(
        self, proxy: nn.Module | lightning.LightningModule
    ) -> torch.optim.Optimizer:
        logger.debug("Instantiating proxy optimizer")
        try:
            proxy_optimizer = instantiate(self.cfg.proxy_optimizer)

        except ConfigAttributeError as e:
            msg = (
                f"'proxy.optimizer' attribute is missing in the configuration file: {e}"
            )
            logger.exception(msg)
            raise e

        proxy_optimizer = proxy_optimizer.get("optimizer", None)
        if proxy_optimizer is None:
            msg = (
                "'optimizer' key is missing in the proxy optimizer configuration file. "
                "Please provide an 'optimizer' configuration."
            )
            logger.error(msg)
            raise ConfigAttributeError(msg)

        return proxy_optimizer(proxy.parameters())

    def instantiate_uncertainty_score(self):
        logger.debug("Instantiating uncertainty score")
        try:
            uncertainty_score = instantiate(self.cfg.uncertainty_score)
        except ConfigAttributeError as e:
            msg = (
                "'uncertainty_score' attribute is missing in the configuration "
                f"file: {e}"
            )
            logger.exception(msg)
            raise e

        uncertainty_score = uncertainty_score.get("uncertainty_score", None)
        if uncertainty_score is None:
            msg = (
                "'uncertainty_score' key is missing in the uncertainty score "
                "configuration file. Please provide an 'uncertainty_score' "
                "configuration."
            )
            logger.error(msg)
            raise ConfigAttributeError(msg)

        return uncertainty_score

    def instantiate_scheduler(
        self, optimizer: torch.optim.Optimizer
    ) -> torch.optim.lr_scheduler.LRScheduler:
        logger.debug("Instantiating scheduler")
        try:
            scheduler = instantiate(self.cfg.scheduler)
        except ConfigAttributeError:
            logger.warning(
                "No scheduler defined in the configuration file. "
                "Consider setting one via the `scheduler` key in the config "
                "file. You can ignore this warning if you don't want to use "
                "a scheduler."
            )
            return

        scheduler = scheduler.get("scheduler", None)
        if scheduler is None:
            msg = (
                "'scheduler' key is missing in the scheduler configuration file. "
                "Please provide a 'scheduler' configuration."
            )
            logger.error(msg)
            raise ConfigAttributeError(msg)

        return scheduler(optimizer=optimizer)

    def instantiate_trainer(
        self, callbacks: list[lightning.Callback]
    ) -> lightning.Trainer:
        logger.debug("Instantiating trainer")

        try:
            trainer_cfg = OmegaConf.to_container(self.cfg.trainer, resolve=True)

            trainer_cfg = instantiate(trainer_cfg)
        except ConfigAttributeError as e:
            msg = f"'trainer' attribute is missing in the configuration file: {e}"
            logger.exception(msg)
            raise e

        try:
            trainer = trainer_cfg.trainer
        except ConfigAttributeError as e:
            msg = (
                "'trainer' key is missing in the trainer configuration file. "
                "Please provide a 'trainer' configuration."
            )
            logger.exception(msg)
            raise e

        try:
            loggers = trainer_cfg.get("logger", [])
            logger_list = [logger for _, logger in loggers.items()]
        except ConfigAttributeError:
            logger_list = []

        return trainer(logger=logger_list, callbacks=callbacks)

    def instantiate_callbacks(self) -> dict[str, lightning.Callback]:
        logger.debug("Instantiating callbacks")

        try:
            callbacks = instantiate(self.cfg.callbacks)
            return callbacks
        except ConfigAttributeError:
            return []

    def instantiate_attacks(
        self,
    ) -> dict[str, nn.Module]:
        logger.debug("Instantiating attacks")
        try:
            attacks = instantiate(self.cfg.attacks)
            for attack in attacks.values():
                # Check if the attack has a loss function attribute
                if hasattr(attack, "loss_fn") and isinstance(
                    attack.loss_fn, DictConfig
                ):
                    attack.loss_fn = attack.loss_fn.get("loss", None)
        except ConfigAttributeError:
            return []

        if attacks is None:
            msg = (
                "'attacks' key is missing in the attacks configuration file. "
                "Please provide an 'attacks' configuration."
            )
            logger.error(msg)
            raise ConfigAttributeError(msg)
        return attacks


class CallbacksHandler:
    """
    Handler for callbacks defined in the hydra configuration files.

    This class provides a class method to retrieve a specific callback by name.
    Handled callbacks include:
    - model_checkpoint
    - early_stopping
    """

    HANDLED_CALLBACKS = {
        "model_checkpoint",
        "early_stopping",
    }

    @classmethod
    def get_callback(
        cls, callbacks: DictConfig, callback_name: str
    ) -> nn.Module | None:
        """
        Get a specific callback by name.
        """
        if callback_name not in cls.HANDLED_CALLBACKS:
            msg = f"Callback '{callback_name}' is not handled."
            logger.error(msg)
            raise ValueError(msg)

        try:
            return callbacks[callback_name]
        except KeyError:
            msg = f"Callback '{callback_name}' is not defined in the configuration."
            logger.warning(msg)
            return None


def config_parsing(cfg: DictConfig) -> tuple[DictConfig, DictConfig]:
    """
    Parses the training configuration and returns the experiment configuration
    and the instantiated configuration.

    Args:
        cfg (DictConfig): The Hydra configuration object.

    Returns:
        tuple: A tuple containing the experiment configuration and the instantiated
            configuration.
    """
    OmegaConf.register_new_resolver("eval", eval)
    experiment = cfg.get("experiment", None)
    if experiment is None:
        msg = (
            "Experiment configuration is missing. Please provide 'experiment' "
            "in the config."
        )
        logger.error(msg)
        raise RuntimeError(msg)

    try:
        ExperimentConfig(**experiment)
    except Exception as e:
        msg = f"Validation problem with the experiment configuration: {e}"
        logger.exception(msg)
        raise e

    try:
        experiment_cfg = OmegaConf.to_container(experiment, resolve=True)
    except Exception as e:
        msg = f"Failed to parse experiment configuration: {e}"
        logger.exception(msg)
        raise e

    environment_loader.load_environment(experiment_cfg["env_file"])  # type: ignore
    logger.info(f"Experiment configuration: {experiment_cfg}")

    instantiator = Instantiator(cfg)
    cfg = instantiator.get_instatiated_cfg()
    return experiment_cfg, cfg
