import os
from pathlib import Path

import hydra
import lightning
import torch.nn as nn
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from ev_at.core.logging import add_handler, get_logger
from ev_at.core.pipe.utils import CallbacksHandler, config_parsing
from ev_at.data.loader import CIFAR10Datamodule, CIFAR100Datamodule
from ev_at.modules import BaseModule, EMFFModule, EvidentialModule
from ev_at.pipe import (
    ATTrainingModule,
    EMFFTradesTrainingModule,
    EVATTrainingModule,
    IKLTrainingModule,
    StandardTrainingModule,
    TradesTrainingModule,
)

logger = get_logger("training-pipe")


@hydra.main(config_path="../configs", config_name="train_config", version_base=None)
def run_train_pipe(cfg: DictConfig) -> None:
    """
    Main function to run the training pipe using Hydra configuration.

    Args:
        config: Hydra configuration object containing all necessary parameters.
    """
    add_handler(
        sink=Path(HydraConfig.get().runtime.output_dir)
        / f"{HydraConfig.get().job.name}.log"
    )
    seed = cfg["experiment"]["seed"]
    if seed is None:
        msg = (
            "seed is not set in configuration file. "
            "Please check the seed in the configuration file."
        )
        logger.error(msg)
        raise RuntimeError(msg)
    logger.debug(f"Reset random state with seed {seed}")
    lightning.seed_everything(seed)

    exp_cfg, cfg = config_parsing(cfg)

    logger.info(f"Output directories set to: {cfg['output_dir']}")

    data_path = Path(os.environ.get("DATA_PATH", "./data"))
    if not data_path.exists():
        msg = (
            f"Data path {data_path} does not exist. Please check the DATA_PATH "
            "environment variable."
        )
        logger.error(msg)
        raise FileNotFoundError(msg)
    logger.info(f"{data_path} loaded successfully")

    if exp_cfg.get("dataset", "cifar10") == "cifar10":
        datamodule = CIFAR10Datamodule(
            data_path,
            batch_size=cfg["dataloaders"]["batch_size"],
            seed=seed,
            num_workers=cfg["dataloaders"]["num_workers"],
            aug=cfg["dataloaders"]["aug"],
        )
    elif exp_cfg["dataset"] == "cifar100":
        datamodule = CIFAR100Datamodule(
            data_path,
            batch_size=cfg["dataloaders"]["batch_size"],
            seed=seed,
            num_workers=cfg["dataloaders"]["num_workers"],
            aug=cfg["dataloaders"]["aug"],
        )

    match exp_cfg["model_type"]:
        case "base":
            module = BaseModule(
                model=cfg["model"],
                loss_fn=cfg["loss"],
                uncertainty_score=cfg["uncertainty_score"],
            )
            proxy = BaseModule(
                model=cfg["proxy"],
                loss_fn=cfg["loss"],
                uncertainty_score=cfg["uncertainty_score"],
            )
        case "evidential":
            module = EvidentialModule(
                model=cfg["model"],
                loss_fn=cfg["loss"],
                activation=exp_cfg["activation"],
                uncertainty_score=cfg["uncertainty_score"],
            )
            proxy = EvidentialModule(
                model=cfg["proxy"],
                loss_fn=cfg["loss"],
                activation=exp_cfg["activation"],
                uncertainty_score=cfg["uncertainty_score"],
            )
        case "emff":
            module = EMFFModule(
                model=cfg["model"],
                loss_fn=cfg["loss"],
                uncertainty_score=cfg["uncertainty_score"],
            )

    match exp_cfg["training_pipe"]:
        case "standard":
            training_module = StandardTrainingModule(
                model=module,
                optimizer=cfg["optimizer"],
                scheduler=cfg["scheduler"],
                num_classes=datamodule.num_classes(),
            )
        case "ikl":
            training_module = IKLTrainingModule(
                test_attack=cfg["attacks"][0],
                model=module,
                proxy=proxy,
                optimizer=cfg["optimizer"],
                proxy_optimizer=cfg["proxy_optimizer"],
                scheduler=cfg["scheduler"],
                beta=exp_cfg["beta"],
                epsilon=exp_cfg["epsilon"],
                num_steps=exp_cfg["num_steps"],
                step_size=exp_cfg["step_size"],
                alpha=exp_cfg["alpha"],
                gamma=exp_cfg["gamma"],
                awp_warmup=exp_cfg["awp_warmup"],
                awp_gamma=exp_cfg["awp_gamma"],
                T=exp_cfg["T"],
                train_budget=exp_cfg["train_budget"],
                num_classes=datamodule.num_classes(),
                max_epochs=exp_cfg["max_epochs"],
                norm=exp_cfg["norm"],
            )
        case "trades":
            training_module = TradesTrainingModule(
                test_attack=cfg["attacks"][0],
                model=module,
                proxy=proxy,
                optimizer=cfg["optimizer"],
                proxy_optimizer=cfg["proxy_optimizer"],
                scheduler=cfg["scheduler"],
                beta=exp_cfg["beta"],
                epsilon=exp_cfg["epsilon"],
                num_steps=exp_cfg["num_steps"],
                step_size=exp_cfg["step_size"],
                awp_warmup=exp_cfg["awp_warmup"],
                awp_gamma=exp_cfg["awp_gamma"],
                num_classes=datamodule.num_classes(),
                norm=exp_cfg["norm"],
            )
        case "at":
            training_module = ATTrainingModule(
                test_attack=cfg["attacks"][0],
                model=module,
                proxy=proxy,
                optimizer=cfg["optimizer"],
                proxy_optimizer=cfg["proxy_optimizer"],
                scheduler=cfg["scheduler"],
                epsilon=exp_cfg["epsilon"],
                num_steps=exp_cfg["num_steps"],
                step_size=exp_cfg["step_size"],
                awp_warmup=exp_cfg["awp_warmup"],
                awp_gamma=exp_cfg["awp_gamma"],
                num_classes=datamodule.num_classes(),
                norm=exp_cfg["norm"],
            )
        case "trades-emff":
            training_module = EMFFTradesTrainingModule(
                test_attack=cfg["attacks"][0],
                model=module,
                optimizer=cfg["optimizer"],
                scheduler=cfg["scheduler"],
                beta=exp_cfg["beta"],
                epsilon=exp_cfg["epsilon"],
                num_steps=exp_cfg["num_steps"],
                step_size=exp_cfg["step_size"],
                num_classes=datamodule.num_classes(),
                norm=exp_cfg["norm"],
            )
        case "ev-at":
            training_module = EVATTrainingModule(
                test_attack=cfg["attacks"][0],
                model=module,
                proxy=proxy,
                optimizer=cfg["optimizer"],
                proxy_optimizer=cfg["proxy_optimizer"],
                scheduler=cfg["scheduler"],
                beta=exp_cfg["beta"],
                epsilon=exp_cfg["epsilon"],
                num_steps=exp_cfg["num_steps"],
                step_size=exp_cfg["step_size"],
                alpha=exp_cfg["alpha"],
                gamma=exp_cfg["gamma"],
                awp_warmup=exp_cfg["awp_warmup"],
                awp_gamma=exp_cfg["awp_gamma"],
                T=exp_cfg["T"],
                train_budget=exp_cfg["train_budget"],
                num_classes=datamodule.num_classes(),
                max_epochs=exp_cfg["max_epochs"],
                norm=exp_cfg["norm"],
            )

    trainer: lightning.Trainer = cfg["trainer"]

    if exp_cfg.get("resume_from_checkpoint", False):
        save_dir = Path(trainer.default_root_dir)

        try:
            all_ckpts = list(save_dir.rglob("*.ckpt"))
            all_ckpts.sort()
        except Exception as e:
            logger.exception(f"Error accessing checkpoints in {save_dir}: {e}")
            raise RuntimeError from e

        if not all_ckpts:
            msg = f"No checkpoints found in {save_dir}."
            msg += " Please check the training process."
            logger.error(msg)
            raise FileNotFoundError(msg)

        last_model_path = all_ckpts[0]

        logger.info(f"Resuming training from checkpoint: {last_model_path}")
        trainer.fit(training_module, datamodule=datamodule, ckpt_path=last_model_path)
        logger.success("Training completed.")

    else:
        logger.info("Starting training...")
        trainer.fit(training_module, datamodule=datamodule)
        logger.success("Training completed.")

    ckpt_cb: nn.Module | None = CallbacksHandler.get_callback(
        cfg["callbacks"], "model_checkpoint"
    )
    if ckpt_cb is not None:
        best_model_path = ckpt_cb.best_model_path
        logger.info(f"Best model saved at: {best_model_path}")
    else:
        logger.warning("No model checkpoint callback found")


if __name__ == "__main__":
    run_train_pipe()
