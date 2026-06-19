import os
from pathlib import Path

import hydra
import lightning
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from ev_at.core.logging import add_handler, get_logger
from ev_at.core.pipe.utils import config_parsing
from ev_at.data.loader import (
    CIFAR10CorruptedDatamodule,
    CIFAR10Datamodule,
    CIFAR100CorruptedDatamodule,
    CIFAR100Datamodule,
)
from ev_at.modules import BaseModule, EMFFModule, EvidentialModule
from ev_at.pipe import CIFARClassificationEvaluationModule

logger = get_logger("eval")


@hydra.main(config_path="../configs", config_name="eval_config", version_base=None)
def run_eval_pipe(cfg: DictConfig):
    """
    Main function to run the evaluation pipelines using Hydra configuration.

    Args:
        config: Hydra configuration object containing all necessary parameters.
    """
    add_handler(
        sink=Path(HydraConfig.get().runtime.output_dir)
        / f"{HydraConfig.get().job.name}.log"
    )
    seed = cfg["experiment"]["seed"]
    if seed is None:
        msg = "Seed is not set in the configuration"
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

    if exp_cfg.get("dataset", "cifar10") == "cifar10":
        datamodule = CIFAR10Datamodule(
            data_path,
            batch_size=cfg["dataloaders"]["batch_size"],
            num_workers=cfg["dataloaders"]["num_workers"],
        )
    elif exp_cfg["dataset"] == "cifar10-c":
        datamodule = CIFAR10CorruptedDatamodule(
            data_path,
            batch_size=cfg["dataloaders"]["batch_size"],
            num_workers=cfg["dataloaders"]["num_workers"],
        )
    elif exp_cfg["dataset"] == "cifar100":
        datamodule = CIFAR100Datamodule(
            data_path,
            batch_size=cfg["dataloaders"]["batch_size"],
            num_workers=cfg["dataloaders"]["num_workers"],
        )
    elif exp_cfg["dataset"] == "cifar100-c":
        datamodule = CIFAR100CorruptedDatamodule(
            data_path,
            batch_size=cfg["dataloaders"]["batch_size"],
            num_workers=cfg["dataloaders"]["num_workers"],
        )

    trainer: lightning.Trainer = cfg["trainer"]
    save_dir = Path(trainer.default_root_dir)

    try:
        all_ckpts = list(save_dir.rglob("*.ckpt"))
        all_ckpts.sort()
    except Exception as e:
        logger.exception(f"Error accessing checkpoints in {save_dir}: {e}")
        raise RuntimeError from e

    if not all_ckpts:
        msg = f"No checkpoints found in {save_dir}. Please check the training process."
        logger.error(msg)
        raise FileNotFoundError(msg)

    best_model_path = all_ckpts[0]

    if cfg.get("official_ckpt", False):
        model = cfg["model"]
        checkpoint = torch.load(best_model_path, map_location="cpu")
        model.load_state_dict(checkpoint)

    match exp_cfg["model_type"]:
        case "base":
            module_best = BaseModule(
                model=model if cfg.get("official_ckpt", False) else cfg["model"],
                loss_fn=cfg["loss"],
                uncertainty_score=cfg["uncertainty_score"],
            )
        case "evidential":
            module_best = EvidentialModule(
                model=model if cfg.get("official_ckpt", False) else cfg["model"],
                loss_fn=cfg["loss"],
                activation=exp_cfg["activation"],
                uncertainty_score=cfg["uncertainty_score"],
            )
        case "emff":
            module_best = EMFFModule(
                model=model if cfg.get("official_ckpt", False) else cfg["model"],
                loss_fn=cfg["loss"],
                uncertainty_score=cfg["uncertainty_score"],
            )
    if cfg.get("official_ckpt", False):
        eval_module = CIFARClassificationEvaluationModule(
            attacks=cfg["attacks"],
            model=module_best,
            num_classes=datamodule.num_classes(),
        )
    else:
        match cfg["ckpt_choice"]:
            case "best":
                ckpt_path = best_model_path
        logger.info(f"Using checkpoint: {ckpt_path}")
        try:
            eval_module = CIFARClassificationEvaluationModule.load_from_checkpoint(
                checkpoint_path=ckpt_path,
                attacks=cfg["attacks"],
                model=module_best,
                num_classes=datamodule.num_classes(),
            )
        except Exception as e:
            logger.exception(f"Error loading model from checkpoint {ckpt_path}: {e}")
            raise RuntimeError from e

    logger.info("Starting testing...")
    trainer.test(eval_module, datamodule=datamodule)
    logger.info("Testing completed.")


if __name__ == "__main__":
    run_eval_pipe()
