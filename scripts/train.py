"""Train a FaceNet-style model using PyTorch Lightning."""

from __future__ import annotations

import argparse
from pathlib import Path

import lightning as L
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

from facenet.config import TrainingConfig, load_config
from facenet.data import FaceDataModule
from facenet.models import FaceNetLightningModule
from facenet.utils.logging import configure_logging
from facenet.utils.seed import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train FaceNet embedding network.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/train.yaml"),
        help="Path to training YAML configuration file.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Override dataset root directory from config.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg: TrainingConfig = load_config(args.config, config_type=TrainingConfig)

    if args.data_root:
        cfg.data.root = str(args.data_root)

    configure_logging(cfg.logging.output_dir)
    seed_everything(cfg.seed)

    data_module = FaceDataModule(cfg.data, seed=cfg.seed)
    model = FaceNetLightningModule(cfg.model, cfg.optimizer, cfg.scheduler)

    checkpoint_dir = Path(cfg.logging.output_dir) / "checkpoints"
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="facenet-{epoch:02d}-{val_loss:.4f}",
        monitor=cfg.logging.monitor,
        mode=cfg.logging.mode,
        save_top_k=cfg.logging.checkpoint_top_k,
        every_n_epochs=cfg.logging.checkpoint_interval,
        auto_insert_metric_name=False,
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")
    logger = CSVLogger(save_dir=cfg.logging.output_dir, name="training_logs")

    trainer = L.Trainer(
        max_epochs=cfg.trainer.max_epochs,
        devices=cfg.trainer.devices,
        accelerator=cfg.trainer.accelerator,
        precision=cfg.trainer.precision,
        gradient_clip_val=cfg.trainer.gradient_clip_val,
        accumulate_grad_batches=cfg.trainer.accumulate_grad_batches,
        limit_train_batches=cfg.trainer.limit_train_batches,
        limit_val_batches=cfg.trainer.limit_val_batches,
        deterministic=cfg.trainer.deterministic,
        enable_progress_bar=cfg.trainer.enable_progress_bar,
        log_every_n_steps=cfg.logging.log_every_n_steps,
        callbacks=[checkpoint_callback, lr_monitor],
        logger=logger,
    )

    trainer.fit(model, datamodule=data_module)


if __name__ == "__main__":
    main()
