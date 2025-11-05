"""Lightning module orchestrating FaceNet training."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, Optional

import lightning as L
import torch
import torch.nn.functional as F
from torch import optim
from torchmetrics import MeanMetric

from ..config import ModelConfig, OptimizerConfig, SchedulerConfig
from ..losses import TripletLoss
from .backbone import build_backbone


class FaceNetLightningModule(L.LightningModule):
    """PyTorch LightningModule encapsulating training, validation, and optimization logic."""

    def __init__(
        self,
        model_cfg: ModelConfig,
        optimizer_cfg: OptimizerConfig,
        scheduler_cfg: Optional[SchedulerConfig] = None,
    ) -> None:
        super().__init__()
        self.model_cfg = model_cfg
        self.optimizer_cfg = optimizer_cfg
        self.scheduler_cfg = scheduler_cfg

        self.model = build_backbone(
            backbone_name=model_cfg.backbone,
            embedding_dim=model_cfg.embedding_dim,
            pretrained=model_cfg.pretrained,
            dropout=model_cfg.dropout,
            train_backbone=model_cfg.train_backbone,
        )
        self.loss_fn = TripletLoss(margin=model_cfg.margin)

        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.val_pos_distance = MeanMetric()
        self.val_neg_distance = MeanMetric()
        self.val_accuracy = MeanMetric()

        self.save_hyperparameters(
            {
                "model": asdict(model_cfg),
                "optimizer": asdict(optimizer_cfg),
                "scheduler": asdict(scheduler_cfg) if scheduler_cfg else None,
            }
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.model(images)

    def _triplet_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        anchor = batch["anchor"]
        positive = batch["positive"]
        negative = batch["negative"]

        anchor_emb = self(anchor)
        positive_emb = self(positive)
        negative_emb = self(negative)

        loss = self.loss_fn(anchor_emb, positive_emb, negative_emb)
        pos_distance = F.pairwise_distance(anchor_emb, positive_emb)
        neg_distance = F.pairwise_distance(anchor_emb, negative_emb)
        accuracy = (neg_distance > pos_distance).float().mean()

        return {
            "loss": loss,
            "pos_distance": pos_distance.mean(),
            "neg_distance": neg_distance.mean(),
            "accuracy": accuracy,
        }

    def training_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:  # type: ignore[override]
        metrics = self._compute_training_metrics(batch)
        self.train_loss.update(metrics["loss"])
        batch_size = metrics["batch_size"]

        self.log(
            "train/loss",
            metrics["loss"],
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            batch_size=batch_size,
        )
        self.log(
            "train/pos_distance",
            metrics["pos_distance"],
            prog_bar=False,
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
        )
        self.log(
            "train/neg_distance",
            metrics["neg_distance"],
            prog_bar=False,
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
        )
        return metrics["loss"]

    def on_train_epoch_end(self) -> None:
        self.train_loss.reset()

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:  # type: ignore[override]
        metrics = self._triplet_step(batch)
        self.val_loss.update(metrics["loss"])
        self.val_pos_distance.update(metrics["pos_distance"])
        self.val_neg_distance.update(metrics["neg_distance"])
        self.val_accuracy.update(metrics["accuracy"])
        batch_size = batch["anchor"].size(0)

        self.log(
            "val/loss",
            metrics["loss"],
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
        )
        self.log(
            "val/pos_distance",
            metrics["pos_distance"],
            prog_bar=False,
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
        )
        self.log(
            "val/neg_distance",
            metrics["neg_distance"],
            prog_bar=False,
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
        )
        self.log(
            "val/accuracy",
            metrics["accuracy"],
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
        )

    def on_validation_epoch_end(self) -> None:
        self.val_loss.reset()
        self.val_pos_distance.reset()
        self.val_neg_distance.reset()
        self.val_accuracy.reset()

    def _compute_training_metrics(
        self, batch: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        if "image" in batch:
            return self._batch_hard_step(batch)
        return {**self._triplet_step(batch), "batch_size": batch["anchor"].size(0)}

    def _batch_hard_step(
        self, batch: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        images = batch["image"]
        labels = batch["label"]
        embeddings = self(images)
        loss, pos_distance, neg_distance, accuracy = self._batch_hard_mining(
            embeddings, labels
        )
        return {
            "loss": loss,
            "pos_distance": pos_distance,
            "neg_distance": neg_distance,
            "accuracy": accuracy,
            "batch_size": images.size(0),
        }

    def _batch_hard_mining(
        self, embeddings: torch.Tensor, labels: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        dot_product = torch.matmul(embeddings, embeddings.T)
        squared_norm = embeddings.pow(2).sum(dim=1, keepdim=True)
        pairwise_dist = squared_norm - 2 * dot_product + squared_norm.T
        pairwise_dist = torch.clamp(pairwise_dist, min=1e-12).sqrt()
        labels = labels.view(-1, 1)
        mask_positive = torch.eq(labels, labels.T)
        # remove self comparisons
        diag_mask = torch.eye(
            mask_positive.size(0), dtype=torch.bool, device=mask_positive.device
        )
        mask_positive = mask_positive & ~diag_mask
        mask_negative = ~mask_positive & ~diag_mask

        if not mask_positive.any():
            raise RuntimeError(
                "Batch-hard mining requires at least two samples per class in a batch."
            )

        pos_dist = pairwise_dist.clone()
        pos_dist[~mask_positive] = float("-inf")
        hardest_pos, _ = pos_dist.max(dim=1)

        neg_dist = pairwise_dist.clone()
        neg_dist[~mask_negative] = float("inf")
        hardest_neg, _ = neg_dist.min(dim=1)

        margin = self.loss_fn.margin
        losses = torch.relu(hardest_pos - hardest_neg + margin)

        accuracy = (hardest_neg > hardest_pos).float().mean()
        return losses.mean(), hardest_pos.mean(), hardest_neg.mean(), accuracy

    def configure_optimizers(self) -> Dict[str, Any]:  # type: ignore[override]
        optimizer = self._build_optimizer()
        if self.scheduler_cfg and self.scheduler_cfg.name:
            scheduler = self._build_scheduler(optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",
                    "monitor": "val/loss",
                },
            }
        return {"optimizer": optimizer}

    def _build_optimizer(self) -> optim.Optimizer:
        cfg = self.optimizer_cfg
        if cfg.name == "adam":
            return optim.Adam(
                self.parameters(),
                lr=cfg.lr,
                weight_decay=cfg.weight_decay,
                betas=cfg.betas,
            )
        if cfg.name == "adamw":
            return optim.AdamW(
                self.parameters(),
                lr=cfg.lr,
                weight_decay=cfg.weight_decay,
                betas=cfg.betas,
            )
        if cfg.name == "sgd":
            return optim.SGD(
                self.parameters(),
                lr=cfg.lr,
                weight_decay=cfg.weight_decay,
                momentum=cfg.momentum,
                nesterov=True,
            )
        raise ValueError(f"Unsupported optimizer: {cfg.name}")

    def _build_scheduler(
        self, optimizer: optim.Optimizer
    ) -> optim.lr_scheduler.LRScheduler:
        assert self.scheduler_cfg is not None
        cfg = self.scheduler_cfg
        if cfg.name == "cosine":
            return optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max(1, cfg.t_max), eta_min=0.0
            )
        if cfg.name == "multistep":
            return optim.lr_scheduler.MultiStepLR(
                optimizer,
                milestones=list(cfg.milestones),
                gamma=cfg.gamma,
            )
        raise ValueError(f"Unsupported scheduler: {cfg.name}")
