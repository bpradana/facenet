"""Evaluate a trained FaceNet model on the validation split."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from facenet.config import (
    ModelConfig,
    OptimizerConfig,
    SchedulerConfig,
    TrainingConfig,
    load_config,
)
from facenet.data import FaceDataModule
from facenet.evaluation import compute_verification_metrics
from facenet.models import FaceNetLightningModule


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate FaceNet checkpoint.")
    parser.add_argument("--config", type=Path, default=Path("configs/train.yaml"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg: TrainingConfig = load_config(args.config, config_type=TrainingConfig)

    data_module = FaceDataModule(cfg.data, seed=cfg.seed)
    data_module.setup("validate")

    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    hyper_params = checkpoint["hyper_parameters"]
    model_cfg = ModelConfig(**hyper_params["model"])
    optimizer_cfg = OptimizerConfig(**hyper_params["optimizer"])
    scheduler_data = hyper_params.get("scheduler")
    scheduler_cfg = SchedulerConfig(**scheduler_data) if scheduler_data else None

    model = FaceNetLightningModule(
        model_cfg,
        optimizer_cfg,
        scheduler_cfg,
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    model.to(args.device)

    val_loader = data_module.val_dataloader()

    anchor_embeddings = []
    positive_embeddings = []
    negative_embeddings = []

    with torch.inference_mode():
        for batch in val_loader:
            anchor = batch["anchor"].to(args.device)
            positive = batch["positive"].to(args.device)
            negative = batch["negative"].to(args.device)

            anchor_emb = model(anchor)
            positive_emb = model(positive)
            negative_emb = model(negative)

            anchor_embeddings.append(anchor_emb.cpu())
            positive_embeddings.append(positive_emb.cpu())
            negative_embeddings.append(negative_emb.cpu())

    anchors = torch.cat(anchor_embeddings)
    positives = torch.cat(positive_embeddings)
    negatives = torch.cat(negative_embeddings)

    same_labels = torch.ones(anchors.size(0))
    diff_labels = torch.zeros(anchors.size(0))

    pairs_a = torch.cat([anchors, anchors])
    pairs_b = torch.cat([positives, negatives])
    labels = torch.cat([same_labels, diff_labels])

    metrics = compute_verification_metrics(pairs_a, pairs_b, labels)

    print("Verification metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")


if __name__ == "__main__":
    main()
