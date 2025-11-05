"""Export a trained FaceNet model to TorchScript for production inference."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from facenet.config import ModelConfig, OptimizerConfig, SchedulerConfig
from facenet.models.lightning_module import FaceNetLightningModule


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export FaceNet checkpoint to TorchScript."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path, default=Path("facenet_embedding_model.ts")
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--image-size", type=int, default=160)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    hyper_params = checkpoint["hyper_parameters"]

    model_cfg = ModelConfig(**hyper_params["model"])
    optimizer_cfg = OptimizerConfig(**hyper_params["optimizer"])
    scheduler_data = hyper_params.get("scheduler")
    scheduler_cfg = SchedulerConfig(**scheduler_data) if scheduler_data else None

    module = FaceNetLightningModule(model_cfg, optimizer_cfg, scheduler_cfg)
    module.load_state_dict(checkpoint["state_dict"])
    module.eval()
    module.to(args.device)

    example_input = torch.randn(
        1, 3, args.image_size, args.image_size, device=args.device
    )
    scripted = torch.jit.trace(module.model, example_input)
    scripted.save(str(args.output))

    print(f"Exported TorchScript model to {args.output}")


if __name__ == "__main__":
    main()
