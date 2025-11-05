"""Run the FaceNet FastAPI inference service."""

from __future__ import annotations

import argparse
from pathlib import Path

import uvicorn

from facenet.config import InferenceConfig, load_config
from facenet.inference import EmbeddingService, create_app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve FaceNet inference API.")
    parser.add_argument("--config", type=Path, default=Path("configs/inference.yaml"))
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg: InferenceConfig = load_config(args.config, config_type=InferenceConfig)
    service = EmbeddingService(cfg)
    app = create_app(service)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
