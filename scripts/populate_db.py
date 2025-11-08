"""Populate the Postgres embedding store from a dataset directory."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np
from PIL import Image

from facenet.config import InferenceConfig, load_config
from facenet.db import PostgresConfig, PostgresEmbeddingStore
from facenet.inference import EmbeddingService


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Populate the embedding database from a dataset directory."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/inference.yaml"),
        help="Path to inference YAML configuration.",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Path to dataset (folders per identity).",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle identities before insertion.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print what would be inserted without touching the database.",
    )
    return parser.parse_args()


def discover_identities(dataset_path: Path) -> List[tuple[str, List[Path]]]:
    identities: List[tuple[str, List[Path]]] = []
    for identity_dir in sorted(p for p in dataset_path.iterdir() if p.is_dir()):
        images = [
            image_path
            for image_path in identity_dir.glob("*")
            if image_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
        ]
        if images:
            identities.append((identity_dir.name.replace("_", " "), images))
    return identities


def build_store(cfg: InferenceConfig, embedding_dim: int) -> PostgresEmbeddingStore:
    db_cfg = cfg.database or {}
    db_host = db_cfg.get("host")
    db_port = int(db_cfg.get("port") or 5432)
    db_name = db_cfg.get("name")
    db_user = db_cfg.get("user")
    db_password = db_cfg.get("password")
    db_table = db_cfg.get("table", "face_embeddings")
    db_identity_table = db_cfg.get("identity_table")
    db_sslmode = db_cfg.get("sslmode")

    if not (db_host and db_name and db_user):
        raise RuntimeError(
            "Database configuration is missing. "
            "Ensure configs/inference.yaml includes the `database` block."
        )

    pg_config = PostgresConfig(
        host=db_host,
        port=db_port,
        database=db_name,
        user=db_user,
        password=db_password or "",
        sslmode=db_sslmode,
    )
    return PostgresEmbeddingStore(
        pg_config,
        db_table,
        embedding_dim=embedding_dim,
        identities_table=db_identity_table,
    )


def load_image(path: Path) -> Image.Image:
    with Image.open(path) as img:
        return img.convert("RGB")


def main() -> None:
    args = parse_args()
    cfg: InferenceConfig = load_config(args.config, config_type=InferenceConfig)
    dataset_path = args.dataset
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_path}")

    service = EmbeddingService(cfg)
    embedding_dim = getattr(service, "embedding_dim", None)
    if embedding_dim is None:
        raise RuntimeError("Unable to infer embedding dimension from model.")

    identities = discover_identities(dataset_path)
    if args.shuffle:
        rng = np.random.default_rng()
        rng.shuffle(identities)

    print(f"Found {len(identities)} identities in {dataset_path}")

    samples: List[tuple[str, List[Path]]] = []
    for identity, image_paths in identities:
        if not image_paths:
            continue
        samples.append((identity, image_paths))

    if args.dry_run:
        for identity, image_paths in samples:
            print(f"[DRY RUN] {identity}: {[str(p) for p in image_paths]}")
        return

    store = build_store(cfg, int(embedding_dim))
    total_inserted = 0

    for identity, image_paths in samples:
        faces = [load_image(path) for path in image_paths]
        embeddings = service.encode_faces(faces)
        inserted = store.register(identity, embeddings)
        total_inserted += inserted
        print(f"Inserted {inserted} embedding(s) for '{identity}'")

    print(f"Finished populating database with {total_inserted} embedding(s).")


if __name__ == "__main__":
    main()
