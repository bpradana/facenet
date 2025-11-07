"""Run live face detection and identification from webcam feed."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

from facenet.config import InferenceConfig, load_config
from facenet.db import PostgresConfig, PostgresEmbeddingStore
from facenet.inference import EmbeddingService

try:
    from ultralytics import YOLO
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "Ultralytics is required for live detection. Install with `pip install ultralytics`."
    ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Live face detection and search.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/inference.yaml"),
        help="Path to inference YAML configuration.",
    )
    parser.add_argument(
        "--camera", type=int, default=0, help="Webcam device index (default: 0)."
    )
    parser.add_argument(
        "--top-k", type=int, default=3, help="Number of nearest neighbors to retrieve."
    )
    parser.add_argument(
        "--min-similarity",
        type=float,
        default=0.75,
        help="Minimum cosine similarity required to label a match.",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=None,
        help="Override detector confidence threshold (uses config by default).",
    )
    parser.add_argument(
        "--show-similarity",
        action="store_true",
        help="Append similarity score to rendered labels.",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=1,
        help="Number of consecutive frames to aggregate per identity (default: 1).",
    )
    return parser.parse_args()


def build_store(cfg: InferenceConfig, embedding_dim: int) -> PostgresEmbeddingStore:
    db_cfg = cfg.database or {}
    db_host = db_cfg.get("host")
    db_port = int(db_cfg.get("port") or 5432)
    db_name = db_cfg.get("name")
    db_user = db_cfg.get("user")
    db_password = db_cfg.get("password")
    db_table = db_cfg.get("table", "face_embeddings")
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
    return PostgresEmbeddingStore(pg_config, db_table, embedding_dim=embedding_dim)


def load_detector(weights_path: Path) -> YOLO:
    if not weights_path.exists():
        raise FileNotFoundError(
            f"YOLO weights not found at {weights_path}. Update detector_weights in config."
        )
    return YOLO(str(weights_path))


def to_square_bbox(
    x1: float, y1: float, x2: float, y2: float, width: int, height: int
) -> Tuple[int, int, int, int]:
    box_w = x2 - x1
    box_h = y2 - y1
    side = min(max(box_w, box_h), float(width), float(height))
    half = side / 2.0
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0

    new_x1 = cx - half
    new_y1 = cy - half
    new_x2 = cx + half
    new_y2 = cy + half

    if new_x1 < 0:
        new_x2 -= new_x1
        new_x1 = 0.0
    if new_y1 < 0:
        new_y2 -= new_y1
        new_y1 = 0.0
    if new_x2 > width:
        shift = new_x2 - width
        new_x1 -= shift
        new_x2 = float(width)
    if new_y2 > height:
        shift = new_y2 - height
        new_y1 -= shift
        new_y2 = float(height)

    new_x1 = max(0.0, new_x1)
    new_y1 = max(0.0, new_y1)
    new_x2 = min(float(width), new_x2)
    new_y2 = min(float(height), new_y2)

    side = min(new_x2 - new_x1, new_y2 - new_y1)
    new_x2 = new_x1 + side
    new_y2 = new_y1 + side

    x1_i = int(np.floor(new_x1))
    y1_i = int(np.floor(new_y1))
    x2_i = int(np.ceil(new_x2))
    y2_i = int(np.ceil(new_y2))

    if x2_i <= x1_i:
        x2_i = min(width, x1_i + 1)
    if y2_i <= y1_i:
        y2_i = min(height, y1_i + 1)

    return x1_i, y1_i, x2_i, y2_i


def main() -> None:
    args = parse_args()
    cfg: InferenceConfig = load_config(args.config, config_type=InferenceConfig)
    service = EmbeddingService(cfg)
    embedding_dim = getattr(service, "embedding_dim", None)
    if embedding_dim is None:
        raise RuntimeError("Unable to infer embedding dimension from model.")

    store = build_store(cfg, int(embedding_dim))
    detector_weights = cfg.detector_weights
    if detector_weights is None:
        raise RuntimeError(
            "detector_weights not configured. YOLO weights are required for live detection."
        )
    detector = load_detector(Path(detector_weights))
    confidence = args.confidence or cfg.detector_confidence

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open camera index {args.camera}.")

    max_frames = max(1, args.frames)
    similarity_buffer: dict[str, List[float]] = {}

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_height, frame_width = frame.shape[:2]
            results = detector.predict(frame, conf=confidence, verbose=False)

            faces: List[Image.Image] = []
            boxes: List[Tuple[int, int, int, int]] = []
            for result in results:
                if result.boxes is None or result.boxes.xyxy is None:
                    continue
                xyxy = result.boxes.xyxy.cpu().numpy()
                confs = (
                    result.boxes.conf.cpu().numpy()
                    if result.boxes.conf is not None
                    else np.ones(xyxy.shape[0])
                )
                for bbox, conf in zip(xyxy, confs):
                    if conf < confidence:
                        continue
                    x1, y1, x2, y2 = bbox.tolist()
                    sx1, sy1, sx2, sy2 = to_square_bbox(
                        x1, y1, x2, y2, frame_width, frame_height
                    )
                    boxes.append((sx1, sy1, sx2, sy2))
                    crop = Image.fromarray(
                        cv2.cvtColor(frame[sy1:sy2, sx1:sx2].copy(), cv2.COLOR_BGR2RGB)
                    )
                    faces.append(crop)

            labels: List[Optional[str]] = [None] * len(boxes)

            if faces:
                embeddings = service.encode_faces(faces)
                current_labels: List[Tuple[str, float]] = []
                for idx, embedding in enumerate(embeddings):
                    matches = store.search(embedding, top_k=args.top_k)
                    if not matches:
                        labels[idx] = "Unknown"
                        continue
                    best = matches[0]
                    if best.similarity < args.min_similarity:
                        labels[idx] = "Unknown"
                        continue

                    labels[idx] = best.identity
                    current_labels.append((best.identity, best.similarity))

                for identity, similarity in current_labels:
                    buf = similarity_buffer.setdefault(identity, [])
                    buf.append(similarity)
                    if len(buf) > max_frames:
                        buf.pop(0)
                # prune old identities not seen in current frame
                tracked = {identity for identity, _ in current_labels}
                for identity in list(similarity_buffer.keys()):
                    if identity not in tracked:
                        buf = similarity_buffer[identity]
                        if len(buf) >= max_frames:
                            similarity_buffer.pop(identity)

                # update labels with averaged similarity
                for idx, identity in enumerate(labels):
                    if identity in similarity_buffer and identity != "Unknown":
                        scores = similarity_buffer[identity]
                        avg_sim = sum(scores) / len(scores)
                        if avg_sim < args.min_similarity:
                            labels[idx] = "Unknown"
                        elif args.show_similarity:
                            labels[idx] = f"{identity} ({avg_sim:.2f})"

            for (sx1, sy1, sx2, sy2), label in zip(boxes, labels):
                color = (0, 255, 0) if label and label != "Unknown" else (0, 0, 255)
                cv2.rectangle(frame, (sx1, sy1), (sx2, sy2), color, 2)
                if label:
                    cv2.putText(
                        frame,
                        label,
                        (sx1, max(20, sy1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        color,
                        2,
                        cv2.LINE_AA,
                    )

            cv2.imshow("FaceNet Live Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
