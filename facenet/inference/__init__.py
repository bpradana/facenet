"""Inference utilities for serving FaceNet embeddings."""

from .service import EmbeddingService, create_app

__all__ = ["EmbeddingService", "create_app"]
