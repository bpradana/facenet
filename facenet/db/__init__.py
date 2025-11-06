"""Database backends for FaceNet."""

from .postgres_store import PostgresConfig, PostgresEmbeddingStore

__all__ = ["PostgresConfig", "PostgresEmbeddingStore"]
