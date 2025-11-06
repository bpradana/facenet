"""PostgreSQL-backed embedding storage using pgvector."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Iterable, Optional

import numpy as np
import psycopg
from pgvector.psycopg import register_vector
from psycopg.rows import dict_row
from psycopg.types.json import Json


@dataclass
class PostgresConfig:
    """Connection configuration for Postgres vector store."""

    host: str = "127.0.0.1"
    port: int = 5432
    database: str = "facenet"
    user: str = "postgres"
    password: str = "postgres"
    options: Optional[str] = None
    sslmode: Optional[str] = None

    def build_dsn(self) -> str:
        parts = [
            f"host={self.host}",
            f"port={self.port}",
            f"dbname={self.database}",
            f"user={self.user}",
        ]
        if self.password:
            parts.append(f"password={self.password}")
        if self.options:
            parts.append(f"options={self.options}")
        if self.sslmode:
            parts.append(f"sslmode={self.sslmode}")
        return " ".join(parts)


@dataclass
class SearchResult:
    id: int
    identity: str
    similarity: float
    distance: float
    metadata: Optional[dict[str, Any]]
    created_at: Any


class PostgresEmbeddingStore:
    """Embedding persistence layer backed by a Postgres database with pgvector."""

    def __init__(
        self,
        config: PostgresConfig,
        table_name: str,
        embedding_dim: int,
        *,
        lists: int = 100,
        distance_metric: str = "cosine",
    ) -> None:
        self.config = config
        self.dsn = config.build_dsn()
        self.table_name = table_name
        self.embedding_dim = embedding_dim
        self.lists = lists
        self.distance_metric = distance_metric
        self._setup_schema()

    def _setup_schema(self) -> None:
        with psycopg.connect(self.dsn, autocommit=True) as conn:
            register_vector(conn)
            with conn.cursor() as cur:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                cur.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS {self.table_name} (
                        id SERIAL PRIMARY KEY,
                        identity TEXT NOT NULL,
                        embedding vector({self.embedding_dim}) NOT NULL,
                        metadata JSONB,
                        created_at TIMESTAMPTZ DEFAULT NOW()
                    );
                    """
                )

    def register(
        self,
        identity: str,
        embeddings: Iterable[np.ndarray],
        *,
        metadata: Optional[dict[str, Any]] = None,
    ) -> int:
        payloads = []
        for embedding in embeddings:
            vector = np.asarray(embedding, dtype=np.float32)
            payloads.append(
                {
                    "identity": identity,
                    "embedding": vector,
                    "metadata": Json(metadata) if metadata is not None else None,
                }
            )

        if not payloads:
            return 0

        with psycopg.connect(self.dsn, autocommit=True) as conn:
            register_vector(conn)
            with conn.cursor() as cur:
                cur.executemany(
                    f"""
                    INSERT INTO {self.table_name} (identity, embedding, metadata)
                    VALUES (%(identity)s, %(embedding)s, %(metadata)s);
                    """,
                    payloads,
                )
        return len(payloads)

    def search(
        self,
        query_embedding: np.ndarray,
        *,
        top_k: int = 5,
    ) -> list[SearchResult]:
        vector = np.asarray(query_embedding, dtype=np.float32)
        operator = {
            "cosine": "<=>",
            "l2": "<->",
            "inner": "<#>",
        }.get(self.distance_metric, "<->")

        with psycopg.connect(self.dsn, autocommit=True, row_factory=dict_row) as conn:
            register_vector(conn)
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    SELECT
                        id,
                        identity,
                        metadata,
                        created_at,
                        embedding {operator} %s AS distance
                    FROM {self.table_name}
                    ORDER BY embedding {operator} %s
                    LIMIT %s;
                    """,
                    (vector, vector, top_k),
                )
                rows = cur.fetchall()

        results: list[SearchResult] = []
        for row in rows:
            distance = float(row["distance"])
            similarity = (
                1.0 - (distance**2) / 2.0
                if self.distance_metric in ("l2")
                else 1.0 - distance
            )
            metadata = row.get("metadata")
            if isinstance(metadata, str):
                try:
                    metadata = json.loads(metadata)
                except json.JSONDecodeError:
                    pass
            results.append(
                SearchResult(
                    id=row["id"],
                    identity=row["identity"],
                    similarity=similarity,
                    distance=distance,
                    metadata=metadata,
                    created_at=row.get("created_at"),
                )
            )
        return results

    def list_identities(self) -> list[str]:
        with psycopg.connect(self.dsn, autocommit=True) as conn:
            register_vector(conn)
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    SELECT DISTINCT identity
                    FROM {self.table_name}
                    ORDER BY identity;
                    """
                )
                rows = cur.fetchall()
        return [row[0] for row in rows]

    def count(self) -> int:
        with psycopg.connect(self.dsn, autocommit=True) as conn:
            register_vector(conn)
            with conn.cursor() as cur:
                cur.execute(f"SELECT COUNT(*) FROM {self.table_name};")
                value = cur.fetchone()
        return int(value[0]) if value else 0
