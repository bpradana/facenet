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
        identities_table: Optional[str] = None,
        lists: int = 100,
        distance_metric: str = "cosine",
    ) -> None:
        self.config = config
        self.dsn = config.build_dsn()
        self.embeddings_table = table_name
        self.identities_table = identities_table or f"{table_name}_identities"
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
                    CREATE TABLE IF NOT EXISTS {self.identities_table} (
                        id SERIAL PRIMARY KEY,
                        identity TEXT UNIQUE NOT NULL,
                        centroid vector({self.embedding_dim}) NOT NULL,
                        embedding_count INTEGER NOT NULL DEFAULT 0,
                        metadata JSONB,
                        created_at TIMESTAMPTZ DEFAULT NOW(),
                        updated_at TIMESTAMPTZ DEFAULT NOW()
                    );
                    """
                )
                cur.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS {self.embeddings_table} (
                        id SERIAL PRIMARY KEY,
                        identity_id INTEGER NOT NULL REFERENCES {self.identities_table}(id) ON DELETE CASCADE,
                        embedding vector({self.embedding_dim}) NOT NULL,
                        metadata JSONB,
                        created_at TIMESTAMPTZ DEFAULT NOW()
                    );
                    """
                )
                cur.execute(
                    f"""
                    CREATE INDEX IF NOT EXISTS {self.embeddings_table}_identity_idx
                    ON {self.embeddings_table} (identity_id);
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
                    "embedding": vector,
                }
            )

        if not payloads:
            return 0

        metadata_json = Json(metadata) if metadata is not None else None

        with psycopg.connect(self.dsn, autocommit=False) as conn:
            register_vector(conn)
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    SELECT id, centroid, embedding_count
                    FROM {self.identities_table}
                    WHERE identity = %s
                    FOR UPDATE;
                    """,
                    (identity,),
                )
                row = cur.fetchone()

                vectors = np.stack([p["embedding"] for p in payloads])
                inserted = vectors.shape[0]
                new_centroid = vectors.mean(axis=0)
                embedding_count = inserted
                identity_id: Optional[int] = None

                if row is not None:
                    identity_id = int(row[0])
                    existing_centroid = np.asarray(row[1], dtype=np.float32)
                    existing_count = int(row[2])
                    total = existing_count + inserted
                    if existing_count > 0:
                        aggregate = existing_centroid * existing_count + vectors.sum(
                            axis=0
                        )
                        new_centroid = aggregate / total
                    embedding_count = total
                    params: list[Any] = [new_centroid, embedding_count]
                    update_sql = f"""
                        UPDATE {self.identities_table}
                        SET centroid = %s,
                            embedding_count = %s,
                            updated_at = NOW()
                    """
                    if metadata_json is not None:
                        update_sql += ", metadata = %s"
                        params.append(metadata_json)
                    update_sql += " WHERE id = %s;"
                    params.append(identity_id)
                    cur.execute(update_sql, tuple(params))
                else:
                    cur.execute(
                        f"""
                        INSERT INTO {self.identities_table} (identity, centroid, embedding_count, metadata)
                        VALUES (%s, %s, %s, %s)
                        RETURNING id;
                        """,
                        (identity, new_centroid, embedding_count, metadata_json),
                    )
                    identity_id = int(cur.fetchone()[0])

                assert identity_id is not None
                embedding_rows = [
                    {
                        "identity_id": identity_id,
                        "embedding": vector,
                        "metadata": Json(metadata) if metadata is not None else None,
                    }
                    for vector in vectors
                ]
                cur.executemany(
                    f"""
                    INSERT INTO {self.embeddings_table} (identity_id, embedding, metadata)
                    VALUES (%(identity_id)s, %(embedding)s, %(metadata)s);
                    """,
                    embedding_rows,
                )
            conn.commit()
        return inserted

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
                        centroid {operator} %s AS distance
                    FROM {self.identities_table}
                    ORDER BY centroid {operator} %s
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
                    SELECT identity
                    FROM {self.identities_table}
                    ORDER BY identity;
                    """
                )
                rows = cur.fetchall()
        return [row[0] for row in rows]

    def count(self) -> int:
        with psycopg.connect(self.dsn, autocommit=True) as conn:
            register_vector(conn)
            with conn.cursor() as cur:
                cur.execute(f"SELECT COUNT(*) FROM {self.identities_table};")
                value = cur.fetchone()
        return int(value[0]) if value else 0
