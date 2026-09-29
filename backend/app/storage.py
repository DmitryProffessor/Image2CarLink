"""PostgreSQL + pgvector access for gallery embedding storage and search."""

from __future__ import annotations

from typing import Any

import numpy as np
import psycopg
from psycopg_pool import ConnectionPool, PoolTimeout

from backend.app.config import settings
from backend.app.embedding import EMBEDDING_DIM, l2_normalize

_pool: ConnectionPool[Any] | None = None


def start_pool(database_url: str | None = None) -> None:
    global _pool
    if _pool is None:
        _pool = ConnectionPool(
            conninfo=database_url or settings.database_url,
            min_size=settings.database_pool_min_size,
            max_size=settings.database_pool_max_size,
            timeout=settings.database_pool_timeout,
            open=False,
        )
        _pool.open(wait=True, timeout=settings.database_pool_timeout)


def close_pool() -> None:
    global _pool
    if _pool is not None:
        _pool.close()
        _pool = None


def _connection_pool() -> ConnectionPool[Any]:
    if _pool is None:
        start_pool()
    if _pool is None:
        raise RuntimeError("PostgreSQL connection pool failed to initialize")
    return _pool


def initialize_schema() -> None:
    pool = _connection_pool()
    with pool.connection() as conn:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS vehicle_gallery_items (
                gallery_id TEXT PRIMARY KEY,
                source_image_id TEXT NOT NULL,
                bbox_x INTEGER NOT NULL,
                bbox_y INTEGER NOT NULL,
                bbox_w INTEGER NOT NULL CHECK (bbox_w > 0),
                bbox_h INTEGER NOT NULL CHECK (bbox_h > 0),
                embedding VECTOR({EMBEDDING_DIM}) NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS vehicle_gallery_items_embedding_cosine_idx
            ON vehicle_gallery_items USING hnsw (embedding vector_cosine_ops)
            """
        )


def database_is_ready() -> bool:
    try:
        with _connection_pool().connection() as conn:
            conn.execute("SELECT 1")
        return True
    except (psycopg.Error, PoolTimeout):
        return False


def _vector_literal(embedding: np.ndarray) -> str:
    vector = l2_normalize(embedding)
    if vector.shape != (EMBEDDING_DIM,):
        raise ValueError(f"embedding must have dimension {EMBEDDING_DIM}")
    if not np.any(vector):
        raise ValueError("embedding must have a non-zero norm")
    return "[" + ",".join(f"{float(value):.9g}" for value in vector) + "]"


def upsert_gallery_embedding(
    gallery_id: str,
    source_image_id: str,
    bbox: tuple[int, int, int, int],
    embedding: np.ndarray,
) -> None:
    vector_text = _vector_literal(embedding)
    with _connection_pool().connection() as conn:
        conn.execute(
            """
            INSERT INTO vehicle_gallery_items (
                gallery_id, source_image_id, bbox_x, bbox_y, bbox_w, bbox_h, embedding
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s::vector)
            ON CONFLICT (gallery_id)
            DO UPDATE SET embedding = EXCLUDED.embedding
            """,
            (gallery_id, source_image_id, *bbox, vector_text),
        )


def upsert_gallery_embeddings(
    items: list[tuple[str, str, tuple[int, int, int, int], np.ndarray]],
) -> None:
    if not items:
        return
    values = [
        (
            gallery_id,
            source_image_id,
            *bbox,
            _vector_literal(embedding),
        )
        for gallery_id, source_image_id, bbox, embedding in items
    ]
    with _connection_pool().connection() as conn:
        with conn.transaction():
            for row in values:
                conn.execute(
                    """
                    INSERT INTO vehicle_gallery_items (
                        gallery_id, source_image_id,
                        bbox_x, bbox_y, bbox_w, bbox_h, embedding
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s::vector)
                    ON CONFLICT (gallery_id)
                    DO UPDATE SET embedding = EXCLUDED.embedding
                    """,
                    (*row[:6], row[6]),
                )


def search_gallery(query_embedding: np.ndarray, limit: int) -> list[dict[str, Any]]:
    vector_text = _vector_literal(query_embedding)
    with _connection_pool().connection() as conn:
        rows = conn.execute(
            """
            SELECT gallery_id,
                   1 - (embedding <=> %s::vector) AS cosine_similarity
            FROM vehicle_gallery_items
            ORDER BY embedding <=> %s::vector, gallery_id
            LIMIT %s
            """,
            (vector_text, vector_text, limit),
        ).fetchall()
    return [{"gallery_id": str(row[0]), "score": float(row[1])} for row in rows]


def gallery_size() -> int:
    with _connection_pool().connection() as conn:
        return int(
            conn.execute("SELECT count(*) FROM vehicle_gallery_items").fetchone()[0]
        )
