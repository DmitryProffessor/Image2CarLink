"""Receiving, processing, ranking, and rejection operations."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from backend.app.config import settings
from backend.app.embedding import (
    cosine_confidence,
    mask_sensitive_regions,
    vehicle_embedding,
)
from backend.app.storage import search_gallery, upsert_gallery_embedding
from backend.app.validation import vehicle_crop


@dataclass(frozen=True)
class CandidateResult:
    gallery_id: str
    score: float
    confidence: float


def rank_embeddings(
    query_embedding: np.ndarray,
    gallery: list[tuple[str, np.ndarray]],
    top_n: int,
) -> list[CandidateResult]:
    query = np.asarray(query_embedding, dtype=np.float32)
    query_norm = float(np.linalg.norm(query))
    if query.ndim != 1 or query_norm < 1e-12:
        raise ValueError("query embedding must be a non-zero vector")
    query = query / query_norm
    scored: list[CandidateResult] = []
    for gallery_id, vector in gallery:
        candidate = np.asarray(vector, dtype=np.float32)
        norm = float(np.linalg.norm(candidate))
        if candidate.shape != query.shape or norm < 1e-12:
            raise ValueError("gallery embedding dimensions must match and be non-zero")
        score = float(np.dot(query, candidate / norm))
        scored.append(CandidateResult(gallery_id, score, cosine_confidence(score)))
    return sorted(scored, key=lambda item: (-item.score, item.gallery_id))[:top_n]


def embed_vehicle(frame_bgr: np.ndarray, x: int, y: int, w: int, h: int) -> np.ndarray:
    crop = vehicle_crop(frame_bgr, x, y, w, h)
    # Inputs are expected to follow the organizer's pre-blurred image protocol.
    # The geometric mask suppresses the usual lower-centre plate area, but cannot
    # identify plates or faces outside that region.
    return vehicle_embedding(mask_sensitive_regions(crop))


def gallery_annotation_id(source_image_id: str, x: int, y: int, w: int, h: int) -> str:
    """Stable identifier for one vehicle annotation, even with shared frame IDs."""
    identity = f"{source_image_id}\0{x}\0{y}\0{w}\0{h}".encode()
    return hashlib.sha256(identity).hexdigest()


def add_gallery_vehicle(
    image_id: str, frame_bgr: np.ndarray, x: int, y: int, w: int, h: int
) -> str:
    embedding = embed_vehicle(frame_bgr, x, y, w, h)
    gallery_id = gallery_annotation_id(image_id, x, y, w, h)
    upsert_gallery_embedding(gallery_id, image_id, (x, y, w, h), embedding)
    return gallery_id


def retrieve_vehicle(
    query_id: str,
    frame_bgr: np.ndarray,
    x: int,
    y: int,
    w: int,
    h: int,
    top_n: int | None = None,
) -> dict:
    embedding = embed_vehicle(frame_bgr, x, y, w, h)
    limit = min(top_n or settings.top_n, settings.top_n)
    ranked_rows = search_gallery(embedding, limit)
    ranked = [
        CandidateResult(
            gallery_id=row["image_id"],
            score=row["score"],
            confidence=cosine_confidence(row["score"]),
        )
        for row in ranked_rows
    ]
    accepted = [
        candidate
        for candidate in ranked
        if candidate.confidence >= settings.rejection_threshold
    ]
    rejected = not accepted
    return {
        "query_id": query_id,
        "candidates": [
            {
                "gallery_id": item.gallery_id,
                "score": item.score,
                "confidence": item.confidence,
            }
            for item in accepted
        ],
        "rejected": rejected,
        "threshold": settings.rejection_threshold,
        "metric": "cosine",
    }
