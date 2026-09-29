from __future__ import annotations

import os
import uuid

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.app import main, storage
from backend.app.embedding import EMBEDDING_DIM

pytestmark = pytest.mark.integration


@pytest.fixture
def disposable_pgvector_database():
    database_url = os.getenv("TEST_DATABASE_URL")
    if not database_url:
        pytest.skip("set TEST_DATABASE_URL to a disposable pgvector-enabled database")
    storage.start_pool(database_url)
    storage.initialize_schema()
    pool = storage._connection_pool()
    with pool.connection() as conn:
        conn.execute("TRUNCATE vehicle_gallery_items")
    try:
        yield
    finally:
        with pool.connection() as conn:
            conn.execute("TRUNCATE vehicle_gallery_items")
        storage.close_pool()


def test_pgvector_empty_gallery_insert_and_multi_bbox_search(
    disposable_pgvector_database,
) -> None:
    assert storage.gallery_size() == 0
    source_frame_id = f"shared-frame-{uuid.uuid4().hex}"
    first_id = f"annotation-front-{uuid.uuid4().hex}"
    second_id = f"annotation-side-{uuid.uuid4().hex}"
    query = np.zeros(EMBEDDING_DIM, dtype=np.float32)
    query[0] = 1.0
    orthogonal = np.zeros(EMBEDDING_DIM, dtype=np.float32)
    orthogonal[1] = 1.0

    storage.upsert_gallery_embeddings(
        [
            (first_id, source_frame_id, (0, 0, 30, 20), query),
            (second_id, source_frame_id, (30, 0, 30, 20), orthogonal),
        ]
    )

    assert storage.gallery_size() == 2
    results = storage.search_gallery(query, limit=2)
    assert [row["gallery_id"] for row in results] == [first_id, second_id]
    assert results[0]["score"] == pytest.approx(1.0)
    assert results[1]["score"] == pytest.approx(0.0)


def test_live_api_empty_rejection_and_two_annotations_same_frame(
    disposable_pgvector_database,
) -> None:
    frame = np.zeros((48, 96, 3), dtype=np.uint8)
    frame[:, :48] = (20, 50, 220)
    frame[:, 48:] = (30, 210, 40)
    success, encoded = cv2.imencode(".jpg", frame)
    assert success
    image_bytes = encoded.tobytes()
    client = TestClient(main.app)
    rejected = client.post(
        "/api/v1/retrieve",
        data={"query_id": "integration-query", "x": 0, "y": 0, "w": 48, "h": 48},
        files={"image": ("frame.jpg", image_bytes, "image/jpeg")},
    )
    assert rejected.status_code == 200
    assert rejected.json()["rejected"] is True
    assert rejected.json()["candidates"] == []

    gallery_responses = []
    for x in (0, 48):
        gallery_responses.append(
            client.post(
                "/api/v1/gallery/items",
                data={
                    "image_id": "same-source-frame",
                    "x": x,
                    "y": 0,
                    "w": 48,
                    "h": 48,
                },
                files={"image": ("frame.jpg", image_bytes, "image/jpeg")},
            )
        )
    assert [response.status_code for response in gallery_responses] == [201, 201]
    annotation_ids = [response.json()["gallery_id"] for response in gallery_responses]
    assert annotation_ids[0] != annotation_ids[1]

    matched = client.post(
        "/api/v1/retrieve",
        data={"query_id": "integration-query", "x": 0, "y": 0, "w": 48, "h": 48},
        files={"image": ("query.jpg", image_bytes, "image/jpeg")},
    )
    assert matched.status_code == 200
    result = matched.json()
    assert result["rejected"] is False
    assert result["candidates"][0]["gallery_id"] == annotation_ids[0]
