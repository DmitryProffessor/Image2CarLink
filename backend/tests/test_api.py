from __future__ import annotations

from io import BytesIO

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.app import main, service
from backend.app.service import gallery_annotation_id


def jpeg_bytes() -> bytes:
    success, encoded = cv2.imencode(".jpg", np.full((40, 60, 3), 110, dtype=np.uint8))
    assert success
    return BytesIO(encoded).getvalue()


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(main, "initialize_schema", lambda: None)
    monkeypatch.setattr(main, "database_is_ready", lambda: True)
    with TestClient(main.app) as test_client:
        yield test_client


def test_health_and_openapi_documentation(client: TestClient) -> None:
    assert client.get("/health").json() == {"status": "ok"}
    schema = client.get("/openapi.json").json()
    assert "/api/v1/retrieve" in schema["paths"]
    assert "/api/v1/gallery/items" in schema["paths"]


def test_retrieve_returns_ranked_match(client: TestClient, monkeypatch) -> None:
    monkeypatch.setattr(
        service,
        "search_gallery",
        lambda embedding, limit: [
            {"image_id": "g1", "score": 0.9},
            {"image_id": "g2", "score": 0.2},
        ][:limit],
    )
    response = client.post(
        "/api/v1/retrieve",
        data={
            "query_id": "q1",
            "x": "1",
            "y": "2",
            "w": "20",
            "h": "25",
            "top_n": "2",
        },
        files={"image": ("query.jpg", jpeg_bytes(), "image/jpeg")},
    )
    assert response.status_code == 200
    assert response.json()["candidates"][0]["gallery_id"] == "g1"
    assert response.json()["candidates"][0]["confidence"] == pytest.approx(0.95)
    assert response.json()["rejected"] is False


def test_gallery_api_distinguishes_two_boxes_in_same_uploaded_frame(
    client: TestClient, monkeypatch
) -> None:
    stored_items = []

    def save_item(image_id, frame, x, y, w, h):
        gallery_id = gallery_annotation_id(image_id, x, y, w, h)
        stored_items.append(gallery_id)
        return gallery_id

    monkeypatch.setattr(main, "add_gallery_vehicle", save_item)
    image = jpeg_bytes()
    responses = []
    for x in (0, 30):
        responses.append(
            client.post(
                "/api/v1/gallery/items",
                data={
                    "image_id": "same-frame",
                    "x": str(x),
                    "y": "0",
                    "w": "25",
                    "h": "30",
                },
                files={"image": ("frame.jpg", image, "image/jpeg")},
            )
        )
    assert [response.status_code for response in responses] == [201, 201]
    assert responses[0].json()["gallery_id"] != responses[1].json()["gallery_id"]
    assert len(set(stored_items)) == 2


def test_retrieve_rejects_bbox_outside_actual_frame(client: TestClient) -> None:
    response = client.post(
        "/api/v1/retrieve",
        data={"query_id": "q1", "x": "59", "y": "0", "w": "2", "h": "1"},
        files={"image": ("query.jpg", jpeg_bytes(), "image/jpeg")},
    )
    assert response.status_code == 400


def test_retrieve_returns_empty_list_on_rejection(
    client: TestClient, monkeypatch
) -> None:
    monkeypatch.setattr(
        service,
        "search_gallery",
        lambda embedding, limit: [{"image_id": "g1", "score": 0.2}],
    )
    response = client.post(
        "/api/v1/retrieve",
        data={"query_id": "q1", "x": "0", "y": "0", "w": "30", "h": "30"},
        files={"image": ("query.jpg", jpeg_bytes(), "image/jpeg")},
    )
    assert response.status_code == 200
    assert response.json()["rejected"] is True
    assert response.json()["candidates"] == []
