from __future__ import annotations

import csv

import cv2
import numpy as np

from backend.app import ingest


def test_ingest_preserves_multiple_vehicle_boxes_from_same_frame(
    tmp_path, monkeypatch
) -> None:
    images = tmp_path / "images"
    images.mkdir()
    pixels = np.full((40, 80, 3), 90, dtype=np.uint8)
    image_path = images / "frame-random-id.png"
    assert cv2.imwrite(str(image_path), pixels)
    annotations = tmp_path / "test_gallery.csv"
    with annotations.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("image_id", "x", "y", "w", "h"))
        writer.writeheader()
        writer.writerows(
            (
                {
                    "image_id": "frame-random-id",
                    "x": 0,
                    "y": 0,
                    "w": 35,
                    "h": 35,
                },
                {
                    "image_id": "frame-random-id",
                    "x": 40,
                    "y": 0,
                    "w": 35,
                    "h": 35,
                },
            )
        )

    saved_items = []
    monkeypatch.setattr(
        ingest,
        "upsert_gallery_embeddings",
        lambda items: saved_items.extend(items),
    )
    assert ingest.import_gallery(annotations, images) == 2
    assert len(saved_items) == 2
    assert saved_items[0][0] != saved_items[1][0]
    assert all(item[1] == "frame-random-id" for item in saved_items)
    assert [item[2] for item in saved_items] == [(0, 0, 35, 35), (40, 0, 35, 35)]
