from __future__ import annotations

import csv

import cv2
import numpy as np

from backend.app.embedding import (
    EMBEDDING_DIM,
    mask_sensitive_regions,
    vehicle_embedding,
)
from reid_pipeline import _crop_vehicle, generate_artifacts


def test_generated_artifacts_match_csv_order_and_expected_shapes(tmp_path) -> None:
    dataset = tmp_path / "dataset"
    image_dir = dataset / "images"
    image_dir.mkdir(parents=True)
    (dataset / "train.csv").write_text(
        "image_id,x,y,w,h,vehicle_id\ntrain-1,0,0,32,24,train-only\n",
        encoding="utf-8",
    )
    query_rows = [
        {"image_id": "query-a", "x": "0", "y": "0", "w": "32", "h": "24"},
        {"image_id": "query-b", "x": "1", "y": "1", "w": "30", "h": "22"},
    ]
    gallery_rows = [
        {
            "image_id": f"gallery-{index:02}",
            "x": "0",
            "y": "0",
            "w": "32",
            "h": "24",
        }
        for index in range(10)
    ]

    for index, row in enumerate(query_rows + gallery_rows):
        pixels = np.full(
            (24, 32, 3),
            (index * 19 % 255, index * 31 % 255, index * 47 % 255),
            dtype=np.uint8,
        )
        assert cv2.imwrite(str(image_dir / f"{row['image_id']}.png"), pixels)

    for filename, rows in (
        ("test_query.csv", query_rows),
        ("test_gallery.csv", gallery_rows),
    ):
        with (dataset / filename).open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=("image_id", "x", "y", "w", "h"))
            writer.writeheader()
            writer.writerows(rows)

    output = tmp_path / "output"
    generate_artifacts(dataset, output, confidence_threshold=0.0)

    embeddings = np.load(output / "embeddings.npy")
    expected_count = len(query_rows) + len(gallery_rows)
    assert embeddings.shape == (expected_count, EMBEDDING_DIM)
    assert embeddings.dtype == np.float32
    assert np.isfinite(embeddings).all()
    assert np.allclose(np.linalg.norm(embeddings, axis=1), 1.0, atol=1e-6)

    first_query = vehicle_embedding(
        mask_sensitive_regions(_crop_vehicle(image_dir, query_rows[0]))
    )
    first_gallery = vehicle_embedding(
        mask_sensitive_regions(_crop_vehicle(image_dir, gallery_rows[0]))
    )
    assert np.array_equal(embeddings[0], first_query)
    assert np.array_equal(embeddings[len(query_rows)], first_gallery)

    with (output / "submission.csv").open(encoding="utf-8", newline="") as stream:
        submission = list(csv.reader(stream))
    assert len(submission) == len(query_rows)
    assert [row[0] for row in submission] == [row["image_id"] for row in query_rows]
    assert all(len(row) == 1 + len(gallery_rows) for row in submission)
    assert all(len(set(row[1:])) == len(gallery_rows) for row in submission)
    gallery_ids = [row["image_id"] for row in gallery_rows]
    for query_index, row in enumerate(submission):
        scores = embeddings[query_index] @ embeddings[len(query_rows) :].T
        expected_order = np.argsort(-scores, kind="stable")
        assert row[1:] == [gallery_ids[index] for index in expected_order]

    with (output / "candidates.csv").open(encoding="utf-8", newline="") as stream:
        candidates = list(csv.DictReader(stream))
    assert len(candidates) == len(query_rows) * len(gallery_rows)
    assert {row["query_id"] for row in candidates} == {
        row["image_id"] for row in query_rows
    }
    for query_id in {row["image_id"] for row in query_rows}:
        query_candidates = [row for row in candidates if row["query_id"] == query_id]
        scores = [float(row["confidence"]) for row in query_candidates]
        assert scores == sorted(scores, reverse=True)
