from __future__ import annotations

from dataclasses import replace
from io import BytesIO

import numpy as np
import pytest
from fastapi import HTTPException
from PIL import Image

from backend.app import validation
from backend.app.embedding import (
    EMBEDDING_DIM,
    cosine_confidence,
    l2_normalize,
    mask_sensitive_regions,
    vehicle_embedding,
)
from backend.app.service import (
    gallery_annotation_id,
    rank_embeddings,
)
from backend.app.validation import decode_image, validate_bbox, vehicle_crop


def test_embedding_is_unit_normalized_and_fixed_dimension() -> None:
    crop = np.full((72, 120, 3), 127, dtype=np.uint8)
    vector = vehicle_embedding(crop)
    assert vector.shape == (EMBEDDING_DIM,)
    assert vector.dtype == np.float32
    assert np.isclose(np.linalg.norm(vector), 1.0, atol=1e-6)


def test_l2_normalize_rejects_non_finite_values() -> None:
    with pytest.raises(ValueError, match="finite"):
        l2_normalize(np.array([1.0, np.nan], dtype=np.float32))


def test_bbox_validation_rejects_bounds_and_empty_crop() -> None:
    frame = np.zeros((20, 30, 3), dtype=np.uint8)
    with pytest.raises(HTTPException) as exc:
        validate_bbox(29, 0, 2, 1, frame)
    assert exc.value.status_code == 400
    with pytest.raises(HTTPException):
        vehicle_crop(frame, 0, 0, 0, 1)


def test_image_validation_rejects_wrong_type_and_corrupt_bytes() -> None:
    with pytest.raises(HTTPException) as exc:
        decode_image(b"data", "text/plain")
    assert exc.value.status_code == 415
    with pytest.raises(HTTPException) as exc:
        decode_image(b"not an image", "image/jpeg")
    assert exc.value.status_code == 400


def test_image_pixel_limit_rejects_dimensions_before_full_decode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = BytesIO()
    Image.new("RGB", (11, 11), color="white").save(data, format="PNG")
    monkeypatch.setattr(
        validation,
        "settings",
        replace(validation.settings, max_image_pixels=100),
    )
    with pytest.raises(HTTPException) as exc:
        decode_image(data.getvalue(), "image/png")
    assert exc.value.status_code == 413


def test_sensitive_plate_mask_is_applied_deterministically() -> None:
    crop = np.full((100, 100, 3), 255, dtype=np.uint8)
    masked = mask_sensitive_regions(crop)
    assert np.all(masked[58:, 18:82] == 0)
    assert np.all(masked[:58] == 255)
    assert np.all(masked[58:, :18] == 255)
    assert np.array_equal(masked, mask_sensitive_regions(crop))


def test_gallery_annotation_ids_are_stable_and_distinguish_bboxes() -> None:
    first = gallery_annotation_id("shared-frame", 0, 0, 30, 20)
    same_annotation = gallery_annotation_id("shared-frame", 0, 0, 30, 20)
    second = gallery_annotation_id("shared-frame", 30, 0, 30, 20)
    assert first == same_annotation
    assert first != second


def test_ranking_is_descending_cosine_similarity() -> None:
    query = np.array([1.0, 0.0], dtype=np.float32)
    gallery = [
        ("second", np.array([0.0, 1.0], dtype=np.float32)),
        ("first", np.array([1.0, 0.0], dtype=np.float32)),
    ]
    result = rank_embeddings(query, gallery, top_n=2)
    assert [item.gallery_id for item in result] == ["first", "second"]
    assert result[0].score == pytest.approx(1.0)


def test_confidence_mapping_and_threshold_comparison() -> None:
    threshold = 0.65
    assert cosine_confidence(0.3) == pytest.approx(threshold)
    assert cosine_confidence(0.29) < threshold
