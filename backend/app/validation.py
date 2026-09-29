"""Image and bounding-box validation for retrieval and gallery uploads."""

from __future__ import annotations

from io import BytesIO

import numpy as np
from fastapi import HTTPException
from PIL import Image, UnidentifiedImageError

from backend.app.config import settings

ALLOWED_MIME_TYPES = {"image/jpeg", "image/png"}
IMAGE_SIGNATURES = {
    "image/jpeg": (b"\xff\xd8\xff",),
    "image/png": (b"\x89PNG\r\n\x1a\n",),
}


def decode_image(data: bytes, content_type: str | None) -> np.ndarray:
    if content_type not in ALLOWED_MIME_TYPES:
        raise HTTPException(
            status_code=415,
            detail="Only JPEG and PNG image uploads are supported",
        )
    if not data:
        raise HTTPException(status_code=400, detail="Uploaded image is empty")
    if len(data) > settings.max_image_bytes:
        raise HTTPException(status_code=413, detail="Image exceeds MAX_IMAGE_BYTES")
    if not any(
        data.startswith(signature) for signature in IMAGE_SIGNATURES[content_type]
    ):
        raise HTTPException(
            status_code=400,
            detail="Image bytes do not match the declared JPEG/PNG content type",
        )

    try:
        with Image.open(BytesIO(data)) as decoded:
            if decoded.format not in {"JPEG", "PNG"}:
                raise HTTPException(
                    status_code=400, detail="Decoded image is not JPEG or PNG"
                )
            width, height = decoded.size
            if width > settings.max_image_width or height > settings.max_image_height:
                raise HTTPException(
                    status_code=413,
                    detail=(
                        f"Image dimensions exceed configured maximum "
                        f"({settings.max_image_width}x{settings.max_image_height})"
                    ),
                )
            if width * height > settings.max_image_pixels:
                raise HTTPException(
                    status_code=413,
                    detail="Image pixel count exceeds MAX_IMAGE_PIXELS",
                )
            rgb = np.asarray(decoded.convert("RGB"), dtype=np.uint8)
    except HTTPException:
        raise
    except (
        UnidentifiedImageError,
        OSError,
        ValueError,
        SyntaxError,
        Image.DecompressionBombError,
    ) as exc:
        raise HTTPException(
            status_code=400, detail="Image bytes could not be safely decoded"
        ) from exc
    return np.ascontiguousarray(rgb[:, :, ::-1])


def validate_bbox(x: int, y: int, w: int, h: int, frame: np.ndarray) -> None:
    frame_height, frame_width = frame.shape[:2]
    if min(x, y) < 0:
        raise HTTPException(status_code=400, detail="BBox x and y must be non-negative")
    if min(w, h) <= 0:
        raise HTTPException(
            status_code=400, detail="BBox width and height must be positive"
        )
    if x + w > frame_width or y + h > frame_height:
        raise HTTPException(
            status_code=400,
            detail=(
                f"BBox exceeds decoded frame dimensions "
                f"({frame_width}x{frame_height})"
            ),
        )


def vehicle_crop(frame: np.ndarray, x: int, y: int, w: int, h: int) -> np.ndarray:
    validate_bbox(x, y, w, h, frame)
    crop = frame[y : y + h, x : x + w].copy()
    if crop.size == 0:
        raise HTTPException(status_code=400, detail="BBox produced an empty crop")
    return crop
