"""Load a test_gallery-style CSV and its images into PostgreSQL pgvector."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2

from backend.app.embedding import mask_sensitive_regions, vehicle_embedding
from backend.app.service import gallery_annotation_id
from backend.app.storage import upsert_gallery_embeddings
from backend.app.validation import vehicle_crop

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")


def resolve_image(image_dir: Path, image_id: str) -> Path:
    path = Path(image_id)
    if path.name != image_id or image_id in {"", ".", ".."}:
        raise ValueError(f"invalid image_id {image_id!r}: IDs must be flat filenames")
    direct = image_dir / image_id
    if direct.is_file():
        return direct
    for suffix in IMAGE_SUFFIXES:
        candidate = image_dir / f"{image_id}{suffix}"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"image for {image_id!r} not found in {image_dir}")


def import_gallery(annotations: Path, image_dir: Path) -> int:
    with annotations.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"image_id", "x", "y", "w", "h"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"{annotations} must contain columns {sorted(required)}")
        rows = list(reader)
    items = []
    for row in rows:
        image_id = row["image_id"].strip()
        image_path = resolve_image(image_dir, image_id)
        frame = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError(f"could not decode image {image_path}")
        bbox = [int(row[key]) for key in ("x", "y", "w", "h")]
        x, y, width, height = bbox
        crop = vehicle_crop(frame, x, y, width, height)
        embedding = vehicle_embedding(mask_sensitive_regions(crop))
        gallery_id = gallery_annotation_id(image_id, x, y, width, height)
        items.append((gallery_id, image_id, (x, y, width, height), embedding))
    upsert_gallery_embeddings(items)
    return len(items)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    args = parser.parse_args()
    imported_count = import_gallery(args.annotations, args.images)
    print(f"Imported/updated {imported_count} gallery items")


if __name__ == "__main__":
    main()
