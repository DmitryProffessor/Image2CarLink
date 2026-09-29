"""Offline vehicle retrieval pipeline for the dataset described in README_data.md.

The pipeline masks a fixed lower-central part of every crop before feature
extraction as a plate-leakage precaution. This heuristic is not plate detection.
"""

from __future__ import annotations

import argparse
import csv
from collections.abc import Iterable
from pathlib import Path

import cv2
import numpy as np

from backend.app.embedding import (
    cosine_confidence,
    mask_sensitive_regions,
    vehicle_embedding,
)

BOX_COLUMNS = ("image_id", "x", "y", "w", "h")
TOP_K = 10


def _read_annotations(path: Path, *, require_vehicle_id: bool) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"{path} is empty")
    required = set(BOX_COLUMNS)
    if require_vehicle_id:
        required.add("vehicle_id")
    missing = required.difference(rows[0])
    if missing:
        raise ValueError(f"{path} is missing columns: {', '.join(sorted(missing))}")
    return rows


def _image_path(images_dir: Path, image_id: str) -> Path:
    # IDs are names, not paths. Reject separators to avoid escaping images_dir.
    if not image_id or Path(image_id).name != image_id:
        raise ValueError(f"Invalid image_id: {image_id!r}")
    direct = images_dir / image_id
    if direct.is_file() and direct.suffix.lower() in {".jpg", ".jpeg", ".png"}:
        return direct
    for suffix in (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"):
        candidate = images_dir / f"{image_id}{suffix}"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"No JPEG or PNG found for image_id={image_id!r} in {images_dir}"
    )


def _crop_vehicle(images_dir: Path, row: dict[str, str]) -> np.ndarray:
    image = cv2.imread(str(_image_path(images_dir, row["image_id"])), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"OpenCV could not decode {row['image_id']!r}")
    try:
        x, y, width, height = (int(row[key]) for key in ("x", "y", "w", "h"))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid bounding box for {row['image_id']!r}") from exc
    if x < 0 or y < 0 or width <= 0 or height <= 0:
        raise ValueError(f"Invalid bounding box for {row['image_id']!r}")
    if x + width > image.shape[1] or y + height > image.shape[0]:
        raise ValueError(f"Bounding box is outside image {row['image_id']!r}")
    return image[y : y + height, x : x + width].copy()


def _embeddings(rows: Iterable[dict[str, str]], images_dir: Path) -> np.ndarray:
    return np.vstack(
        [
            vehicle_embedding(mask_sensitive_regions(_crop_vehicle(images_dir, row)))
            for row in rows
        ]
    ).astype(np.float32)


def generate_artifacts(
    dataset_dir: Path,
    output_dir: Path,
    *,
    confidence_threshold: float = 0.65,
) -> None:
    images_dir = dataset_dir / "images"
    query = _read_annotations(dataset_dir / "test_query.csv", require_vehicle_id=False)
    gallery = _read_annotations(
        dataset_dir / "test_gallery.csv", require_vehicle_id=False
    )
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Missing image directory: {images_dir}")
    if not 0.0 <= confidence_threshold <= 1.0:
        raise ValueError("confidence_threshold must be between 0 and 1")
    query_embeddings = _embeddings(query, images_dir)
    gallery_embeddings = _embeddings(gallery, images_dir)
    similarities = query_embeddings @ gallery_embeddings.T
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(
        output_dir / "embeddings.npy", np.vstack((query_embeddings, gallery_embeddings))
    )

    gallery_ids = [row["image_id"] for row in gallery]
    with (output_dir / "submission.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.writer(stream, lineterminator="\n")
        for row, scores in zip(query, similarities):
            order = np.argsort(-scores, kind="stable")[:TOP_K]
            writer.writerow([row["image_id"], *(gallery_ids[index] for index in order)])

    with (output_dir / "candidates.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(("query_id", "gallery_id", "confidence"))
        for row, scores in zip(query, similarities):
            confidences = np.asarray(
                [cosine_confidence(float(score)) for score in scores],
                dtype=np.float32,
            )
            accepted = np.flatnonzero(confidences >= confidence_threshold)
            accepted = accepted[np.argsort(-scores[accepted], kind="stable")]
            for index in accepted:
                writer.writerow(
                    (
                        row["image_id"],
                        gallery_ids[index],
                        f"{confidences[index]:.6f}",
                    )
                )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate vehicle-retrieval submission artifacts."
    )
    parser.add_argument("--dataset", type=Path, default=Path("dataset"))
    parser.add_argument("--output", type=Path, default=Path("."))
    parser.add_argument("--confidence-threshold", type=float, default=0.65)
    args = parser.parse_args()
    generate_artifacts(
        args.dataset, args.output, confidence_threshold=args.confidence_threshold
    )


if __name__ == "__main__":
    main()
