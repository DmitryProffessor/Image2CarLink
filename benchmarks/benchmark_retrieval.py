"""Measure per-annotation latency, sequential throughput, and sampled RSS."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import threading
import time
from pathlib import Path

import psutil

from backend.app.embedding import mask_sensitive_regions, vehicle_embedding
from reid_pipeline import _crop_vehicle


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    index = min(int((len(ordered) - 1) * quantile), len(ordered) - 1)
    return ordered[index]


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"image_id", "x", "y", "w", "h"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"{path} must contain columns {sorted(required)}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"{path} contains no annotations")
    return rows


def benchmark(
    annotations: Path,
    images_dir: Path,
    *,
    warmup_count: int = 5,
    sample_interval: float = 0.02,
) -> dict[str, float | int | str]:
    rows = load_rows(annotations)
    process = psutil.Process()
    for row in rows[:warmup_count]:
        vehicle_embedding(mask_sensitive_regions(_crop_vehicle(images_dir, row)))

    stop_sampling = threading.Event()
    peak_rss = [process.memory_info().rss]

    def sample_memory() -> None:
        while not stop_sampling.wait(sample_interval):
            peak_rss[0] = max(peak_rss[0], process.memory_info().rss)

    sampler = threading.Thread(target=sample_memory, name="rss-sampler", daemon=True)
    sampler.start()
    start_rss = process.memory_info().rss
    per_item_seconds: list[float] = []
    total_start = time.perf_counter()
    try:
        for row in rows:
            item_start = time.perf_counter()
            vehicle_embedding(mask_sensitive_regions(_crop_vehicle(images_dir, row)))
            per_item_seconds.append(time.perf_counter() - item_start)
    finally:
        elapsed = time.perf_counter() - total_start
        stop_sampling.set()
        sampler.join()
    peak_rss[0] = max(peak_rss[0], process.memory_info().rss)

    latencies_ms = [duration * 1000 for duration in per_item_seconds]
    return {
        "annotations": len(rows),
        "elapsed_seconds": elapsed,
        "throughput_annotations_per_second": len(rows) / elapsed,
        "batch_size": 1,
        "latency_ms_p50": statistics.median(latencies_ms),
        "latency_ms_p95": percentile(latencies_ms, 0.95),
        "latency_ms_max": max(latencies_ms),
        "rss_start_mib": start_rss / (1024 * 1024),
        "rss_peak_sampled_mib": peak_rss[0] / (1024 * 1024),
        "rss_peak_increase_mib": max(0, peak_rss[0] - start_rss) / (1024 * 1024),
        "measurement_note": (
            "Sequential batch-size-1 CPU measurement; no queue. "
            "RSS peak is sampled and may miss short-lived peaks."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--warmup-count", type=int, default=5)
    args = parser.parse_args()
    if args.warmup_count < 0:
        parser.error("--warmup-count must be non-negative")
    print(
        json.dumps(
            benchmark(
                args.annotations,
                args.images,
                warmup_count=args.warmup_count,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
