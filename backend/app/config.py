"""Environment-based application configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass


def _positive_int(name: str, default: int) -> int:
    value = int(os.getenv(name, str(default)))
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


@dataclass(frozen=True)
class Settings:
    database_url: str = os.getenv(
        "DATABASE_URL",
        "postgresql://appuser:apppass@localhost:5432/vehicle_reid",
    )
    rejection_threshold: float = float(os.getenv("REJECTION_THRESHOLD", "0.65"))
    top_n: int = _positive_int("TOP_N", 10)
    max_image_bytes: int = _positive_int("MAX_IMAGE_BYTES", 10 * 1024 * 1024)
    max_image_width: int = _positive_int("MAX_IMAGE_WIDTH", 8192)
    max_image_height: int = _positive_int("MAX_IMAGE_HEIGHT", 8192)
    max_image_pixels: int = _positive_int("MAX_IMAGE_PIXELS", 20000000)
    database_pool_min_size: int = _positive_int("DATABASE_POOL_MIN_SIZE", 1)
    database_pool_max_size: int = _positive_int("DATABASE_POOL_MAX_SIZE", 8)
    database_pool_timeout: float = float(os.getenv("DATABASE_POOL_TIMEOUT", "5"))
    gallery_csv: str = os.getenv("GALLERY_CSV", "")
    gallery_images_dir: str = os.getenv("GALLERY_IMAGES_DIR", "")
    cors_origins: tuple[str, ...] = tuple(
        origin.strip()
        for origin in os.getenv(
            "CORS_ORIGINS", "http://localhost:8080,http://127.0.0.1:8080"
        ).split(",")
        if origin.strip()
    )

    def validate(self) -> None:
        if not 0.0 <= self.rejection_threshold <= 1.0:
            raise ValueError("REJECTION_THRESHOLD must be between 0 and 1")
        if self.database_pool_max_size < self.database_pool_min_size:
            raise ValueError("DATABASE_POOL_MAX_SIZE must be >= DATABASE_POOL_MIN_SIZE")
        if self.database_pool_timeout <= 0:
            raise ValueError("DATABASE_POOL_TIMEOUT must be positive")


settings = Settings()
settings.validate()
