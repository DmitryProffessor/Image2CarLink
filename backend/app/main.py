"""FastAPI vehicle retrieval service with automatically generated OpenAPI docs."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from backend.app.config import settings
from backend.app.ingest import import_gallery
from backend.app.service import add_gallery_vehicle, retrieve_vehicle
from backend.app.storage import (
    close_pool,
    database_is_ready,
    gallery_size,
    initialize_schema,
)
from backend.app.validation import decode_image

logger = logging.getLogger(__name__)


class Candidate(BaseModel):
    gallery_id: str
    score: float = Field(description="Cosine similarity; larger values rank higher")
    confidence: float = Field(
        description="Linear score mapping (score + 1) / 2; not a calibrated probability"
    )


class RetrievalResponse(BaseModel):
    query_id: str
    candidates: list[Candidate]
    rejected: bool
    threshold: float
    metric: str = "cosine"


class GalleryItemResponse(BaseModel):
    gallery_id: str
    image_id: str
    status: str = "stored"


@asynccontextmanager
async def lifespan(_: FastAPI):
    initialize_schema()
    try:
        if settings.gallery_csv and settings.gallery_images_dir:
            csv_path = Path(settings.gallery_csv)
            image_dir = Path(settings.gallery_images_dir)
            if csv_path.is_file() and image_dir.is_dir():
                imported = await run_in_threadpool(import_gallery, csv_path, image_dir)
                logger.info("Imported or refreshed %d gallery records", imported)
            else:
                logger.warning("Gallery files missing; starting with an empty gallery")
        yield
    finally:
        close_pool()


app = FastAPI(
    title="Vehicle Retrieval API",
    description=(
        "Visual vehicle re-identification from a supplied image and bounding box. "
        "No detection, ALPR, tracking, camera, time, or location signals are used."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.cors_origins),
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/health", tags=["Operations"])
def health() -> dict[str, str]:
    if not database_is_ready():
        raise HTTPException(status_code=503, detail="database unavailable")
    return {"status": "ok"}


@app.get("/api/v1/gallery", tags=["Gallery"])
def get_gallery_status() -> dict[str, int]:
    try:
        return {"items": gallery_size()}
    except Exception as exc:
        logger.exception("Failed to read gallery size")
        raise HTTPException(
            status_code=503, detail="gallery database unavailable"
        ) from exc


@app.post(
    "/api/v1/gallery/items",
    response_model=GalleryItemResponse,
    tags=["Gallery"],
    status_code=201,
)
async def add_gallery_item(
    image_id: str = Form(..., min_length=1, max_length=255),
    x: int = Form(..., ge=0),
    y: int = Form(..., ge=0),
    w: int = Form(..., gt=0),
    h: int = Form(..., gt=0),
    image: UploadFile = File(...),
) -> GalleryItemResponse:
    image_bytes = await image.read(settings.max_image_bytes + 1)
    frame = await run_in_threadpool(decode_image, image_bytes, image.content_type)
    try:
        gallery_id = await run_in_threadpool(
            add_gallery_vehicle, image_id, frame, x, y, w, h
        )
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Failed to store gallery embedding")
        raise HTTPException(
            status_code=503, detail="gallery database unavailable"
        ) from exc
    return GalleryItemResponse(gallery_id=gallery_id, image_id=image_id)


@app.post(
    "/api/v1/retrieve",
    response_model=RetrievalResponse,
    tags=["Retrieval"],
    summary="Find visually similar gallery vehicles",
)
async def retrieve(
    image: UploadFile = File(..., description="JPEG or PNG full frame"),
    x: int = Form(..., ge=0, description="BBox left coordinate in source pixels"),
    y: int = Form(..., ge=0, description="BBox top coordinate in source pixels"),
    w: int = Form(..., gt=0, description="BBox width in source pixels"),
    h: int = Form(..., gt=0, description="BBox height in source pixels"),
    query_id: str = Form("query", min_length=1, max_length=255),
    top_n: int | None = Form(None, ge=1, le=100),
) -> RetrievalResponse:
    image_bytes = await image.read(settings.max_image_bytes + 1)
    frame = await run_in_threadpool(decode_image, image_bytes, image.content_type)
    try:
        result = await run_in_threadpool(
            retrieve_vehicle, query_id, frame, x, y, w, h, top_n
        )
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Vehicle retrieval failed")
        raise HTTPException(
            status_code=503, detail="retrieval database unavailable"
        ) from exc
    return RetrievalResponse(**result)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.app.main:app", host="0.0.0.0", port=8000)
