# Vehicle Image Retrieval / Re-identification

[![Python](https://img.shields.io/badge/Python-3.12-blue)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688)](https://fastapi.tiangolo.com/)
[![PostgreSQL](https://img.shields.io/badge/DB-PostgreSQL%2015-336791)](https://www.postgresql.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

## Overview

This project provides an offline evaluation pipeline and a runnable vehicle retrieval service. Given a full image and a supplied vehicle bounding box (BBox), it computes a visual feature vector, searches a gallery, and returns the highest-scoring gallery annotations or an empty candidate list when the configured rejection threshold is not met.

Automatic license plate recognition (ALPR) can fail when plates are dirty, occluded, affected by weather or glare, or outside the camera view. This project uses visual appearance for vehicle re-identification instead. It does not run ALPR, face recognition, detection, tracking, or use camera/time/location metadata. Input imagery must satisfy the dataset anonymization protocol (blurred plates and faces). A fixed lower-central crop area is masked before feature extraction; this is a heuristic and cannot guarantee exclusion of plate pixels outside that area. The generic visual descriptor also has no face-specific feature, but blurred face pixels within the remaining crop may affect generic color/spatial statistics. Strict pixel-level exclusion would require supplied privacy masks or an approved masking/detection policy.

**In scope:** BBox-based image cropping, embedding generation, gallery search, ranking, and rejection.

**Out of scope:** vehicle detection, ALPR, and tracking. BBox coordinates must be supplied by the caller.

The current descriptor is a deterministic handcrafted baseline, not a trained deep neural re-identification model. No learned accuracy or threshold calibration is claimed.

## Features

- FastAPI inference backend with generated OpenAPI schema and Swagger UI
- Thin browser client for adding gallery examples and querying matches
- PostgreSQL 15 with pgvector cosine-distance search
- Image and BBox validation with HTTP error responses
- L2-normalized, 386-dimensional `float32` embeddings
- Configurable top-N and rejection threshold
- Gallery loading from a CSV + image directory or through the API
- Offline generation of `embeddings.npy`, `submission.csv`, and `candidates.csv`
- Evaluation tooling for mAP, Rank-1/Rank-5, and candidate-proposal metrics
- Docker Compose stack for local demonstration

## Architecture and pipeline

```text
Browser (web service :8080)
        |
        | multipart form request
        v
FastAPI inference service (:8000)
  1. Receiving: decode JPEG/PNG and validate BBox against decoded frame size
  2. Processing: crop, mask lower-central region, produce normalized 386-D vector
  3. Analysis: search PostgreSQL/pgvector using cosine distance
  4. Result: return accepted top-N matches or an empty rejection
        |
        v
PostgreSQL 15 + pgvector (:5432)
```

The browser client calls the API from the user’s browser. The gallery stores visual vectors, source image IDs, BBoxes, and an ingestion `created_at` timestamp. It does not store capture camera/time/location metadata or `vehicle_id`, and those fields are not used for matching.

## Project structure

```text
.
├── backend/
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── app/
│   │   ├── config.py
│   │   ├── embedding.py
│   │   ├── ingest.py
│   │   ├── main.py
│   │   ├── service.py
│   │   ├── storage.py
│   │   └── validation.py
│   └── tests/
├── db/
│   └── init.sql
├── web/
│   ├── Dockerfile
│   └── index.html
├── dataset/                  # Supply test_gallery.csv and images/ here for Compose auto-import
├── benchmarks/
│   └── benchmark_retrieval.py
├── docker-compose.yml
├── .env.example
├── .dockerignore
├── pyproject.toml
├── reid_pipeline.py          # Offline submission-artifact generator
├── evaluate.py
├── requirements.txt
├── requirements-data.txt
├── README_data.md
└── LICENSE
```

## Requirements and launch

### Docker Compose (recommended)

Requirements: Docker Engine/Desktop with the Compose plugin. From the repository root:

```powershell
Copy-Item .env.example .env
docker compose up --build -d
docker compose ps
```

On macOS/Linux, use `cp .env.example .env` for the first command. Compose can also start with its built-in defaults without a `.env` file. The stack exposes:

| Service | Local URL / port | Purpose |
|---|---|---|
| `web` | http://localhost:8080 | Browser client (unprivileged nginx) |
| `api` | http://localhost:8000 | Inference API |
| `api` docs | http://localhost:8000/docs | Swagger UI |
| `api` OpenAPI | http://localhost:8000/openapi.json | OpenAPI JSON |
| `db` | `localhost:5432` | PostgreSQL + pgvector |

Docker Compose starts the database, then the API, then the web client. Health checks are configured for all services. To stop the stack:

```powershell
docker compose down
```

To also remove the persisted database volume (this deletes indexed gallery vectors):

```powershell
docker compose down -v
```

### Local API development

Python 3.12 is used by the service image. Create an environment and install pinned dependencies:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r backend\requirements.txt
```

The local API expects a running PostgreSQL instance with the pgvector extension and a suitable `DATABASE_URL`. The default local URL is `postgresql://appuser:apppass@localhost:5432/vehicle_reid`. Run with:

```powershell
python -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000
```

### Offline artifact generation

The separate offline pipeline uses NumPy and OpenCV:

```powershell
python -m pip install -r requirements-data.txt
python reid_pipeline.py --dataset .\dataset --output .\artifacts --confidence-threshold 0.65
```

This installs the dependencies for artifact generation and evaluation. The commands require the dataset structure described below and write the three challenge artifacts under `artifacts/`.

## Configuration

Compose reads an optional `.env` file in the project root. See [.env.example](.env.example).

| Variable | Default | Used by | Description |
|---|---:|---|---|
| `POSTGRES_DB` | `vehicle_reid` | `db`, `api` | Database name |
| `POSTGRES_USER` | `appuser` | `db`, `api` | Database user |
| `POSTGRES_PASSWORD` | `apppass` | `db`, `api` | Local database password; replace before non-local deployment |
| `POSTGRES_PORT` | `5432` | `db` | Host port for PostgreSQL |
| `API_PORT` | `8000` | `api` | Host port for FastAPI |
| `WEB_PORT` | `8080` | `web`, `api` | Host port for the browser client and allowed CORS origin |
| `REJECTION_THRESHOLD` | `0.65` | `api` | Minimum mapped similarity accepted as a candidate |
| `TOP_N` | `10` | `api` | Maximum number of ranked candidates searched/returned |
| `MAX_IMAGE_BYTES` | `10485760` | `api` | Maximum upload size (10 MiB) |
| `MAX_IMAGE_WIDTH` | `8192` | `api` | Maximum decoded source-frame width |
| `MAX_IMAGE_HEIGHT` | `8192` | `api` | Maximum decoded source-frame height |
| `MAX_IMAGE_PIXELS` | `20000000` | `api` | Maximum decoded source-frame pixel count |
| `DATABASE_POOL_MIN_SIZE` | `1` | `api` | Minimum PostgreSQL connection-pool size |
| `DATABASE_POOL_MAX_SIZE` | `8` | `api` | Bounded maximum number of PostgreSQL connections |
| `DATABASE_POOL_TIMEOUT` | `5` | `api` | Seconds to wait for a pooled connection |
| `DATABASE_URL` | Compose-generated | `api` | PostgreSQL connection URL; override for local execution |
| `CORS_ORIGINS` | localhost web origins | `api` | Comma-separated browser origins permitted to call the API |
| `GALLERY_CSV` | `/data/test_gallery.csv` in Compose | `api` | Gallery annotation CSV loaded on API startup when present |
| `GALLERY_IMAGES_DIR` | `/data/images` in Compose | `api` | Directory of source frames for startup gallery import |

For Compose gallery initialization, mount files with this layout:

```text
dataset/
├── images/
│   ├── <image_id>.jpg
│   └── ...
└── test_gallery.csv
```

Gallery startup import expects `image_id,x,y,w,h` columns. If the CSV or image directory is absent, the API starts with an empty gallery and the browser/API gallery-add operation can populate it. The entire CSV is validated and embedded before its rows are upserted in a single database transaction; import errors prevent normal startup rather than silently skipping invalid rows.

Startup import is idempotent for unchanged annotation IDs, but it does not remove gallery entries omitted from a later CSV. To replace the gallery completely, stop the stack and remove the persisted database volume (`docker compose down -v`), then start again with the replacement gallery data. Legacy vectors in the former `vehicle_gallery` table are not migrated: that schema did not retain BBoxes and therefore cannot distinguish multiple vehicle annotations from one frame.

## Dataset and image input

The offline challenge dataset uses:

```text
dataset/
├── images/                  # Flat directory of JPEG/PNG frames
├── train.csv                # image_id,x,y,w,h,vehicle_id
├── test_query.csv           # image_id,x,y,w,h
└── test_gallery.csv         # image_id,x,y,w,h
```

`x` and `y` are the top-left pixel coordinates, and `w` and `h` are positive pixel dimensions in the original frame. A row represents one vehicle annotation; multiple annotations may have the same `image_id`. The API derives frame width and height from the decoded uploaded image; every BBox must fit completely within those actual dimensions. The API accepts JPEG and PNG. Gallery ingestion resolves either an exact filename in `images/` or an image ID with `.jpg`, `.jpeg`, or `.png` extensions. Configurable limits reject decoded width above 8192, height above 8192, or more than 20,000,000 pixels by default, in addition to the upload byte cap.

The challenge protocol says that train and test `vehicle_id` sets do not overlap. Inference does not read `train.csv` or consume training labels; neither `vehicle_id` nor `camera_id` is inserted into or used by the gallery service. The service searches gallery annotations added through the API or imported from `test_gallery.csv`. Gallery identity is a deterministic SHA-256 of source `image_id` and BBox coordinates, so multiple boxes from the same frame are distinct entries. The source image ID and BBox are retained for annotation traceability, not used as similarity features.

## API

All image endpoints use `multipart/form-data`. Image fields must have `image/jpeg` or `image/png` content type. Swagger UI is served at `/docs`.

| Method and path | Purpose | Success |
|---|---|---|
| `GET /health` | Check API/database availability | `200 {"status":"ok"}` |
| `GET /api/v1/gallery` | Return indexed gallery count | `200 {"items": <count>}` |
| `POST /api/v1/gallery/items` | Add or replace one gallery-annotation embedding | `201` with annotation ID, source image ID, and status |
| `POST /api/v1/retrieve` | Rank gallery candidates for a query crop | `200` with candidates or rejection |

### Add a gallery item

Multipart fields: `image_id`, `image` (JPEG/PNG), `x`, `y`, `w`, and `h`.

Example:

```powershell
curl.exe -X POST http://localhost:8000/api/v1/gallery/items `
  -F "image_id=gallery-001" `
  -F "x=20" -F "y=30" -F "w=300" -F "h=180" `
  -F "image=@.\vehicle.jpg;type=image/jpeg"
```

Example response:

```json
{
  "gallery_id": "a deterministic annotation hash",
  "image_id": "gallery-001",
  "status": "stored"
}
```

`image_id` identifies the source frame; the service derives a stable per-annotation `gallery_id` from that ID and the BBox. Different BBoxes in one frame therefore remain separate gallery entries. Reposting the same frame ID and BBox updates the existing annotation vector.

### Retrieve candidates

Multipart fields: `image` (JPEG/PNG), `x`, `y`, `w`, `h`, optional `query_id`, optional `top_n`.

```powershell
curl.exe -X POST http://localhost:8000/api/v1/retrieve `
  -F "query_id=query-001" `
  -F "x=20" -F "y=30" -F "w=300" -F "h=180" `
  -F "top_n=10" `
  -F "image=@.\query.jpg;type=image/jpeg"
```

Matched response:

```json
{
  "query_id": "query-001",
  "candidates": [
    {
      "gallery_id": "gallery-001",
      "score": 0.82,
      "confidence": 0.91
    }
  ],
  "rejected": false,
  "threshold": 0.65,
  "metric": "cosine"
}
```

Rejection response:

```json
{
  "query_id": "query-001",
  "candidates": [],
  "rejected": true,
  "threshold": 0.65,
  "metric": "cosine"
}
```

`score` is cosine similarity. `confidence = (score + 1) / 2` is a monotonic score mapping, **not a calibrated probability**. A query is rejected when no candidate in the searched top-N reaches `REJECTION_THRESHOLD`. The default `0.65` is retained from the offline baseline, not selected from validation evidence; production/hackathon use should calibrate it on a labeled validation set and report F1/TNR/PR-AUC.

### HTTP errors

| Status | Meaning |
|---:|---|
| `400` | Invalid/empty image bytes, invalid BBox, or BBox outside decoded frame |
| `413` | Upload exceeds `MAX_IMAGE_BYTES` |
| `415` | Unsupported image content type |
| `422` | Missing or malformed multipart fields |
| `503` | Database or retrieval store unavailable |

## Browser client

Open http://localhost:8080 after `docker compose up --build -d`. The page provides forms to register gallery images and submit a query image/BBox. It displays the JSON response, including an empty candidate list when rejected. The browser page expects the API at `http://localhost:8000`; change `apiBase` in [web/index.html](web/index.html) if deploying the two services under different public addresses. Each API request reads the multipart upload asynchronously; decoding, feature extraction, and database access run in the worker thread pool. Database access uses a bounded psycopg connection pool.

## Embedding and similarity

The online and offline pipelines use the same deterministic descriptor:

1. Extract the caller-provided BBox crop.
2. Mask the lower-central crop band (starts at 58% of crop height; spans 18% to 82% of crop width).
3. Resize to `128x128`.
4. Concatenate 96 HSV histogram values, 288 spatial channel mean/std values, and 2 Sobel edge statistics.
5. L2-normalize the 386-value `float32` vector.

PostgreSQL stores the vectors as `VECTOR(386)` in `vehicle_gallery_items`, keyed by per-annotation ID and with source image ID/BBox retained for traceability. Search orders by pgvector cosine distance (`<=>`) and returns `1 - cosine_distance` as similarity. Gallery vectors are normalized before insert/search. Equal scores are ordered by gallery ID for deterministic ranking. This baseline uses visual color/layout/edge statistics; it is not a learned appearance model and may not reliably separate vehicles with similar appearance or large viewpoint/lighting variation.

## Offline submissions and evaluation

Generate challenge output artifacts:

```powershell
python reid_pipeline.py --dataset .\dataset --output .\artifacts --confidence-threshold 0.65
```

The generator produces:

- `submission.csv`: each query followed by up to 10 gallery IDs in descending cosine similarity
- `embeddings.npy`: query embeddings first, then gallery embeddings, preserving CSV row order
- `candidates.csv`: accepted query-gallery pairs with confidence; a query with no accepted pair has no row

The checked-in root-level `embeddings.npy`, `submission.csv`, and `candidates.csv` are legacy files and do **not** correspond to the root CSV sizes or the current 386-dimensional pipeline. Do not submit them as current results. The checkout does not include the required `dataset/images/` frames, so these binaries cannot be correctly regenerated here. Supply the complete organizer dataset before generating submission artifacts.

Use the organizer’s ground-truth file to calculate metrics:

```powershell
python evaluate.py `
  --gt .\test_ground_truth.csv `
  --submission .\artifacts\submission.csv `
  --candidates .\artifacts\candidates.csv `
  --embeddings .\artifacts\embeddings.npy `
  --query .\dataset\test_query.csv `
  --gallery .\dataset\test_gallery.csv `
  --json .\artifacts\report.json
```

The evaluator reports mAP@10, Rank-1, Rank-5, full-ranking mAP/mINP, and candidate precision, recall, F1, TNR, and PR-AUC where the provided ground truth supports them. TNR is not available if there are no open-set queries. There are no benchmark scores claimed in this README.

The repository's current CSV files include an extra `camera_id` column in `train.csv`; the generator only requires the annotation columns plus `vehicle_id` and does not use camera metadata. The checked-in `README_data.md` describes the challenge's `test_query.csv` / `test_gallery.csv` protocol. A generic `test.csv` has no query/gallery distinction and is not accepted as an equivalent input format.

## Tests and quality checks

Install the pinned development dependencies, then run:

```powershell
python -m pip install -r requirements-dev.txt
python -m pytest -q backend\tests
ruff check backend benchmarks reid_pipeline.py
ruff format --check backend benchmarks reid_pipeline.py
```

The standard test suite includes synthetic end-to-end artifact shape/order checks, API validation/rejection checks, and multiple-BBox-per-frame ingestion coverage. `test_storage_integration.py` contains real PostgreSQL + pgvector integration tests; they are skipped unless `TEST_DATABASE_URL` is set, and require a disposable test database because they truncate the service table.

Run the optional real pgvector integration tests only against a disposable database. They truncate the `vehicle_gallery_items` table. With the default local Compose database, create a separate test database first; update the connection URL if you changed the Compose credentials or published port:

```powershell
docker compose exec -T db createdb -U appuser vehicle_reid_test
```

```powershell
$env:TEST_DATABASE_URL = "postgresql://appuser:apppass@localhost:5432/vehicle_reid_test"
python -m pytest -q -m integration backend\tests\test_storage_integration.py
```

Docker stack smoke check:

```powershell
docker compose up --build -d
docker compose ps
curl.exe http://localhost:8000/health
curl.exe http://localhost:8000/openapi.json
```

Expected: three containers report healthy, `/health` returns `{"status":"ok"}`, and OpenAPI JSON includes `/api/v1/retrieve` and `/api/v1/gallery/items`. A retrieval against an empty gallery should return HTTP 200 with `rejected: true` and `candidates: []`.

### Performance measurement

On the evaluation machine, provision the full labeled image set and record the hardware/software versions, then run the batch-size-1 sequential benchmark. It measures per-annotation p50/p95/max latency, throughput, and sampled process RSS; it creates no work queue. The RSS is sampled and may miss transient peaks, so use OS/container peak-memory monitoring for formal OOM verification.

```powershell
python -m pip install -r requirements-dev.txt
python benchmarks\benchmark_retrieval.py `
  --annotations .\dataset\test_gallery.csv `
  --images .\dataset\images
```

No speed or memory figures are asserted here: the dataset images and organizer hardware are not available in this checkout.

## Training, dependencies, and reproducibility

No learned-model training loop or neural checkpoint is used by the retrieval service. The descriptor is deterministic and the API does not use random sampling. The dependency versions are pinned in [backend/requirements.txt](backend/requirements.txt); `requirements.txt` includes the backend and evaluator dependencies. Container images use versioned base tags. For offline deployment, pre-build/pull the images and retain the Python package wheels and source before disconnecting from the network; no model weight download is required.

The original `weights/yolov8m.pt` detector is not used by this service and is deliberately excluded from the re-identification path. No external pretrained weights or third-party image datasets are used by the retrieval implementation.

## Limitations

- Handcrafted descriptors are a lightweight baseline; cross-camera accuracy is not established by supplied test results.
- The linear score mapping is not probabilistic calibration. Threshold `0.65` needs validation against the target gallery distribution.
- PostgreSQL stores per-annotation IDs, source image IDs, BBoxes, and vectors; uploaded source images are not persisted by the API.
- The browser client requires the user to provide BBox coordinates.
- No authentication, authorization, TLS termination, rate limiting, or production secrets management is configured. Do not expose this demo stack directly to an untrusted network.
- Dataset files and organizer ground truth are not bundled here. A populated gallery is needed for positive-match demonstrations.
- Inputs are expected to be pre-anonymized. The geometric lower-centre plate mask has a unit test but is not a license-plate detector and cannot guarantee that all plate pixels outside its mask have zero influence. Likewise, face pixels that remain in the pre-blurred crop may affect generic image statistics; strict face-pixel exclusion requires annotated privacy masks or an approved policy.

## License

Licensed under the MIT License; see [LICENSE](LICENSE).

## Contacts and citation

Created by White_Coffee team.
