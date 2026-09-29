from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd
import pytest

from evaluate import (
    candidate_metrics,
    json_safe,
    load_embeddings,
    ranking_metrics,
)


def test_ranking_metrics_excludes_junk_and_open_set_queries() -> None:
    query = pd.DataFrame(
        [
            {"vehicle_id": "vehicle-1", "camera_id": "camera-a"},
            {"vehicle_id": "vehicle-3", "camera_id": "camera-a"},
        ],
        index=["query-match", "query-open"],
    )
    gallery = pd.DataFrame(
        [
            {"vehicle_id": "vehicle-1", "camera_id": "camera-a"},
            {"vehicle_id": "vehicle-2", "camera_id": "camera-a"},
            {"vehicle_id": "vehicle-1", "camera_id": "camera-b"},
        ],
        index=["junk", "negative", "positive"],
    )

    metrics = ranking_metrics(
        query,
        gallery,
        {"query-match": ["junk", "negative", "positive"]},
        top_k=2,
    )

    assert metrics == {
        "n_scored": 1,
        "n_openset_excluded": 1,
        "mAP@2": 0.5,
        "Rank-1": 0.0,
        "Rank-5": 1.0,
    }


def test_ranking_metrics_rejects_non_positive_top_k() -> None:
    query = pd.DataFrame(
        [{"vehicle_id": "vehicle-1", "camera_id": "camera-a"}],
        index=["query"],
    )
    gallery = pd.DataFrame(
        [{"vehicle_id": "vehicle-1", "camera_id": "camera-b"}],
        index=["gallery"],
    )

    with pytest.raises(ValueError, match="positive"):
        ranking_metrics(query, gallery, {}, top_k=0)


def test_candidate_metrics_counts_top_result_per_query() -> None:
    query = pd.DataFrame(
        [
            {"vehicle_id": "vehicle-1", "camera_id": "camera-a"},
            {"vehicle_id": "vehicle-3", "camera_id": "camera-a"},
        ],
        index=["query-match", "query-open"],
    )
    gallery = pd.DataFrame(
        [
            {"vehicle_id": "vehicle-1", "camera_id": "camera-b"},
            {"vehicle_id": "vehicle-2", "camera_id": "camera-b"},
        ],
        index=["positive", "negative"],
    )

    metrics = candidate_metrics(
        query,
        gallery,
        {
            "query-match": [("positive", 0.9), ("negative", 0.8)],
            "query-open": [("negative", 0.7)],
        },
    )

    assert (metrics["TP"], metrics["FP"], metrics["FN"], metrics["TN"]) == (
        1,
        1,
        0,
        0,
    )
    assert metrics["TNR"] == 0.0


def test_json_safe_converts_non_finite_metrics_to_null() -> None:
    safe_report = json_safe({"TNR": float("nan"), "PR-AUC": float("inf")})

    assert safe_report == {"TNR": None, "PR-AUC": None}
    assert json.loads(json.dumps(safe_report, allow_nan=False)) == safe_report


def test_load_embeddings_rejects_non_finite_values(tmp_path) -> None:
    query_csv = tmp_path / "query.csv"
    gallery_csv = tmp_path / "gallery.csv"
    query_csv.write_text("image_id\nquery-1\n", encoding="utf-8")
    gallery_csv.write_text("image_id\ngallery-1\n", encoding="utf-8")
    embeddings = tmp_path / "embeddings.npy"
    np.save(embeddings, np.array([[np.nan], [1.0]], dtype=np.float32))

    with pytest.raises(SystemExit, match="non-finite"):
        load_embeddings(embeddings, query_csv, gallery_csv)


def test_cli_rejects_non_positive_top_k(monkeypatch: pytest.MonkeyPatch) -> None:
    from evaluate import main

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate.py",
            "--gt",
            "gt.csv",
            "--submission",
            "submission.csv",
            "--top-k",
            "0",
        ],
    )

    with pytest.raises(SystemExit) as error:
        main()

    assert error.value.code == 2


def test_cli_generates_report_for_ranking_and_candidates(
    tmp_path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from evaluate import main

    ground_truth = tmp_path / "ground_truth.csv"
    ground_truth.write_text(
        "image_id,vehicle_id,camera_id,split\n"
        "query-match,vehicle-1,camera-a,query\n"
        "query-open,vehicle-3,camera-a,query\n"
        "gallery-positive,vehicle-1,camera-b,gallery\n"
        "gallery-negative,vehicle-2,camera-b,gallery\n",
        encoding="utf-8",
    )
    submission = tmp_path / "submission.csv"
    submission.write_text(
        "query-match,gallery-positive,gallery-negative\nquery-open\n",
        encoding="utf-8",
    )
    candidates = tmp_path / "candidates.csv"
    candidates.write_text(
        "query_id,gallery_id,confidence\n" "query-match,gallery-positive,0.9\n",
        encoding="utf-8",
    )
    report_path = tmp_path / "report.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate.py",
            "--gt",
            str(ground_truth),
            "--submission",
            str(submission),
            "--candidates",
            str(candidates),
            "--json",
            str(report_path),
        ],
    )

    main()

    output = capsys.readouterr().out
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert "mAP@10" in output
    assert report["ranking"]["Rank-1"] == 1.0
    assert report["candidates"]["TNR"] == 1.0
    assert report["candidates"]["PR-AUC"] == 1.0
