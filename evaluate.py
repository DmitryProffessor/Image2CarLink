"""Evaluate vehicle-retrieval artifacts against organizer ground truth.

Same-vehicle/same-camera gallery items are excluded before top-k scoring.
Open-set queries are excluded from ranking metrics; ties preserve input order.
"""

import argparse
import csv
import json
import math
import sys

import numpy as np
import pandas as pd

TOP_K = 10


def load_gt(path):
    gt = pd.read_csv(path, dtype={"image_id": str})
    for col in ("image_id", "vehicle_id", "camera_id", "split"):
        if col not in gt.columns:
            sys.exit(f"{path} is missing required column '{col}'")
    if gt[["image_id", "vehicle_id", "camera_id", "split"]].isna().any().any():
        sys.exit(f"{path} contains missing ground-truth values")
    if not gt["split"].isin(("query", "gallery")).all():
        sys.exit(f"{path} has split values other than 'query' or 'gallery'")
    query = gt[gt.split == "query"].set_index("image_id")
    gallery = gt[gt.split == "gallery"].set_index("image_id")
    if query.empty or gallery.empty:
        sys.exit("Ground truth must contain at least one query and one gallery item")
    return query, gallery


def load_submission(path, gallery_ids):
    """Read ranked gallery IDs by query, discarding unknown and repeated IDs."""
    ranked, unknown, dupes = {}, 0, 0
    with open(path, encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        for vals in reader:
            vals = [v.strip() for v in vals if v is not None and v.strip()]
            if not vals:
                continue
            qid, preds = vals[0], vals[1:]

            seen, clean = set(), []
            for p in preds:
                if p not in gallery_ids:
                    unknown += 1
                    continue
                if p in seen:
                    dupes += 1
                    continue
                seen.add(p)
                clean.append(p)
            ranked[qid] = clean

    if unknown:
        print(f"  ! {unknown} submission gallery IDs are not in ground truth; ignored")
    if dupes:
        print(f"  ! {dupes} duplicate gallery IDs in submission rows; ignored")
    return ranked


def load_candidates(path):
    """Read candidate rows and sort each query's results by descending confidence."""
    df = pd.read_csv(path, dtype={"query_id": str, "gallery_id": str})
    for col in ("query_id", "gallery_id", "confidence"):
        if col not in df.columns:
            sys.exit(f"{path} is missing required column '{col}'")
    if df[["query_id", "gallery_id", "confidence"]].isna().any().any():
        sys.exit(f"{path} contains missing candidate values")
    df["query_id"] = df["query_id"].str.strip()
    df["gallery_id"] = df["gallery_id"].str.strip()
    if (df["query_id"].eq("") | df["gallery_id"].eq("")).any():
        sys.exit(f"{path} contains an empty query_id or gallery_id")

    df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce")
    if not np.isfinite(df["confidence"].to_numpy(dtype=np.float64)).all():
        sys.exit(f"{path} contains a non-numeric or non-finite confidence")

    out = {}
    for qid, grp in df.groupby("query_id", sort=False):
        pairs = sorted(
            ((r.gallery_id, float(r.confidence)) for r in grp.itertuples()),
            key=lambda t: -t[1],
        )
        out[str(qid)] = pairs
    return out


def load_embeddings(path, query_csv, gallery_csv):
    """Load query-then-gallery embeddings in the corresponding CSV row order."""
    emb = np.load(path)
    q_df = pd.read_csv(query_csv, dtype={"image_id": str})
    g_df = pd.read_csv(gallery_csv, dtype={"image_id": str})
    for filename, frame in ((query_csv, q_df), (gallery_csv, g_df)):
        if "image_id" not in frame.columns:
            sys.exit(f"{filename} is missing required column 'image_id'")
        if frame["image_id"].isna().any() or frame["image_id"].str.strip().eq("").any():
            sys.exit(f"{filename} contains a missing or empty image_id")
    n_q, n_g = len(q_df), len(g_df)

    if emb.ndim != 2:
        sys.exit(f"embeddings.npy must be 2D; got shape {emb.shape}")
    if emb.shape[0] != n_q + n_g:
        sys.exit(
            f"embeddings.npy has {emb.shape[0]} rows; expected "
            f"{n_q} query + {n_g} gallery = {n_q + n_g}"
        )

    emb = emb.astype(np.float32, copy=False)
    if not np.isfinite(emb).all():
        sys.exit("embeddings.npy contains non-finite values")
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    if np.any(norms == 0):
        sys.exit("embeddings.npy contains a zero-length embedding")
    emb = emb / np.clip(norms, 1e-12, None)
    return emb[:n_q], emb[n_q:], q_df.image_id.tolist(), g_df.image_id.tolist()


def valid_positives(query_row, gallery):
    """Count same-vehicle gallery items from a different camera."""
    same_vid = gallery.vehicle_id == query_row.vehicle_id
    same_cam = gallery.camera_id == query_row.camera_id
    return int((same_vid & ~same_cam).sum())


def strip_junk(ranked_list, query_row, gal_vid, gal_cam):
    """Exclude same-vehicle/same-camera entries while preserving ranking order."""
    out = []
    for gid in ranked_list:
        if gid not in gal_vid:
            continue
        if gal_vid[gid] == query_row.vehicle_id and gal_cam[gid] == query_row.camera_id:
            continue
        out.append(gid)
    return out


def ranking_metrics(query, gallery, ranked, top_k=TOP_K, ranks=(1, 5)):
    if top_k <= 0:
        raise ValueError("top_k must be a positive integer")
    if any(rank <= 0 for rank in ranks):
        raise ValueError("rank cutoffs must be positive integers")

    gal_vid = gallery.vehicle_id.to_dict()
    gal_cam = gallery.camera_id.to_dict()

    aps, hits = [], {k: [] for k in ranks}
    n_openset, n_missing = 0, 0

    for qid, row in query.iterrows():
        n_pos = valid_positives(row, gallery)
        if n_pos == 0:
            n_openset += 1
            continue

        if qid not in ranked:
            n_missing += 1
            aps.append(0.0)
            for k in ranks:
                hits[k].append(False)
            continue

        clean = strip_junk(ranked[qid], row, gal_vid, gal_cam)[:top_k]
        rel = np.array([gal_vid[g] == row.vehicle_id for g in clean], dtype=bool)

        if rel.any():
            cum = np.cumsum(rel)
            prec = cum / (np.arange(len(rel)) + 1)
            aps.append(float((prec * rel).sum() / min(n_pos, top_k)))
        else:
            aps.append(0.0)

        for k in ranks:
            hits[k].append(bool(rel[:k].any()))

    if n_missing:
        print(
            f"  ! {n_missing} queries are missing from submission.csv; scored as AP=0"
        )
    return {
        "n_scored": len(aps),
        "n_openset_excluded": n_openset,
        f"mAP@{top_k}": float(np.mean(aps)) if aps else 0.0,
        **{f"Rank-{k}": float(np.mean(hits[k])) if hits[k] else 0.0 for k in ranks},
    }


def full_ranking_metrics(q_emb, g_emb, q_ids, g_ids, query, gallery):
    """Calculate full-gallery mAP and mINP from embeddings."""
    gal_vid = gallery.vehicle_id.to_dict()
    gal_cam = gallery.camera_id.to_dict()

    g_vid = np.array([gal_vid.get(g, -1) for g in g_ids])
    g_cam = np.array([gal_cam.get(g, -1) for g in g_ids])

    sims = q_emb @ g_emb.T
    aps, inps = [], []

    for i, qid in enumerate(q_ids):
        if qid not in query.index:
            continue
        row = query.loc[qid]
        junk = (g_vid == row.vehicle_id) & (g_cam == row.camera_id)
        keep = ~junk
        if not keep.any():
            continue

        order = np.argsort(-sims[i][keep], kind="stable")
        rel = (g_vid[keep] == row.vehicle_id)[order]
        n_pos = int(rel.sum())
        if n_pos == 0:
            continue

        cum = np.cumsum(rel)
        prec = cum / (np.arange(len(rel)) + 1)
        aps.append(float((prec * rel).sum() / n_pos))

        hardest = int(np.max(np.nonzero(rel)[0])) + 1
        inps.append(n_pos / hardest)

    return {
        "mAP_full": float(np.mean(aps)) if aps else 0.0,
        "mINP": float(np.mean(inps)) if inps else 0.0,
        "n_scored": len(aps),
    }


def candidate_metrics(query, gallery, candidates):
    """Score accept/reject decisions per query using its highest-confidence ID."""
    gal_vid = gallery.vehicle_id.to_dict()
    tp = fp = fn = tn = 0
    fp_openset = 0
    scores, labels = [], []

    for qid, row in query.iterrows():
        has_match = valid_positives(row, gallery) > 0
        returned = candidates.get(qid, [])

        labels.append(1 if has_match else 0)
        scores.append(returned[0][1] if returned else float("-inf"))

        if returned:
            top_gid = returned[0][0]
            correct = gal_vid.get(top_gid, None) == row.vehicle_id
            if has_match and correct:
                tp += 1
            else:
                fp += 1
                if not has_match:
                    fp_openset += 1
        else:
            if has_match:
                fn += 1
            else:
                tn += 1

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    tnr = tn / (tn + fp_openset) if (tn + fp_openset) else float("nan")

    return {
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
        "n_openset_queries": tn + fp_openset,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "TNR": tnr,
        "PR-AUC": pr_auc(np.array(scores), np.array(labels)),
    }


def pr_auc(scores, labels):
    """Calculate average precision for confidence-based open/closed-set scores."""
    finite = np.isfinite(scores)
    if labels.sum() == 0 or not finite.any():
        return float("nan")

    s = np.where(finite, scores, np.min(scores[finite]) - 1.0)
    order = np.argsort(-s, kind="stable")
    y = labels[order]

    cum_tp = np.cumsum(y)
    prec = cum_tp / (np.arange(len(y)) + 1)
    rec = cum_tp / labels.sum()

    ap, prev_rec = 0.0, 0.0
    for p, r in zip(prec, rec):
        ap += p * (r - prev_rec)
        prev_rec = r
    return float(ap)


def positive_int(value):
    """argparse type for options that must be greater than zero."""
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def json_safe(value):
    """Replace non-finite floats with null so reports comply with standard JSON."""
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate vehicle retrieval artifacts against ground truth"
    )
    parser.add_argument("--gt", required=True, help="ground-truth CSV")
    parser.add_argument("--submission", required=True, help="ranked submission CSV")
    parser.add_argument("--candidates", help="optional candidate CSV")
    parser.add_argument("--embeddings", help="optional NumPy embeddings file")
    parser.add_argument("--query", help="query CSV used to generate embeddings")
    parser.add_argument("--gallery", help="gallery CSV used to generate embeddings")
    parser.add_argument("--top-k", type=positive_int, default=TOP_K)
    parser.add_argument("--json", help="optional path for a JSON report")
    args = parser.parse_args()

    query, gallery = load_gt(args.gt)
    print(
        f"ground truth   : {len(query)} query, {len(gallery)} gallery, "
        f"{gallery.vehicle_id.nunique()} vehicle identities"
    )

    report = {}

    print("\n--- Ranking (submission.csv) ---")
    ranked = load_submission(args.submission, set(gallery.index))
    rm = ranking_metrics(query, gallery, ranked, top_k=args.top_k)
    report["ranking"] = rm
    print(f"scored queries   : {rm['n_scored']}")
    if rm["n_openset_excluded"]:
        print(f"open-set excluded: {rm['n_openset_excluded']}")
    print(f"mAP@{args.top_k:<2}          : {rm[f'mAP@{args.top_k}']:.4f}")
    print(f"Rank-1           : {rm['Rank-1']:.4f}")
    print(f"Rank-5           : {rm['Rank-5']:.4f}")

    if args.embeddings:
        if not (args.query and args.gallery):
            parser.error("--embeddings requires both --query and --gallery")
        print("\n--- Full ranking (embeddings.npy, diagnostic) ---")
        q_emb, g_emb, q_ids, g_ids = load_embeddings(
            args.embeddings, args.query, args.gallery
        )
        fm = full_ranking_metrics(q_emb, g_emb, q_ids, g_ids, query, gallery)
        report["full_ranking"] = fm
        print(f"full mAP          : {fm['mAP_full']:.4f}")
        print(f"mINP              : {fm['mINP']:.4f}")

    if args.candidates:
        print("\n--- Candidate acceptance (candidates.csv) ---")
        cands = load_candidates(args.candidates)
        cm = candidate_metrics(query, gallery, cands)
        report["candidates"] = cm
        print(f"TP/FP/FN/TN        : {cm['TP']}/{cm['FP']}/{cm['FN']}/{cm['TN']}")
        print(f"Precision          : {cm['Precision']:.4f}")
        print(f"Recall             : {cm['Recall']:.4f}")
        print(f"F1                 : {cm['F1']:.4f}")
        if cm["n_openset_queries"]:
            tnr = f"{cm['TNR']:.4f}"
        else:
            tnr = "n/a"
        print(f"TNR                : {tnr}")
        pr_auc_value = cm["PR-AUC"]
        pr_auc_text = f"{pr_auc_value:.4f}" if math.isfinite(pr_auc_value) else "n/a"
        print(f"PR-AUC             : {pr_auc_text}")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(
                json_safe(report),
                f,
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )
        print(f"\nReport saved to {args.json}")


if __name__ == "__main__":
    main()
