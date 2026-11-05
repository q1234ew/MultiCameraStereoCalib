"""Offline CO-Calib-inspired selection ablation using frozen historical observations.

This is not a CO-Calib reproduction: existing ChArUco detections, rectangular
image mask, common-point caches, and the application's OpenCV solvers are used.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from multicamera.board.charuco_board import CharucoBoard
from multicamera.board.charuco_board import CharucoBoardConfig
from multicamera.board.detector import CharucoDetector
from multicamera.board.detector import DetectionResult
from multicamera.calibration.intrinsic import IntrinsicCalibrator
from multicamera.calibration.models import CameraModel
from multicamera.calibration.selection import image_scores, top_score_indices
from multicamera.calibration.stereo import StereoCalibrator


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def collect_pairs(directory):
    board = CharucoBoard(CharucoBoardConfig())
    detector = CharucoDetector(board)
    records = []
    size = None
    for path in sorted(directory.glob("*_left.png")):
        paths = [path, path.with_name(path.name.replace("_left.png", "_right.png"))]
        detections = []
        for source in paths:
            image = cv2.imread(str(source))
            if image is None:
                raise ValueError(f"Unreadable source {source}")
            current_size = image.shape[1::-1]
            if size is not None and size != current_size:
                raise ValueError("Inconsistent image sizes")
            size = current_size
            detections.append(detector.detect(image))
        if not all(d.valid for d in detections):
            raise ValueError("Undetected pair; explicit exclusion protocol required")
        ids, a, b = np.intersect1d(*(d.charuco_ids.ravel() for d in detections),
                                 return_indices=True)
        records.append({"capture_id": path.stem.removesuffix("_left"),
                        "paths": [p.name for p in paths], "sha256": [digest(p) for p in paths],
                        "ids": ids.tolist(), "left": detections[0].charuco_corners.reshape(-1, 2)[a].tolist(),
                        "right": detections[1].charuco_corners.reshape(-1, 2)[b].tolist(),
                        "usable": len(ids) >= 6})
    return {"board": board.to_dict(), "image_size": list(size), "records": records,
            "object_points_m": board.object_points.astype(float).tolist()}


def frame_metrics(points, objects, size):
    p = np.asarray(points, float).reshape(-1, 2)
    obj = np.asarray(objects, float).reshape(-1, 3)[:, :2]
    center = np.asarray(size, float) / 2
    radius = np.linalg.norm(center)  # Frozen rectangular valid-image approximation.
    vectors = p - center
    axis = vectors.mean(axis=0)
    if np.linalg.norm(axis) <= 1e-6 * radius:
        axis = np.linalg.svd(vectors - vectors.mean(axis=0), full_matrices=False)[2][0]
    else:
        axis /= np.linalg.norm(axis)
    span = float(np.ptp(vectors @ axis) / (2 * radius))
    h, _ = cv2.findHomography(obj, p, 0)
    if h is None or not np.isfinite(h).all():
        raise ValueError("Homography unavailable")
    h /= h[2, 2]
    isotropy = []
    for x, y in obj:
        q = h @ [x, y, 1]
        if abs(q[2]) < 1e-12:
            raise ValueError("Singular homography")
        jac = (h[:2, :2] * q[2] - np.outer(q[:2], h[2, :2])) / q[2] ** 2
        singular = np.linalg.svd(jac, compute_uv=False)
        isotropy.append(float(singular[-1] / singular[0]))
    bins = np.clip((p / size * [8, 6]).astype(int), 0, [7, 5])
    cells = sorted(set((bins[:, 1] * 8 + bins[:, 0]).tolist()))
    projected = cv2.perspectiveTransform(obj.reshape(-1, 1, 2), h).reshape(-1, 2)
    return {"span": span, "iso": float(np.percentile(isotropy, 10)),
            "center_fraction": float(np.mean(np.linalg.norm(vectors, axis=1) <= .2 * radius)),
            "homography_rms_px": float(np.sqrt(np.mean(np.sum((p - projected) ** 2, axis=1)))),
            "cells": cells, "points": len(p)}


def select(records, metrics, policy, budget):
    if budget > len(records):
        raise ValueError("Insufficient training captures")
    if policy == "uniform":
        return np.linspace(0, len(records) - 1, budget, dtype=int).tolist()
    if policy == "corners":
        return sorted(sorted(range(len(records)), key=lambda i: (-len(records[i]["ids"]), i))[:budget])
    if policy in {"radial", "radial_iso"}:
        def score(i):
            radial = min(m["span"] for m in metrics[i])
            iso = min(m["iso"] for m in metrics[i])
            return (radial if policy == "radial" else radial * iso, -i)
        return sorted(sorted(range(len(records)), key=score, reverse=True)[:budget])
    if policy != "coverage_radial_iso":
        raise ValueError(policy)
    # Experimental fixed-budget selector: not an upstream algorithm/threshold.
    counts = np.zeros((2, 48), int)
    selected = []
    for step in range(budget):
        def score(i):
            radial = min(m["span"] for m in metrics[i])
            iso = min(m["iso"] for m in metrics[i])
            gain = sum(float(np.sum(1 / (1 + counts[c, m["cells"]])))
                       for c, m in enumerate(metrics[i]))
            # Two initial quality anchors, then repeated-view coverage gain.
            return ((radial * iso, gain, -i) if step < 2
                    else (gain, radial * iso, -i))
        best = max((i for i in range(len(records)) if i not in selected), key=score)
        selected.append(best)
        for c, m in enumerate(metrics[best]):
            counts[c, m["cells"]] += 1
    return sorted(selected)


def upstream_roles(metrics):
    """Paired-cache subset of upstream default allocation, excluding mono-only data."""
    available = set(range(len(metrics)))
    covered = [set(), set()]
    roles = []
    while available:
        eligible = [i for i in available if all(m["iso"] >= .3 and m["points"] >= 12
                                               for m in metrics[i])]
        if not eligible:
            break
        def key(i):
            return (sum(len(set(m["cells"]) - covered[c]) for c, m in enumerate(metrics[i])),
                    metrics[i][0]["points"], min(m["iso"] for m in metrics[i]),
                    min(m["span"] for m in metrics[i]), -i)
        best = max(eligible, key=key)
        if roles and key(best)[0] == 0:
            break
        roles.append(best)
        available.remove(best)
        for c, m in enumerate(metrics[best]):
            covered[c].update(m["cells"])
    return {"covisible_indices": roles,
            "anchor_eligible_per_camera": [
                [i for i, pair in enumerate(metrics) if pair[c]["span"] >= .25
                 and pair[c]["iso"] >= .5 and pair[c]["center_fraction"] >= .05]
                for c in range(2)],
            "limits": "Paired cache only; no full upstream role allocation or mono-fill reproduction"}


def calibrators(data, records, model, session=None):
    board = CharucoBoard.from_dict(data["board"])
    left, right = [IntrinsicCalibrator(board, model, min_frames=6) for _ in range(2)]
    stereo = StereoCalibrator(board, model, min_frames=6)
    for r in records:
        dets = [DetectionResult(np.asarray(r[side], np.float32).reshape(-1, 1, 2),
                                np.asarray(r["ids"], np.int32).reshape(-1, 1), [], None,
                                tuple(data["image_size"])) for side in ("left", "right")]
        for c, cal in enumerate((left, right)):
            image = cv2.imread(str(session / r["paths"][c])) if session is not None else None
            if session is not None and image is None:
                raise ValueError("Missing quality-baseline source image")
            if not cal.add_detection(dets[c], image=image, frame_id=r["capture_id"]):
                raise ValueError(cal.last_rejection_reason)
        if not stereo.add_detection_pair(*dets, frame_id=r["capture_id"]):
            raise ValueError("Stereo observation rejected")
    return left, right, stereo


def stats(values):
    a = np.asarray(values, float)
    return {"points": len(a), "rms_px": float(np.sqrt(np.mean(a ** 2))),
            "p95_px": float(np.percentile(a, 95))} if len(a) else None


def evaluate(data, records, left, right, result, board):
    errors = [[], []]
    edge_errors = [[], []]
    per_frame = []
    for record in records:
        obj = board.object_points[record["ids"]].astype(float).reshape(-1, 1, 3)
        xy = [np.asarray(record[s], float).reshape(-1, 1, 2) for s in ("left", "right")]
        row = {"capture_id": record["capture_id"]}
        for source in (0, 1):
            target = 1 - source
            intr = (left, right)
            fish = result.model == CameraModel.FISHEYE
            p = cv2.fisheye.undistortPoints(xy[source], intr[source].camera_matrix,
                                          intr[source].dist_coeffs) if fish else xy[source]
            ok, rv, tv = cv2.solvePnP(obj, p, np.eye(3) if fish else intr[source].camera_matrix,
                                     None if fish else intr[source].dist_coeffs)
            if not ok:
                raise ValueError("Held-out source pose failed")
            rot = cv2.Rodrigues(rv)[0]
            if np.any((obj.reshape(-1, 3) @ rot.T + tv.ravel())[:, 2] <= 0):
                raise ValueError("Source points behind camera")
            rt = result.R if source == 0 else result.R.T
            tt = result.T.reshape(3, 1) if source == 0 else -result.R.T @ result.T.reshape(3, 1)
            rot, tv = rt @ rot, rt @ tv + tt
            if np.any((obj.reshape(-1, 3) @ rot.T + tv.ravel())[:, 2] <= 0):
                raise ValueError("Target points behind camera")
            project = cv2.fisheye.projectPoints if fish else cv2.projectPoints
            predicted = project(obj, cv2.Rodrigues(rot)[0], tv, intr[target].camera_matrix,
                                intr[target].dist_coeffs)[0].reshape(-1, 2)
            distances = np.linalg.norm(predicted - xy[target].reshape(-1, 2), axis=1)
            if not np.isfinite(distances).all():
                raise ValueError("Nonfinite held-out prediction")
            normalized = xy[target].reshape(-1, 2) / data["image_size"]
            edge = np.any((normalized < .2) | (normalized > .8), axis=1)
            errors[source].extend(distances.tolist())
            edge_errors[source].extend(distances[edge].tolist())
            row[("left_to_right", "right_to_left")[source]] = stats(distances)
        per_frame.append(row)
    return {"all": stats(errors[0] + errors[1]),
            "edge": stats(edge_errors[0] + edge_errors[1]),
            "left_to_right": stats(errors[0]), "right_to_left": stats(errors[1]),
            "per_frame": per_frame}


def solve(data, train, holdout, model):
    started = time.perf_counter()
    row = {"selected_ids": [r["capture_id"] for r in train]}
    try:
        left, right, stereo = calibrators(data, train, model)
        # Mirror the explicit application's fisheye_common compatibility policy.
        if model == CameraModel.FISHEYE:
            stereo = stereo.copy_with_common_ids()
        li, ri = left.calibrate(), right.calibrate()
        result = stereo.calibrate(li, ri)
        if not all(np.isfinite(a).all() for a in (li.camera_matrix, ri.camera_matrix,
                                                  li.dist_coeffs, ri.dist_coeffs, result.R, result.T)):
            raise ValueError("Nonfinite solution")
        row.update(status="ok", train_rms_px=result.rms_error,
                   baseline_m=float(np.linalg.norm(result.T)), R=result.R.tolist(),
                   T_m=result.T.ravel().tolist(),
                   shared_stereo_ids=stereo.frame_corner_ids[0].tolist(),
                   evaluation=evaluate(data, holdout, li, ri, result, left.board))
    except (cv2.error, ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
        row.update(status="failed", error=str(exc))
    row["seconds"] = time.perf_counter() - started
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--random-repeats", type=int, default=30)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    sources = [
        ("test1", Path("data/results/mrcal_test1_retrained_verified"),
         Path("data/test1/calib_20261019_152613")),
        ("historical", Path("data/results/mrcal_opencv5_final"),
         Path("offline/calib_20260520_104534")),
    ]
    protocol = {"created": "2026-10-28", "opencv": cv2.__version__, "numpy": np.__version__,
                "script_sha256": digest(__file__), "random_repeats": args.random_repeats,
                "seed": 20261028, "budgets": [8, 10, 12, 16],
                "solver_min_frames": 6, "production_intrinsic_min_frames": 10,
                "random_budget": 12, "upstream_commit": "8b894eef46015ae432477268040f32b913b761c1",
                "mask": "full rectangular image, no brightness-derived mask",
                "limits": ["Historical reused holdouts, not blind external metrology",
                           "Cached cross-camera common points, no native mono-fill test",
                           "6-frame diagnostic minimum below production default 10",
                           "Geometry selectors ignore image clarity; quality baseline uses source images",
                           "No threshold tuning against evaluation; no solver or model selection",
                           "coverage_radial_iso is an original fixed-budget heuristic, not CO-Calib"]}
    write(args.output / "protocol.json", protocol)
    pairs_cache = args.output / "pairs_observations.json"
    pair_data = collect_pairs(Path("data/pairs"))
    write(pairs_cache, pair_data)
    pair_ids = [r["capture_id"] for r in pair_data["records"]]
    pair_split = {"strategy": "chronological_tail_with_two_guard_captures",
                  "train": pair_ids[:-6], "guard": pair_ids[-6:-4],
                  "validation": pair_ids[-4:]}
    write(args.output / "pairs_split.json", pair_split)
    sources.append(("pairs_fisheye", None, Path("data/pairs")))
    report = {"protocol": protocol, "datasets": {}}
    logging.disable(logging.CRITICAL)
    for name, directory, session in sources:
        cache = directory / "observations.json" if directory else pairs_cache
        data = json.loads(cache.read_text())
        if directory:
            original = json.loads((directory / "report.json").read_text())
            split = original.get("split", original.get("protocol", {}).get("split"))
        else:
            split = pair_split
        train = [r for r in data["records"] if r["capture_id"] in split["train"] and r["usable"]]
        holdout = [r for r in data["records"] if r["capture_id"] in split["validation"] and r["usable"]]
        if set(r["capture_id"] for r in train) & set(r["capture_id"] for r in holdout):
            raise ValueError("Train/holdout overlap")
        for r in data["records"]:
            for relative, expected in zip(r["paths"], r["sha256"]):
                if digest(session / relative) != expected:
                    raise ValueError("Source image hash changed")
        objects = np.array(data["object_points_m"])
        metrics = [[frame_metrics(r[s], objects[r["ids"]], data["image_size"])
                    for s in ("left", "right")] for r in train]
        left_cal, right_cal, _ = calibrators(data, train, CameraModel.PINHOLE, session)
        quality = []
        for cal in (left_cal, right_cal):
            counts = cal.frame_corner_counts
            quality.append([.55 * c / max(max(counts), 1) + .45 * q
                            for c, q in zip(counts, image_scores(cal))])
        pair_quality = np.minimum(*quality)
        ds = {"cache_sha256": digest(cache),
              "report_sha256": digest(directory / "report.json") if directory else None,
              "split": split, "train_usable_count": len(train), "validation_count": len(holdout),
              "frame_metrics": {r["capture_id"]: m for r, m in zip(train, metrics)},
              "upstream_default_gate_audit": upstream_roles(metrics), "models": {}}
        report["datasets"][name] = ds
        write(args.output / "report.json", report)
        for model in CameraModel:
            runs = []
            ds["models"][model.value] = runs
            selections = [("all", len(train), list(range(len(train))))]
            for budget in protocol["budgets"]:
                if budget > len(train):
                    continue
                for policy in ("uniform", "corners", "radial", "radial_iso", "coverage_radial_iso"):
                    selections.append((policy, budget, select(train, metrics, policy, budget)))
                # Matched-budget version of current score; no implicit 80% keep ratio.
                indices = top_score_indices(pair_quality.tolist(), budget, keep_ratio=0)
                selections.append(("quality_matched", budget, indices))
            rng = np.random.default_rng(protocol["seed"])
            random_budget = min(12, len(train) - 2)
            for repeat in range(args.random_repeats):
                indices = sorted(rng.choice(len(train), random_budget, replace=False).tolist())
                selections.append((f"random_{repeat:03d}", random_budget, indices))
            memo = {}
            for policy, budget, indices in selections:
                key = tuple(indices)
                if key not in memo:
                    memo[key] = solve(data, [train[i] for i in indices], holdout, model)
                row = {"policy": policy, "budget": budget, **memo[key]}
                runs.append(row)
                value = row.get("evaluation", {}).get("all", {}).get("rms_px")
                print(name, model.value, policy, budget, row["status"], value, flush=True)
                write(args.output / "report.json", report)
        for source_record in data["records"]:
            for relative, expected in zip(source_record["paths"], source_record["sha256"]):
                if digest(session / relative) != expected:
                    raise ValueError("Source changed during experiment")
        ds["source_images_verified_after"] = True
        write(args.output / "report.json", report)
    print("Report:", args.output / "report.json", flush=True)


if __name__ == "__main__":
    main()
