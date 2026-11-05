#!/usr/bin/env python3
"""Estimate fixed landmark-board extrinsics from a handheld AprilTag video."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


TAG_DICT = cv2.aruco.DICT_APRILTAG_36h11


def load_boards(board_dir: Path) -> tuple[dict[int, str], dict[int, np.ndarray], dict[str, list[int]]]:
    tag_to_board: dict[int, str] = {}
    tag_to_corners: dict[int, np.ndarray] = {}
    board_to_tags: dict[str, list[int]] = {}
    for path in sorted(board_dir.glob("*.json")):
        meta = json.loads(path.read_text(encoding="utf-8"))
        if "tags" not in meta:
            continue
        board = path.stem
        board_to_tags[board] = []
        for tag in meta["tags"]:
            tag_id = int(tag["id"])
            tag_to_board[tag_id] = board
            tag_to_corners[tag_id] = np.asarray(tag["corners_board_m_top_left_origin"], dtype=np.float64)
            board_to_tags[board].append(tag_id)
    if not tag_to_board:
        raise FileNotFoundError(f"No landmark board JSON files found in {board_dir}")
    return tag_to_board, tag_to_corners, board_to_tags


def make_camera_matrix(width: int, height: int, focal_px: float) -> np.ndarray:
    return np.asarray(
        [
            [focal_px, 0.0, width * 0.5],
            [0.0, focal_px, height * 0.5],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def rt_to_t(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    rmat, _ = cv2.Rodrigues(rvec)
    t = np.eye(4, dtype=np.float64)
    t[:3, :3] = rmat
    t[:3, 3] = tvec.reshape(3)
    return t


def invert_t(t: np.ndarray) -> np.ndarray:
    inv = np.eye(4, dtype=np.float64)
    inv[:3, :3] = t[:3, :3].T
    inv[:3, 3] = -inv[:3, :3] @ t[:3, 3]
    return inv


def rotation_angle_deg(r: np.ndarray) -> float:
    cos_theta = (float(np.trace(r)) - 1.0) * 0.5
    cos_theta = max(-1.0, min(1.0, cos_theta))
    return math.degrees(math.acos(cos_theta))


def rotation_to_quat(r: np.ndarray) -> np.ndarray:
    q = np.empty(4, dtype=np.float64)
    trace = np.trace(r)
    if trace > 0:
        s = math.sqrt(trace + 1.0) * 2.0
        q[0] = 0.25 * s
        q[1] = (r[2, 1] - r[1, 2]) / s
        q[2] = (r[0, 2] - r[2, 0]) / s
        q[3] = (r[1, 0] - r[0, 1]) / s
    else:
        idx = int(np.argmax(np.diag(r)))
        if idx == 0:
            s = math.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2.0
            q[0] = (r[2, 1] - r[1, 2]) / s
            q[1] = 0.25 * s
            q[2] = (r[0, 1] + r[1, 0]) / s
            q[3] = (r[0, 2] + r[2, 0]) / s
        elif idx == 1:
            s = math.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2.0
            q[0] = (r[0, 2] - r[2, 0]) / s
            q[1] = (r[0, 1] + r[1, 0]) / s
            q[2] = 0.25 * s
            q[3] = (r[1, 2] + r[2, 1]) / s
        else:
            s = math.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2.0
            q[0] = (r[1, 0] - r[0, 1]) / s
            q[1] = (r[0, 2] + r[2, 0]) / s
            q[2] = (r[1, 2] + r[2, 1]) / s
            q[3] = 0.25 * s
    return q / np.linalg.norm(q)


def quat_to_rotation(q: np.ndarray) -> np.ndarray:
    q = q / np.linalg.norm(q)
    w, x, y, z = q
    return np.asarray(
        [
            [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
            [2 * x * y + 2 * z * w, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * x * w],
            [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x * x - 2 * y * y],
        ],
        dtype=np.float64,
    )


def average_rotation(rotations: list[np.ndarray]) -> np.ndarray:
    quats = []
    ref = None
    for r in rotations:
        q = rotation_to_quat(r)
        if ref is None:
            ref = q
        elif np.dot(ref, q) < 0:
            q = -q
        quats.append(q)
    a = np.zeros((4, 4), dtype=np.float64)
    for q in quats:
        a += np.outer(q, q)
    vals, vecs = np.linalg.eigh(a)
    q_avg = vecs[:, int(np.argmax(vals))]
    if q_avg[0] < 0:
        q_avg = -q_avg
    return quat_to_rotation(q_avg)


def reprojection_error(obj: np.ndarray, img: np.ndarray, rvec: np.ndarray, tvec: np.ndarray, k: np.ndarray) -> float:
    projected, _ = cv2.projectPoints(obj, rvec, tvec, k, None)
    err = np.linalg.norm(projected.reshape(-1, 2) - img.reshape(-1, 2), axis=1)
    return float(np.mean(err))


def collect_detections(
    video: Path,
    tag_to_board: dict[int, str],
    tag_to_corners: dict[int, np.ndarray],
    stride: int,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    dictionary = cv2.aruco.getPredefinedDictionary(TAG_DICT)
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    detector = cv2.aruco.ArucoDetector(dictionary, params)

    detections: list[dict[str, object]] = []
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % stride != 0:
            frame_idx += 1
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids_arr, _ = detector.detectMarkers(gray)
        per_board: dict[str, dict[str, object]] = {}
        if ids_arr is not None:
            for tag_id, img_corners in zip(ids_arr.flatten().tolist(), corners):
                tag_id = int(tag_id)
                if tag_id not in tag_to_board:
                    continue
                board = tag_to_board[tag_id]
                if board not in per_board:
                    per_board[board] = {"object_points": [], "image_points": [], "tag_ids": []}
                per_board[board]["object_points"].extend(tag_to_corners[tag_id].tolist())
                per_board[board]["image_points"].extend(img_corners.reshape(4, 2).astype(np.float64).tolist())
                per_board[board]["tag_ids"].append(tag_id)
        if per_board:
            detections.append({"frame": frame_idx, "time_s": frame_idx / fps if fps else None, "boards": per_board})
        frame_idx += 1
    cap.release()
    info = {
        "width": width,
        "height": height,
        "fps": fps,
        "frame_count": frame_count,
        "duration_s": frame_count / fps if fps else None,
    }
    return detections, info


def solve_board_poses(
    detections: list[dict[str, object]],
    k: np.ndarray,
    min_tags: int,
    max_reproj_px: float,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    solved: list[dict[str, object]] = []
    stats = {"attempted": 0, "accepted": 0, "rejected_reprojection": 0, "rejected_points": 0}
    for det in detections:
        frame_poses = {}
        boards = det["boards"]
        assert isinstance(boards, dict)
        for board, data in boards.items():
            tag_ids = sorted(set(int(x) for x in data["tag_ids"]))
            obj = np.asarray(data["object_points"], dtype=np.float64)
            img = np.asarray(data["image_points"], dtype=np.float64)
            if len(tag_ids) < min_tags or len(obj) < 4:
                stats["rejected_points"] += 1
                continue
            stats["attempted"] += 1
            ok, rvec, tvec = cv2.solvePnP(obj, img, k, None, flags=cv2.SOLVEPNP_ITERATIVE)
            if not ok:
                stats["rejected_reprojection"] += 1
                continue
            err = reprojection_error(obj, img, rvec, tvec, k)
            if err > max_reproj_px:
                stats["rejected_reprojection"] += 1
                continue
            frame_poses[board] = {
                "T_cam_board": rt_to_t(rvec, tvec),
                "reprojection_error_px": err,
                "tag_ids": tag_ids,
            }
            stats["accepted"] += 1
        if frame_poses:
            solved.append({"frame": det["frame"], "time_s": det["time_s"], "poses": frame_poses})
    return solved, stats


def pairwise_relatives(solved: list[dict[str, object]]) -> dict[tuple[str, str], list[dict[str, object]]]:
    pairs: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for frame in solved:
        poses = frame["poses"]
        assert isinstance(poses, dict)
        boards = sorted(poses)
        for i, a in enumerate(boards):
            for b in boards[i + 1 :]:
                ta = poses[a]["T_cam_board"]
                tb = poses[b]["T_cam_board"]
                assert isinstance(ta, np.ndarray)
                assert isinstance(tb, np.ndarray)
                t_a_b = invert_t(ta) @ tb
                t_b_a = invert_t(tb) @ ta
                pairs[(a, b)].append(
                    {
                        "frame": frame["frame"],
                        "time_s": frame["time_s"],
                        "T_a_b": t_a_b,
                        "T_b_a": t_b_a,
                        "err_a_px": poses[a]["reprojection_error_px"],
                        "err_b_px": poses[b]["reprojection_error_px"],
                    }
                )
    return pairs


def aggregate_transforms(samples: list[np.ndarray]) -> tuple[np.ndarray, dict[str, float]]:
    translations = np.asarray([t[:3, 3] for t in samples], dtype=np.float64)
    med_t = np.median(translations, axis=0)
    distances = np.linalg.norm(translations - med_t, axis=1)
    keep = distances <= np.percentile(distances, 80) if len(samples) >= 8 else np.ones(len(samples), dtype=bool)
    kept = [sample for sample, ok in zip(samples, keep) if ok]
    translations_kept = np.asarray([t[:3, 3] for t in kept], dtype=np.float64)
    rotations = [t[:3, :3] for t in kept]

    out = np.eye(4, dtype=np.float64)
    out[:3, 3] = np.median(translations_kept, axis=0)
    out[:3, :3] = average_rotation(rotations)

    rot_errors = [rotation_angle_deg(out[:3, :3].T @ r) for r in rotations]
    trans_errors = [float(np.linalg.norm(t[:3, 3] - out[:3, 3])) for t in kept]
    stats = {
        "samples": float(len(samples)),
        "kept_samples": float(len(kept)),
        "translation_median_m": float(np.linalg.norm(out[:3, 3])),
        "translation_mad_m": float(np.median(trans_errors)) if trans_errors else 0.0,
        "translation_p90_m": float(np.percentile(trans_errors, 90)) if trans_errors else 0.0,
        "rotation_median_deg": float(np.median(rot_errors)) if rot_errors else 0.0,
        "rotation_p90_deg": float(np.percentile(rot_errors, 90)) if rot_errors else 0.0,
    }
    return out, stats


def evaluate_focal(
    detections: list[dict[str, object]],
    width: int,
    height: int,
    focal: float,
    min_tags: int,
    max_reproj_px: float,
) -> dict[str, float]:
    k = make_camera_matrix(width, height, focal)
    solved, _ = solve_board_poses(detections, k, min_tags, max_reproj_px)
    pairs = pairwise_relatives(solved)
    score = 0.0
    pair_count = 0
    sample_count = 0
    for samples in pairs.values():
        if len(samples) < 5:
            continue
        transforms = [s["T_a_b"] for s in samples]
        _avg, stats = aggregate_transforms(transforms)
        score += stats["translation_p90_m"] + 0.01 * stats["rotation_p90_deg"]
        pair_count += 1
        sample_count += len(samples)
    if pair_count == 0:
        score = float("inf")
    else:
        score /= pair_count
    return {"focal_px": focal, "score": score, "pair_count": float(pair_count), "pair_samples": float(sample_count)}


def choose_focal(
    detections: list[dict[str, object]],
    width: int,
    height: int,
    min_tags: int,
    max_reproj_px: float,
    user_focal: float | None,
) -> tuple[float, list[dict[str, float]]]:
    if user_focal is not None:
        return user_focal, []
    lo = max(width, height) * 0.55
    hi = max(width, height) * 1.85
    coarse = np.linspace(lo, hi, 18)
    coarse_scores = [evaluate_focal(detections, width, height, float(f), min_tags, max_reproj_px) for f in coarse]
    best = min(coarse_scores, key=lambda x: x["score"])
    span = (hi - lo) / 17.0
    fine = np.linspace(max(300.0, best["focal_px"] - span), best["focal_px"] + span, 13)
    fine_scores = [evaluate_focal(detections, width, height, float(f), min_tags, max_reproj_px) for f in fine]
    scores = sorted(coarse_scores + fine_scores, key=lambda x: x["focal_px"])
    best = min(scores, key=lambda x: x["score"])
    return float(best["focal_px"]), scores


def build_world_transforms(
    pair_aggregates: dict[tuple[str, str], dict[str, object]],
    ref_board: str,
) -> tuple[dict[str, np.ndarray], list[str]]:
    adjacency: dict[str, list[tuple[str, np.ndarray, float]]] = defaultdict(list)
    for (a, b), data in pair_aggregates.items():
        t_a_b = data["T_a_b"]
        t_b_a = data["T_b_a"]
        confidence = float(data["stats_a_b"]["kept_samples"])
        assert isinstance(t_a_b, np.ndarray)
        assert isinstance(t_b_a, np.ndarray)
        adjacency[a].append((b, t_a_b, confidence))
        adjacency[b].append((a, t_b_a, confidence))

    world = {ref_board: np.eye(4, dtype=np.float64)}
    visited_order = [ref_board]
    queue = [ref_board]
    while queue:
        current = queue.pop(0)
        neighbors = sorted(adjacency[current], key=lambda x: -x[2])
        for nxt, t_current_nxt, _confidence in neighbors:
            if nxt in world:
                continue
            world[nxt] = world[current] @ t_current_nxt
            visited_order.append(nxt)
            queue.append(nxt)
    return world, visited_order


def serialise_t(t: np.ndarray) -> list[list[float]]:
    return [[float(v) for v in row] for row in t.tolist()]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--boards", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--ref-board", default="landmark_a4_asym_board_01_ids100-105")
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--focal-px", type=float, default=None)
    parser.add_argument("--min-tags", type=int, default=2)
    parser.add_argument("--max-reproj-px", type=float, default=5.0)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    tag_to_board, tag_to_corners, board_to_tags = load_boards(args.boards)
    detections, video_info = collect_detections(args.video, tag_to_board, tag_to_corners, args.stride)

    focal, focal_scores = choose_focal(
        detections,
        int(video_info["width"]),
        int(video_info["height"]),
        args.min_tags,
        args.max_reproj_px,
        args.focal_px,
    )
    k = make_camera_matrix(int(video_info["width"]), int(video_info["height"]), focal)
    solved, solve_stats = solve_board_poses(detections, k, args.min_tags, args.max_reproj_px)
    pairs = pairwise_relatives(solved)

    pair_aggregates: dict[tuple[str, str], dict[str, object]] = {}
    pair_json: dict[str, object] = {}
    for pair, samples in sorted(pairs.items()):
        if len(samples) < 3:
            continue
        t_ab, stats_ab = aggregate_transforms([s["T_a_b"] for s in samples])
        t_ba, stats_ba = aggregate_transforms([s["T_b_a"] for s in samples])
        pair_aggregates[pair] = {
            "T_a_b": t_ab,
            "T_b_a": t_ba,
            "stats_a_b": stats_ab,
            "stats_b_a": stats_ba,
        }
        key = f"{pair[0]}__to__{pair[1]}"
        pair_json[key] = {
            "from": pair[0],
            "to": pair[1],
            "T_from_to": serialise_t(t_ab),
            "T_to_from": serialise_t(t_ba),
            "stats_from_to": stats_ab,
            "stats_to_from": stats_ba,
        }

    world, visited_order = build_world_transforms(pair_aggregates, args.ref_board)
    world_json = {
        board: {
            "T_world_board": serialise_t(t),
            "translation_world_m": [float(v) for v in t[:3, 3]],
        }
        for board, t in sorted(world.items())
    }
    missing = sorted(set(board_to_tags) - set(world))

    result = {
        "video": str(args.video),
        "boards_dir": str(args.boards),
        "reference_board": args.ref_board,
        "video_info": video_info,
        "assumptions": {
            "camera_model": "pinhole",
            "principal_point": "image center",
            "distortion": "none",
            "focal_px": focal,
            "note": "Use real phone intrinsics for metric-grade results; this estimate is sufficient for checking landmark connectivity and approximate geometry.",
        },
        "settings": {
            "stride": args.stride,
            "min_tags": args.min_tags,
            "max_reprojection_error_px": args.max_reproj_px,
        },
        "detection_frames": len(detections),
        "solve_stats": solve_stats,
        "focal_search": focal_scores,
        "pairwise_transforms": pair_json,
        "world_transforms": world_json,
        "connected_boards_in_order": visited_order,
        "unconnected_boards": missing,
    }
    out_path = args.out / "board_extrinsics.json"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({
        "focal_px": focal,
        "connected_boards": visited_order,
        "unconnected_boards": missing,
        "pair_count": len(pair_json),
        "output": str(out_path),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
