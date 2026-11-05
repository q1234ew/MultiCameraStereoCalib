#!/usr/bin/env python3
"""Evaluate deployed Orbbec extrinsics against visible AprilTag landmarks."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np


TAG_DICT = cv2.aruco.DICT_APRILTAG_36h11


def load_boards(board_dir: Path) -> tuple[dict[int, str], dict[int, np.ndarray]]:
    tag_to_board: dict[int, str] = {}
    tag_to_corners: dict[int, np.ndarray] = {}
    for path in sorted(board_dir.glob("*.json")):
        meta = json.loads(path.read_text(encoding="utf-8"))
        if "tags" not in meta:
            continue
        board = path.stem
        for tag in meta["tags"]:
            tag_id = int(tag["id"])
            tag_to_board[tag_id] = board
            tag_to_corners[tag_id] = np.asarray(tag["corners_board_m_top_left_origin"], dtype=np.float64)
    if not tag_to_board:
        raise FileNotFoundError(f"No board JSON files with tags found in {board_dir}")
    return tag_to_board, tag_to_corners


def load_board_world(path: Path) -> dict[str, np.ndarray]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {board: np.asarray(payload["T_world_board"], dtype=np.float64) for board, payload in data["world_transforms"].items()}


def load_camera_world(path: Path) -> dict[str, np.ndarray]:
    data = json.loads(path.read_text(encoding="utf-8"))
    poses = {}
    for serial, payload in data["cameras"].items():
        pose = payload.get("T_world_camera")
        if pose is not None:
            poses[serial] = np.asarray(pose, dtype=np.float64)
    return poses


def camera_matrix(camera_json: Path) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    meta = json.loads(camera_json.read_text(encoding="utf-8"))
    k = meta["color"]["intrinsic"]
    d = meta["color"]["distortion"]
    K = np.asarray(
        [
            [float(k["fx"]), 0.0, float(k["cx"])],
            [0.0, float(k["fy"]), float(k["cy"])],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    dist = np.asarray(
        [
            float(d.get("k1", 0.0)),
            float(d.get("k2", 0.0)),
            float(d.get("p1", 0.0)),
            float(d.get("p2", 0.0)),
            float(d.get("k3", 0.0)),
            float(d.get("k4", 0.0)),
            float(d.get("k5", 0.0)),
            float(d.get("k6", 0.0)),
        ],
        dtype=np.float64,
    )
    return K, dist, meta


def frame_name(index: int) -> str:
    return f"{index:06d}.png"


def invert_t(t: np.ndarray) -> np.ndarray:
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = t[:3, :3].T
    out[:3, 3] = -out[:3, :3] @ t[:3, 3]
    return out


def rt_to_t(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    rmat, _ = cv2.Rodrigues(rvec)
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = rmat
    out[:3, 3] = tvec.reshape(3)
    return out


def rotation_error_deg(a: np.ndarray, b: np.ndarray) -> float:
    delta = a[:3, :3].T @ b[:3, :3]
    cos_theta = (float(np.trace(delta)) - 1.0) * 0.5
    cos_theta = max(-1.0, min(1.0, cos_theta))
    return math.degrees(math.acos(cos_theta))


def reprojection_error(obj: np.ndarray, img: np.ndarray, rvec: np.ndarray, tvec: np.ndarray, K: np.ndarray, dist: np.ndarray) -> float:
    projected, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
    err = np.linalg.norm(projected.reshape(-1, 2) - img.reshape(-1, 2), axis=1)
    return float(np.mean(err))


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=np.float64), p))


def evaluate_capture(
    capture_dir: Path,
    tag_to_board: dict[int, str],
    tag_to_corners: dict[int, np.ndarray],
    board_world: dict[str, np.ndarray],
    camera_world: dict[str, np.ndarray],
    frame_start: int,
    frame_count: int,
    frame_step: int,
    min_tags: int,
    max_reproj_px: float,
    annotated_dir: Path,
) -> dict[str, object]:
    K, dist, camera_meta = camera_matrix(capture_dir / "camera.json")
    serial = camera_meta["device"]["serial_number"]
    deployed_pose = camera_world.get(serial)
    if deployed_pose is None:
        raise ValueError(f"No deployed T_world_camera for {serial}")

    dictionary = cv2.aruco.getPredefinedDictionary(TAG_DICT)
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    detector = cv2.aruco.ArucoDetector(dictionary, params)

    reproj_errors: list[float] = []
    trans_errors: list[float] = []
    rot_errors: list[float] = []
    observations = []
    board_counts: dict[str, int] = {}
    best_frame = None
    best_count = 0

    frame_indices = list(range(frame_start, frame_start + frame_count, max(1, frame_step)))
    for frame_index in frame_indices:
        image_path = capture_dir / "color" / frame_name(frame_index)
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        corners, ids_arr, _ = detector.detectMarkers(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
        per_board: dict[str, dict[str, object]] = {}
        if ids_arr is not None:
            for tag_id, img_corners in zip(ids_arr.flatten().tolist(), corners):
                tag_id = int(tag_id)
                board = tag_to_board.get(tag_id)
                if board is None or board not in board_world:
                    continue
                if board not in per_board:
                    per_board[board] = {"object_points": [], "image_points": [], "tag_ids": []}
                per_board[board]["object_points"].extend(tag_to_corners[tag_id].tolist())
                per_board[board]["image_points"].extend(img_corners.reshape(4, 2).astype(np.float64).tolist())
                per_board[board]["tag_ids"].append(tag_id)

        if sum(len(v["tag_ids"]) for v in per_board.values()) > best_count:
            best_count = sum(len(v["tag_ids"]) for v in per_board.values())
            best_frame = (image.copy(), corners, ids_arr, frame_index)

        for board, payload in sorted(per_board.items()):
            tag_ids = sorted(set(int(x) for x in payload["tag_ids"]))
            if len(tag_ids) < min_tags:
                continue
            obj = np.asarray(payload["object_points"], dtype=np.float64)
            img = np.asarray(payload["image_points"], dtype=np.float64)
            ok, rvec, tvec = cv2.solvePnP(obj, img, K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
            if not ok:
                continue
            err_px = reprojection_error(obj, img, rvec, tvec, K, dist)
            accepted = err_px <= max_reproj_px
            T_cam_board = rt_to_t(rvec, tvec)
            T_world_camera_observed = board_world[board] @ invert_t(T_cam_board)
            trans_err = float(np.linalg.norm(T_world_camera_observed[:3, 3] - deployed_pose[:3, 3]))
            rot_err = rotation_error_deg(deployed_pose, T_world_camera_observed)
            observation = {
                "frame": frame_index,
                "board": board,
                "tag_ids": tag_ids,
                "tag_count": len(tag_ids),
                "reprojection_error_px": err_px,
                "translation_error_m": trans_err,
                "rotation_error_deg": rot_err,
                "accepted": accepted,
            }
            observations.append(observation)
            board_counts[board] = board_counts.get(board, 0) + 1
            if accepted:
                reproj_errors.append(err_px)
                trans_errors.append(trans_err)
                rot_errors.append(rot_err)

    annotated_path = None
    if best_frame is not None and best_frame[2] is not None:
        image, corners, ids_arr, frame_index = best_frame
        cv2.aruco.drawDetectedMarkers(image, corners, ids_arr)
        annotated_dir.mkdir(parents=True, exist_ok=True)
        annotated_path = annotated_dir / f"{serial}_frame{frame_index:06d}_tags.jpg"
        cv2.imwrite(str(annotated_path), image)

    status = "ok" if reproj_errors else "no_accepted_landmark_observations"
    return {
        "serial": serial,
        "capture_dir": str(capture_dir),
        "frames_checked": len(frame_indices),
        "observations_total": len(observations),
        "observations_accepted": len(reproj_errors),
        "status": status,
        "board_observation_counts": board_counts,
        "annotated_best_frame": None if annotated_path is None else str(annotated_path),
        "metrics": {
            "reprojection_error_px_median": percentile(reproj_errors, 50),
            "reprojection_error_px_p90": percentile(reproj_errors, 90),
            "translation_error_m_median": percentile(trans_errors, 50),
            "translation_error_m_p90": percentile(trans_errors, 90),
            "rotation_error_deg_median": percentile(rot_errors, 50),
            "rotation_error_deg_p90": percentile(rot_errors, 90),
        },
        "observations": observations,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--captures", nargs="+", required=True, type=Path)
    parser.add_argument("--boards", required=True, type=Path)
    parser.add_argument("--board-extrinsics", required=True, type=Path)
    parser.add_argument("--camera-extrinsics", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--frame-start", type=int, default=0)
    parser.add_argument("--frame-count", type=int, default=60)
    parser.add_argument("--frame-step", type=int, default=5)
    parser.add_argument("--min-tags", type=int, default=2)
    parser.add_argument("--max-reproj-px", type=float, default=4.0)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    tag_to_board, tag_to_corners = load_boards(args.boards)
    board_world = load_board_world(args.board_extrinsics)
    camera_world = load_camera_world(args.camera_extrinsics)

    camera_results = [
        evaluate_capture(
            capture,
            tag_to_board,
            tag_to_corners,
            board_world,
            camera_world,
            args.frame_start,
            args.frame_count,
            args.frame_step,
            args.min_tags,
            args.max_reproj_px,
            args.out / "annotated",
        )
        for capture in args.captures
    ]

    accepted = sum(int(result["observations_accepted"]) for result in camera_results)
    all_reproj = [
        obs["reprojection_error_px"]
        for result in camera_results
        for obs in result["observations"]
        if obs["accepted"]
    ]
    all_trans = [
        obs["translation_error_m"]
        for result in camera_results
        for obs in result["observations"]
        if obs["accepted"]
    ]
    all_rot = [
        obs["rotation_error_deg"]
        for result in camera_results
        for obs in result["observations"]
        if obs["accepted"]
    ]
    summary = {
        "captures": [str(p) for p in args.captures],
        "boards": str(args.boards),
        "board_extrinsics": str(args.board_extrinsics),
        "camera_extrinsics": str(args.camera_extrinsics),
        "settings": {
            "frame_start": args.frame_start,
            "frame_count": args.frame_count,
            "frame_step": args.frame_step,
            "min_tags": args.min_tags,
            "max_reprojection_error_px": args.max_reproj_px,
        },
        "status": "ok" if accepted else "no_accepted_landmark_observations",
        "accepted_observations": accepted,
        "aggregate_metrics": {
            "reprojection_error_px_median": percentile(all_reproj, 50),
            "reprojection_error_px_p90": percentile(all_reproj, 90),
            "translation_error_m_median": percentile(all_trans, 50),
            "translation_error_m_p90": percentile(all_trans, 90),
            "rotation_error_deg_median": percentile(all_rot, 50),
            "rotation_error_deg_p90": percentile(all_rot, 90),
        },
        "cameras": camera_results,
    }
    out_path = args.out / "summary.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(out_path), "status": summary["status"], "accepted_observations": accepted}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
