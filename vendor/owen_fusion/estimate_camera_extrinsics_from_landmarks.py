#!/usr/bin/env python3
"""Estimate Orbbec camera poses in the AprilTag landmark world frame."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
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
        raise FileNotFoundError(f"No landmark board JSON files found in {board_dir}")
    return tag_to_board, tag_to_corners


def load_world_boards(path: Path) -> dict[str, np.ndarray]:
    data = json.loads(path.read_text(encoding="utf-8"))
    world = {}
    for board, payload in data["world_transforms"].items():
        world[board] = np.asarray(payload["T_world_board"], dtype=np.float64)
    return world


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
    # OpenCV rational model order: k1,k2,p1,p2,k3,k4,k5,k6.
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


def rt_to_t(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    rmat, _ = cv2.Rodrigues(rvec)
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = rmat
    out[:3, 3] = tvec.reshape(3)
    return out


def invert_t(t: np.ndarray) -> np.ndarray:
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = t[:3, :3].T
    out[:3, 3] = -out[:3, :3] @ t[:3, 3]
    return out


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


def average_transforms(transforms: list[np.ndarray]) -> tuple[np.ndarray, dict[str, float]]:
    if not transforms:
        raise ValueError("No transforms to average")
    translations = np.asarray([t[:3, 3] for t in transforms], dtype=np.float64)
    out = np.eye(4, dtype=np.float64)
    out[:3, 3] = np.median(translations, axis=0)
    quats = []
    ref = None
    for t in transforms:
        q = rotation_to_quat(t[:3, :3])
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
    out[:3, :3] = quat_to_rotation(q_avg)
    trans_errors = [float(np.linalg.norm(t[:3, 3] - out[:3, 3])) for t in transforms]
    return out, {
        "samples": float(len(transforms)),
        "translation_mad_m": float(np.median(trans_errors)),
        "translation_p90_m": float(np.percentile(trans_errors, 90)),
    }


def reprojection_error(obj: np.ndarray, img: np.ndarray, rvec: np.ndarray, tvec: np.ndarray, K: np.ndarray, dist: np.ndarray) -> float:
    projected, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
    err = np.linalg.norm(projected.reshape(-1, 2) - img.reshape(-1, 2), axis=1)
    return float(np.mean(err))


def detect_image(
    image_path: Path,
    tag_to_board: dict[int, str],
    tag_to_corners: dict[int, np.ndarray],
    K: np.ndarray,
    dist: np.ndarray,
    world_boards: dict[str, np.ndarray],
    min_tags: int,
    max_reproj_px: float,
    annotated_path: Path | None,
) -> dict[str, object]:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(image_path)

    dictionary = cv2.aruco.getPredefinedDictionary(TAG_DICT)
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    detector = cv2.aruco.ArucoDetector(dictionary, params)
    corners, ids_arr, _ = detector.detectMarkers(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))

    per_board: dict[str, dict[str, object]] = defaultdict(lambda: {"object_points": [], "image_points": [], "tag_ids": []})
    if ids_arr is not None:
        for tag_id, img_corners in zip(ids_arr.flatten().tolist(), corners):
            tag_id = int(tag_id)
            if tag_id not in tag_to_board:
                continue
            board = tag_to_board[tag_id]
            if board not in world_boards:
                continue
            per_board[board]["object_points"].extend(tag_to_corners[tag_id].tolist())
            per_board[board]["image_points"].extend(img_corners.reshape(4, 2).astype(np.float64).tolist())
            per_board[board]["tag_ids"].append(tag_id)

    board_results = {}
    world_camera_samples = []
    for board, payload in sorted(per_board.items()):
        tag_ids = sorted(set(int(x) for x in payload["tag_ids"]))
        if len(tag_ids) < min_tags:
            continue
        obj = np.asarray(payload["object_points"], dtype=np.float64)
        img = np.asarray(payload["image_points"], dtype=np.float64)
        ok, rvec, tvec = cv2.solvePnP(obj, img, K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            continue
        err = reprojection_error(obj, img, rvec, tvec, K, dist)
        if err > max_reproj_px:
            continue
        T_cam_board = rt_to_t(rvec, tvec)
        T_world_camera = world_boards[board] @ invert_t(T_cam_board)
        world_camera_samples.append(T_world_camera)
        board_results[board] = {
            "tag_ids": tag_ids,
            "reprojection_error_px": err,
            "T_cam_board": T_cam_board.tolist(),
            "T_world_camera_from_this_board": T_world_camera.tolist(),
        }

    if annotated_path is not None:
        annotated = image.copy()
        if ids_arr is not None:
            cv2.aruco.drawDetectedMarkers(annotated, corners, ids_arr)
        for board, payload in board_results.items():
            pts = []
            for tag_id in payload["tag_ids"]:
                if ids_arr is None:
                    continue
                for detected_id, corner in zip(ids_arr.flatten().tolist(), corners):
                    if int(detected_id) == tag_id:
                        pts.append(corner.reshape(4, 2).mean(axis=0))
            if pts:
                xy = np.mean(np.asarray(pts), axis=0).astype(int)
                cv2.putText(annotated, board.replace("landmark_a4_asym_", ""), tuple(xy), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
        annotated_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(annotated_path), annotated)

    if world_camera_samples:
        T_world_camera, stats = average_transforms(world_camera_samples)
        T_camera_world = invert_t(T_world_camera)
    else:
        T_world_camera, T_camera_world, stats = None, None, {}

    return {
        "image": str(image_path),
        "detected_boards": board_results,
        "T_world_camera": None if T_world_camera is None else T_world_camera.tolist(),
        "T_camera_world": None if T_camera_world is None else T_camera_world.tolist(),
        "translation_world_m": None if T_world_camera is None else [float(v) for v in T_world_camera[:3, 3]],
        "aggregate_stats": stats,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--captures", nargs="+", required=True, type=Path, help="One or more Orbbec raw capture directories.")
    parser.add_argument("--boards", required=True, type=Path)
    parser.add_argument("--board-extrinsics", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--frame", default="000000.png")
    parser.add_argument("--min-tags", type=int, default=2)
    parser.add_argument("--max-reproj-px", type=float, default=4.0)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    tag_to_board, tag_to_corners = load_boards(args.boards)
    world_boards = load_world_boards(args.board_extrinsics)

    cameras = {}
    for capture_dir in args.captures:
        K, dist, camera_meta = camera_matrix(capture_dir / "camera.json")
        serial = camera_meta["device"]["serial_number"]
        image_path = capture_dir / "color" / args.frame
        annotated_path = args.out / f"{capture_dir.name}_{serial}_landmarks.jpg"
        result = detect_image(
            image_path,
            tag_to_board,
            tag_to_corners,
            K,
            dist,
            world_boards,
            args.min_tags,
            args.max_reproj_px,
            annotated_path,
        )
        result["capture_dir"] = str(capture_dir)
        result["camera_json"] = str(capture_dir / "camera.json")
        result["device"] = camera_meta["device"]
        result["color_intrinsic"] = camera_meta["color"]["intrinsic"]
        result["color_distortion"] = camera_meta["color"]["distortion"]
        result["annotated_image"] = str(annotated_path)
        cameras[serial] = result

    serials = sorted(cameras)
    pairwise = {}
    for i, a in enumerate(serials):
        Ta = cameras[a]["T_world_camera"]
        if Ta is None:
            continue
        Ta = np.asarray(Ta, dtype=np.float64)
        for b in serials[i + 1 :]:
            Tb = cameras[b]["T_world_camera"]
            if Tb is None:
                continue
            Tb = np.asarray(Tb, dtype=np.float64)
            T_a_b = invert_t(Ta) @ Tb
            pairwise[f"{a}__to__{b}"] = {
                "from_serial": a,
                "to_serial": b,
                "T_from_to": T_a_b.tolist(),
                "translation_m": [float(v) for v in T_a_b[:3, 3]],
                "distance_m": float(np.linalg.norm(T_a_b[:3, 3])),
            }

    output = {
        "reference": {
            "boards_dir": str(args.boards),
            "board_extrinsics": str(args.board_extrinsics),
            "world_frame": "board_01 landmark frame from board_extrinsics.json",
        },
        "settings": {
            "frame": args.frame,
            "min_tags": args.min_tags,
            "max_reprojection_error_px": args.max_reproj_px,
        },
        "cameras": cameras,
        "pairwise_camera_transforms": pairwise,
    }
    out_path = args.out / "orbbec_camera_extrinsics.json"
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(out_path), "cameras": serials, "pairwise": list(pairwise)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
