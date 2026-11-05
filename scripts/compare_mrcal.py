"""Reproducible, offline OPENCV5 stereo comparison; no application state is written."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
from pathlib import Path
import sys
import time

import cv2
import numpy as np
from scipy.optimize import least_squares

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from multicamera.board.charuco_board import CharucoBoard
from multicamera.board.detector import CharucoDetector

OPENCV_MAX_ITERATIONS = 10000


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def collect_observations(session, pair_name, cache=None):
    """Detect once, retaining the exact same common corner set for both solvers."""
    metadata = json.loads((session / "session.json").read_text(encoding="utf-8"))
    board = CharucoBoard.from_dict(metadata["board"])
    objects = board.object_points.astype(np.float64)
    # Both backends use the first INNER corner as the board origin.
    origin = objects[0].copy()
    objects -= origin
    width, height = board.config.cols - 1, board.config.rows - 1
    expected = np.zeros((height, width, 3))
    expected[..., :2] = np.stack(np.meshgrid(np.arange(width), np.arange(height)), axis=-1)
    expected *= board.config.square_length
    if not np.allclose(objects, expected.reshape(-1, 3), atol=1e-7, rtol=0):
        raise ValueError("ChArUco IDs do not map to the expected row-major regular grid")
    # Use exact grid coordinates rather than OpenCV's float32 board geometry.
    objects = expected.reshape(-1, 3)
    directory = session / "images" / pair_name
    sources = []
    for left in sorted(directory.glob("left_*.png")):
        right = directory / left.name.replace("left_", "right_", 1)
        if not right.is_file():
            raise ValueError(f"Unpaired image: {left}")
        sources.append({"capture_id": left.stem.removeprefix("left_"),
                        "paths": [str(p.relative_to(session)) for p in (left, right)],
                        "sha256": [digest(p) for p in (left, right)]})
    if not sources:
        raise ValueError(f"No image pairs in {directory}")
    identity = {"sources": sources, "board": metadata["board"], "pair_name": pair_name}
    dataset_hash = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    if cache:
        result = json.loads(cache.read_text(encoding="utf-8"))
        if result["dataset_hash"] != dataset_hash or result["schema_version"] != 1:
            raise ValueError("Observation cache does not match source images / board / pair")
        return result
    detector = CharucoDetector(board)
    records, image_size = [], None
    for source in sources:
        detections = []
        for relative in source["paths"]:
            image = cv2.imread(str(session / relative))
            if image is None:
                raise ValueError(f"Unreadable image: {relative}")
            size = (image.shape[1], image.shape[0])
            if image_size is not None and size != image_size:
                raise ValueError("All images must have the same size")
            image_size = size
            detections.append(detector.detect(image))
        left, right = detections
        record = dict(source)
        record["detected_counts"] = [d.num_corners for d in detections]
        if left.valid and right.valid:
            ids, il, ir = np.intersect1d(left.charuco_ids.ravel(), right.charuco_ids.ravel(),
                                       return_indices=True)
            record.update(ids=ids.tolist(),
                          left=left.charuco_corners.reshape(-1, 2)[il].tolist(),
                          right=right.charuco_corners.reshape(-1, 2)[ir].tolist())
        else:
            record.update(ids=[], left=[], right=[])
        record["usable"] = len(record["ids"]) >= 6
        # A point set confined to a line cannot constrain a planar pose.
        if record["usable"]:
            xy = objects[record["ids"], :2]
            record["usable"] = bool(np.linalg.matrix_rank(xy - xy.mean(axis=0)) == 2)
        records.append(record)
        print(f"detect {record['capture_id']}: common={len(record['ids'])}", flush=True)
    return {"schema_version": 1, "dataset_hash": dataset_hash, "pair_name": pair_name,
            "board": metadata["board"], "board_origin_offset_m": origin.tolist(),
            "object_points_m": objects.tolist(), "image_size": list(image_size),
            "detector": {"name": "MultiCamera.CharucoDetector", "opencv": cv2.__version__,
                         "source_sha256": digest(Path(__file__).resolve().parents[1]
                                                 / "src/multicamera/board/detector.py")},
            "records": records}


def split_records(records, validation_fraction=0.25, gap=2):
    """Split whole captures BEFORE validity filtering; keep a chronological guard gap."""
    if not 0 < validation_fraction < 0.5 or gap < 0:
        raise ValueError("Require 0 < validation_fraction < 0.5 and gap >= 0")
    count = max(2, math.ceil(len(records) * validation_fraction))
    boundary = len(records) - count
    train = records[:max(0, boundary - gap)]
    guard = records[max(0, boundary - gap):boundary]
    validation = records[boundary:]
    manifest = {"strategy": "chronological_tail_with_guard_gap",
                "validation_fraction": validation_fraction, "gap_captures": gap,
                "train": [r["capture_id"] for r in train],
                "guard": [r["capture_id"] for r in guard],
                "validation": [r["capture_id"] for r in validation],
                "unusable": [r["capture_id"] for r in records if not r["usable"]]}
    train = [r for r in train if r["usable"]]
    validation = [r for r in validation if r["usable"]]
    if len(train) < 6 or len(validation) < 2:
        raise ValueError("Require >=6 usable training captures and >=2 validation captures")
    return train, validation, manifest


def points(record, objects):
    return (objects[record["ids"]], np.array(record["left"], dtype=float),
            np.array(record["right"], dtype=float))


def solve_opencv(train, objects, size):
    obj, left, right = zip(*(points(r, objects) for r in train))
    obj = [o.astype(np.float32) for o in obj]
    imgs = [[p.astype(np.float32) for p in side] for side in (left, right)]
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER,
                OPENCV_MAX_ITERATIONS, 1e-10)
    started = time.perf_counter()
    singles = [cv2.calibrateCamera(obj, side, size, None, None, criteria=criteria)
               for side in imgs]
    initial = [(s[1].copy(), s[2].reshape(5).copy()) for s in singles]
    initialization_seconds = time.perf_counter() - started
    results = {}
    for mode in ("fixed", "joint"):
        t0 = time.perf_counter()
        flags = cv2.CALIB_FIX_INTRINSIC if mode == "fixed" else cv2.CALIB_USE_INTRINSIC_GUESS
        answer = cv2.stereoCalibrate(obj, imgs[0], imgs[1],
                                    initial[0][0].copy(), initial[0][1].copy(),
                                    initial[1][0].copy(), initial[1][1].copy(),
                                    size, flags=flags, criteria=criteria)
        rms, kl, dl, kr, dr, rotation, translation = answer[:7]
        results[f"opencv_{mode}"] = {
            "K": [kl, kr], "D": [dl.reshape(5), dr.reshape(5)],
            "R": rotation, "T": translation.reshape(3),
            "solver_seconds": time.perf_counter() - t0, "native_rms": rms}
    results["opencv_fixed"]["initial_intrinsics_seconds"] = initialization_seconds
    return results


def pose_from_left(obj, pixels, model, side=0):
    ok, rvec, tvec = cv2.solvePnP(obj, pixels, model["K"][side], model["D"][side],
                                flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        raise ValueError("Board pose estimation failed")
    if np.any((obj @ cv2.Rodrigues(rvec)[0].T + tvec.reshape(3))[:, 2] <= 0):
        raise ValueError("Board pose has points behind the camera")
    return np.r_[rvec.ravel(), tvec.ravel()]


def solve_mrcal(train, objects, data, seed, output):
    import mrcal
    width, height = data["board"]["cols"] - 1, data["board"]["rows"] - 1
    observations = np.zeros((len(train) * 2, height, width, 3), dtype=float)
    observations[..., 2] = -1
    indices = []
    poses = []
    for i, record in enumerate(train):
        obj, left, right = points(record, objects)
        poses.append(pose_from_left(obj, left, seed))
        for camera, xy in enumerate((left, right)):
            flat = observations[2 * i + camera].reshape(-1, 3)
            flat[record["ids"], :2] = xy
            flat[record["ids"], 2] = 1
            indices.append((i, camera, camera - 1))
    intrinsics = np.array([np.r_[k[0, 0], k[1, 1], k[0, 2], k[1, 2], d]
                           for k, d in zip(seed["K"], seed["D"])])
    # Enforce projection compatibility independently of optimizer convergence.
    rays = np.array([[x, y, 1.] for x in (-.7, 0., .7) for y in (-.5, 0., .5)])
    for camera in range(2):
        actual = mrcal.project(rays, "LENSMODEL_OPENCV5", intrinsics[camera])
        expected = cv2.projectPoints(rays, np.zeros(3), np.zeros(3),
                                     seed["K"][camera], seed["D"][camera])[0].reshape(-1, 2)
        np.testing.assert_allclose(actual, expected, atol=1e-8, rtol=1e-12)
    results = {}
    for mode in ("fixed", "joint"):
        inputs = dict(
            intrinsics=intrinsics.copy(),
            rt_cam_ref=np.array([np.r_[cv2.Rodrigues(seed["R"])[0].ravel(), seed["T"]]]),
            rt_ref_frame=np.array(poses),
            observations_board=observations.copy(),
            indices_frame_camintrinsics_camextrinsics=np.array(indices, dtype=np.int32),
            lensmodel="LENSMODEL_OPENCV5", imagersizes=np.array([data["image_size"]] * 2, dtype=np.int32),
            calibration_object_spacing=data["board"]["square_length"],
            do_optimize_intrinsics_core=mode == "joint",
            do_optimize_intrinsics_distortions=mode == "joint",
            do_optimize_extrinsics=True, do_optimize_frames=True,
            do_optimize_calobject_warp=False, do_apply_outlier_rejection=False,
            do_apply_regularization=False)
        t0 = time.perf_counter()
        stats = mrcal.optimize(**inputs)
        elapsed = time.perf_counter() - t0
        params = inputs["intrinsics"]
        kl, kr = [np.array([[p[0], 0., p[2]], [0., p[1], p[3]], [0., 0., 1.]]) for p in params]
        rt = inputs["rt_cam_ref"][0]
        results[f"mrcal_{mode}"] = {
            "K": [kl, kr], "D": [p[4:].copy() for p in params],
            "R": cv2.Rodrigues(rt[:3])[0], "T": rt[3:].copy(),
            "solver_seconds": elapsed,
            "input_missing_corner_slots": int(np.count_nonzero(observations[..., 2] < 0)),
            "new_rejected_corners": int(np.count_nonzero(
                (observations[..., 2] >= 0) & (inputs["observations_board"][..., 2] < 0))),
            "solver_stats": {k: float(v) for k, v in stats.items() if np.isscalar(v)}}
        if not np.array_equal(inputs["observations_board"][..., 2], observations[..., 2]):
            raise ValueError("mrcal unexpectedly changed the observation mask")
        for camera in range(2):
            mrcal.cameramodel(optimization_inputs=inputs, icam_intrinsics=camera).write(
                str(output / f"mrcal_{mode}_{camera}.cameramodel"))
    return results


def project_camera(obj, pose, model, side):
    xyz = obj @ cv2.Rodrigues(pose[:3])[0].T + pose[3:]
    if side:
        xyz = xyz @ model["R"].T + model["T"]
    if np.any(xyz[:, 2] <= 0):
        raise ValueError("Prediction contains points behind the camera")
    return cv2.projectPoints(xyz, np.zeros(3), np.zeros(3),
                             model["K"][side], model["D"][side])[0].reshape(-1, 2)


def stereo_residual(pose, obj, left, right, model):
    return np.concatenate([(project_camera(obj, pose, model, side) - xy).ravel()
                           for side, xy in enumerate((left, right))])


def error_summary(vectors):
    vectors = np.asarray(vectors).reshape(-1, 2)
    if not len(vectors):
        return {"points": 0, "rms_px": None, "median_px": None, "p95_px": None}
    distances = np.linalg.norm(vectors, axis=1)
    return {"points": len(distances), "rms_px": float(np.sqrt(np.mean(distances ** 2))),
            "median_px": float(np.median(distances)), "p95_px": float(np.percentile(distances, 95))}


def evaluate(records, objects, size, model):
    """Refit only a nuisance board pose, never camera parameters, on evaluation captures."""
    all_errors, transfer, edge, vertical, frames = [], [], [], [], []
    r1, r2, p1, p2, *_ = cv2.stereoRectify(
        model["K"][0], model["D"][0], model["K"][1], model["D"][1], size,
        model["R"], model["T"], flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)
    for record in records:
        obj, left, right = points(record, objects)
        left_pose = pose_from_left(obj, left, model)
        right_pose = pose_from_left(obj, right, model, side=1)
        # Left-only board fit -> right prediction; right-only fit -> left prediction.
        forward = project_camera(obj, left_pose, model, 1) - right
        rr = cv2.Rodrigues(right_pose[:3])[0]
        backward_pose = np.r_[cv2.Rodrigues(model["R"].T @ rr)[0].ravel(),
                              model["R"].T @ (right_pose[3:] - model["T"])]
        backward = project_camera(obj, backward_pose, model, 0) - left
        joint_pose = least_squares(stereo_residual, left_pose,
                                  args=(obj, left, right, model), method="lm",
                                  max_nfev=300, ftol=1e-10, xtol=1e-10, gtol=1e-10)
        if not joint_pose.success:
            raise ValueError(f"Evaluation pose did not converge: {record['capture_id']}")
        residual = stereo_residual(joint_pose.x, obj, left, right, model).reshape(2, -1, 2)
        all_errors.extend(residual.reshape(-1, 2))
        transfer.extend(np.concatenate((forward, backward)))
        for side, xy in enumerate((left, right)):
            normalized = xy / np.array(size)
            mask = np.any((normalized < .2) | (normalized > .8), axis=1)
            edge.extend(residual[side][mask])
        rect = [cv2.undistortPoints(xy.reshape(-1, 1, 2), model["K"][c], model["D"][c],
                                    R=r, P=p).reshape(-1, 2)
                for c, xy, r, p in ((0, left, r1, p1), (1, right, r2, p2))]
        dy = rect[0][:, 1] - rect[1][:, 1]
        vertical.extend(dy.tolist())
        frames.append({"capture_id": record["capture_id"], "common_corners": len(obj),
                       "joint_pose": error_summary(residual),
                       "left_fit_right_predict": error_summary(forward),
                       "right_fit_left_predict": error_summary(backward)})
    vertical = np.abs(vertical)
    return {"captures": len(records), "joint_pose": error_summary(all_errors),
            "cross_camera_transfer": error_summary(transfer), "edge_joint_pose": error_summary(edge),
            "rectified_vertical": {"points": len(vertical),
                                   "rms_px": float(np.sqrt(np.mean(vertical ** 2))),
                                   "median_px": float(np.median(vertical)),
                                   "p95_px": float(np.percentile(vertical, 95))},
            "per_capture": frames}


def serializable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {k: serializable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [serializable(v) for v in value]
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument("--pair", default="rgb_pair_1")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--observations", type=Path, help="Reuse a hash-verified detection cache")
    parser.add_argument("--validation-fraction", type=float, default=.25)
    parser.add_argument("--gap", type=int, default=2)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Output directory must be empty; preserve prior experiments")
    args.output.mkdir(parents=True, exist_ok=True)
    try:
        importlib.metadata.version("mrcal")  # Fail before detection if backend is unavailable.
        data = collect_observations(args.session.resolve(), args.pair, args.observations)
        write_json(args.output / "observations.json", data)
        train, validation, split = split_records(data["records"], args.validation_fraction, args.gap)
        write_json(args.output / "split.json", split)
        print(f"usable split: train={len(train)}, validation={len(validation)}, guard={len(split['guard'])}", flush=True)
        objects = np.array(data["object_points_m"])
        size = tuple(data["image_size"])
        models = solve_opencv(train, objects, size)
        print("OpenCV fixed/joint complete", flush=True)
        models.update(solve_mrcal(train, objects, data, models["opencv_fixed"], args.output))
        report = {"status": "passed", "schema_version": 1, "dataset_hash": data["dataset_hash"],
                  "lensmodel": "OPENCV5", "split": split,
                  "environment": {"python": sys.version, "platform": platform.platform(),
                                  "opencv": cv2.__version__, "mrcal": importlib.metadata.version("mrcal"),
                                  "numpy": np.__version__, "scipy": importlib.metadata.version("scipy"),
                                  "opencv_distribution": "opencv-python-headless",
                                  "script_sha256": digest(__file__)},
                  "protocol": {"point_rms": "sqrt(mean(dx^2 + dy^2))",
                               "opencv_termination": {"max_iterations": OPENCV_MAX_ITERATIONS,
                                                      "epsilon": 1e-10},
                               "corner_set": "same common ChArUco IDs for all backends",
                               "outlier_rejection": False, "regularization": False, "board_warp": False,
                               "evaluation_pose": "same scipy LM, board pose only; camera parameters frozen",
                               "transfer": "fit one camera's board pose, predict the other; both directions",
                               "mrcal_seed": "training-only opencv_fixed result, same initial K/D as opencv_joint",
                               "timing": "stereo solve only; excludes shared detection, initialization and evaluation",
                               "uncertainty": "not evaluated in this minimal experiment"},
                  "limitations": ["Single existing session; no independent recapture or measured ground truth",
                                  "Capture indices establish order, not exposure synchronization",
                                  "Guard gap reduces adjacency leakage but cannot prove pose independence",
                                  "Rectified pixel error uses each model's own alpha=0 scale; auxiliary only"],
                  "models": {}}
        for name, model in models.items():
            for array in (*model["K"], *model["D"], model["R"], model["T"]):
                if not np.isfinite(array).all():
                    raise ValueError(f"Nonfinite model: {name}")
            np.testing.assert_allclose(model["R"].T @ model["R"], np.eye(3), atol=1e-8)
            if np.linalg.det(model["R"]) < 0 or np.linalg.norm(model["T"]) <= 0:
                raise ValueError(f"Invalid rotation/baseline: {name}")
            result = serializable(model)
            result["baseline_mm"] = float(np.linalg.norm(model["T"]) * 1000)
            result["train"] = evaluate(train, objects, size, model)
            result["validation"] = evaluate(validation, objects, size, model)
            report["models"][name] = result
            print(name, "validation:", result["validation"]["cross_camera_transfer"], flush=True)
        write_json(args.output / "report.json", report)
        rows = ["# OPENCV5 离线比较", "", f"数据哈希：`{data['dataset_hash']}`", "",
                f"训练 {len(train)} 对；留出验证 {len(validation)} 对；隔离 {len(split['guard'])} 对。",
                "", "| 后端/策略 | 训练点 RMS | 留出点 RMS | 留出跨目预测 RMS | 留出跨目 P95 | 基线 mm | 求解秒 |",
                "|---|---:|---:|---:|---:|---:|---:|"]
        for name, result in report["models"].items():
            val = result["validation"]
            rows.append(f"| {name} | {result['train']['joint_pose']['rms_px']:.4f} | "
                        f"{val['joint_pose']['rms_px']:.4f} | {val['cross_camera_transfer']['rms_px']:.4f} | "
                        f"{val['cross_camera_transfer']['p95_px']:.4f} | {result['baseline_mm']:.3f} | "
                        f"{result['solver_seconds']:.3f} |")
        rows += ["", "误差单位为 px；统一二维点 RMS。留出集只估计板姿态，不更新内外参。",
                 "跨目预测从单目估计板姿态，再预测另一目，合并两个方向。",
                 "", "仅比较相同 OPENCV5、同一角点和同一划分；关闭异常剔除、正则与板形变。",
                 "单会话结果不能证明泛化精度或绝对基线准确；同步和重复姿态仍需独立采集验证。",
                 "本实验不输出质量合格结论，status=passed 仅代表协议与数值检查完成。"]
        (args.output / "summary.md").write_text("\n".join(rows) + "\n", encoding="utf-8")
    except Exception as exc:
        write_json(args.output / "failure.json", {"status": "failed", "error": str(exc)})
        raise


if __name__ == "__main__":
    main()
