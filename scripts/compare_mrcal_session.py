"""Train fixed OPENCV5/spline configurations on one session and evaluate one common holdout."""
from __future__ import annotations

import argparse
import copy
import importlib.metadata
from pathlib import Path
import sys

import cv2
import mrcal
import numpy as np

import compare_mrcal as common
import compare_mrcal_spline as spline
from validate_mrcal_fov140 import FOV140

MODELS = {"opencv5": "LENSMODEL_OPENCV5", "spline12x8_fov120": spline.SPLINES["spline12x8"],
          "spline16x10_fov140": FOV140}


def training_inputs(train, data, seed):
    """Build solver state from training captures only, with no original-session parameters."""
    height, width = data["board"]["rows"] - 1, data["board"]["cols"] - 1
    observations = np.zeros((2 * len(train), height, width, 3), dtype=float)
    observations[..., 2] = -1
    objects = np.array(data["object_points_m"])
    poses, indices = [], []
    for frame, record in enumerate(train):
        obj, left, right = common.points(record, objects)
        poses.append(common.pose_from_left(obj, left, seed))
        for camera, xy in enumerate((left, right)):
            flat = observations[2 * frame + camera].reshape(-1, 3)
            flat[record["ids"], :2] = xy
            flat[record["ids"], 2] = 1
            indices.append((frame, camera, camera - 1))
    return {"intrinsics": np.array([np.r_[k[0, 0], k[1, 1], k[0, 2], k[1, 2], d]
                                     for k, d in zip(seed["K"], seed["D"])]),
            "rt_cam_ref": np.array([np.r_[cv2.Rodrigues(seed["R"])[0].ravel(), seed["T"]]]),
            "rt_ref_frame": np.array(poses), "observations_board": observations,
            "indices_frame_camintrinsics_camextrinsics": np.array(indices, dtype=np.int32),
            "points": None, "observations_point": None, "indices_point_camintrinsics_camextrinsics": None,
            "lensmodel": "LENSMODEL_OPENCV5", "imagersizes": np.array([data["image_size"]] * 2, dtype=np.int32),
            "calibration_object_spacing": data["board"]["square_length"],
            "do_optimize_intrinsics_core": True, "do_optimize_intrinsics_distortions": True,
            "do_optimize_extrinsics": True, "do_optimize_frames": True,
            "do_optimize_calobject_warp": False, "do_apply_outlier_rejection": False,
            "do_apply_regularization": True, "verbose": False}


def summarize(report):
    lines = ["# test1 会话内固定模型比较", "",
             f"训练 {report['train_captures']} 对、留出 {report['validation_captures']} 对、隔离 {len(report['protocol']['split']['guard'])} 对。",
             "左右物理角色可能反向；保持原文件角色，三种模型统一重新求解相机关系，不自动交换或要求外参方向与旧会话相同。", "",
             "| 模型 | 训练点 RMS px | 留出跨目 RMS px | 边缘 RMS px | 边缘 P95 px | 角区 RMS px |",
             "|---|---:|---:|---:|---:|---:|"]
    for name, model in report["models"].items():
        if model["status"] != "completed":
            lines.append(f"| {name} | 失败 | — | — | — | — |")
            continue
        val = model["validation"]["transfer"]
        values = [model["train"]["joint_pose"]["all"]["rms_px"], val["all"]["rms_px"],
                  val["edge"]["rms_px"], val["edge"]["p95_px"], val["corners"]["rms_px"]]
        lines.append(f"| {name} | " + " | ".join("—" if v is None else f"{v:.4f}" for v in values) + " |")
    lines += ["", "三者共用角点、划分、初始训练数据和评价程序，开启弱正则、关闭异常剔除与板形变。",
              "仅训练集用于相机参数和初始化；验证仅估计板姿态。未用留出结果换网格、改 FOV、删点或重选划分。",
              "边缘为任一方向外侧 20%，角区为两方向同时处于外侧 20%；统一二维点 RMS。",
              "相机物理安装及同步条件仍未确认，少量留出图片上的比较不代表实测测距精度或最终发布验收。"]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Output directory must be empty")
    args.output.mkdir(parents=True, exist_ok=True)
    try:
        data = common.collect_observations(args.session.resolve(), "rgb_pair_1", args.observations)
        train, holdout, split = common.split_records(data["records"], .25, 2)
        protocol = {"models": MODELS, "split": split, "regularization": True,
                    "outlier_rejection": False, "board_warp": False, "edge_fraction": .2,
                    "camera_initialization": "new session training captures only, OpenCV OPENCV5 seed",
                    "labels": "file roles preserved; physical left/right possibly reversed per user",
                    "image_swapping": False, "new_session_parameters_retrained": True,
                    "holdout_use": "board pose fitting and evaluation only, no camera fitting or tuning",
                    "uncertainty_and_parameter_stability": "not evaluated in this first comparison"}
        # Commit the exact split and configurations before running any camera solve.
        common.write_json(args.output / "protocol.json", protocol)
        common.write_json(args.output / "observations.json", data)
        objects, size = np.array(data["object_points_m"]), tuple(data["image_size"])
        print(f"fixed split: training={len(train)}, holdout={len(holdout)}, guard={len(split['guard'])}", flush=True)
        initial = common.solve_opencv(train, objects, size)["opencv_fixed"]
        base, base_solver = spline.solve(training_inputs(train, data, initial))
        candidates = {"opencv5": (base, {"solver": base_solver})}
        for name in ("spline12x8_fov120", "spline16x10_fov140"):
            candidate, init = spline.spline_seed(base, MODELS[name])
            candidates[name] = (candidate, {"initialization": init})
        report = {"status": "completed", "protocol": protocol, "dataset_hash": data["dataset_hash"],
                  "script_sha256": common.digest(__file__),
                  "helper_sha256": {m.__file__: common.digest(m.__file__) for m in (common, spline)},
                  "environment": {n: importlib.metadata.version(n) for n in ("mrcal", "numpy", "scipy", "opencv-python-headless")},
                  "python": sys.version, "train_captures": len(train), "validation_captures": len(holdout),
                  "train_coverage": spline.coverage(train, size), "validation_coverage": spline.coverage(holdout, size),
                  "models": {}}
        for name, (candidate, details) in candidates.items():
            print("train/evaluate", name, flush=True)
            try:
                if name == "opencv5":
                    fitted = candidate
                else:
                    fitted, details["solver"] = spline.solve(candidate)
                training_weights = fitted["observations_board"][..., 2].copy()
                frozen_parameters = (fitted["intrinsics"].tobytes(), fitted["rt_cam_ref"].tobytes())
                # Same training-only OPENCV5 pose seed for all three evaluation paths.
                result = {"status": "completed", "lensmodel": fitted["lensmodel"], **details,
                          "train": spline.evaluate(fitted, train, objects, size, initial),
                          "validation": spline.evaluate(fitted, holdout, objects, size, initial),
                          "intrinsics": fitted["intrinsics"].tolist(), "rt_cam_ref": fitted["rt_cam_ref"].tolist(),
                          "baseline_mm": float(np.linalg.norm(fitted["rt_cam_ref"][0, 3:]) * 1000)}
                if frozen_parameters != (fitted["intrinsics"].tobytes(), fitted["rt_cam_ref"].tobytes()):
                    raise ValueError("Evaluation changed camera parameters")
                np.testing.assert_array_equal(training_weights, fitted["observations_board"][..., 2])
                for camera in (0, 1):
                    mrcal.cameramodel(optimization_inputs=copy.deepcopy(fitted), icam_intrinsics=camera).write(
                        str(args.output / f"{name}_{camera}.cameramodel"))
                report["models"][name] = result
                print(name, "holdout", result["validation"]["transfer"]["all"], flush=True)
            except Exception as exc:
                report["models"][name] = {"status": "failed", "error": str(exc)}
            common.write_json(args.output / "report.partial.json", report)
        common.write_json(args.output / "report.json", report)
        (args.output / "summary.md").write_text(summarize(report), encoding="utf-8")
        print("Completed", args.output, flush=True)
    except Exception as exc:
        common.write_json(args.output / "failure.json", {"status": "failed", "error": str(exc)})
        raise


if __name__ == "__main__":
    main()
