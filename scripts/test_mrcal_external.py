"""Evaluate frozen camera models on a new session; never calibrate on test images."""
from __future__ import annotations

import argparse
import copy
import importlib.metadata
import json
from pathlib import Path

import mrcal
import numpy as np

import compare_mrcal as common
import compare_mrcal_spline as spline
import diagnose_mrcal_spline as diagnosis
import validate_mrcal as validation


def model_fingerprint(inputs):
    """Only camera parameters are frozen; board poses are nuisance estimates."""
    return (inputs["lensmodel"], inputs["intrinsics"].tobytes(), inputs["rt_cam_ref"].tobytes())


def monocular_diagnostics(inputs, records, objects, size, seed):
    """Separate monocular residuals from fixed-rig transfer; pose shifts are diagnostics only."""
    all_errors, coordinates, frames = [], [], []
    fixed_rt = inputs["rt_cam_ref"][0]
    for record in records:
        obj, left, right = common.points(record, objects)
        observations = (left, right)
        initial = common.pose_from_left(obj, left, seed)
        poses = [spline.fit_pose(inputs, obj, observations, initial, (camera,)) for camera in range(2)]
        errors = [spline.project(inputs, obj, poses[camera], camera) - observations[camera] for camera in range(2)]
        for error, xy in zip(errors, observations):
            all_errors.extend(error)
            coordinates.extend(xy)
        # Right-only pose was parameterized in the left frame using frozen extrinsics.
        # Undo that parameterization to obtain independent board-to-right and board-to-left.
        right_board = mrcal.compose_rt(fixed_rt, poses[1])
        observed_relative = mrcal.compose_rt(right_board, mrcal.invert_rt(poses[0]))
        difference = mrcal.compose_rt(observed_relative, mrcal.invert_rt(fixed_rt))
        frames.append({"capture_id": record["capture_id"],
                       "monocular": [common.error_summary(e) for e in errors],
                       "independent_pose_relative_rt": observed_relative.tolist(),
                       "rotation_difference_deg": float(np.linalg.norm(difference[:3]) * 180 / np.pi),
                       "translation_vector_difference_mm": float(np.linalg.norm(observed_relative[3:] - fixed_rt[3:]) * 1000)})
    errors, xy = np.array(all_errors), np.array(coordinates)
    rotations = [f["rotation_difference_deg"] for f in frames]
    translations = [f["translation_vector_difference_mm"] for f in frames]
    return {"residuals": {region: common.error_summary(errors[mask])
                          for region, mask in spline.region_masks(xy, size).items()},
            "relative_pose_diagnostic": {"rotation_difference_deg_median": float(np.median(rotations)),
                                         "rotation_difference_deg_p95": float(np.percentile(rotations, 95)),
                                         "translation_vector_difference_mm_median": float(np.median(translations)),
                                         "translation_vector_difference_mm_p95": float(np.percentile(translations, 95)),
                                         "interpretation": "nuisance pose diagnostic only; no extrinsics refit or adoption"},
            "per_capture": frames}


def evaluate_frozen(inputs, records, objects, size, seed):
    before = model_fingerprint(inputs)
    result = {"stereo": spline.evaluate(inputs, records, objects, size, seed),
              "monocular": monocular_diagnostics(inputs, records, objects, size, seed)}
    if model_fingerprint(inputs) != before:
        raise ValueError("Frozen camera parameters changed during external evaluation")
    return result


def summarize(report):
    lines = ["# test1 冻结模型跨会话测试", "",
             "相机安装/对焦/视场是否保持不变：用户不确定。因此这是迁移诊断，不是最终泛化验收。", "",
             f"完整图片对 {report['provided_pairs']}；有效 {report['usable_pairs']}；未使用 {report['unusable_ids']}。",
             "只拟合每次标定板姿态，不重新拟合内参、畸变、双目外参或样条；不调整分辨率、不交换左右。", "",
             "| 模型 | 跨目 RMS px | 边缘跨目 RMS px | 边缘跨目 P95 px | 单目拟合 RMS px | 单目边缘 RMS px |",
             "|---|---:|---:|---:|---:|---:|"]
    for name, result in report["models"].items():
        if result["status"] != "completed":
            lines.append(f"| {name} | 失败 | — | — | — | — |")
            continue
        transfer = result["stereo"]["transfer"]
        mono = result["monocular"]["residuals"]
        lines.append(f"| {name} | {transfer['all']['rms_px']:.4f} | {transfer['edge']['rms_px']:.4f} | "
                     f"{transfer['edge']['p95_px']:.4f} | {mono['all']['rms_px']:.4f} | {mono['edge']['rms_px']:.4f} |")
    lines += ["", "## 每帧独立板姿态的相机关系诊断", "",
              "| 模型 | 相对旋转差中位数 deg | 平移向量差中位数 mm |",
              "|---|---:|---:|"]
    for name, result in report["models"].items():
        if result["status"] == "completed":
            pose = result["monocular"]["relative_pose_diagnostic"]
            lines.append(f"| {name} | {pose['rotation_difference_deg_median']:.4f} | "
                         f"{pose['translation_vector_difference_mm_median']:.4f} |")
    lines += ["", "上述相对姿态差不是实测外参真值，受内参误差、板姿态估计、同步与检测误差影响。",
              "未采用新的相机关系，也没有据此调整冻结模型。跨目误差较大时，不能在设备条件未知的情况下归因于 mrcal 泛化失败。",
              "新图像与原会话没有完全相同的文件，仅能排除字节完全重复，不能证明采集过程独立。"]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True,
                        help="frozen three-model FOV140 comparison directory")
    parser.add_argument("--original-session", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Output directory must be empty")
    args.output.mkdir(parents=True, exist_ok=True)
    try:
        previous = json.loads((args.baseline / "report.json").read_text())
        old_metadata = json.loads((args.original_session / "session.json").read_text())
        new_metadata = json.loads((args.session / "session.json").read_text())
        if old_metadata["board"] != new_metadata["board"]:
            raise ValueError("External board specification differs; frozen test rejected")
        protocol = {"source_dataset_hash": previous["dataset_hash"],
                    "models": previous["protocol"]["models"], "external_session": args.session.name,
                    "rig_state": "unknown: user could not confirm unchanged installation/focus/FOV",
                    "classification": "cross-session transfer diagnostic, not final generalization acceptance",
                    "camera_parameters_frozen": True, "fit_board_poses_only": True,
                    "test_split": "all supplied usable capture pairs; no training on external session",
                    "outlier_rejection": False, "image_rescaling": False, "swap_left_right": False,
                    "tuning": "none", "edge_fraction": .2}
        common.write_json(args.output / "protocol.json", protocol)
        data = common.collect_observations(args.session.resolve(), "rgb_pair_1")
        common.write_json(args.output / "observations.json", data)
        records = [r for r in data["records"] if r["usable"]]
        if len(records) < 2:
            raise ValueError("Insufficient usable external observations")
        original_hashes = {common.digest(p) for p in (args.original_session / "images/rgb_pair_1").glob("*.png")}
        duplicates = [path for r in data["records"] for path, digest in zip(r["paths"], r["sha256"])
                      if digest in original_hashes]
        if duplicates:
            raise ValueError(f"External images duplicate original session files: {duplicates}")
        loaded = {}
        for name, expected in previous["protocol"]["models"].items():
            path = args.baseline / f"{name}_0.cameramodel"
            inputs = diagnosis.load_inputs(path)
            if inputs["lensmodel"] != expected:
                raise ValueError("Frozen model type mismatch")
            if not np.array_equal(inputs["imagersizes"], [data["image_size"]] * 2):
                raise ValueError("Image resolution mismatch; no implicit rescaling allowed")
            loaded[name] = (inputs, path)
        seed = validation.opencv_model(loaded["opencv5"][0])
        report = {"status": "completed", "protocol": protocol, "dataset_hash": data["dataset_hash"],
                  "baseline_report_sha256": common.digest(args.baseline / "report.json"),
                  "script_sha256": common.digest(__file__),
                  "helper_sha256": {m.__file__: common.digest(m.__file__) for m in (common, spline, diagnosis, validation)},
                  "environment": {n: importlib.metadata.version(n) for n in ("mrcal", "numpy", "scipy")},
                  "image_size": data["image_size"], "provided_pairs": len(data["records"]),
                  "usable_pairs": len(records), "unusable_ids": [r["capture_id"] for r in data["records"] if not r["usable"]],
                  "exact_original_file_duplicates": [], "coverage": spline.coverage(records, tuple(data["image_size"])),
                  "models": {}}
        for name, (inputs, path) in loaded.items():
            print("external frozen evaluation", name, flush=True)
            try:
                result = evaluate_frozen(copy.deepcopy(inputs), records, np.array(data["object_points_m"]),
                                          tuple(data["image_size"]), seed)
                result.update(status="completed", lensmodel=inputs["lensmodel"], source_model_sha256=common.digest(path))
                report["models"][name] = result
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
