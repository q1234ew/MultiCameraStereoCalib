"""Offline capture-quality and model-fit audit; no recalibration or sample filtering."""
from __future__ import annotations

import argparse
import copy
import importlib.metadata
import json
from pathlib import Path

import cv2
import mrcal
import numpy as np

import compare_mrcal as common
import compare_mrcal_spline as spline
import diagnose_mrcal_spline as diagnosis
import test_mrcal_external as external
import validate_mrcal as validation
from validate_mrcal_fov140 import inverse_checks


def image_metrics(image, corners):
    """Metrics describe board ROI only; printed black/white cells confound clipping rates."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    points = np.asarray(corners).reshape(-1, 2).astype(np.float32)
    hull = cv2.convexHull(points)
    mask = np.zeros(gray.shape, np.uint8)
    cv2.fillConvexPoly(mask, np.round(hull).astype(np.int32), 1)
    values = gray[mask > 0]
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    area = cv2.contourArea(hull)
    return {"corner_hull_image_fraction": float(area / gray.size),
            "board_roi_pixels": len(values), "roi_laplacian_variance": float(np.var(laplacian[mask > 0])),
            "roi_gray_p05_p50_p95": np.percentile(values, [5, 50, 95]).tolist(),
            "roi_dark_le5_fraction": float(np.mean(values <= 5)),
            "roi_bright_ge250_fraction": float(np.mean(values >= 250)),
            "interpretation": "relative sharpness/exposure descriptors, not blur or clipping pass/fail"}


def pose_descriptor(objects, corners, intrinsics, camera, board_center):
    pose = common.pose_from_left(objects, np.asarray(corners), intrinsics, side=camera)
    rotation = cv2.Rodrigues(pose[:3])[0]
    normal = rotation[:, 2].copy()
    if normal[2] < 0:
        normal *= -1
    center = rotation @ board_center + pose[3:]
    return {"distance_to_board_center_m": float(np.linalg.norm(center)), "center_depth_m": float(center[2]),
            "tilt_deg": float(np.degrees(np.arccos(np.clip(normal[2], -1, 1)))),
            "horizontal_tilt_deg": float(np.degrees(np.arctan2(normal[0], normal[2]))),
            "vertical_tilt_deg": float(np.degrees(np.arctan2(normal[1], normal[2])))}


def coverage(records, size):
    result = spline.coverage(records, size)
    for camera, side in enumerate(("left", "right")):
        capture_counts = np.zeros((6, 8), int)
        for record in records:
            xy = np.asarray(record[side])
            counts, _, _ = np.histogram2d(xy[:, 1], xy[:, 0], bins=(6, 8), range=((0, size[1]), (0, size[0])))
            capture_counts += counts > 0
        result[camera]["grid_6x8_distinct_capture_counts"] = capture_counts.tolist()
        result[camera]["empty_cells"] = int(np.count_nonzero(capture_counts == 0))
        result[camera]["cells_seen_in_fewer_than_3_captures"] = int(np.count_nonzero(capture_counts < 3))
    return result


def residual_field(inputs, records, objects, size, seed):
    """Common board-pose protocol, frozen camera parameters, residuals in observed pixel bins."""
    residuals = [[], []]
    coordinates = [[], []]
    per_capture = []
    for record in records:
        obj, left, right = common.points(record, objects)
        initial = common.pose_from_left(obj, left, seed)
        pose = spline.fit_pose(inputs, obj, (left, right), initial, (0, 1))
        errors = [spline.project(inputs, obj, pose, c) - xy for c, xy in enumerate((left, right))]
        per_capture.append({"capture_id": record["capture_id"],
                            "joint_pose_point_rms_px": common.error_summary(np.concatenate(errors))["rms_px"]})
        for c, xy in enumerate((left, right)):
            residuals[c].extend(errors[c])
            coordinates[c].extend(xy)
    cameras = []
    for camera in (0, 1):
        error, xy = np.array(residuals[camera]), np.array(coordinates[camera])
        cells = []
        bins = np.minimum((xy / np.array(size) * [8, 6]).astype(int), [7, 5])
        for row in range(6):
            for col in range(8):
                chosen = error[(bins[:, 0] == col) & (bins[:, 1] == row)]
                if len(chosen):
                    mean = chosen.mean(axis=0)
                    cells.append({"row": row, "col": col, "points": len(chosen),
                                  "mean_vector_px": mean.tolist(), "mean_magnitude_px": float(np.linalg.norm(mean)),
                                  "point_rms_px": common.error_summary(chosen)["rms_px"]})
        cameras.append({"camera": camera, "cells": cells,
                        "largest_mean_bias_cells_min20points": sorted([c for c in cells if c["points"] >= 20],
                                                                      key=lambda c: -c["mean_magnitude_px"])[:5]})
    return {"per_capture": per_capture, "cameras": cameras,
            "interpretation": "spatial residual bias is diagnostic; not proof of a specific physical error"}


def masked_uncertainty(inputs, sigma=.2):
    width, height = inputs["imagersizes"][0]
    pixels = np.array([(x * (width - 1), y * (height - 1)) for y in np.linspace(.025, .975, 11)
                       for x in np.linspace(.025, .975, 17)])
    regions = spline.region_masks(pixels, (width, height))
    results = []
    for camera in (0, 1):
        model = mrcal.cameramodel(optimization_inputs=copy.deepcopy(inputs), icam_intrinsics=camera)
        rays = mrcal.unproject(pixels, *model.intrinsics(), normalize=True)
        valid = diagnosis.valid_inverse(rays, mrcal.project(rays, *model.intrinsics()), pixels)
        item = {"camera": camera, "query_count": len(pixels), "valid_inverse_queries": int(valid.sum()),
                "invalid_pixels": pixels[~valid].tolist(), "levels": []}
        for distance in (1., 3., None):
            try:
                covariance = mrcal.projection_uncertainty(
                    rays[valid] if distance is None else rays[valid] * distance, model=model,
                    atinfinity=distance is None, method="mean-pcam", observed_pixel_uncertainty=sigma)
                good = np.isfinite(covariance).all(axis=(-1, -2))
                eigenvalues = np.full((len(covariance), 2), np.nan)
                eigenvalues[good] = np.linalg.eigvalsh(covariance[good])
                good &= (eigenvalues[:, 0] >= -1e-8)
                magnitudes = np.full(len(pixels), np.nan)
                positions = np.flatnonzero(valid)[good]
                magnitudes[positions] = np.sqrt(np.maximum(eigenvalues[good, -1], 0))
                valid_covariance = np.isfinite(magnitudes)
                item["levels"].append({"distance_m": distance if distance else "infinity",
                    "invalid_covariance_after_valid_inverse": int(np.count_nonzero(~good)),
                    "worst_direction_stdev_px": [float(x) if np.isfinite(x) else None for x in magnitudes],
                    "regions": {region: {"queries": int(mask.sum()),
                        "valid_covariance_queries": int(np.count_nonzero(mask & valid_covariance)),
                        "median_px": float(np.nanmedian(magnitudes[mask])) if (mask & valid_covariance).any() else None,
                        "p95_px": float(np.nanpercentile(magnitudes[mask], 95)) if (mask & valid_covariance).any() else None,
                        "max_px": float(np.nanmax(magnitudes[mask])) if (mask & valid_covariance).any() else None}
                        for region, mask in regions.items()}})
            except Exception as exc:
                item["levels"].append({"distance_m": distance if distance else "infinity", "error": str(exc)})
        results.append(item)
    return {"sigma_px_per_coordinate": sigma, "method": "mean-pcam", "cameras": results,
            "meaning": "conditional sensitivity under assumed iid noise; no real noise or stereo range accuracy guarantee"}


def summarize(report):
    lines = ["# test1 采图质量与模型适配性", "",
             "不重标定、不删帧、不调模型；沿用 16/6/2 划分，采图分析覆盖全部提供图片。", "",
             "## 图片和姿态", ""]
    for camera in (0, 1):
        frames = [c for r in report["images"] for c in r["cameras"] if c["camera"] == camera]
        sharpness = [f["image"]["roi_laplacian_variance"] for f in frames]
        distance = [f["pose"]["distance_to_board_center_m"] for f in frames]
        tilts = [f["pose"]["tilt_deg"] for f in frames]
        lines.append(f"- 目 {camera}：ROI 清晰度描述值 {min(sharpness):.1f}–{max(sharpness):.1f}，"
                     f"板中心距离 {min(distance):.3f}–{max(distance):.3f}m，倾斜 {min(tilts):.1f}–{max(tilts):.1f}°。")
    lines += ["", "清晰度/灰度指标受板的大小、黑白图案和照明影响，不以固定阈值直接判模糊或过曝。", "",
              "## 模型条件不确定度，1m、σ=0.2px", "",
              "| 模型 | 目 | 有效反投影 | 边缘 σ 中位数 px | 边缘 σ P95 px | 上边缘 σ P95 px |",
              "|---|---:|---:|---:|---:|---:|"]
    for name, result in report["models"].items():
        for camera in result["uncertainty"]["cameras"]:
            level = camera["levels"][0]
            if "regions" not in level:
                lines.append(f"| {name} | {camera['camera']} | {camera['valid_inverse_queries']}/187 | 失败 | — | — |")
                continue
            regions = level["regions"]
            numbers = [regions["edge"]["median_px"], regions["edge"]["p95_px"], regions["top"]["p95_px"]]
            lines.append(f"| {name} | {camera['camera']} | {camera['valid_inverse_queries']}/187 | "
                         + " | ".join("—" if n is None else f"{n:.4f}" for n in numbers) + " |")
    lines += ["", "无效查询单独计数，不填零；表内统计只涵盖有效查询。",
              "协方差不覆盖真实镜头模型误差；没有时间戳证据，不能从离线图像证明曝光同步。",
              "覆盖直方图、每格独立采集数、每帧残差及姿态差见完整 report.json。"]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Output directory must be empty")
    args.output.mkdir(parents=True, exist_ok=True)
    try:
        baseline = json.loads((args.baseline / "report.json").read_text())
        data = common.collect_observations(args.session.resolve(), "rgb_pair_1", args.baseline / "observations.json")
        if data["dataset_hash"] != baseline["dataset_hash"]:
            raise ValueError("Dataset hash mismatch")
        records = [r for r in data["records"] if r["usable"]]
        split = baseline["protocol"]["split"]
        subsets = {name: [r for r in records if r["capture_id"] in ids]
                   for name, ids in split.items() if name in ("train", "validation", "guard")}
        objects, size = np.array(data["object_points_m"]), tuple(data["image_size"])
        model_paths = {name: args.baseline / f"{name}_0.cameramodel" for name in baseline["models"]}
        inputs = {name: diagnosis.load_inputs(path) for name, path in model_paths.items()}
        seed = validation.opencv_model(inputs["opencv5"])
        common.write_json(args.output / "protocol.json", {
            "split": split, "new_fitting": "board poses only, no camera recalibration",
            "image_quality": "board convex-hull ROI, descriptive not acceptance thresholds",
            "uncertainty_sigma": .2, "source_report_sha256": common.digest(args.baseline / "report.json"),
            "synchronization": "unverified; no exposure timestamps in archive"})
        report = {"status": "completed", "dataset_hash": data["dataset_hash"], "split": split,
                  "script_sha256": common.digest(__file__),
                  "helper_sha256": {m.__file__: common.digest(m.__file__) for m in (common, spline, diagnosis, external, validation)},
                  "environment": {n: importlib.metadata.version(n) for n in ("mrcal", "numpy", "scipy", "opencv-python-headless")},
                  "coverage": {"all": coverage(records, size), **{n: coverage(s, size) for n, s in subsets.items()}},
                  "images": [], "models": {}}
        for record in records:
            frame = {"capture_id": record["capture_id"], "common_corners": len(record["ids"]), "cameras": []}
            for camera, side in enumerate(("left", "right")):
                image = cv2.imread(str(args.session / record["paths"][camera]))
                frame["cameras"].append({"camera": camera,
                    "image": image_metrics(image, record[side]),
                    "pose": pose_descriptor(objects[record["ids"]], record[side], seed, camera, objects.mean(axis=0))})
            report["images"].append(frame)
        for name, model in inputs.items():
            print("audit", name, flush=True)
            fingerprint = external.model_fingerprint(model)
            _, _, jacobian, factor = mrcal.optimizer_callback(**model)
            result = {"source_model_sha256": common.digest(model_paths[name]),
                      "inverse": inverse_checks(model, subsets["validation"], inputs["opencv5"]),
                      "uncertainty": masked_uncertainty(model), "knot_support": spline.knot_support(model),
                      "native_rcond_estimate": float(factor.rcond()) if factor else None,
                      "jacobian_finite": bool(np.isfinite(jacobian.data).all()),
                      "residuals": {subset: residual_field(model, rs, objects, size, seed)
                                    for subset, rs in subsets.items()},
                      "monocular_pose_diagnostics": external.monocular_diagnostics(model, records, objects, size, seed)}
            if fingerprint != external.model_fingerprint(model):
                raise ValueError("Audit changed frozen camera parameters")
            report["models"][name] = result
            common.write_json(args.output / "report.partial.json", report)
        common.write_json(args.output / "report.json", report)
        (args.output / "summary.md").write_text(summarize(report), encoding="utf-8")
        print("Completed", args.output, flush=True)
    except Exception as exc:
        common.write_json(args.output / "failure.json", {"status": "failed", "error": str(exc)})
        raise


if __name__ == "__main__":
    main()
