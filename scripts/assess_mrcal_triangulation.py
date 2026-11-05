"""Frozen-model holdout triangulation, board geometry and conditional 3-D uncertainty."""
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
import test_mrcal_external as external


def statistics(values):
    values = np.asarray(values, dtype=float)
    if not len(values):
        return {"count": 0, "mean": None, "rms": None, "median_abs": None,
                "p95_abs": None, "max_abs": None}
    if not np.isfinite(values).all():
        raise ValueError("Non-finite metric; invalid samples must be counted explicitly")
    return {"count": len(values), "mean": float(values.mean()),
            "rms": float(np.sqrt(np.mean(values**2))),
            "median_abs": float(np.median(np.abs(values))),
            "p95_abs": float(np.percentile(np.abs(values), 95)),
            "max_abs": float(np.max(np.abs(values)))}


def plane_residuals(points):
    """Orthogonal distance to best-fit plane; no rescaling or board shape fit."""
    centered = points - points.mean(axis=0)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    normal = vt[-1]
    if normal[2] < 0:
        normal = -normal
    return centered @ normal


def pair_geometry(points, objects):
    i, j = np.triu_indices(len(points), k=1)
    delta = points[i] - points[j]
    reconstructed = np.linalg.norm(delta, axis=-1)
    nominal = np.linalg.norm(objects[i] - objects[j], axis=-1)
    if (reconstructed <= 0).any():
        raise ValueError("Degenerate reconstructed board distance")
    return i, j, delta / reconstructed[:, None], nominal, reconstructed - nominal


def pair_variance(covariance, i, j, direction):
    """Distance variance including calibration correlations across board corners."""
    blocks = covariance[i, :, i, :] + covariance[j, :, j, :]
    blocks -= covariance[i, :, j, :] + covariance[j, :, i, :]
    return np.einsum("ni,nij,nj->n", direction, blocks, direction)


def diagonal_blocks(covariance):
    indices = np.arange(covariance.shape[0])
    return covariance[indices, :, indices, :]


def covariance_diagnostics(covariance):
    matrix = covariance.reshape(covariance.shape[0] * 3, -1)
    if not np.isfinite(matrix).all():
        raise ValueError("Non-finite 3-D covariance")
    scale = max(float(np.max(np.abs(matrix))), 1e-20)
    symmetry = float(np.max(np.abs(matrix - matrix.T)) / scale)
    eigenvalues = np.linalg.eigvalsh((matrix + matrix.T) / 2)
    minimum = float(eigenvalues[0])
    if symmetry > 1e-7 or minimum < -scale * 1e-7:
        raise ValueError(f"Invalid covariance: asymmetry={symmetry}, min eigenvalue={minimum}")
    return {"relative_asymmetry": symmetry, "min_eigenvalue_m2": minimum,
            "max_eigenvalue_m2": float(eigenvalues[-1])}


def rigid_board_residuals(points, objects):
    """Diagnostic only: rigid board alignment without scale, not independent depth truth."""
    a, b = objects - objects.mean(axis=0), points - points.mean(axis=0)
    u, _, vt = np.linalg.svd(a.T @ b)
    correction = np.eye(3)
    correction[-1, -1] = np.linalg.det(u @ vt)
    fitted = a @ (u @ correction @ vt) + points.mean(axis=0)
    return points - fitted


def observation_sampling(q, models, predicted, size, samples, seed):
    """Check observation-only depth variance on fixed cameras, center and edge queries."""
    radius = np.max(np.abs(q[:, 0] / np.asarray(size) - .5), axis=-1)
    selected = np.unique([np.argmin(radius), np.argmax(radius)])
    rng = np.random.default_rng(seed)
    noisy = q[selected][None] + rng.normal(0, .2, size=(samples, len(selected), 2, 2))
    sampled = mrcal.triangulate(noisy, models, method=mrcal.triangulate_leecivera_mid2)
    valid = np.isfinite(sampled).all(axis=-1) & (sampled[..., 2] > 0)
    results = []
    for k, index in enumerate(selected):
        if not valid[:, k].all():
            results.append({"point_index": int(index), "valid_samples": int(valid[:, k].sum()),
                            "requested_samples": samples, "variance_check": "failed_invalid_samples"})
            continue
        variance = float(np.var(sampled[:, k, 2], ddof=1))
        expected = float(predicted[index, 2, 2])
        # Independent trials per query, family correction across 3 models x 6 frames x 2 queries.
        from scipy.stats import chi2
        lower, upper = chi2.ppf([.05 / (2 * 36), 1 - .05 / (2 * 36)], samples - 1) / (samples - 1)
        ratio = variance / expected
        results.append({"point_index": int(index), "valid_samples": samples,
                        "requested_samples": samples, "depth_stdev_ratio": float(np.sqrt(ratio)),
                        "variance_ratio_interval": [float(lower), float(upper)],
                        "passed": bool(lower <= ratio <= upper)})
    return results


def assess_frame(record, objects, inputs, models, size, output, name, samples):
    obj, left, right = common.points(record, objects)
    q = np.stack((left, right), axis=1)
    valid_inverse = np.ones(len(q), bool)
    for camera in (0, 1):
        ray = mrcal.unproject(q[:, camera], *models[camera].intrinsics(), normalize=True)
        valid_inverse &= diagnosis.valid_inverse(ray, mrcal.project(ray, *models[camera].intrinsics()), q[:, camera])
    if not valid_inverse.all():
        # Do not silently change common evaluation samples between configurations.
        raise ValueError(f"Invalid inverse for corner IDs {np.array(record['ids'])[~valid_inverse].tolist()}")
    p, cal, obs, joint = mrcal.triangulate(
        q, models, q_calibration_stdev=.2, q_observation_stdev=.2,
        q_observation_stdev_correlation=0., stabilize_coords=True,
        method=mrcal.triangulate_leecivera_mid2)
    finite = np.isfinite(p).all(axis=-1)
    other = mrcal.transform_point_rt(inputs["rt_cam_ref"][0], p)
    positive = (p[:, 2] > 0) & (other[:, 2] > 0)
    if not (finite & positive).all():
        raise ValueError(f"Invalid 3-D points: finite={finite.sum()}, positive_depth_both={positive.sum()}/{len(p)}")
    covariance_checks = {"calibration": covariance_diagnostics(cal), "joint": covariance_diagnostics(joint)}
    if not np.isfinite(obs).all() or np.linalg.eigvalsh(obs).min() < -1e-12:
        raise ValueError("Invalid observation covariance")
    np.testing.assert_allclose(diagonal_blocks(joint), diagonal_blocks(cal) + obs, rtol=1e-7, atol=1e-14)
    i, j, direction, nominal, errors = pair_geometry(p, obj)
    spacing = inputs["calibration_object_spacing"]
    adjacent = np.isclose(nominal, spacing, rtol=1e-5, atol=1e-8)
    pair_cal = pair_variance(cal, i, j, direction)
    pair_obs = np.einsum("ni,nij,nj->n", direction, obs[i] + obs[j], direction)
    if min(float(pair_cal.min()), float(pair_obs.min())) < -1e-12:
        raise ValueError("Negative distance variance")
    pair_sigma = np.sqrt(np.maximum(pair_cal + pair_obs, 0))
    plane = plane_residuals(p)
    rigid = rigid_board_residuals(p, obj)
    reprojection = np.concatenate((mrcal.project(p, *models[0].intrinsics()) - left,
                                   mrcal.project(other, *models[1].intrinsics()) - right))
    # Edge if either camera sees the point in its outer 20%; one 3-D point counted once.
    masks = {key: spline.region_masks(left, size)[key] | spline.region_masks(right, size)[key]
             for key in ("edge", "corners", "top", "bottom", "left", "right")}
    masks["all"] = np.ones(len(p), bool)
    masks["center"] = ~masks["edge"]
    cal_diag, joint_diag = diagonal_blocks(cal), diagonal_blocks(joint)
    depths = {"calibration": np.sqrt(np.maximum(cal_diag[:, 2, 2], 0)) * 1000,
              "observation": np.sqrt(np.maximum(obs[:, 2, 2], 0)) * 1000,
              "joint": np.sqrt(np.maximum(joint_diag[:, 2, 2], 0)) * 1000}
    frame = {"capture_id": record["capture_id"], "point_count": len(p),
             "valid_inverse_points": int(valid_inverse.sum()), "positive_depth_both_cameras": int(positive.sum()),
             "depth_m": {"min": float(p[:, 2].min()), "median": float(np.median(p[:, 2])), "max": float(p[:, 2].max())},
             "distance_all_pairs_mm": statistics(errors * 1000),
             "distance_adjacent_mm": statistics(errors[adjacent] * 1000),
             "distance_all_pairs_relative_percent": statistics(errors / nominal * 100),
             "distance_joint_sigma_all_pairs_mm": statistics(pair_sigma * 1000),
             "distance_joint_sigma_adjacent_mm": statistics(pair_sigma[adjacent] * 1000),
             "plane_mm": statistics(plane * 1000),
             "triangulated_point_reprojection": common.error_summary(reprojection),
             "rigid_fit_depth_residual_mm_diagnostic_only": statistics(rigid[:, 2] * 1000),
             "covariance_diagnostics": covariance_checks,
             "regions": {region: {"points": int(mask.sum()), "plane_mm": statistics(plane[mask] * 1000),
                 "depth_sigma_mm": {source: statistics(values[mask]) for source, values in depths.items()},
                 "distance_pairs_touching_region_mm": statistics(errors[mask[i] | mask[j]] * 1000)}
                         for region, mask in masks.items()},
             "observation_only_monte_carlo": observation_sampling(q, models, obs, size, samples, 20261027 + int(record["capture_id"]))}
    np.savez_compressed(output / f"{name}_{record['capture_id']}.npz", ids=record["ids"], pixels=q,
                        points_m=p, objects_m=obj, covariance_calibration_m2=cal,
                        covariance_observation_m2=obs, covariance_joint_m2=joint,
                        pair_i=i, pair_j=j, pair_nominal_m=nominal, pair_error_m=errors,
                        pair_joint_sigma_m=pair_sigma, adjacent=adjacent, plane_residual_m=plane,
                        **{f"mask_{key}": mask for key, mask in masks.items()})
    arrays = {"distance_all_pairs_mm": errors * 1000, "distance_adjacent_mm": errors[adjacent] * 1000,
              "plane_mm": plane * 1000, "distance_all_pairs_relative_percent": errors / nominal * 100,
              "distance_joint_sigma_all_pairs_mm": pair_sigma * 1000,
              "distance_joint_sigma_adjacent_mm": pair_sigma[adjacent] * 1000}
    return frame, arrays, masks, depths


def summarize(report):
    lines = ["# 留出三角测量与三维不确定度", "",
             f"冻结训练模型；仅评价固定留出 {report['holdout_captures']} 对，"
             f"共 {report['holdout_points']} 个三维角点。不做尺度修正或删点。", "",
             "| 模型 | 相邻距离 RMS mm | 全部点对距离 RMS mm | 平面 RMS mm | 深度 σ 中位数 mm | 边缘深度 σ P95 mm |",
             "|---|---:|---:|---:|---:|---:|"]
    for name, model in report["models"].items():
        if model["status"] != "completed":
            lines.append(f"| {name} | 失败 | — | — | — | — |")
            continue
        a = model["aggregate"]
        numbers = [a["distance_adjacent_mm"]["rms"], a["distance_all_pairs_mm"]["rms"],
                   a["plane_mm"]["rms"], model["regions"]["all"]["depth_sigma_mm"]["joint"]["median_abs"],
                   model["regions"]["edge"]["depth_sigma_mm"]["joint"]["p95_abs"]]
        lines.append(f"| {name} | " + " | ".join(f"{x:.4f}" for x in numbers) + " |")
    lines += ["", "深度为文件 camera0 坐标 Z；深度 σ 为 1σ，标定及观测噪声各假设每坐标 0.2px、左右观测独立。",
              "模型坐标系稳定化开启，距离不确定度保留点间标定相关性。",
              "边缘指任一相机看到的外侧 20%；三维点只计一次，各区域覆盖见完整报告。",
              "板规格是名义参考，未独立量测板尺寸/平整度或实际距离；平面拟合消除了整体姿态，不能证明绝对深度准确。",
              "全部点对共享角点，非独立样本；不据此计算独立样本置信区间。",
              "协方差仅表示假设高斯采样噪声的条件敏感度，不包括模型偏差、不同步、板弯曲及尺寸误差。",
              "本轮 Monte Carlo 只验证固定相机下的观测噪声传播，没有重新验证完整重标定三维协方差。"]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=1000)
    args = parser.parse_args()
    if args.samples < 100:
        parser.error("At least 100 observation sampling trials required")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Output directory must be empty")
    args.output.mkdir(parents=True, exist_ok=True)
    try:
        baseline = json.loads((args.baseline / "report.json").read_text())
        data = common.collect_observations(args.session.resolve(), "rgb_pair_1", args.baseline / "observations.json")
        if data["dataset_hash"] != baseline["dataset_hash"]:
            raise ValueError("Dataset hash mismatch")
        split = baseline["protocol"]["split"]
        records = [r for r in data["records"] if r["capture_id"] in split["validation"]]
        if len(records) != 6 or not all(r["usable"] for r in records):
            raise ValueError("Expected unchanged six-capture holdout")
        if [r["capture_id"] for r in records] != split["validation"]:
            raise ValueError("Holdout capture order changed")
        if (set(split["train"]) & set(split["validation"]) or
                set(split["guard"]) & (set(split["train"]) | set(split["validation"]))):
            raise ValueError("Split subsets overlap")
        objects, size = np.array(data["object_points_m"]), tuple(data["image_size"])
        names = ("opencv5", "spline12x8_fov120", "spline16x10_fov140")
        protocol = {"split": split, "evaluation_subset": "validation_only", "new_camera_fitting": False,
                    "triangulation": "mrcal.triangulate_leecivera_mid2", "stabilize_coords": True,
                    "noise_sigma_calibration_px": .2, "noise_sigma_observation_px": .2,
                    "observation_correlation": 0., "depth_definition": "camera0 Z in meters; file roles preserved",
                    "reference": "nominal board geometry, no external dimensional or range truth",
                    "plane": "unweighted orthogonal least-squares plane per capture",
                    "distance": "all unordered corner pairs and grid-adjacent 40mm pairs; no scale alignment",
                    "uncertainty": "first-order conditional covariance; calibration cross-point correlations retained",
                    "sampling": {"observation_only_trials": args.samples, "seed_base": 20261027,
                                 "familywise_alpha": .05, "maximum_queries": 36},
                    "source_report_sha256": common.digest(args.baseline / "report.json")}
        common.write_json(args.output / "protocol.json", protocol)
        report = {"status": "completed", "protocol": protocol, "dataset_hash": data["dataset_hash"],
                  "holdout_captures": len(records),
                  "holdout_points": sum(len(record["ids"]) for record in records),
                  "script_sha256": common.digest(__file__),
                  "helper_sha256": {m.__file__: common.digest(m.__file__) for m in (common, spline, diagnosis, external)},
                  "environment": {n: importlib.metadata.version(n) for n in ("mrcal", "numpy", "scipy", "opencv-python-headless")},
                  "models": {}}
        for name in names:
            print("triangulate", name, flush=True)
            paths = [args.baseline / f"{name}_{c}.cameramodel" for c in (0, 1)]
            hashes = [common.digest(path) for path in paths]
            inputs = diagnosis.load_inputs(paths[0])
            if len(inputs["rt_ref_frame"]) != len(split["train"]):
                raise ValueError("Frozen model does not contain exactly the training frames")
            fingerprint = external.model_fingerprint(inputs)
            models = tuple(mrcal.cameramodel(optimization_inputs=copy.deepcopy(inputs), icam_intrinsics=c) for c in (0, 1))
            result = {"status": "completed", "source_model_sha256": hashes, "frames": []}
            accumulated, region_depths, region_planes = {}, {}, {}
            try:
                for record in records:
                    print(name, record["capture_id"], flush=True)
                    frame, arrays, masks, depths = assess_frame(record, objects, inputs, models, size, args.output, name, args.samples)
                    result["frames"].append(frame)
                    for key, values in arrays.items():
                        accumulated.setdefault(key, []).extend(values)
                    for region, mask in masks.items():
                        region_planes.setdefault(region, []).extend(arrays["plane_mm"][mask])
                        for source, values in depths.items():
                            region_depths.setdefault(region, {}).setdefault(source, []).extend(values[mask])
                result["aggregate"] = {key: statistics(values) for key, values in accumulated.items()}
                result["regions"] = {region: {"points": len(region_planes[region]),
                    "plane_mm": statistics(region_planes[region]),
                    "depth_sigma_mm": {key: statistics(values) for key, values in sources.items()}}
                    for region, sources in region_depths.items()}
                checks = [c for f in result["frames"] for c in f["observation_only_monte_carlo"]]
                result["observation_only_sampling_passed"] = all(c.get("passed", False) for c in checks)
            except Exception as exc:
                result.update(status="failed", error=str(exc))
            if fingerprint != external.model_fingerprint(inputs) or hashes != [common.digest(path) for path in paths]:
                raise ValueError("Evaluation changed frozen camera models")
            report["models"][name] = result
            common.write_json(args.output / "report.partial.json", report)
        if any(r["status"] != "completed" for r in report["models"].values()):
            report["status"] = "completed_with_failures"
        common.write_json(args.output / "report.json", report)
        (args.output / "summary.md").write_text(summarize(report), encoding="utf-8")
        print("Completed", args.output, flush=True)
    except Exception as exc:
        common.write_json(args.output / "failure.json", {"error": str(exc)})
        raise


if __name__ == "__main__":
    main()
