"""Compare native mrcal spline and OPENCV5 models on a frozen offline split."""
from __future__ import annotations

import argparse
import copy
import importlib.metadata
import json
from pathlib import Path
import sys
import time

import mrcal
import numpy as np
from scipy.optimize import least_squares

import compare_mrcal as comparison
import validate_mrcal as validation

# Declared before examining holdout: main grid and one density sensitivity check.
SPLINES = {
    "spline12x8": "LENSMODEL_SPLINED_STEREOGRAPHIC_order=3_Nx=12_Ny=8_fov_x_deg=120",
    "spline16x10": "LENSMODEL_SPLINED_STEREOGRAPHIC_order=3_Nx=16_Ny=10_fov_x_deg=120",
}


def region_masks(xy, size):
    uv = np.asarray(xy) / np.asarray(size)
    outer = (uv < .2) | (uv > .8)
    return {"all": np.ones(len(xy), dtype=bool), "edge": outer.any(axis=1),
            "center": ~outer.any(axis=1), "corners": outer.all(axis=1),
            "left": uv[:, 0] < .2, "right": uv[:, 0] > .8,
            "top": uv[:, 1] < .2, "bottom": uv[:, 1] > .8}


def spline_seed(source, lensmodel):
    """Fit an initial spline to training-only OPENCV5 rays, with core held fixed.

    Correction coefficients enter projection linearly when core and rays are fixed.
    These pseudo-observations are ONLY an initialization, never solver measurements.
    """
    result = validation.copy_inputs(source)
    result["lensmodel"] = lensmodel
    result["do_optimize_intrinsics_core"] = False
    result["do_apply_regularization"] = True
    seeds, diagnostics = [], []
    for camera in range(2):
        width, height = source["imagersizes"][camera]
        pixels = np.array([(x, y) for y in np.linspace(0, height - 1, 17)
                           for x in np.linspace(0, width - 1, 25)])
        rays = mrcal.unproject(pixels, source["lensmodel"], source["intrinsics"][camera], normalize=True)
        intrinsics = np.zeros(mrcal.lensmodel_num_params(lensmodel))
        intrinsics[:4] = source["intrinsics"][camera, :4]
        predicted, _, jacobian = mrcal.project(rays, lensmodel, intrinsics, get_gradients=True)
        design = jacobian[..., 4:].reshape(-1, len(intrinsics) - 4)
        intrinsics[4:], _, rank, _ = np.linalg.lstsq(design, (pixels - predicted).ravel(), rcond=1e-10)
        error = mrcal.project(rays, lensmodel, intrinsics) - pixels
        diagnostics.append({"camera": camera, "seed_grid": [25, 17], "rank": int(rank),
                            "coefficient_count": len(intrinsics) - 4,
                            "approximation": comparison.error_summary(error)})
        seeds.append(intrinsics)
    result["intrinsics"] = np.array(seeds)
    return result, diagnostics


def solve(inputs):
    # validation.fit intentionally disables regularization; preserve it here for native spline solves.
    candidate = copy.deepcopy(inputs)
    started = time.perf_counter()
    stats = mrcal.optimize(**candidate)
    elapsed = time.perf_counter() - started
    for name in ("intrinsics", "rt_cam_ref", "rt_ref_frame"):
        if not np.isfinite(candidate[name]).all():
            raise ValueError(f"Nonfinite optimized {name}")
    np.testing.assert_array_equal(candidate["observations_board"][..., 2], inputs["observations_board"][..., 2])
    return candidate, {"seconds": elapsed,
                       "stats": {k: float(v) for k, v in stats.items() if np.isscalar(v)}}


def project(inputs, objects, pose, camera):
    xyz = mrcal.transform_point_rt(pose, objects)
    if camera:
        xyz = mrcal.transform_point_rt(inputs["rt_cam_ref"][0], xyz)
    if np.any(xyz[:, 2] <= 0):
        raise ValueError("Board prediction behind camera")
    return mrcal.project(xyz, inputs["lensmodel"], inputs["intrinsics"][camera])


def fit_pose(inputs, objects, observations, initial, cameras):
    def residual(pose):
        return np.concatenate([(project(inputs, objects, pose, camera) - observations[camera]).ravel()
                               for camera in cameras])
    answer = least_squares(residual, initial, method="lm", max_nfev=500,
                           ftol=1e-10, xtol=1e-10, gtol=1e-10)
    if not answer.success:
        raise ValueError("Evaluation board pose did not converge")
    return answer.x


def evaluate(inputs, records, objects, size, pose_seed):
    transfer, joint, coordinates, per_capture = [], [], [], []
    for record in records:
        obj, left, right = comparison.points(record, objects)
        observations = (left, right)
        initial = comparison.pose_from_left(obj, left, pose_seed)
        pose_left = fit_pose(inputs, obj, observations, initial, (0,))
        # Use the same board-to-left parameterization for both independent monocular fits.
        pose_right = fit_pose(inputs, obj, observations, initial, (1,))
        pose_joint = fit_pose(inputs, obj, observations, initial, (0, 1))
        residual_transfer = np.concatenate((project(inputs, obj, pose_right, 0) - left,
                                            project(inputs, obj, pose_left, 1) - right))
        residual_joint = np.concatenate([project(inputs, obj, pose_joint, c) - observations[c]
                                         for c in range(2)])
        xy = np.concatenate(observations)
        transfer.extend(residual_transfer)
        joint.extend(residual_joint)
        coordinates.extend(xy)
        per_capture.append({"capture_id": record["capture_id"],
                            "transfer": {region: comparison.error_summary(residual_transfer[mask])
                                         for region, mask in region_masks(xy, size).items()}})
    transfer, joint, coordinates = map(np.asarray, (transfer, joint, coordinates))
    return {"captures": len(records),
            "transfer": {region: comparison.error_summary(transfer[mask])
                         for region, mask in region_masks(coordinates, size).items()},
            "joint_pose": {region: comparison.error_summary(joint[mask])
                           for region, mask in region_masks(coordinates, size).items()},
            "per_capture": per_capture}


def coverage(records, size):
    result = []
    for camera, name in enumerate(("left", "right")):
        xy = np.concatenate([np.asarray(r[name]) for r in records])
        hist, _, _ = np.histogram2d(xy[:, 1], xy[:, 0], bins=(6, 8), range=((0, size[1]), (0, size[0])))
        result.append({"camera": camera, "grid_6x8_counts": hist.astype(int).tolist(),
                       "regions": {region: int(mask.sum()) for region, mask in region_masks(xy, size).items()}})
    return result


def checked_unproject(pixels, lensmodel, intrinsics):
    rays = mrcal.unproject(pixels, lensmodel, intrinsics, normalize=True)
    good = np.isfinite(rays).all(axis=-1) & (np.linalg.norm(rays, axis=-1) > 1e-12)
    if not good.all():
        raise ValueError(f"Invalid inverse ray at query indices {np.flatnonzero(~good).tolist()}")
    projected = mrcal.project(rays, lensmodel, intrinsics)
    good = np.isfinite(projected).all(axis=-1) & (np.linalg.norm(projected - pixels, axis=-1) <= .01)
    if not good.all():
        raise ValueError(f"Invalid inverse roundtrip at query indices {np.flatnonzero(~good).tolist()}")
    return rays


def uncertainty_map(inputs, sigma=.2):
    width, height = inputs["imagersizes"][0]
    pixels = np.array([(x * (width - 1), y * (height - 1))
                       for y in np.linspace(.025, .975, 11) for x in np.linspace(.025, .975, 17)])
    masks = region_masks(pixels, (width, height))
    result = {"sigma_px_per_coordinate": sigma, "method": "mean-pcam",
              "grid": [17, 11], "pixels": pixels.tolist(), "cameras": []}
    for camera in range(2):
        model = mrcal.cameramodel(optimization_inputs=inputs, icam_intrinsics=camera)
        rays = checked_unproject(pixels, *model.intrinsics())
        levels = []
        for distance in (1., 3., None):
            covariance = mrcal.projection_uncertainty(
                rays if distance is None else rays * distance, model=model, method="mean-pcam",
                atinfinity=distance is None, what="covariance", observed_pixel_uncertainty=sigma)
            eigenvalues = np.linalg.eigvalsh(covariance)
            if not np.isfinite(covariance).all() or np.any(eigenvalues < -1e-8):
                raise ValueError("Invalid projection covariance")
            deviations = np.sqrt(np.maximum(eigenvalues[:, -1], 0))
            levels.append({"distance_m": distance if distance else "infinity",
                           "worst_stdev_px": deviations.tolist(),
                           "regions": {region: {"queries": int(mask.sum()),
                                                  "median_px": float(np.median(deviations[mask])),
                                                  "p95_px": float(np.percentile(deviations[mask], 95)),
                                                  "max_px": float(np.max(deviations[mask]))}
                                       for region, mask in masks.items()}})
        result["cameras"].append({"camera": camera, "levels": levels})
    return result


def knot_support(inputs):
    if not inputs["lensmodel"].startswith("LENSMODEL_SPLINED"):
        return None
    result = []
    grid = mrcal.ref_calibration_object(inputs["observations_board"].shape[2],
                                         inputs["observations_board"].shape[1], inputs["calibration_object_spacing"])
    for camera in range(2):
        support = np.zeros(len(inputs["intrinsics"][camera]) - 4, dtype=int)
        for obs, (frame, cam, ext) in enumerate(inputs["indices_frame_camintrinsics_camextrinsics"]):
            if cam != camera:
                continue
            xyz = mrcal.transform_point_rt(inputs["rt_ref_frame"][frame], grid)
            if ext >= 0:
                xyz = mrcal.transform_point_rt(inputs["rt_cam_ref"][ext], xyz)
            xyz = xyz[inputs["observations_board"][obs, ..., 2] > 0]
            _, _, jacobian = mrcal.project(xyz, inputs["lensmodel"], inputs["intrinsics"][camera], get_gradients=True)
            support += np.any(np.abs(jacobian[..., 4:]) > 1e-10, axis=1).sum(axis=0)
        result.append({"camera": camera, "coefficient_count": len(support),
                       "zero_direct_support_coefficients": int(np.count_nonzero(support == 0)),
                       "fewer_than_5_observations": int(np.count_nonzero(support < 5)),
                       "observations_per_coefficient": support.tolist()})
    return result


def monte_carlo(inputs, output, name, samples=200, *,
                query_fractions=(.1, .5, .9), family_queries=108):
    """Check spline conditional covariance under known noise; preserve regularization."""
    ideal = validation.render_observations(inputs)
    # The earlier OPENCV5 helper resets regularization; retain this experiment's flags.
    ideal["do_apply_regularization"] = inputs["do_apply_regularization"]
    baseline, _ = solve(ideal)
    width, height = baseline["imagersizes"][0]
    pixels = np.array([(x * (width - 1), y * (height - 1))
                       for y in query_fractions for x in query_fractions])
    distances = (1., 3., None)
    sigma = .2
    covariance = []
    for camera in range(2):
        model = mrcal.cameramodel(optimization_inputs=baseline, icam_intrinsics=camera)
        rays = checked_unproject(pixels, *model.intrinsics())
        covariance.append([mrcal.projection_uncertainty(
            rays if d is None else rays * d, model=model, atinfinity=d is None,
            method="mean-pcam", observed_pixel_uncertainty=sigma) for d in distances])
    covariance = np.array(covariance)
    if not np.isfinite(covariance).all():
        raise ValueError("Nonfinite Monte Carlo reference covariance")
    rng = np.random.default_rng(20261027)
    projections, failures = [], []
    for trial in range(samples):
        perturbed = copy.deepcopy(baseline)
        mask = perturbed["observations_board"][..., 2] > 0
        perturbed["observations_board"][..., :2][mask] += rng.normal(0, sigma, (mask.sum(), 2))
        try:
            fitted, _ = solve(perturbed)
            q = validation.projection_samples(baseline, fitted, pixels, distances)
            if not np.isfinite(q).all():
                raise ValueError("Nonfinite Monte Carlo reprojection")
            projections.append(q)
        except Exception as exc:
            failures.append({"trial": trial, "error": str(exc)})
        if (trial + 1) % 50 == 0:
            print(name, "Monte Carlo", trial + 1, "/", samples, flush=True)
    if len(projections) < 30:
        return {"status": "failed", "failures": failures, "successful_samples": len(projections)}
    projections = np.array(projections)
    np.savez_compressed(output / f"{name}_monte_carlo.npz", projections=projections,
                        predicted_covariance=covariance, pixels=pixels)
    report = validation.covariance_check(projections, covariance, pixels, sigma, family_queries=family_queries)
    report.update(status="completed", requested_samples=samples, failures=failures,
                  seed=20261027, regularization=baseline["do_apply_regularization"],
                  query_fractions=list(query_fractions), family_queries=family_queries,
                  truth="exact projections using fitted model and original training geometry",
                  checks_pass=not failures and report["variance_checks_pass"] and report["coverage_checks_pass"])
    return report


def summarize(report):
    lines = ["# mrcal 样条与 OPENCV5 比较", "",
             "保持原 24 对训练、9 对验证和角点掩码；边缘定义为任一方向位于图像外侧 20%。", "",
             "| 模型 | 训练点 RMS | 留出跨目 RMS | 边缘跨目 RMS | 边缘 P95 | 角区跨目 RMS |",
             "|---|---:|---:|---:|---:|---:|"]
    for name, result in report["models"].items():
        if result["status"] != "completed":
            lines.append(f"| {name} | 失败 | — | — | — | — |")
            continue
        val = result["validation"]["transfer"]
        lines.append(f"| {name} | {result['train']['joint_pose']['all']['rms_px']:.4f} | "
                     f"{val['all']['rms_px']:.4f} | {val['edge']['rms_px']:.4f} | "
                     f"{val['edge']['p95_px']:.4f} | {val['corners']['rms_px']:.4f} |")
    lines += ["", "## 条件投影不确定度", "",
              "相同已知角点噪声 σ=0.2 px；原始模型各自的 mean-pcam，非完整测距误差。", "",
              "| 模型 | 目 | 距离 m | 中心标准差中位数 px | 边缘标准差中位数 px | 边缘标准差 P95 px |",
              "|---|---:|---|---:|---:|---:|"]
    for name, result in report["models"].items():
        uncertainty = result.get("uncertainty", {})
        for camera in uncertainty.get("cameras", []):
            for level in camera["levels"]:
                region = level["regions"]
                lines.append(f"| {name} | {camera['camera']} | {level['distance_m']} | "
                             f"{region['center']['median_px']:.4f} | {region['edge']['median_px']:.4f} | "
                             f"{region['edge']['p95_px']:.4f} |")
    lines += ["", "样条核心固定，样条系数/双目外参/板姿态联合求解。主要比较采用默认弱正则的两种模型；",
              "原 OPENCV5 无正则结果另列作锚点。不剔除角点、不拟合板形变、不按留出结果选择样条密度。",
              "样条有更高的不确定度不一定更差，可能揭示低参数模型掩盖的数据约束不足。",
              "同一旧会话的验证集已被使用；结果仅用于模型诊断，不能当作新盲测或证明真实不确定度。", "",
              "## 不确定度加噪核对", ""]
    for name, result in report["models"].items():
        mc = result.get("monte_carlo")
        if mc:
            if mc.get("status") == "completed":
                ratios = [q["stdev_ratio"] for q in mc["queries"]]
                lines.append(f"- {name}：{mc['samples']} 次成功，实测/预测标准差比 "
                             f"{min(ratios):.3f}–{max(ratios):.3f}，统计检查 {mc['checks_pass']}。")
            else:
                lines.append(f"- {name}：{mc.get('status')}，{mc.get('error', '见完整报告')}。")
        if result.get("uncertainty_error"):
            lines.append(f"- {name} 原生不确定度无效：{result['uncertainty_error']}，不作可靠性结论。")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=200)
    args = parser.parse_args()
    if args.samples < 30:
        parser.error("Require at least 30 Monte Carlo samples")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Output directory must be empty")
    args.output.mkdir(parents=True, exist_ok=True)
    try:
        previous = json.loads((args.baseline / "report.json").read_text())
        data = comparison.collect_observations(args.session.resolve(), "rgb_pair_1", args.baseline / "observations.json")
        if data["dataset_hash"] != previous["dataset_hash"]:
            raise ValueError("Dataset mismatch")
        records = {r["capture_id"]: r for r in data["records"]}
        train = [records[i] for i in previous["split"]["train"] if records[i]["usable"]]
        holdout = [records[i] for i in previous["split"]["validation"] if records[i]["usable"]]
        objects, size = np.array(data["object_points_m"]), tuple(data["image_size"])
        source_path = args.baseline / "mrcal_joint_0.cameramodel"
        source = validation.copy_inputs(mrcal.cameramodel(str(source_path)).optimization_inputs())
        seed = validation.opencv_model(source)
        training_fov = []
        for camera in range(2):
            width = source["imagersizes"][camera, 0]
            cy = source["intrinsics"][camera, 3]
            rays = mrcal.unproject(np.array([[0., cy], [width - 1., cy]]), source["lensmodel"],
                                   source["intrinsics"][camera], normalize=True)
            training_fov.append(float(np.diff(np.degrees(np.arctan2(rays[:, 0], rays[:, 2])))[0]))
        protocol = {"lensmodels": SPLINES, "spline_core_optimized": False,
                    "regularization": "OPENCV5 off anchor + OPENCV5 on + splines on",
                    "outlier_rejection": False, "board_warp": False,
                    "edge_fraction": .2, "known_sigma_px": .2, "split": previous["split"],
                    "fov_selection": "120 deg fixed before holdout; training-only horizontal FOV below",
                    "training_horizontal_fov_degrees": training_fov,
                    "monte_carlo": {"samples": args.samples, "sigma_px": .2, "seed": 20261027,
                                    "models": ["opencv5_regularized", "spline12x8"],
                                    "familywise_queries": 108},
                    "model_selection": "both spline grids reported; no holdout selection"}
        comparison.write_json(args.output / "protocol.json", protocol)
        report = {"status": "completed", "dataset_hash": data["dataset_hash"], "protocol": protocol,
                  "script_sha256": comparison.digest(__file__), "source_model_sha256": comparison.digest(source_path),
                  "environment": {name: importlib.metadata.version(name) for name in ("mrcal", "numpy", "scipy")},
                  "python": sys.version, "train_coverage": coverage(train, size),
                  "validation_coverage": coverage(holdout, size), "models": {}}
        candidates = {"opencv5_unregularized": copy.deepcopy(source), "opencv5_regularized": copy.deepcopy(source)}
        candidates["opencv5_regularized"]["do_apply_regularization"] = True
        for name, lensmodel in SPLINES.items():
            candidates[name], diagnostic = spline_seed(source, lensmodel)
            report.setdefault("initialization", {})[name] = diagnostic
        for name, candidate in candidates.items():
            print("fit", name, flush=True)
            try:
                fitted, timing = solve(candidate)
                result = {"status": "completed", "lensmodel": fitted["lensmodel"], "solver": timing,
                          "intrinsics_parameters_per_camera": fitted["intrinsics"].shape[1],
                          "intrinsics": fitted["intrinsics"].tolist(),
                          "rt_cam_ref": fitted["rt_cam_ref"].tolist(),
                          "regularization": fitted["do_apply_regularization"],
                          "train": evaluate(fitted, train, objects, size, seed),
                          "validation": evaluate(fitted, holdout, objects, size, seed),
                          "knot_support": knot_support(fitted)}
                for camera in range(2):
                    mrcal.cameramodel(optimization_inputs=fitted, icam_intrinsics=camera).write(
                        str(args.output / f"{name}_{camera}.cameramodel"))
                try:
                    result["uncertainty"] = uncertainty_map(fitted)
                except Exception as exc:
                    result["uncertainty_error"] = str(exc)
                if name in ("opencv5_regularized", "spline12x8"):
                    try:
                        result["monte_carlo"] = monte_carlo(fitted, args.output, name, args.samples)
                    except Exception as exc:
                        result["monte_carlo"] = {"status": "failed", "error": str(exc)}
                report["models"][name] = result
                print(name, "holdout edge", result["validation"]["transfer"]["edge"], flush=True)
                if name in SPLINES:
                    # Diagnostic only: unconstrained exterior knots can make this singular.
                    free = copy.deepcopy(fitted)
                    free["do_apply_regularization"] = False
                    try:
                        no_reg, diag = solve(free)
                        result["unregularized_diagnostic"] = {"status": "completed", "solver": diag,
                            "validation": evaluate(no_reg, holdout, objects, size, seed)}
                    except Exception as exc:
                        result["unregularized_diagnostic"] = {"status": "failed", "error": str(exc)}
            except Exception as exc:
                report["models"][name] = {"status": "failed", "error": str(exc)}
            comparison.write_json(args.output / "report.partial.json", report)
        comparison.write_json(args.output / "report.json", report)
        (args.output / "summary.md").write_text(summarize(report), encoding="utf-8")
        print("Completed", args.output, flush=True)
    except Exception as exc:
        comparison.write_json(args.output / "failure.json", {"status": "failed", "error": str(exc)})
        raise


if __name__ == "__main__":
    main()
