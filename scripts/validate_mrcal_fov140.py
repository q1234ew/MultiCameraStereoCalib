"""Frozen-split comparison of 16x10/FOV140, 12x8/FOV120 and OPENCV5."""
from __future__ import annotations

import argparse
import copy
import importlib.metadata
import json
from pathlib import Path
import sys

import mrcal
import numpy as np

import compare_mrcal as common
import compare_mrcal_spline as spline
import diagnose_mrcal_spline as diagnosis
import validate_mrcal as validation

FOV140 = "LENSMODEL_SPLINED_STEREOGRAPHIC_order=3_Nx=16_Ny=10_fov_x_deg=140"


def inverse_checks(inputs, holdout, reference):
    """Same pixel grid for inverse checks; same physical rays for fold checks."""
    width, height = inputs["imagersizes"][0]
    pixels = np.array([(x, y) for y in np.linspace(0, height - 1, 37)
                       for x in np.linspace(0, width - 1, 65)])
    result = {"dense_grid": [65, 37], "grid_includes_pixel_border": True,
              "roundtrip_tolerance_px": 1e-5, "cameras": []}
    for camera, side in enumerate(("left", "right")):
        lens, params = inputs["lensmodel"], inputs["intrinsics"][camera]
        rays = mrcal.unproject(pixels, lens, params, normalize=True)
        projected = mrcal.project(rays, lens, params)
        valid = diagnosis.valid_inverse(rays, projected, pixels)
        observed = np.concatenate([np.array(record[side]) for record in holdout])
        observed_rays = mrcal.unproject(observed, lens, params, normalize=True)
        observed_valid = diagnosis.valid_inverse(observed_rays, mrcal.project(observed_rays, lens, params), observed)
        # Native inverse can pick another branch. Check local orientation using the
        # same training-reference physical rays rather than each model's own inverse.
        reference_rays = spline.checked_unproject(pixels, reference["lensmodel"], reference["intrinsics"][camera])
        uv = mrcal.project_stereographic(reference_rays)
        unit, dv_du = mrcal.unproject_stereographic(uv, get_gradients=True)
        _, dq_dv, _ = mrcal.project(unit, lens, params, get_gradients=True)
        jacobian = dq_dv @ dv_du
        determinant = np.linalg.det(jacobian)
        singular = np.linalg.svd(jacobian, compute_uv=False)
        historical = np.array([[1247.025, 17.975]])
        h_ray = mrcal.unproject(historical, lens, params, normalize=True)
        h_good = diagnosis.valid_inverse(h_ray, mrcal.project(h_ray, lens, params), historical)
        result["cameras"].append({
            "camera": camera, "grid_queries": len(pixels), "grid_valid": int(valid.sum()),
            "invalid_pixels": pixels[~valid].tolist(), "zero_ray_count": int(np.count_nonzero(np.linalg.norm(rays, axis=-1) <= 1e-12)),
            "holdout_corners": len(observed), "holdout_inverse_valid": int(observed_valid.sum()),
            "holdout_invalid_pixels": observed[~observed_valid].tolist(),
            "historical_failure_pixel_valid": bool(h_good[0]),
            "common_physical_ray_orientation": {
                "queries": len(determinant), "negative_determinants": int(np.count_nonzero(determinant < 0)),
                "min_determinant": float(determinant.min()),
                "min_jacobian_singular_value": float(singular[:, -1].min())}})
    return result


def summary(report):
    lines = ["# 16×10/FOV140 同划分离线验证", "",
             "24 对训练、9 对留出；弱正则开启、异常剔除与板形变关闭；不按本轮结果重选配置。", "",
             "| 模型 | 留出 RMS px | 边缘 RMS px | 边缘 P95 px | 右边缘 RMS px | 角区 RMS px |",
             "|---|---:|---:|---:|---:|---:|"]
    for name, model in report["models"].items():
        if model["status"] != "completed":
            lines.append(f"| {name} | 失败 | — | — | — | — |")
            continue
        regions = model["validation"]["transfer"]
        lines.append(f"| {name} | {regions['all']['rms_px']:.4f} | {regions['edge']['rms_px']:.4f} | "
                     f"{regions['edge']['p95_px']:.4f} | {regions['right']['rms_px']:.4f} | {regions['corners']['rms_px']:.4f} |")
    lines += ["", "## 严格反投影检查", "",
              "包含图像像素边界的 65×37 网格；有限非零射线、有限往返误差 ≤1e-5 px。", "",
              "| 模型 | 目 | 网格有效/总数 | 留出角点有效/总数 | 共同物理射线上的负行列式数 |",
              "|---|---:|---:|---:|---:|"]
    for name, model in report["models"].items():
        for c in model.get("inverse", {}).get("cameras", []):
            lines.append(f"| {name} | {c['camera']} | {c['grid_valid']}/{c['grid_queries']} | "
                         f"{c['holdout_inverse_valid']}/{c['holdout_corners']} | "
                         f"{c['common_physical_ray_orientation']['negative_determinants']} |")
    lines += ["", "## 条件不确定度（σ=0.2 px，1m）", "",
              "| 模型 | 目 | 边缘标准差中位数 px | 边缘标准差 P95 px |",
              "|---|---:|---:|---:|"]
    for name, model in report["models"].items():
        for c in model.get("uncertainty", {}).get("cameras", []):
            region = c["levels"][0]["regions"]["edge"]
            lines.append(f"| {name} | {c['camera']} | {region['median_px']:.4f} | {region['p95_px']:.4f} |")
        if model.get("uncertainty_error"):
            lines.append(f"- {name} 不确定度错误：{model['uncertainty_error']}。")
    lines += ["", "## 边缘加噪核对", "",
              "查询为每方向 2.5%、50%、97.5%，每目 3×3，距离 1m/3m/无穷远；",
              "三个模型合计 162 个查询作多重比较校正。使用模型一致精确投影，不证明真实模型误差消失。", ""]
    for name, model in report["models"].items():
        mc = model.get("monte_carlo", {})
        if mc.get("status") == "completed":
            ratios = [q["stdev_ratio"] for q in mc["queries"]]
            lines.append(f"- {name}：{mc['samples']}/{mc['requested_samples']} 次成功，标准差比 "
                         f"{min(ratios):.3f}–{max(ratios):.3f}，统计检查 {mc['checks_pass']}。")
        elif mc:
            lines.append(f"- {name}：{mc.get('status')}，{mc.get('error', '见原始报告')}。")
    lines += ["", "密集网格通过不是连续全视场单射的证明。留出集已用于多轮研究，属于诊断比较而非新盲测。",
              "不覆盖真实重采、测量真值、同步误差或 Windows 发布。"]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--opencv-baseline", type=Path, required=True)
    parser.add_argument("--spline-baseline", type=Path, required=True)
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=200)
    args = parser.parse_args()
    if args.samples < 30:
        parser.error("Require at least 30 samples")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Output directory must be empty")
    args.output.mkdir(parents=True, exist_ok=True)
    try:
        prior = json.loads((args.opencv_baseline / "report.json").read_text())
        old_splines = json.loads((args.spline_baseline / "report.json").read_text())
        source_path = args.opencv_baseline / "mrcal_joint_0.cameramodel"
        if old_splines["source_model_sha256"] != common.digest(source_path):
            raise ValueError("Source model provenance mismatch")
        data = common.collect_observations(args.session.resolve(), "rgb_pair_1", args.opencv_baseline / "observations.json")
        if data["dataset_hash"] != prior["dataset_hash"] or data["dataset_hash"] != old_splines["dataset_hash"]:
            raise ValueError("Dataset mismatch")
        records = {r["capture_id"]: r for r in data["records"]}
        train = [records[i] for i in prior["split"]["train"] if records[i]["usable"]]
        holdout = [records[i] for i in prior["split"]["validation"] if records[i]["usable"]]
        source = diagnosis.load_inputs(source_path)
        objects, size = np.array(data["object_points_m"]), tuple(data["image_size"])
        protocol = {"split": prior["split"], "models": {
            "opencv5": "LENSMODEL_OPENCV5", "spline12x8_fov120": spline.SPLINES["spline12x8"],
            "spline16x10_fov140": FOV140}, "regularization": True,
            "outlier_rejection": False, "board_warp": False, "edge_fraction": .2,
            "inverse_grid": [65, 37], "roundtrip_tolerance_px": 1e-5,
            "uncertainty_sigma_px": .2, "uncertainty_distances_m": [1., 3., "infinity"],
            "monte_carlo": {"samples_per_model": args.samples, "query_fractions": [.025, .5, .975],
                            "seed": 20261027, "family_queries": 162},
            "selection": "fixed requested candidate, no holdout-based tuning"}
        common.write_json(args.output / "protocol.json", protocol)
        report = {"status": "completed", "protocol": protocol, "dataset_hash": data["dataset_hash"],
                  "script_sha256": common.digest(__file__), "python": sys.version,
                  "helper_sha256": {m.__file__: common.digest(m.__file__) for m in (common, spline, diagnosis, validation)},
                  "source_report_sha256": common.digest(args.spline_baseline / "report.json"),
                  "environment": {n: importlib.metadata.version(n) for n in ("mrcal", "numpy", "scipy")},
                  "models": {}}
        candidates = {}
        for name, old_name in (("opencv5", "opencv5_regularized"), ("spline12x8_fov120", "spline12x8")):
            path = args.spline_baseline / f"{old_name}_0.cameramodel"
            inputs = diagnosis.load_inputs(path)
            np.testing.assert_allclose(inputs["intrinsics"], old_splines["models"][old_name]["intrinsics"], rtol=0, atol=1e-12)
            candidates[name] = (inputs, {"reused_model_sha256": common.digest(path)})
        initial, diagnostic = spline.spline_seed(source, FOV140)
        fitted, solver = spline.solve(initial)
        candidates["spline16x10_fov140"] = (fitted, {"initialization": diagnostic, "solver": solver})
        seed = validation.opencv_model(source)
        for name, (inputs, provenance) in candidates.items():
            print("evaluate", name, flush=True)
            try:
                assert inputs["do_apply_regularization"]
                result = {"status": "completed", "lensmodel": inputs["lensmodel"], "provenance": provenance,
                          "train": spline.evaluate(inputs, train, objects, size, seed),
                          "validation": spline.evaluate(inputs, holdout, objects, size, seed),
                          "inverse": inverse_checks(inputs, holdout, source),
                          "knot_support": spline.knot_support(inputs)}
                for camera in range(2):
                    mrcal.cameramodel(optimization_inputs=copy.deepcopy(inputs), icam_intrinsics=camera).write(
                        str(args.output / f"{name}_{camera}.cameramodel"))
                try:
                    result["uncertainty"] = spline.uncertainty_map(inputs)
                except Exception as exc:
                    result["uncertainty_error"] = str(exc)
                try:
                    result["monte_carlo"] = spline.monte_carlo(inputs, args.output, name, args.samples,
                                                               query_fractions=(.025, .5, .975), family_queries=162)
                except Exception as exc:
                    result["monte_carlo"] = {"status": "failed", "error": str(exc)}
                report["models"][name] = result
            except Exception as exc:
                report["models"][name] = {"status": "failed", "error": str(exc)}
            common.write_json(args.output / "report.partial.json", report)
        common.write_json(args.output / "report.json", report)
        (args.output / "summary.md").write_text(summary(report), encoding="utf-8")
        print("Completed", args.output, flush=True)
    except Exception as exc:
        common.write_json(args.output / "failure.json", {"status": "failed", "error": str(exc)})
        raise


if __name__ == "__main__":
    main()
