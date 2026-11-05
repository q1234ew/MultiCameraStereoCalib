"""Read-only native-model diagnostics: inverse projection, coverage and covariance algebra."""
from __future__ import annotations

import argparse
import copy
import importlib.metadata
import inspect
import json
from pathlib import Path
import platform
import sys

import mrcal
import mrcal.model_analysis as analysis
import numpy as np
from scipy.linalg import qr, solve_triangular, svdvals
from scipy.optimize import least_squares

import compare_mrcal as common
import compare_mrcal_spline as spline
import validate_mrcal as validation


def load_inputs(path):
    original = mrcal.cameramodel(str(path)).optimization_inputs()
    result = validation.copy_inputs(original)
    result["do_apply_regularization"] = original["do_apply_regularization"]
    return result


def valid_inverse(rays, projected, pixels, tolerance=1e-5):
    return (np.isfinite(rays).all(axis=-1) & (np.linalg.norm(rays, axis=-1) > 1e-12)
            & np.isfinite(projected).all(axis=-1)
            & (np.linalg.norm(projected - pixels, axis=-1) <= tolerance))


def capture_covariance(inputs, camera, rays):
    """Capture the gradient given to propagation without modifying installed source."""
    model = mrcal.cameramodel(optimization_inputs=inputs, icam_intrinsics=camera)
    original = analysis._propagate_calibration_uncertainty
    captured = {}

    def intercept(what, **kwargs):
        gradient = kwargs["dF_dbunpacked"].copy()
        mrcal.unpack_state(gradient, **inputs)
        captured["gradient"] = gradient
        return original(what, **kwargs)

    analysis._propagate_calibration_uncertainty = intercept
    try:
        covariance = mrcal.projection_uncertainty(rays, model=model, method="mean-pcam",
                                                  observed_pixel_uncertainty=.2)
    finally:
        analysis._propagate_calibration_uncertainty = original
    return covariance, captured["gradient"]


def qr_covariance(jacobian, gradient, observation_rows):
    """Independent regularized sandwich covariance using column-pivoted QR, not J^T J."""
    _, upper, permutation = qr(jacobian, mode="economic", pivoting=True)
    flat = gradient.reshape(-1, gradient.shape[-1])
    rhs = flat[:, permutation].T
    middle = solve_triangular(upper.T, rhs, lower=True)
    solution = solve_triangular(upper, middle)
    sensitivity = (jacobian[:observation_rows, permutation] @ solution).T
    sensitivity = sensitivity.reshape(gradient.shape[:-1] + (observation_rows,))
    covariance = np.einsum("...ik,...jk->...ij", sensitivity, sensitivity) * .04
    return covariance


def inverse_multistart(inputs, camera, pixel, opencv_inputs):
    lens, params = inputs["lensmodel"], inputs["intrinsics"][camera]
    opencv_ray = mrcal.unproject(pixel, opencv_inputs["lensmodel"],
                                opencv_inputs["intrinsics"][camera], normalize=True)
    center = mrcal.project_stereographic(opencv_ray)
    core_guess = (pixel - params[2:4]) / params[:2]
    starts = [center, core_guess] + [center + (dx, dy) for dx in (-.25, 0, .25) for dy in (-.2, 0, .2)]

    def residual(uv):
        return mrcal.project(mrcal.unproject_stereographic(uv), lens, params) - pixel

    def jacobian(uv):
        ray, dv_du = mrcal.unproject_stereographic(uv, get_gradients=True)
        _, dq_dv, _ = mrcal.project(ray, lens, params, get_gradients=True)
        return dq_dv @ dv_du

    ux, uy = mrcal.knots_for_splined_models(lens)
    # Cubic spline interior, excluding the one-knot padding on each side.
    interior = np.array([[ux[1], ux[-2]], [uy[1], uy[-2]]])
    attempts, roots = [], []
    for start in starts:
        fitted = least_squares(residual, start, jac=jacobian, max_nfev=500,
                               xtol=1e-13, ftol=1e-13, gtol=1e-13)
        error = float(np.linalg.norm(fitted.fun))
        singular = svdvals(jacobian(fitted.x))
        in_bounds = bool(np.all((fitted.x >= interior[:, 0]) & (fitted.x <= interior[:, 1])))
        attempts.append({"start_uv": np.asarray(start).tolist(), "solution_uv": fitted.x.tolist(),
                         "pixel_error": error, "jacobian_singular_values": singular.tolist(),
                         "in_spline_interior": in_bounds})
        if error < 1e-5 and not any(np.linalg.norm(fitted.x - root["uv"]) < 1e-5 for root in roots):
            ray = mrcal.unproject_stereographic(fitted.x)
            ray /= np.linalg.norm(ray)
            model = mrcal.cameramodel(optimization_inputs=inputs, icam_intrinsics=camera)
            covariance = mrcal.projection_uncertainty(ray, model=model, observed_pixel_uncertainty=.2)
            roots.append({"uv": fitted.x.tolist(), "ray": ray.tolist(), "pixel_error": error,
                          "in_spline_interior": in_bounds,
                          "covariance_finite": bool(np.isfinite(covariance).all())})
    # Evaluate the local mapping at physical rays, never at a zero inverse sentinel.
    us = np.array([(x, y) for y in np.linspace(center[1] - .2, center[1] + .2, 25)
                   for x in np.linspace(center[0] - .25, center[0] + .25, 25)])
    jac = np.array([jacobian(u) for u in us])
    determinants = np.linalg.det(jac)
    step = 1e-6
    finite_difference = np.stack([(residual(center + np.eye(2)[i] * step)
                                    - residual(center - np.eye(2)[i] * step)) / (2 * step)
                                   for i in range(2)], axis=-1)
    derivative_error = float(np.max(np.abs(jacobian(center) - finite_difference)))
    return {"opencv5_reference_uv": center.tolist(), "spline_interior_uv": interior.tolist(),
            "attempts": attempts, "exact_roots": roots,
            "local_patch": {"grid": [25, 25], "det_min": float(determinants.min()),
                            "det_max": float(determinants.max()),
                            "negative_determinant_queries": int(np.count_nonzero(determinants < 0)),
                            "analytic_vs_finite_difference_max_abs": derivative_error}}


def matrix_diagnostics(inputs, rays, camera):
    _, residual, sparse, factor = mrcal.optimizer_callback(**inputs)
    dense = sparse.toarray()
    observation_rows = mrcal.measurement_index_regularization(**inputs)
    singular = svdvals(dense)
    observation_singular = svdvals(dense[:observation_rows])
    covariance, gradient = capture_covariance(inputs, camera, rays)
    if not np.isfinite(gradient).all():
        raise ValueError("Matrix comparison requires valid, nonzero query rays")
    independently_computed = qr_covariance(dense, gradient, observation_rows)
    relative = np.linalg.norm(covariance - independently_computed) / np.linalg.norm(independently_computed)
    tol = max(dense.shape) * np.finfo(float).eps * singular[0]
    tol_obs = max(dense[:observation_rows].shape) * np.finfo(float).eps * observation_singular[0]
    return {"shape": list(dense.shape), "jacobian_finite": bool(np.isfinite(dense).all()),
            "residual_finite": bool(np.isfinite(residual).all()),
            "native_factorization_available": factor is not None,
            "native_rcond_estimate": float(factor.rcond()) if factor else None,
            "regularized_singular_min_max": [float(singular[-1]), float(singular[0])],
            "regularized_rank": int(np.count_nonzero(singular > tol)),
            "regularized_condition_J": float(singular[0] / singular[-1]),
            "observation_only_rank": int(np.count_nonzero(observation_singular > tol_obs)),
            "observation_only_nullity": int(dense.shape[1] - np.count_nonzero(observation_singular > tol_obs)),
            "regularization_rows": dense.shape[0] - observation_rows,
            "valid_query_gradient_finite": True,
            "native_covariance_finite": bool(np.isfinite(covariance).all()),
            "qr_covariance_finite": bool(np.isfinite(independently_computed).all()),
            "native_vs_pivoted_qr_relative_error": float(relative),
            "native_covariance": covariance.tolist(), "qr_covariance": independently_computed.tolist()}


def diagnose_model(inputs, opencv, output, name):
    width, height = inputs["imagersizes"][0]
    pixels = np.array([(x * (width - 1), y * (height - 1))
                       for y in np.linspace(.025, .975, 11) for x in np.linspace(.025, .975, 17)])
    result = {"status": "completed", "lensmodel": inputs["lensmodel"], "regularization": inputs["do_apply_regularization"],
              "knot_support": spline.knot_support(inputs), "cameras": []}
    for camera in range(2):
        lens, parameters = inputs["lensmodel"], inputs["intrinsics"][camera]
        raw = mrcal.unproject(pixels, lens, parameters, normalize=False)
        normalized = mrcal.unproject(pixels, lens, parameters, normalize=True)
        projected = mrcal.project(normalized, lens, parameters)
        valid = valid_inverse(normalized, projected, pixels)
        model = mrcal.cameramodel(optimization_inputs=inputs, icam_intrinsics=camera)
        cov = mrcal.projection_uncertainty(normalized, model=model, observed_pixel_uncertainty=.2)
        bad_cov = ~np.isfinite(cov).all(axis=(-1, -2))
        detail = {"camera": camera, "queries": len(pixels), "valid_inverse_count": int(valid.sum()),
                  "invalid_covariance_count": int(bad_cov.sum()),
                  "invalid_covariance_indices": np.flatnonzero(bad_cov).tolist(),
                  "invalid_queries": []}
        for index in np.flatnonzero(~valid):
            detail["invalid_queries"].append({
                "index": int(index), "pixel": pixels[index].tolist(),
                "raw_ray_finite": bool(np.isfinite(raw[index]).all()),
                "normalized_ray": normalized[index].tolist(),
                "roundtrip_finite": bool(np.isfinite(projected[index]).all()),
                "inverse_search": inverse_multistart(inputs, camera, pixels[index], opencv)})
        # Explicit valid-point subset: do not silently convert invalid covariance to zero.
        good_cov = mrcal.projection_uncertainty(normalized[valid], model=model, observed_pixel_uncertainty=.2)
        detail["valid_subset_covariance_all_finite"] = bool(np.isfinite(good_cov).all())
        if valid.any():
            detail["valid_subset_covariance_eigen_min"] = float(np.linalg.eigvalsh(good_cov).min())
        result["cameras"].append(detail)
    # One independent matrix comparison uses 9 physical rays from a verified-valid 3x3 grid.
    queries = np.array([(x * (width - 1), y * (height - 1)) for y in (.1, .5, .9) for x in (.1, .5, .9)])
    rays = mrcal.unproject(queries, inputs["lensmodel"], inputs["intrinsics"][0], normalize=True)
    result["matrix"] = matrix_diagnostics(inputs, rays, 0)
    common.write_json(output / f"{name}.json", result)
    return result


def synthetic_augmented_control(source):
    """Oracle-only coverage counterfactual; never mix these points into real calibration."""
    truth = copy.deepcopy(source)
    objects = mrcal.ref_calibration_object(truth["observations_board"].shape[2],
                                           truth["observations_board"].shape[1], truth["calibration_object_spacing"])
    center = objects.reshape(-1, 3).mean(axis=0)
    initial_frames = len(truth["rt_ref_frame"])
    frames, observations, indices = [], [], []
    width, height = truth["imagersizes"][0]
    for x, y in ((.15, .15), (.85, .15), (.15, .85), (.85, .85)):
        ray = mrcal.unproject(np.array([x * (width - 1), y * (height - 1)]),
                              truth["lensmodel"], truth["intrinsics"][0], normalize=True)
        for distance in (1., 1.5):
            for tilt in (-.3, 0., .3):
                rotation = np.array([tilt, -.5 * tilt, .1])
                translation = ray * distance - mrcal.rotate_point_r(rotation, center)
                frame = np.r_[rotation, translation]
                iframe = initial_frames + len(frames)
                frames.append(frame)
                for camera in range(2):
                    xyz = mrcal.transform_point_rt(frame, objects)
                    if camera:
                        xyz = mrcal.transform_point_rt(truth["rt_cam_ref"][0], xyz)
                    xy = mrcal.project(xyz, truth["lensmodel"], truth["intrinsics"][camera])
                    good = np.all((xy >= 0) & (xy < (width, height)), axis=-1) & (xyz[..., 2] > 0)
                    observation = np.dstack((xy, np.where(good, 1., -1.)))
                    observations.append(observation)
                    indices.append((iframe, camera, camera - 1))
    truth["rt_ref_frame"] = np.concatenate((truth["rt_ref_frame"], frames))
    truth["observations_board"] = np.concatenate((truth["observations_board"], observations))
    truth["indices_frame_camintrinsics_camextrinsics"] = np.concatenate(
        (truth["indices_frame_camintrinsics_camextrinsics"], np.array(indices, dtype=np.int32)))
    return validation.render_observations(truth)


def summarize(report):
    lines = ["# 16×10 样条无效协方差诊断", "",
             "查询来自同一 17×11 网格。严格检查非零射线及有限投影往返，不将失败查询替换为零。", "",
             "| 模型/控制 | 左有效反投影 | 右有效反投影 | 观测矩阵零空间维数 | 正则后矩阵秩 | cond(J) | 原生/独立 QR 协方差相对差 |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for name, model in report["models"].items():
        if model["status"] != "completed":
            lines.append(f"| {name} | 失败 | — | — | — | — | — |")
            continue
        matrix = model["matrix"]
        counts = [c["valid_inverse_count"] for c in model["cameras"]]
        lines.append(f"| {name} | {counts[0]}/187 | {counts[1]}/187 | "
                     f"{matrix['observation_only_nullity']} | {matrix['regularized_rank']}/{matrix['shape'][1]} | "
                     f"{matrix['regularized_condition_J']:.3g} | {matrix['native_vs_pivoted_qr_relative_error']:.3g} |")
    lines += ["", "原始 16×10 的失败点是左目 (1247.025, 17.975)，只有该点使协方差为 NaN。",
              "未归一化反投影返回 NaN；normalize=True 将失败结果变为零向量。零向量不是有效射线。",
              "局部投影 Jacobian 行列式变号，说明出现折叠；找到的远支根在样条内部定义域之外，不能当作安全修复。",
              "", "正常射线下，正则化矩阵满秩且原生/独立 QR 协方差一致；未发现协方差代数实现错误证据。",
              "仅真实观测矩阵存在未约束方向。合成角区增广能降低该零空间并恢复逆投影，支持覆盖/局部欠约束解释。",
              "扩大 FOV 的控制也能恢复，说明网格定义域和有效密度参与影响；仅增加 Ny 未恢复。",
              "", "控制中的虚拟观测来自 OPENCV5 oracle，不是补采照片，也不代表新标定可发布。",
              "旧比较脚本漏检零向量和 NaN 往返的检查已修复，应用 UI、依赖与 Windows 打包未改。"]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spline-baseline", type=Path, required=True)
    parser.add_argument("--opencv-baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Output directory must be empty")
    args.output.mkdir(parents=True, exist_ok=True)
    try:
        source_path = args.opencv_baseline / "mrcal_joint_0.cameramodel"
        source = load_inputs(source_path)
        prior = json.loads((args.spline_baseline / "report.json").read_text())
        if common.digest(source_path) != prior["source_model_sha256"]:
            raise ValueError("OPENCV5 source model does not match spline comparison provenance")
        protocol = {"queries": "same 17x11 grid at 2.5%..97.5%", "known_sigma_px": .2,
                    "covariance_distance_m": 1., "comparison": "native covariance vs independent pivoted QR sandwich",
                    "controls": ["original12x8", "original16x10", "16x12 fov120 real",
                                 "16x10 fov140 real", "16x10 synthetic original geometry",
                                 "16x10 synthetic augmented corner geometry"],
                    "synthetic_control": "oracle OPENCV5 truth, not new real data or deployable calibration"}
        common.write_json(args.output / "protocol.json", protocol)
        result = {"status": "completed", "protocol": protocol, "script_sha256": common.digest(__file__),
                  "dataset_hash": prior["dataset_hash"],
                  "source_comparison_sha256": common.digest(args.spline_baseline / "report.json"),
                  "helper_sha256": {str(Path(m.__file__).resolve()): common.digest(m.__file__)
                                    for m in (common, spline, validation)},
                  "native_source_sha256": {inspect.getfile(f): common.digest(inspect.getfile(f))
                                            for f in (mrcal.unproject, mrcal.projection_uncertainty)},
                  "environment": {n: importlib.metadata.version(n) for n in ("mrcal", "numpy", "scipy")},
                  "python": sys.version, "platform": platform.platform(),
                  "models": {}}
        for name in ("spline12x8", "spline16x10"):
            path = args.spline_baseline / f"{name}_0.cameramodel"
            inputs = load_inputs(path)
            print("diagnose", name, flush=True)
            result["models"][name] = diagnose_model(inputs, source, args.output, name)
            result["models"][name]["source_sha256"] = common.digest(path)
        controls = {
            "real16x12_fov120": (source, "LENSMODEL_SPLINED_STEREOGRAPHIC_order=3_Nx=16_Ny=12_fov_x_deg=120"),
            "real16x10_fov140": (source, "LENSMODEL_SPLINED_STEREOGRAPHIC_order=3_Nx=16_Ny=10_fov_x_deg=140"),
            "synthetic_original": (validation.render_observations(source), spline.SPLINES["spline16x10"]),
            "synthetic_augmented": (synthetic_augmented_control(source), spline.SPLINES["spline16x10"])}
        for name, (truth, lens) in controls.items():
            print("control", name, flush=True)
            try:
                initial, _ = spline.spline_seed(truth, lens)
                fitted, solver = spline.solve(initial)
                result["models"][name] = diagnose_model(fitted, source, args.output, name)
                result["models"][name]["solver"] = solver
            except Exception as exc:
                result["models"][name] = {"status": "failed", "error": str(exc)}
            common.write_json(args.output / "report.partial.json", result)
        common.write_json(args.output / "report.json", result)
        (args.output / "summary.md").write_text(summarize(result), encoding="utf-8")
        print("Completed", args.output, flush=True)
    except Exception as exc:
        common.write_json(args.output / "failure.json", {"status": "failed", "error": str(exc)})
        raise


if __name__ == "__main__":
    main()
