"""Offline fault injection and Monte Carlo validation of mrcal's OPENCV5 backend."""
from __future__ import annotations

import argparse
import copy
import importlib.metadata
import json
from pathlib import Path
import platform
import sys

import cv2
import mrcal
import numpy as np
from scipy.stats import binomtest, chi2

import compare_mrcal as comparison


def copy_inputs(inputs):
    """Remove serialized legacy error sentinels; fill board-only API optional keys."""
    result = copy.deepcopy(inputs)
    for key in ("frames_rt_toref", "extrinsics_rt_fromref"):
        result.pop(key, None)
    for key in ("points", "observations_point", "indices_point_camintrinsics_camextrinsics"):
        result.setdefault(key, None)
    result.update(do_apply_outlier_rejection=False, do_apply_regularization=False,
                  do_optimize_calobject_warp=False, verbose=False)
    return result


def render_observations(inputs):
    """Generate exact OPENCV5 observations at existing capture poses, retaining missing slots."""
    result = copy_inputs(inputs)
    grid = mrcal.ref_calibration_object(
        result["observations_board"].shape[2], result["observations_board"].shape[1],
        result["calibration_object_spacing"])
    for i, (frame, camera, extrinsic) in enumerate(
            result["indices_frame_camintrinsics_camextrinsics"]):
        xyz = mrcal.transform_point_rt(result["rt_ref_frame"][frame], grid)
        if extrinsic >= 0:
            xyz = mrcal.transform_point_rt(result["rt_cam_ref"][extrinsic], xyz)
        result["observations_board"][i, ..., :2] = mrcal.project(
            xyz, result["lensmodel"], result["intrinsics"][camera])
    return result


def add_noise(inputs, sigma, rng):
    result = copy_inputs(inputs)
    valid = result["observations_board"][..., 2] > 0
    result["observations_board"][..., :2][valid] += rng.normal(0, sigma, (valid.sum(), 2))
    return result


def inject_outliers(inputs, fraction, magnitude, rng):
    """Inject exact-magnitude random-direction shifts only at originally valid corner slots."""
    result = copy_inputs(inputs)
    valid = result["observations_board"][..., 2] > 0
    slots = np.argwhere(valid)
    count = int(round(len(slots) * fraction))
    selected = slots[rng.choice(len(slots), count, replace=False)]
    truth = np.zeros(valid.shape, dtype=bool)
    if count:
        location = tuple(selected.T)
        theta = rng.uniform(0, 2 * np.pi, count)
        result["observations_board"][..., :2][location] += magnitude * np.c_[np.cos(theta), np.sin(theta)]
        truth[location] = True
    return result, truth


def detection_metrics(original_valid, injected, output_weights):
    rejected = original_valid & (output_weights < 0)
    tp = int(np.count_nonzero(rejected & injected))
    fp = int(np.count_nonzero(rejected & ~injected))
    fn = int(np.count_nonzero(injected & ~rejected))
    negatives = int(np.count_nonzero(original_valid & ~injected))
    return {"injected": int(injected.sum()), "true_positive": tp, "false_positive": fp,
            "false_negative": fn, "precision": tp / (tp + fp) if tp + fp else None,
            "recall": tp / (tp + fn) if tp + fn else None,
            "false_positive_rate": fp / negatives if negatives else None,
            "rejected_slots": np.argwhere(rejected).tolist()}


def fit(inputs, rejection=False):
    result = copy_inputs(inputs)
    result["do_apply_outlier_rejection"] = rejection
    stats = mrcal.optimize(**result)
    for name in ("intrinsics", "rt_cam_ref", "rt_ref_frame"):
        if not np.isfinite(result[name]).all():
            raise ValueError(f"Nonfinite optimized {name}")
    return result, {k: float(v) for k, v in stats.items() if np.isscalar(v)}


def opencv_model(inputs):
    params = inputs["intrinsics"]
    return {"K": [np.array([[p[0], 0, p[2]], [0, p[1], p[3]], [0, 0, 1.]]) for p in params],
            "D": [p[4:].copy() for p in params],
            "R": cv2.Rodrigues(inputs["rt_cam_ref"][0, :3])[0],
            "T": inputs["rt_cam_ref"][0, 3:].copy()}


def camera_rt(inputs, camera):
    return np.zeros(6) if camera == 0 else inputs["rt_cam_ref"][camera - 1]


def reproject_mean_pcam(baseline, sampled, pixels, camera, distance):
    """Match mrcal's mean-pcam uncertainty convention, including frame-motion compensation.

    See upstream test/test-projection-uncertainty.py: reproject_perturbed__common.
    No covariance/Jacobian predictions are used to generate these Monte Carlo samples.
    """
    rays = mrcal.unproject(pixels, baseline["lensmodel"], baseline["intrinsics"][camera], normalize=True)
    rt0, rt1 = camera_rt(baseline, camera), camera_rt(sampled, camera)
    if distance is None:
        reference = mrcal.rotate_point_r(-rt0[:3], rays)
        frames = mrcal.rotate_point_r(-baseline["rt_ref_frame"][:, None, :3], reference)
        moved = mrcal.rotate_point_r(sampled["rt_ref_frame"][:, None, :3], frames)
        camera_points = mrcal.rotate_point_r(rt1[:3], moved)
    else:
        reference = mrcal.transform_point_rt(mrcal.invert_rt(rt0), rays * distance)
        frames = mrcal.transform_point_rt(mrcal.invert_rt(baseline["rt_ref_frame"])[:, None, :], reference)
        moved = mrcal.transform_point_rt(sampled["rt_ref_frame"][:, None, :], frames)
        camera_points = mrcal.transform_point_rt(rt1, moved)
    return mrcal.project(camera_points.mean(axis=0), sampled["lensmodel"], sampled["intrinsics"][camera])


def projection_samples(baseline, sampled, pixels, distances):
    return np.array([[reproject_mean_pcam(baseline, sampled, pixels, c, d)
                      for d in distances] for c in range(2)])


def outlier_experiment(real, ideal, validation, objects, size, output, seed):
    results = []
    original_valid = real["observations_board"][..., 2] > 0
    for domain, base in (("model_consistent", ideal), ("real_observations", real)):
        # Known false positives apply only to model-consistent data. Original real points
        # can already contain unknown errors; record extra removals without claiming truth.
        for trial in range(3):
            noisy = (add_noise(base, .2, np.random.default_rng(seed + trial))
                     if domain == "model_consistent" else copy_inputs(base))
            for fraction, magnitude in ((0., 0.), (.01, 8.), (.05, 20.)):
                corrupted, injected = inject_outliers(
                    noisy, fraction, magnitude, np.random.default_rng(seed + 100 + trial))
                case = {"domain": domain, "trial": trial, "fraction": fraction,
                        "shift_px": magnitude, "injected_slots": np.argwhere(injected).tolist(),
                        "false_positive_ground_truth_known": domain == "model_consistent",
                        "fits": {}}
                for rejection in (False, True):
                    label = "rejection_on" if rejection else "rejection_off"
                    try:
                        solved, stats = fit(corrupted, rejection)
                        measured = detection_metrics(original_valid, injected,
                                                     solved["observations_board"][..., 2])
                        val = comparison.evaluate(validation, objects, size, opencv_model(solved))
                        case["fits"][label] = {"status": "completed", "detection": measured,
                                              "validation_transfer": val["cross_camera_transfer"],
                                              "baseline_mm": float(np.linalg.norm(solved["rt_cam_ref"][0, 3:]) * 1000),
                                              "solver_stats": stats}
                    except Exception as exc:
                        case["fits"][label] = {"status": "failed", "error": str(exc)}
                results.append(case)
                print(f"outliers {domain} trial={trial} fraction={fraction} complete", flush=True)
                comparison.write_json(output / "outliers.partial.json", results)
    return results


def covariance_check(samples, expected, query_pixels, sigma, family_queries):
    """Finite-sample checks with familywise correction; queries are not independent trials."""
    count = len(samples)
    alpha = .05 / (family_queries * 2)  # two whitened coordinate variances per query
    bounds = chi2.ppf([alpha / 2, 1 - alpha / 2], count - 1) / (count - 1)
    reports = []
    for index in np.ndindex(expected.shape[:-2]):
        cov = expected[index]
        eigenvalues = np.linalg.eigvalsh(cov)
        if eigenvalues[0] <= 0 or not np.isfinite(cov).all():
            raise ValueError("Predicted covariance is not positive definite")
        errors = samples[(slice(None),) + index] - query_pixels[index[-1]]
        chol = np.linalg.cholesky(cov)
        whitened = np.linalg.solve(chol, errors.T).T
        variances = np.var(whitened, axis=0, ddof=1)
        observed_cov = np.cov(errors, rowvar=False, ddof=1)
        mahalanobis2 = np.sum(whitened ** 2, axis=1)
        covered = int(np.count_nonzero(mahalanobis2 <= chi2.ppf(.95, 2)))
        # Each query has N independent solver samples; no pooling across correlated query points.
        coverage_p = binomtest(covered, count, .95).pvalue
        reports.append({"camera": index[0], "distance_index": index[1], "pixel_index": index[2],
                        "predicted_covariance_px2": cov.tolist(),
                        "empirical_covariance_px2": observed_cov.tolist(),
                        "predicted_worst_stdev_px": float(np.sqrt(eigenvalues[-1])),
                        "empirical_worst_stdev_px": float(np.sqrt(np.linalg.eigvalsh(observed_cov)[-1])),
                        "stdev_ratio": float(np.sqrt(np.linalg.eigvalsh(observed_cov)[-1] / eigenvalues[-1])),
                        "whitened_variances": variances.tolist(),
                        "whitened_mean": whitened.mean(axis=0).tolist(),
                        "variance_check": bool(np.all((variances >= bounds[0]) & (variances <= bounds[1]))),
                        "ellipse95_coverage": covered / count,
                        "ellipse95_binomial_pvalue": float(coverage_p),
                        "coverage_check": bool(coverage_p >= .05 / family_queries)})
    return {"sigma_px_per_coordinate": sigma, "samples": count,
            "variance_ratio_bounds_familywise": bounds.tolist(),
            "queries": reports, "variance_checks_pass": all(q["variance_check"] for q in reports),
            "coverage_checks_pass": all(q["coverage_check"] for q in reports)}


def uncertainty_experiment(ideal, output, count, seed):
    # Re-solve noise-free observations so prediction and MC share an exact model-consistent optimum.
    baseline, _ = fit(ideal)
    width, height = baseline["imagersizes"][0]
    pixels = np.array([(x * (width - 1), y * (height - 1))
                       for y in (.1, .5, .9) for x in (.1, .5, .9)])
    distances = (1., 3., None)
    sigmas = (.1, .3, .6)
    covariance_unit = []
    for camera in range(2):
        model = mrcal.cameramodel(optimization_inputs=baseline, icam_intrinsics=camera)
        rays = mrcal.unproject(pixels, *model.intrinsics(), normalize=True)
        covariance_unit.append([
            mrcal.projection_uncertainty(rays if d is None else rays * d, model=model,
                                        method="mean-pcam", atinfinity=d is None,
                                        what="covariance", observed_pixel_uncertainty=1.)
            for d in distances])
    covariance_unit = np.array(covariance_unit)
    # Identity mapping must reproduce every query, including at infinity.
    np.testing.assert_allclose(projection_samples(baseline, baseline, pixels, distances),
                               np.broadcast_to(pixels, (2, len(distances), *pixels.shape)), atol=1e-7)
    summaries, failures = [], []
    for level, sigma in enumerate(sigmas):
        rng = np.random.default_rng(np.random.SeedSequence([seed, level]))
        samples = []
        for trial in range(count):
            try:
                solved, _ = fit(add_noise(baseline, sigma, rng), rejection=False)
                np.testing.assert_array_equal(solved["observations_board"][..., 2],
                                              baseline["observations_board"][..., 2])
                samples.append(projection_samples(baseline, solved, pixels, distances))
            except Exception as exc:
                failures.append({"sigma": sigma, "trial": trial, "error": str(exc)})
            if (trial + 1) % 50 == 0:
                print(f"uncertainty sigma={sigma}: {trial+1}/{count}", flush=True)
        samples = np.array(samples)
        np.savez_compressed(output / f"uncertainty_sigma_{sigma}.npz", projections=samples,
                            predicted_covariance=covariance_unit * sigma ** 2,
                            pixels=pixels, distances=np.array([1., 3., np.inf]))
        if len(samples) < 30:
            raise ValueError("Too few successful Monte Carlo trials")
        summaries.append(covariance_check(samples, covariance_unit * sigma ** 2, pixels,
                                          sigma, family_queries=len(sigmas) * 2 * len(distances) * len(pixels)))
    return {"method": "mean-pcam", "truth": "noise-free reprojection at original training geometry",
            "pixels": pixels.tolist(), "distances_m": [1., 3., "infinity"],
            "samples_requested_per_sigma": count, "noise_seed": seed,
            "noise": "independent zero-mean Gaussian, known sigma per x/y coordinate, valid slots only",
            "outlier_rejection": False, "predicted_sigma": "known injected sigma, not inferred from residuals",
            "familywise_alpha": .05, "failures": failures, "levels": summaries,
            "checks_pass": not failures and all(s["variance_checks_pass"] and s["coverage_checks_pass"]
                                                for s in summaries)}


def summarize(report):
    lines = ["# mrcal 异常角点与不确定度验证", "",
             "仅离线实验；没有修改原图、应用 UI、应用依赖或 Windows 打包。", "",
             "## 异常角点", "",
             "合成域沿用真实采图姿态和缺点掩码，先生成精确 OPENCV5 投影，加 0.2 px 高斯噪声后注入错点。",
             "真实域直接在原角点上注入错点；非注入点仍可能有原始错误，不能把其额外剔除一概视为误报。", "",
             "| 数据域 | 注入比例/偏移 | 完成求解 | 注入召回率 | 额外剔除数 | 关闭→开启剔除的留出跨目 RMS px |",
             "|---|---|---:|---:|---:|---:|"]
    for domain in ("model_consistent", "real_observations"):
        for fraction, shift in ((0., 0.), (.01, 8.), (.05, 20.)):
            cases = [c for c in report["outliers"] if c["domain"] == domain and c["fraction"] == fraction]
            done = [c for c in cases if all(f["status"] == "completed" for f in c["fits"].values())]
            if not done:
                lines.append(f"| {domain} | {fraction:.0%}/{shift:g}px | 0/{len(cases)} | — | — | — |")
                continue
            recalls = [c["fits"]["rejection_on"]["detection"]["recall"] for c in done]
            recall = "—" if recalls[0] is None else f"{np.mean(recalls):.1%}"
            extras = [c["fits"]["rejection_on"]["detection"]["false_positive"] for c in done]
            rms = [np.mean([c["fits"][f"rejection_{state}"]["validation_transfer"]["rms_px"] for c in done])
                   for state in ("off", "on")]
            lines.append(f"| {domain} | {fraction:.0%}/{shift:g}px | {len(done)}/{len(cases)} | {recall} | "
                         f"{min(extras)}–{max(extras)} | {rms[0]:.4f} → {rms[1]:.4f} |")
    lines += ["", "## 不确定度", "",
              "相同训练几何、精确投影真值；每坐标注入已知高斯噪声，独立重标定。",
              "按官方 mean-pcam 方法补偿坐标漂移，比较预测协方差与实际投影分布。",
              "每目 3×3 像素网格，距离 1m、3m、无穷远；统计检验对全部查询和噪声级别作多重比较校正。", "",
              "| 每坐标噪声 σ px | 成功样本 | 实测/预测标准差范围 | 方差检查 | 95% 椭圆覆盖检查 |",
              "|---|---:|---:|---|---|"]
    for level in report["uncertainty"]["levels"]:
        ratios = [q["stdev_ratio"] for q in level["queries"]]
        lines.append(f"| {level['sigma_px_per_coordinate']} | {level['samples']} | "
                     f"{min(ratios):.3f}–{max(ratios):.3f} | {level['variance_checks_pass']} | {level['coverage_checks_pass']} |")
    lines += ["", f"Monte Carlo 求解失败数：{len(report['uncertainty']['failures'])}。",
              "", "这些检查只验证给定 OPENCV5 和独立高斯角点噪声假设下的采样误差传播。",
              "不能证明真实采图不存在同步、模糊、检测偏差或镜头模型误差，也不代表完整双目测距不确定度。",
              "原始验证集仅用于固定协议下的恢复诊断，没有用来调剔除阈值或重新选择模型。",
              "本轮不验证样条模型或 Windows 部署。"]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True,
                        help="Directory from compare_mrcal.py, containing report and models")
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20261026)
    args = parser.parse_args()
    if args.samples < 30:
        parser.error("Require at least 30 samples per noise level")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Output directory must be empty")
    args.output.mkdir(parents=True, exist_ok=True)
    try:
        previous = json.loads((args.baseline / "report.json").read_text())
        data = comparison.collect_observations(args.session.resolve(), "rgb_pair_1",
                                               args.baseline / "observations.json")
        if data["dataset_hash"] != previous["dataset_hash"]:
            raise ValueError("Baseline dataset mismatch")
        records = {r["capture_id"]: r for r in data["records"]}
        validation = [records[i] for i in previous["split"]["validation"] if records[i]["usable"]]
        model_path = args.baseline / "mrcal_joint_0.cameramodel"
        real = copy_inputs(mrcal.cameramodel(str(model_path)).optimization_inputs())
        ideal = render_observations(real)
        valid = ideal["observations_board"][..., 2] > 0
        report = {"schema_version": 1, "status": "completed", "dataset_hash": data["dataset_hash"],
                  "baseline_model_sha256": comparison.digest(model_path),
                  "baseline_report_sha256": comparison.digest(args.baseline / "report.json"),
                  "script_sha256": comparison.digest(__file__),
                  "comparison_script_sha256": comparison.digest(comparison.__file__),
                  "environment": {"python": sys.version, "platform": platform.platform(),
                                  **{name: importlib.metadata.version(name)
                                     for name in ("mrcal", "numpy", "scipy", "opencv-python-headless")}},
                  "seed": args.seed, "valid_corner_slots": int(valid.sum()),
                  "input_missing_corner_slots": int((~valid).sum()),
                  "protocol": {"lensmodel": "OPENCV5", "board_warp": False, "regularization": False,
                               "outlier_trials_per_case": 3, "outlier_threshold": "mrcal built-in default",
                               "outlier_cases": [[0., 0.], [.01, 8.], [.05, 20.]],
                               "uncertainty_sigmas_px": [.1, .3, .6], "uncertainty_model": "mean-pcam",
                               "uncertainty_does_not_measure": "model errors or total stereo range uncertainty"}}
        comparison.write_json(args.output / "protocol.json", report)
        report["outliers"] = outlier_experiment(real, ideal, validation, np.array(data["object_points_m"]),
                                                tuple(data["image_size"]), args.output, args.seed)
        report["uncertainty"] = uncertainty_experiment(ideal, args.output, args.samples, args.seed)
        report["outlier_all_solves_completed"] = all(
            f["status"] == "completed" for c in report["outliers"] for f in c["fits"].values())
        comparison.write_json(args.output / "report.json", report)
        (args.output / "summary.md").write_text(summarize(report), encoding="utf-8")
        print("Completed:", args.output, "uncertainty checks:", report["uncertainty"]["checks_pass"], flush=True)
    except Exception as exc:
        comparison.write_json(args.output / "failure.json", {"status": "failed", "error": str(exc)})
        raise


if __name__ == "__main__":
    main()
