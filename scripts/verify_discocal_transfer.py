"""Independent DiscoCal-inspired measurement study; no circular detector code is reused.

Image perturbations measure conditional detector sensitivity, NOT physical sensor noise.
Weights use training images only; all original observations and frozen splits are retained.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

import compare_mrcal as common
import verify_calibration_information as ba


def polygon_centroid(vertices):
    """Area centroid of a simple polygon, independent of orientation."""
    following = np.roll(vertices, -1, axis=0)
    cross = vertices[:, 0] * following[:, 1] - following[:, 0] * vertices[:, 1]
    if abs(cross.sum()) < 1e-12:
        raise ValueError("Degenerate projected region")
    return ((vertices + following) * cross[:, None]).sum(axis=0) / (3 * cross.sum())


def circle_control(samples=8192):
    """Numerical region integration, not DiscoCal's analytic conic estimator."""
    k = np.array([[800., 0., 640.], [0., 800., 360.], [0., 0., 1.]])
    center = np.array([.32, .12, 0.])
    radius = .04
    results = {}
    for name, rv, distortion in (
        ("frontoparallel_undistorted", [0., 0., 0.], [0.] * 5),
        ("perspective_only", [.7, -.4, 0.], [0.] * 5),
        ("radial_only", [0., 0., 0.], [-.3, .08, 0., 0., 0.]),
        ("opencv5_combined", [.7, -.4, 0.], [-.3, .08, .002, -.001, 0.]),
    ):
        rv, distortion = np.array(rv), np.array(distortion)
        projected_center = cv2.projectPoints(center[None], rv, np.array([0., 0., .7]), k, distortion)[0].reshape(2)
        centroids = []
        for count in (samples // 2, samples):
            angle = np.arange(count) * 2 * np.pi / count
            boundary = center + radius * np.c_[np.cos(angle), np.sin(angle), np.zeros(count)]
            projected = cv2.projectPoints(boundary, rv, np.array([0., 0., .7]), k, distortion)[0].reshape(-1, 2)
            centroids.append(polygon_centroid(projected))
        results[name] = {"center_to_region_centroid_px": float(np.linalg.norm(centroids[-1] - projected_center)),
                         "integration_refinement_delta_px": float(np.linalg.norm(centroids[1] - centroids[0]))}
    return results


def covariance_statistics(draws, floor_variance=1e-4, shrinkage=.1):
    """Per-corner covariance; incomplete corners never disappear from calibration."""
    valid = np.isfinite(draws).all(axis=-1)
    count = valid.sum(axis=0)
    mean = np.where(valid[..., None], draws, 0.).sum(axis=0) / np.maximum(count[:, None], 1)
    delta = np.where(valid[..., None], draws - mean, 0.)
    covariance = np.einsum("sni,snj->nij", delta, delta) / np.maximum(count[:, None, None] - 1, 1)
    covariance = ((1 - shrinkage) * covariance
                  + shrinkage * np.trace(covariance, axis1=-2, axis2=-1)[:, None, None] / 2 * np.eye(2)
                  + floor_variance * np.eye(2))
    usable = count >= int(np.ceil(.75 * len(draws)))
    return mean, covariance, usable, count


def covariance_check(train_draws, check_draws):
    """Independent perturbation draws check covariance, including estimated mean error."""
    mean, cov, usable, count = covariance_statistics(train_draws)
    delta = check_draws - mean
    finite = np.isfinite(delta).all(axis=-1) & usable[None]
    adjusted = cov * (1 + 1 / np.maximum(count[:, None, None], 1))
    nees = np.einsum("sni,nij,snj->sn", np.nan_to_num(delta), np.linalg.inv(adjusted), np.nan_to_num(delta))
    values = nees[finite]
    stable = np.isfinite(train_draws).all(axis=(0, 2))
    ratio = None
    if stable.sum() >= 2:
        subset = train_draws[:, stable]
        common_mode = np.var(subset.mean(axis=1), axis=0, ddof=1).sum()
        independent = np.var(subset, axis=0, ddof=1).sum() / stable.sum() ** 2
        if independent > 0:
            ratio = float(common_mode / independent)
    return {"samples": len(values), "nees_sum": float(values.sum()),
            "inside_95_count": int((values <= 5.991464547).sum()),
            "mean_nees": float(values.mean()) if len(values) else None,
            "coverage_95": float((values <= 5.991464547).mean()) if len(values) else None,
            "frame_mean_variance_over_independent": ratio}


def detector_draws(gray, detector, ids, rng, draws=64, noise_sigma=2.):
    samples = np.full((draws, len(ids), 2), np.nan)
    lookup = {int(corner_id): i for i, corner_id in enumerate(ids)}
    # Re-run the complete application detector, including markers and interpolation.
    for index in range(draws):
        perturbed = np.clip(np.rint(gray.astype(np.float32) + rng.normal(0., noise_sigma, gray.shape)), 0, 255).astype(np.uint8)
        detected = detector.detect(perturbed)
        if detected.charuco_ids is None:
            continue
        for corner_id, xy in zip(detected.charuco_ids.ravel(), detected.charuco_corners.reshape(-1, 2)):
            if int(corner_id) in lookup:
                samples[index, lookup[int(corner_id)]] = xy
    return samples


def training_weights(session, records, board_config, output, draws=64):
    detector = common.CharucoDetector(common.CharucoBoard.from_dict(board_config))
    rng = np.random.default_rng(20261101)
    raw, failures, diagnostics, arrays = [], [], [], {}
    for frame, record in enumerate(records):
        frame_raw, frame_bad = [], []
        for camera, side in enumerate(("left", "right")):
            # Match original color decode -> gray conversion exactly.
            image = cv2.imread(str(session / record["paths"][camera]))
            if image is None:
                raise ValueError("Missing training image")
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            baseline = detector.detect(gray)
            mapping = {} if baseline.charuco_ids is None else dict(zip(baseline.charuco_ids.ravel(), baseline.charuco_corners.reshape(-1, 2)))
            if any(i not in mapping or np.linalg.norm(mapping[i] - xy) > 1e-4
                   for i, xy in zip(record["ids"], record[side])):
                raise ValueError("Detector differs from frozen observation cache")
            samples = detector_draws(gray, detector, record["ids"], rng, draws)
            half = draws // 2
            mean, cov, good, count = covariance_statistics(samples[:half])
            precision = np.linalg.inv(cov)
            frame_raw.append(precision)
            frame_bad.append(~good)
            check = covariance_check(samples[:half], samples[half:])
            check.update(capture_id=record["capture_id"], camera=camera,
                         missing_draw_observations=int((~np.isfinite(samples).all(axis=-1)).sum()),
                         fallback_corners=int((~good).sum()), corners=len(good),
                         median_detector_shift_px=float(np.median(np.linalg.norm(mean[good] - np.array(record[side])[good], axis=1))) if good.any() else None,
                         median_sigma_px=float(np.median(np.sqrt(np.trace(cov[good], axis1=-2, axis2=-1) / 2))) if good.any() else None)
            diagnostics.append(check)
            arrays[f"frame{frame}_cam{camera}_draws"] = samples
            arrays[f"frame{frame}_cam{camera}_covariance"] = cov
            arrays[f"frame{frame}_cam{camera}_counts"] = count
        raw.append(frame_raw)
        failures.append(frame_bad)
        print("perturbation", record["capture_id"], f"{frame + 1}/{len(records)}", flush=True)
    strengths = np.concatenate([np.trace(m, axis1=-2, axis2=-1)[~bad] / 2
                                for frame, bad_frame in zip(raw, failures) for m, bad in zip(frame, bad_frame)])
    if not len(strengths):
        raise ValueError("No usable training covariance estimates")
    scale = float(np.median(strengths))
    normalized = [[ba.normalize_information(m, scale, bad)[0] for m, bad in zip(frame, bad_frame)]
                  for frame, bad_frame in zip(raw, failures)]
    np.savez_compressed(output / "training_perturbations.npz", **arrays)
    samples_total = sum(d["samples"] for d in diagnostics)
    return normalized, {"normalization_scale": scale, "per_image": diagnostics,
                        "perturbation_check_mean_nees": sum(d["nees_sum"] for d in diagnostics) / samples_total,
                        "perturbation_check_coverage_95": sum(d["inside_95_count"] for d in diagnostics) / samples_total,
                        "fallback_corners": sum(d["fallback_corners"] for d in diagnostics),
                        "image_observations": sum(d["corners"] for d in diagnostics),
                        "missing_draw_observations": sum(d["missing_draw_observations"] for d in diagnostics)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cv2.setNumThreads(1)
    args.output.mkdir(parents=True, exist_ok=False)
    data = common.collect_observations(args.session.resolve(), "rgb_pair_1", args.baseline / "observations.json")
    train, holdout, split = common.split_records(data["records"], .25, 2)
    historical = json.loads((args.baseline / "report.json").read_text())
    if split != historical.get("split", historical.get("protocol", {}).get("split")):
        raise ValueError("Historical split changed")
    protocol = {"dataset_hash": data["dataset_hash"], "split": split,
                "draws": 64, "covariance_fit_draws": 32, "independent_check_draws": 32,
                "seed": 20261101, "noise_sigma_DN": 2., "noise": "rounded clipped additive Gaussian in decoded grayscale",
                "covariance_floor_px2": .0001, "isotropic_shrinkage": .1,
                "minimum_detection_fraction": .75, "precision_eigenvalue_bounds": [.25, 4.],
                "normalization": "training-only median half precision trace",
                "loss": "quadratic Mahalanobis, no residual-dependent outlier rejection",
                "holdout": "fixed historical split; unweighted evaluation only; no tuning",
                "meaning": "conditional detector sensitivity to specified artificial noise; no physical confidence claim",
                "cross_corner_covariance": "diagnosed, not inverted or included in BA",
                "disco_source_commit": "500a665d7469efb8f61f3c50f95337420a762149",
                "script_sha256": common.digest(__file__), "ba_sha256": common.digest(ba.__file__),
                "common_sha256": common.digest(common.__file__), "opencv": cv2.__version__, "numpy": np.__version__}
    common.write_json(args.output / "protocol.json", protocol)
    precision, weights = training_weights(args.session, train, data["board"], args.output)
    report = {"protocol": protocol, "circle_control": circle_control(), "weights": weights, "models": {}}
    common.write_json(args.output / "report.partial.json", report)
    objects, size = np.array(data["object_points_m"]), tuple(data["image_size"])
    initial = common.solve_opencv(train, objects, size)["opencv_joint"]
    for mode in ("uniform", "scalar", "anisotropic"):
        if mode == "uniform":
            information = [[np.tile(np.eye(2), (len(m), 1, 1)) for m in frame] for frame in precision]
        elif mode == "scalar":
            information = [[np.trace(m, axis1=-2, axis2=-1)[:, None, None] / 2 * np.eye(2) for m in frame] for frame in precision]
        else:
            information = precision
        print("BA", mode, flush=True)
        state, model, solver = ba.fit(train, objects, initial, information)
        report["models"][mode] = {"solver": solver, "model": common.serializable(model),
                                  "training": common.evaluate(train, objects, size, model),
                                  "validation": common.evaluate(holdout, objects, size, model)}
        np.savez_compressed(args.output / f"{mode}_state.npz", state=state)
        common.write_json(args.output / "report.partial.json", report)
        print(mode, report["models"][mode]["validation"]["cross_camera_transfer"], flush=True)
    report["status"] = "completed"
    common.write_json(args.output / "report.json", report)


if __name__ == "__main__":
    main()
