"""Offline TartanCalib-inspired ChArUco refinement; not a native AprilGrid reproduction.

All variants retain the original training IDs, split, OPENCV5 and unweighted BA.
Primary evaluation uses frozen raw holdout observations. No application writes.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import cv2
import numpy as np

import compare_mrcal as common
import verify_calibration_information as ba

SOURCE_COMMIT = "78d3f26c9f458bc60c26012399c06614d8d45763"


def nearest_spacing(pixels):
    xy = np.asarray(pixels, dtype=float)
    distances = np.linalg.norm(xy[:, None] - xy[None], axis=-1)
    np.fill_diagonal(distances, np.inf)
    spacing = distances.min(axis=1)
    if not np.isfinite(spacing).all() or (spacing <= 0).any():
        raise ValueError("Degenerate projected corner spacing")
    return spacing


def window_sizes(pixels, fraction, minimum):
    return np.clip(np.floor(nearest_spacing(pixels) * fraction), minimum, 8).astype(int)


def refine_pixels(gray, originals, seeds, windows):
    """OpenCV 4.13 ChArUco coordinates require -0.5 / +0.5 around cornerSubPix.

    Failure retains the original observation, never a model prediction. Fixed 2px
    displacement gate is image-domain safety, not a fitted-residual filter.
    """
    originals, seeds = np.asarray(originals, float), np.asarray(seeds, float)
    result = originals.copy()
    reasons, movements = [], []
    for i, (original, seed, window) in enumerate(zip(originals, seeds, windows)):
        if not np.isfinite(seed).all() or np.linalg.norm(seed - original) > 2.:
            reasons.append("invalid_seed")
            continue
        point = seed - .5
        if np.any(point < window + 2) or np.any(point > np.array(gray.shape[::-1]) - window - 3):
            reasons.append("border")
            continue
        patch = cv2.getRectSubPix(gray, (2 * window + 3,) * 2, tuple(point)).astype(np.float32) / 255.
        gx = cv2.Sobel(patch, cv2.CV_32F, 1, 0, ksize=3, scale=1 / 8.)[1:-1, 1:-1]
        gy = cv2.Sobel(patch, cv2.CV_32F, 0, 1, ksize=3, scale=1 / 8.)[1:-1, 1:-1]
        tensor = np.array([[np.mean(gx * gx), np.mean(gx * gy)], [np.mean(gx * gy), np.mean(gy * gy)]])
        if np.linalg.eigvalsh(tensor)[0] < 1e-6:
            reasons.append("no_corner_texture")
            continue
        refined = cv2.cornerSubPix(gray, point.astype(np.float32).reshape(1, 1, 2),
                                  (int(window), int(window)), (-1, -1),
                                  (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 40, .001)).reshape(2).astype(float) + .5
        movement = float(np.linalg.norm(refined - original))
        if not np.isfinite(refined).all() or movement > 2.:
            reasons.append("large_movement")
            continue
        result[i] = refined
        reasons.append("accepted")
        movements.append(movement)
    return result, {reason: reasons.count(reason) for reason in sorted(set(reasons))}, movements


def gray_images(session, records):
    images = {}
    for record in records:
        for camera in (0, 1):
            image = cv2.imread(str(session / record["paths"][camera]))
            if image is None:
                raise ValueError("Missing image")
            images[record["capture_id"], camera] = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return images


def calibrated_detector(board, k, distortion):
    parameters = cv2.aruco.DetectorParameters()
    parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    charuco = cv2.aruco.CharucoParameters()
    charuco.cameraMatrix, charuco.distCoeffs = k.copy(), distortion.copy()
    return cv2.aruco.CharucoDetector(board.board, charuco, parameters)


def refine_records(records, images, objects, model, fraction, minimum,
                   board=None, image_only=False, fixed_window=None):
    refined = copy.deepcopy(records)
    diagnostics = []
    detectors = ([calibrated_detector(board, model["K"][camera], model["D"][camera]) for camera in (0, 1)]
                 if board is not None else None)
    for original, output in zip(records, refined):
        obj, left, right = common.points(original, objects)
        for camera, side, observed in ((0, "left", left), (1, "right", right)):
            if fixed_window is not None:
                windows = np.full(len(observed), fixed_window, dtype=int)
            elif image_only:
                # Shared secondary holdout reference; independent of all models.
                windows = window_sizes(observed, fraction, minimum)
            else:
                pose = common.pose_from_left(obj, observed, model, side=camera)
                projected = cv2.projectPoints(objects, pose[:3], pose[3:], model["K"][camera], model["D"][camera])[0].reshape(-1, 2)
                windows = window_sizes(projected, fraction, minimum)[original["ids"]]
            gray = images[original["capture_id"], camera]
            seeds = observed.copy()
            missing, extra = 0, 0
            if detectors is not None:
                corners, ids, *_ = detectors[camera].detectBoard(gray)
                mapping = {} if ids is None else dict(zip(ids.ravel(), corners.reshape(-1, 2)))
                original_ids = set(original["ids"])
                extra = len(set(mapping) - original_ids)
                for i, corner_id in enumerate(original["ids"]):
                    if corner_id in mapping:
                        seeds[i] = mapping[corner_id]
                    else:
                        missing += 1
                # Missing detections remain original, including after refinement.
            xy, reasons, _ = refine_pixels(gray, observed, seeds, windows)
            if detectors is not None:
                missing_mask = np.array([corner_id not in mapping for corner_id in original["ids"]])
                xy[missing_mask] = observed[missing_mask]
            output[side] = xy.tolist()
            diagnostics.append({"capture_id": original["capture_id"], "camera": camera,
                                "corners": len(observed), "reasons": reasons,
                                "redetection_missing_original_ids": missing,
                                "redetection_extra_ids_not_used": extra,
                                "window_min_median_max": [int(windows.min()), float(np.median(windows)), int(windows.max())]})
    displacement = np.concatenate([np.linalg.norm(np.array(b[side]) - np.array(a[side]), axis=1)
                                   for a, b in zip(records, refined) for side in ("left", "right")])
    return refined, {"per_image": diagnostics,
                     "original_observations_retained": len(displacement),
                     "final_displacement_median_p95_max_px": np.percentile(displacement, [50, 95, 100]).tolist()}


def fit(records, objects, initial):
    information = [[np.tile(np.eye(2), (len(record["ids"]), 1, 1))] * 2 for record in records]
    return ba.fit(records, objects, initial, information)


def evaluations(train_raw, holdout_raw, holdout_shared, objects, size, model):
    return {"training_raw": common.evaluate(train_raw, objects, size, model),
            "validation_raw": common.evaluate(holdout_raw, objects, size, model),
            "validation_shared_image_only": common.evaluate(holdout_shared, objects, size, model)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if tuple(map(int, cv2.__version__.split(".")[:2])) != (4, 13):
        raise ValueError("This protocol pins OpenCV 4.13 ChArUco pixel conventions")
    cv2.setNumThreads(1)
    args.output.mkdir(parents=True, exist_ok=False)
    data = common.collect_observations(args.session.resolve(), "rgb_pair_1", args.baseline / "observations.json")
    train, holdout, split = common.split_records(data["records"], .25, 2)
    historical = json.loads((args.baseline / "report.json").read_text())
    if split != historical.get("split", historical.get("protocol", {}).get("split")):
        raise ValueError("Historical split changed")
    safe_fraction = .8 * (data["board"]["square_length"] - data["board"]["marker_length"]) / (2 * data["board"]["square_length"])
    protocol = {"dataset_hash": data["dataset_hash"], "split": split,
                "source_commit": SOURCE_COMMIT, "opencv": cv2.__version__,
                "script_sha256": common.digest(__file__), "ba_sha256": common.digest(ba.__file__),
                "common_sha256": common.digest(common.__file__),
                "native_fraction_min_max": [.4, 2, 8], "charuco_fraction_min_max": [safe_fraction, 1, 8],
                "fixed_window_control_halfwidth": 5,
                "subpix_iterations_epsilon": [40, .001], "pixel_convention": "Charuco -0.5 before cornerSubPix, +0.5 after",
                "max_original_displacement_px": 2., "minimum_patch_gradient_eigenvalue": 1e-6,
                "iterations": 3, "point_set": "original IDs only; all failures retain original raw corner",
                "primary_holdout": "frozen original pixels, unweighted, no detection feedback",
                "secondary_holdout": "one shared image-only refinement, independent of every fitted model",
                "model_selection": "report all variants and all iterations; no holdout tuning",
                "solver": "same full stereo OPENCV5 BA and termination for every variant"}
    common.write_json(args.output / "protocol.json", protocol)
    objects, size = np.array(data["object_points_m"]), tuple(data["image_size"])
    images = gray_images(args.session, train)
    holdout_images = gray_images(args.session, holdout)
    shared, secondary_diagnostics = refine_records(holdout, holdout_images, objects, None, safe_fraction, 1, image_only=True)
    common.write_json(args.output / "shared_holdout_observations.json", shared)
    initial = common.solve_opencv(train, objects, size)["opencv_joint"]
    state, baseline, solver = fit(train, objects, initial)
    report = {"protocol": protocol, "secondary_holdout_diagnostics": secondary_diagnostics,
              "models": {"baseline": {"solver": solver, "model": common.serializable(baseline),
                                       **evaluations(train, holdout, shared, objects, size, baseline)}}}
    np.savez_compressed(args.output / "baseline_state.npz", state=state)
    for label, fraction, minimum in (("fixed_window_once", .4, 2), ("native_window_once", .4, 2),
                                     ("charuco_window_once", safe_fraction, 1)):
        observations, diagnostic = refine_records(train, images, objects, baseline, fraction, minimum,
                                                   fixed_window=5 if label == "fixed_window_once" else None)
        state, model, solver = fit(observations, objects, baseline)
        report["models"][label] = {"solver": solver, "refinement": diagnostic,
                                   "training_refined": common.evaluate(observations, objects, size, model),
                                   "model": common.serializable(model),
                                   **evaluations(train, holdout, shared, objects, size, model)}
        common.write_json(args.output / f"{label}_observations.json", observations)
        np.savez_compressed(args.output / f"{label}_state.npz", state=state)
        common.write_json(args.output / "report.partial.json", report)
        print(label, report["models"][label]["validation_raw"]["cross_camera_transfer"], flush=True)
    board = common.CharucoBoard.from_dict(data["board"])
    current = baseline
    for iteration in range(1, 4):
        # Always re-read image evidence and gate against ORIGINAL measurements.
        observations, diagnostic = refine_records(train, images, objects, current, safe_fraction, 1, board=board)
        state, current, solver = fit(observations, objects, current)
        label = f"iterative_calibrated_{iteration}"
        report["models"][label] = {"solver": solver, "refinement": diagnostic,
                                   "training_refined": common.evaluate(observations, objects, size, current),
                                   "model": common.serializable(current),
                                   **evaluations(train, holdout, shared, objects, size, current)}
        common.write_json(args.output / f"{label}_observations.json", observations)
        np.savez_compressed(args.output / f"{label}_state.npz", state=state)
        common.write_json(args.output / "report.partial.json", report)
        print(label, report["models"][label]["validation_raw"]["cross_camera_transfer"], flush=True)
    report["status"] = "completed"
    common.write_json(args.output / "report.json", report)


if __name__ == "__main__":
    main()
