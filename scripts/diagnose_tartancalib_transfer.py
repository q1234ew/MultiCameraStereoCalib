"""Native ChArUco re-detection ablation and training-only stability challenges.

Consumes the frozen main TartanCalib transfer experiment; never selects on holdout.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import cv2
import numpy as np

import compare_mrcal as common
import verify_tartancalib_transfer as study


def redetect(records, images, board, model):
    output = copy.deepcopy(records)
    detectors = [study.calibrated_detector(board, model["K"][c], model["D"][c]) for c in (0, 1)]
    diagnostics = []
    displacements = []
    for original, target in zip(records, output):
        detected_ids = []
        for camera, side in enumerate(("left", "right")):
            corners, ids, *_ = detectors[camera].detectBoard(images[original["capture_id"], camera])
            mapping = {} if ids is None else dict(zip(ids.ravel(), corners.reshape(-1, 2)))
            detected_ids.append(set(mapping))
            counts = {"accepted": 0, "missing": 0, "invalid_or_over_2px": 0}
            for index, corner_id in enumerate(original["ids"]):
                if corner_id not in mapping:
                    counts["missing"] += 1
                    continue
                xy = mapping[corner_id].astype(float)
                delta = np.linalg.norm(xy - np.array(original[side][index]))
                if not np.isfinite(xy).all() or delta > 2.:
                    counts["invalid_or_over_2px"] += 1
                    continue
                target[side][index] = xy.tolist()
                counts["accepted"] += 1
            displacements.extend(np.linalg.norm(np.array(target[side]) - np.array(original[side]), axis=1))
            diagnostics.append({"capture_id": original["capture_id"], "camera": camera, "counts": counts})
        common_ids = detected_ids[0] & detected_ids[1]
        diagnostics[-1]["new_common_ids_not_used"] = sorted(common_ids - set(original["ids"]))
        diagnostics[-1]["missing_common_original_ids"] = sorted(set(original["ids"]) - common_ids)
    return output, {"per_image": diagnostics,
                    "original_observations_retained": len(displacements),
                    "displacement_median_p95_max_px": np.percentile(displacements, [50, 95, 100]).tolist(),
                    "new_common_ids_not_used": sum(len(d.get("new_common_ids_not_used", [])) for d in diagnostics)}


def difference(records_a, records_b):
    values = np.concatenate([np.linalg.norm(np.array(a[side]) - np.array(b[side]), axis=1)
                             for a, b in zip(records_a, records_b) for side in ("left", "right")])
    return {"observations": len(values), "median_p95_max_px": np.percentile(values, [50, 95, 100]).tolist(),
            "over_005_px": int((values > .05).sum()), "over_025_px": int((values > .25).sum())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--experiment", type=Path, required=True)
    args = parser.parse_args()
    cv2.setNumThreads(1)
    data = common.collect_observations(args.session.resolve(), "rgb_pair_1", args.baseline / "observations.json")
    train, holdout, split = common.split_records(data["records"], .25, 2)
    prior = json.loads((args.experiment / "report.json").read_text())
    if data["dataset_hash"] != prior["protocol"]["dataset_hash"] or split != prior["protocol"]["split"]:
        raise ValueError("Experiment dataset/split changed")
    output = args.experiment / "native_ablation"
    output.mkdir(exist_ok=False)
    common.write_json(output / "protocol.json", {
        "dataset_hash": data["dataset_hash"], "split": split,
        "iterations": 3, "refinement": "native OpenCV 4.13 ChArUco marker-limited window, no extra cornerSubPix",
        "selection": "fixed 3 iterations; no best-on-holdout selection",
        "training_only_stability": "focal length +/-5%, same original images and IDs, no refit; seed x +/-0.5px",
        "max_original_displacement_px": 2., "added_observations": False,
        "script_sha256": common.digest(__file__), "study_sha256": common.digest(study.__file__)})
    objects, size = np.array(data["object_points_m"]), tuple(data["image_size"])
    images = study.gray_images(args.session, train)
    shared = json.loads((args.experiment / "shared_holdout_observations.json").read_text())
    board = common.CharucoBoard.from_dict(data["board"])
    current = study.ba.unpack_model(np.load(args.experiment / "baseline_state.npz")["state"][:24])
    result = {"dataset_hash": data["dataset_hash"], "models": {}}
    for iteration in range(1, 4):
        observations, diagnostic = redetect(train, images, board, current)
        state, current, solver = study.fit(observations, objects, current)
        label = f"native_calibrated_{iteration}"
        result["models"][label] = {"solver": solver, "refinement": diagnostic,
                                    "training_refined": common.evaluate(observations, objects, size, current),
                                    "model": common.serializable(current),
                                    **study.evaluations(train, holdout, shared, objects, size, current)}
        common.write_json(output / f"{label}_observations.json", observations)
        np.savez_compressed(output / f"{label}_state.npz", state=state)
        common.write_json(output / "report.partial.json", result)
        print(label, result["models"][label]["validation_raw"]["cross_camera_transfer"], flush=True)
    nominal, nominal_diagnostic = redetect(train, images, board, current)
    stability = {"reference": nominal_diagnostic, "focal_challenges": {}}
    for label, factor in (("minus_5_percent", .95), ("plus_5_percent", 1.05)):
        biased = copy.deepcopy(current)
        for k in biased["K"]:
            k[0, 0] *= factor
            k[1, 1] *= factor
        challenged, diagnostic = redetect(train, images, board, biased)
        stability["focal_challenges"][label] = {"difference_from_nominal": difference(nominal, challenged),
                                                "refinement": diagnostic}
    # Image-only initialization sensitivity: same fixed window for +/-0.5px seeds.
    minus, plus = copy.deepcopy(train), copy.deepcopy(train)
    counts = []
    fraction = prior["protocol"]["charuco_fraction_min_max"][0]
    for source, a, b in zip(train, minus, plus):
        for camera, side in enumerate(("left", "right")):
            raw = np.array(source[side])
            windows = study.window_sizes(raw, fraction, 1)
            for sign, target in ((-1, a), (1, b)):
                refined, reasons, _ = study.refine_pixels(images[source["capture_id"], camera], raw, raw + [sign * .5, 0.], windows)
                target[side] = refined.tolist()
                counts.append(reasons)
    stability["opposite_seed_challenge"] = {"difference": difference(minus, plus), "attempt_counts": counts}
    result.update(status="completed", stability=stability)
    common.write_json(output / "report.json", result)


if __name__ == "__main__":
    main()
