"""Compare actual application pinhole solves at fixed termination criteria and holdouts."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import time
from unittest.mock import patch

import cv2
import numpy as np

import compare_mrcal as common
from multicamera.board.charuco_board import CharucoBoard
from multicamera.board.detector import DetectionResult
from multicamera.calibration.intrinsic import IntrinsicCalibrator
from multicamera.calibration.stereo import StereoCalibrator


def calibrators(data, records):
    board = CharucoBoard.from_dict(data["board"])
    left, right = IntrinsicCalibrator(board), IntrinsicCalibrator(board)
    stereo = StereoCalibrator(board)
    for record in records:
        detections = [DetectionResult(np.asarray(record[side], np.float32).reshape(-1, 1, 2),
                                     np.asarray(record["ids"], np.int32).reshape(-1, 1),
                                     [], None, tuple(data["image_size"])) for side in ("left", "right")]
        assert left.add_detection(detections[0], frame_id=record["capture_id"])
        assert right.add_detection(detections[1], frame_id=record["capture_id"])
        assert stereo.add_detection_pair(*detections, frame_id=record["capture_id"])
    return left, right, stereo


def run(data, train, holdout, criteria, name):
    left_cal, right_cal, stereo_cal = calibrators(data, train)
    calls = []

    def wrapper(original, stage):
        def solve(*args, **kwargs):
            if criteria == "defaults":
                kwargs.pop("criteria", None)  # Reproduce the pre-change OpenCV defaults.
            elif criteria is not None:
                kwargs["criteria"] = criteria
            calls.append({"stage": stage, "criteria": kwargs.get("criteria"),
                          "flags": int(kwargs.get("flags", 0))})
            return original(*args, **kwargs)
        return solve

    result = {"criteria_requested": criteria, "calls": calls, "strategies": {}}
    with patch.object(cv2, "calibrateCamera", wrapper(cv2.calibrateCamera, "intrinsic")), \
            patch.object(cv2, "stereoCalibrate", wrapper(cv2.stereoCalibrate, "stereo")):
        start = time.perf_counter()
        left_initial, right_initial = left_cal.calibrate(), right_cal.calibrate()
        result["intrinsic_seconds"] = time.perf_counter() - start
        result["intrinsic_rms_px"] = [left_initial.rms_error, right_initial.rms_error]
        for strategy, flags in (("fixed", cv2.CALIB_FIX_INTRINSIC),
                                ("joint", cv2.CALIB_USE_INTRINSIC_GUESS)):
            left, right = deepcopy(left_initial), deepcopy(right_initial)
            start = time.perf_counter()
            stereo = stereo_cal.calibrate(left, right, flags=flags)
            seconds = time.perf_counter() - start
            if strategy == "joint":
                # Same post-solve metric refresh as the application worker.
                left_cal.recompute_errors(left)
                right_cal.recompute_errors(right)
            model = {"K": [left.camera_matrix, right.camera_matrix],
                     "D": [left.dist_coeffs.ravel(), right.dist_coeffs.ravel()],
                     "R": stereo.R, "T": stereo.T.ravel()}
            if not all(np.isfinite(a).all() for a in [*model["K"], *model["D"], model["R"], model["T"]]):
                raise ValueError("Non-finite solution")
            objects = np.array(data["object_points_m"])
            item = {"stereo_training_rms_px": stereo.rms_error, "solve_and_maps_seconds": seconds,
                    "baseline_mm": float(np.linalg.norm(stereo.T) * 1000),
                    "model": common.serializable(model),
                    "validation": common.evaluate(holdout, objects, tuple(data["image_size"]), model)}
            result["strategies"][strategy] = item
            print(name, strategy, "train", stereo.rms_error, "val transfer",
                  item["validation"]["cross_camera_transfer"]["rms_px"], "sec", seconds, flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--application-only", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    data = common.collect_observations(args.session.resolve(), "rgb_pair_1", args.baseline / "observations.json")
    train, holdout, split = common.split_records(data["records"], .25, 2)
    baseline = json.loads((args.baseline / "report.json").read_text())
    original_split = baseline.get("split", baseline.get("protocol", {}).get("split"))
    if split != original_split:
        raise ValueError("Historical split changed")
    source = Path(__file__).resolve().parents[1] / "src/multicamera/calibration"
    variants = {"defaults": "defaults", "tight100": (3, 100, 1e-10),
                "tight1000": (3, 1000, 1e-10), "reference10000": (3, 10000, 1e-10)}
    if args.application_only:
        variants = {"application": None, "reference10000": (3, 10000, 1e-10)}
    report = {"dataset_hash": data["dataset_hash"], "split": split, "opencv": cv2.__version__,
              "source_sha256": {name: common.digest(source / name) for name in ("intrinsic.py", "stereo.py")},
              "script_sha256": common.digest(__file__),
              "protocol": "cached common corners, actual application calibrators; no holdout camera fitting or selection; no default strategy change",
              "variants": {}}
    common.write_json(args.output / "protocol.json", {"split": split, "variants": variants})
    for name, criteria in variants.items():
        report["variants"][name] = run(data, train, holdout, criteria, name)
        common.write_json(args.output / "report.partial.json", report)
    common.write_json(args.output / "report.json", report)


if __name__ == "__main__":
    main()
