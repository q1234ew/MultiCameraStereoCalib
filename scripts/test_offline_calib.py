"""Offline stereo calibration test using pre-captured image pairs.

Usage:
    python scripts/test_offline_calib.py
"""
import glob
import json
import logging
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from multicamera.board.charuco_board import CharucoBoard, CharucoBoardConfig
from multicamera.calibration.intrinsic import IntrinsicCalibrator
from multicamera.calibration.models import CameraModel
from multicamera.calibration.stereo import StereoCalibrator

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "pairs")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "results")


def find_pairs(data_dir: str):
    lefts = sorted(glob.glob(os.path.join(data_dir, "*_left.png")))
    pairs = []
    for lp in lefts:
        rp = lp.replace("_left.png", "_right.png")
        if os.path.isfile(rp):
            idx = os.path.basename(lp).split("_")[0]
            pairs.append((idx, lp, rp))
    return pairs


def sharpness(img: np.ndarray) -> float:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def main():
    board_cfg = CharucoBoardConfig(
        cols=15, rows=10,
        square_length=0.04, marker_length=0.03,
    )
    board = CharucoBoard(board_cfg)

    pairs = find_pairs(DATA_DIR)
    log.info("Found %d image pairs in %s", len(pairs), DATA_DIR)
    if not pairs:
        return

    # --- Phase 1: load & filter by sharpness ---
    SHARP_THRESH = 350.0
    good_pairs = []
    t0 = time.time()
    for idx, lp, rp in pairs:
        img_l = cv2.imread(lp)
        img_r = cv2.imread(rp)
        if img_l is None or img_r is None:
            continue
        sl, sr = sharpness(img_l), sharpness(img_r)
        ok = sl >= SHARP_THRESH and sr >= SHARP_THRESH
        status = "OK" if ok else "SKIP(blur)"
        log.info("  [%s] sharpL=%.0f  sharpR=%.0f  → %s", idx, sl, sr, status)
        if ok:
            good_pairs.append((idx, img_l, img_r))

    log.info("After sharpness filter: %d / %d pairs", len(good_pairs), len(pairs))

    # --- Phase 2: detect & collect ---
    calib_left = IntrinsicCalibrator(board, model=CameraModel.PINHOLE, min_frames=6)
    calib_right = IntrinsicCalibrator(board, model=CameraModel.PINHOLE, min_frames=6)
    stereo_cal = StereoCalibrator(board, model=CameraModel.PINHOLE, min_frames=6)

    for idx, img_l, img_r in good_pairs:
        det_l = calib_left.add_frame(img_l)
        det_r = calib_right.add_frame(img_r)
        det_sl, det_sr = stereo_cal.add_frame_pair(img_l, img_r)
        log.info(
            "  [%s]  L=%3d  R=%3d  stereo=%d",
            idx,
            det_l.num_corners if det_l.valid else 0,
            det_r.num_corners if det_r.valid else 0,
            stereo_cal.num_frames,
        )

    log.info("Left=%d  Right=%d  Stereo=%d frames",
             calib_left.num_frames, calib_right.num_frames, stereo_cal.num_frames)

    if not calib_left.ready or not calib_right.ready or not stereo_cal.ready:
        log.error("Not enough valid frames.")
        return

    # --- Phase 3: intrinsic calibration ---
    t1 = time.time()
    intr_l = calib_left.calibrate()
    intr_r = calib_right.calibrate()
    intr_dt = time.time() - t1

    log.info("=" * 60)
    log.info("LEFT  RMS: %.4f px", intr_l.rms_error)
    log.info("  fx=%.1f  fy=%.1f  cx=%.1f  cy=%.1f",
             intr_l.camera_matrix[0, 0], intr_l.camera_matrix[1, 1],
             intr_l.camera_matrix[0, 2], intr_l.camera_matrix[1, 2])
    log.info("  dist: %s", np.array2string(intr_l.dist_coeffs.flatten(), precision=5))
    if intr_l.per_frame_errors:
        log.info("  per-frame: %s",
                 [f"{e:.2f}" for e in intr_l.per_frame_errors])

    log.info("RIGHT RMS: %.4f px", intr_r.rms_error)
    log.info("  fx=%.1f  fy=%.1f  cx=%.1f  cy=%.1f",
             intr_r.camera_matrix[0, 0], intr_r.camera_matrix[1, 1],
             intr_r.camera_matrix[0, 2], intr_r.camera_matrix[1, 2])
    log.info("  dist: %s", np.array2string(intr_r.dist_coeffs.flatten(), precision=5))
    if intr_r.per_frame_errors:
        log.info("  per-frame: %s",
                 [f"{e:.2f}" for e in intr_r.per_frame_errors])
    log.info("Intrinsic time: %.2fs", intr_dt)

    # --- Phase 4: stereo calibration ---
    t2 = time.time()
    stereo_result = stereo_cal.calibrate(intr_l, intr_r)
    stereo_dt = time.time() - t2

    baseline = np.linalg.norm(stereo_result.T) * 1000
    log.info("=" * 60)
    log.info("STEREO RMS: %.4f px", stereo_result.rms_error)
    log.info("  Baseline: %.2f mm", baseline)
    log.info("  R:\n%s", np.array2string(stereo_result.R, precision=6))
    log.info("  T: %s", np.array2string(stereo_result.T.flatten(), precision=6))
    log.info("Stereo time: %.2fs", stereo_dt)

    # --- Phase 5: rectification visual check ---
    os.makedirs(OUT_DIR, exist_ok=True)
    if stereo_result.map1_left is not None:
        log.info("=" * 60)
        log.info("Generating rectification visualizations...")
        for i, (idx, img_l, img_r) in enumerate(good_pairs[:4]):
            rect_l = cv2.remap(img_l, stereo_result.map1_left,
                               stereo_result.map2_left, cv2.INTER_LINEAR)
            rect_r = cv2.remap(img_r, stereo_result.map1_right,
                               stereo_result.map2_right, cv2.INTER_LINEAR)
            combo = np.hstack([rect_l, rect_r])
            h = combo.shape[0]
            for y in range(0, h, h // 16):
                cv2.line(combo, (0, y), (combo.shape[1], y), (0, 255, 0), 1)
            path = os.path.join(OUT_DIR, f"rectified_{idx}.jpg")
            cv2.imwrite(path, combo, [cv2.IMWRITE_JPEG_QUALITY, 90])
            log.info("  → %s", path)

    # --- Save result ---
    from multicamera.calibration.models import StereoPairCalibration
    full = StereoPairCalibration(
        pair_name="cam0", left_id="left", right_id="right",
        left_intrinsics=intr_l, right_intrinsics=intr_r,
        stereo=stereo_result,
    )
    result_json = os.path.join(OUT_DIR, "calibration.json")
    with open(result_json, "w") as f:
        json.dump(full.to_dict(), f, indent=2)
    log.info("Saved calibration → %s", result_json)

    log.info("=" * 60)
    log.info("DONE  total: %.2fs", time.time() - t0)


if __name__ == "__main__":
    main()
