"""Termination settings shared by pinhole intrinsic and stereo calibration."""

import cv2

# A ceiling, not a forced iteration count. Keep the EPS stop for converged solves.
# On the fixed offline splits, 100 iterations stopped early; 1000 matched 10000.
PINHOLE_CALIBRATION_CRITERIA = (
    cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 1000, 1e-10,
)
