"""Stereo matching: disparity and depth computation from rectified image pairs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from ..calibration.models import StereoCalibResult


@dataclass
class StereoMatchParams:
    min_disparity: int = 0
    num_disparities: int = 128  # must be divisible by 16
    block_size: int = 5
    p1_multiplier: int = 8
    p2_multiplier: int = 32
    disp12_max_diff: int = 1
    pre_filter_cap: int = 63
    uniqueness_ratio: int = 10
    speckle_window_size: int = 100
    speckle_range: int = 2
    use_wls_filter: bool = True
    wls_lambda: float = 8000.0
    wls_sigma: float = 1.5


class StereoMatcher:
    """Compute disparity / depth from a rectified stereo pair."""

    def __init__(self, params: Optional[StereoMatchParams] = None):
        self.params = params or StereoMatchParams()
        self._rebuild_matcher()

    def _rebuild_matcher(self):
        p = self.params
        self._left_matcher = cv2.StereoSGBM_create(
            minDisparity=p.min_disparity,
            numDisparities=p.num_disparities,
            blockSize=p.block_size,
            P1=p.p1_multiplier * p.block_size ** 2,
            P2=p.p2_multiplier * p.block_size ** 2,
            disp12MaxDiff=p.disp12_max_diff,
            preFilterCap=p.pre_filter_cap,
            uniquenessRatio=p.uniqueness_ratio,
            speckleWindowSize=p.speckle_window_size,
            speckleRange=p.speckle_range,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
        )
        if p.use_wls_filter:
            self._right_matcher = cv2.ximgproc.createRightMatcher(self._left_matcher)
            self._wls_filter = cv2.ximgproc.createDisparityWLSFilter(self._left_matcher)
            self._wls_filter.setLambda(p.wls_lambda)
            self._wls_filter.setSigmaColor(p.wls_sigma)
        else:
            self._right_matcher = None
            self._wls_filter = None

    def update_params(self, params: StereoMatchParams):
        self.params = params
        self._rebuild_matcher()

    def rectify(
        self,
        left: np.ndarray,
        right: np.ndarray,
        stereo_result: StereoCalibResult,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply stereo rectification to an image pair."""
        rect_left = cv2.remap(
            left,
            stereo_result.map1_left,
            stereo_result.map2_left,
            cv2.INTER_LINEAR,
        )
        rect_right = cv2.remap(
            right,
            stereo_result.map1_right,
            stereo_result.map2_right,
            cv2.INTER_LINEAR,
        )
        return rect_left, rect_right

    def compute_disparity(
        self,
        rect_left: np.ndarray,
        rect_right: np.ndarray,
    ) -> np.ndarray:
        """Compute disparity map from rectified images.

        Returns float32 disparity in pixels.
        """
        gray_l = self._to_gray(rect_left)
        gray_r = self._to_gray(rect_right)

        disp_left = self._left_matcher.compute(gray_l, gray_r)

        if self._wls_filter is not None and self._right_matcher is not None:
            disp_right = self._right_matcher.compute(gray_r, gray_l)
            disp = self._wls_filter.filter(disp_left, gray_l, disparity_map_right=disp_right)
        else:
            disp = disp_left

        return disp.astype(np.float32) / 16.0

    def compute_depth(
        self,
        disparity: np.ndarray,
        Q: np.ndarray,
    ) -> np.ndarray:
        """Reproject disparity to 3D; returns (H, W, 3) point cloud."""
        points_3d = cv2.reprojectImageTo3D(disparity, Q, handleMissingValues=True)
        return points_3d

    @staticmethod
    def _to_gray(img: np.ndarray) -> np.ndarray:
        if len(img.shape) == 3:
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img
