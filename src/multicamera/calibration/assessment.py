"""Data usability assessment and calibration quality evaluation."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .models import CameraIntrinsics, CameraModel, StereoCalibResult, StereoPairCalibration

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
#  Data Usability Assessment
# ═══════════════════════════════════════════════════════════════

class Grade(str, Enum):
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"

    @property
    def label(self) -> str:
        return {
            "excellent": "优秀",
            "good": "良好",
            "fair": "一般",
            "poor": "较差",
        }[self.value]

    @property
    def color(self) -> str:
        return {
            "excellent": "#2d6a4f",
            "good": "#40916c",
            "fair": "#e09f3e",
            "poor": "#d00000",
        }[self.value]


@dataclass
class FrameQuality:
    """Quality metrics for a single captured frame."""

    frame_idx: int
    sharpness: float  # Laplacian variance (higher = sharper)
    brightness: float  # mean intensity [0, 255]
    num_corners: int
    is_blurry: bool
    is_overexposed: bool
    is_underexposed: bool

    @property
    def usable(self) -> bool:
        return (
            not self.is_blurry
            and not self.is_overexposed
            and not self.is_underexposed
            and self.num_corners >= 4
        )


@dataclass
class DataAssessment:
    """Aggregate data usability assessment for a camera."""

    camera_id: str
    total_frames: int = 0
    usable_frames: int = 0
    coverage: float = 0.0
    distribution_uniformity: float = 0.0
    avg_sharpness: float = 0.0
    frame_qualities: List[FrameQuality] = field(default_factory=list)
    coverage_heatmap: Optional[np.ndarray] = None
    image_coverage: Optional[float] = None

    @property
    def grade(self) -> Grade:
        score = self._compute_score()
        if score >= 85:
            return Grade.EXCELLENT
        if score >= 65:
            return Grade.GOOD
        if score >= 45:
            return Grade.FAIR
        return Grade.POOR

    def _compute_score(self) -> float:
        if self.total_frames == 0:
            return 0.0
        usable_ratio = self.usable_frames / self.total_frames
        s = 0.0
        s += min(usable_ratio, 1.0) * 25
        coverage = self.image_coverage if self.image_coverage is not None else self.coverage
        s += min(coverage, 1.0) * 30
        s += min(self.distribution_uniformity, 1.0) * 25
        s += min(self.total_frames / 20.0, 1.0) * 20
        return s

    @property
    def score(self) -> float:
        return self._compute_score()

    @property
    def suggestions(self) -> List[str]:
        tips = []
        if self.total_frames < 15:
            tips.append(f"帧数不足 ({self.total_frames}/15)，建议继续采集")
        if self.coverage < 0.7:
            tips.append(f"板角点完整度偏低 ({self.coverage:.0%})，请确保整板可见")
        if self.image_coverage is not None and self.image_coverage < .7:
            tips.append(f"视场网格覆盖不足 ({self.image_coverage:.0%})，请补拍边缘和角落")
        if self.distribution_uniformity < 0.5:
            tips.append("角点分布不均匀，请在图像各区域都拍摄标定板")
        blurry = sum(1 for fq in self.frame_qualities if fq.is_blurry)
        if blurry > 0:
            tips.append(f"有 {blurry} 帧运动模糊，移动标定板时请放慢速度")
        overexp = sum(1 for fq in self.frame_qualities if fq.is_overexposed)
        underexp = sum(1 for fq in self.frame_qualities if fq.is_underexposed)
        if overexp > 0:
            tips.append(f"有 {overexp} 帧过曝，请降低环境光或相机曝光")
        if underexp > 0:
            tips.append(f"有 {underexp} 帧欠曝，请增加环境光照")
        if not tips:
            tips.append("基础图像检查通过；仍需检查姿态多样性和标定误差")
        return tips


class DataAssessor:
    """Evaluates captured frame quality and data completeness."""

    BLUR_THRESHOLD = 50.0
    BRIGHTNESS_LOW = 40.0
    BRIGHTNESS_HIGH = 220.0

    def assess_frame(
        self, image: np.ndarray, num_corners: int, frame_idx: int
    ) -> FrameQuality:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        brightness = float(gray.mean())

        return FrameQuality(
            frame_idx=frame_idx,
            sharpness=sharpness,
            brightness=brightness,
            num_corners=num_corners,
            is_blurry=sharpness < self.BLUR_THRESHOLD,
            is_overexposed=brightness > self.BRIGHTNESS_HIGH,
            is_underexposed=brightness < self.BRIGHTNESS_LOW,
        )

    def assess_dataset(
        self,
        camera_id: str,
        frame_qualities: List[FrameQuality],
        coverage: float,
        corner_distribution: Optional[np.ndarray],
    ) -> DataAssessment:
        uniformity = 0.0
        if corner_distribution is not None and corner_distribution.sum() > 0:
            normed = corner_distribution.astype(np.float64)
            normed = normed / normed.sum()
            ideal = 1.0 / normed.size
            max_dev = 2.0 * (1.0 - ideal)
            actual_dev = np.sum(np.abs(normed - ideal))
            uniformity = max(0.0, 1.0 - actual_dev / max_dev) if max_dev else 1.

        return DataAssessment(
            camera_id=camera_id,
            total_frames=len(frame_qualities),
            usable_frames=sum(1 for fq in frame_qualities if fq.usable),
            coverage=coverage,
            distribution_uniformity=uniformity,
            avg_sharpness=float(np.mean([fq.sharpness for fq in frame_qualities]))
            if frame_qualities
            else 0.0,
            frame_qualities=frame_qualities,
            coverage_heatmap=corner_distribution,
            image_coverage=(float(np.count_nonzero(corner_distribution) / corner_distribution.size)
                            if corner_distribution is not None and corner_distribution.size else None),
        )


# ═══════════════════════════════════════════════════════════════
#  Calibration Quality Evaluation
# ═══════════════════════════════════════════════════════════════

@dataclass
class CalibrationQuality:
    """Comprehensive calibration quality report."""

    camera_or_pair_id: str
    model: CameraModel
    rms_error: float
    grade: Grade
    score: float
    per_frame_errors: List[float] = field(default_factory=list)
    epipolar_errors: Optional[List[float]] = None
    max_frame_error: float = 0.0
    mean_frame_error: float = 0.0
    details: Dict[str, str] = field(default_factory=dict)
    suggestions: List[str] = field(default_factory=list)


class CalibrationEvaluator:
    """Evaluates the quality of completed calibrations."""

    # RMS thresholds (pixels) for grading
    INTRINSIC_THRESHOLDS = {
        CameraModel.PINHOLE: (0.3, 0.5, 1.0),
        CameraModel.FISHEYE: (0.5, 0.8, 1.5),
    }
    STEREO_THRESHOLDS = {
        CameraModel.PINHOLE: (0.4, 0.7, 1.2),
        CameraModel.FISHEYE: (0.6, 1.0, 2.0),
    }

    def evaluate_intrinsic(self, intrinsics: CameraIntrinsics) -> CalibrationQuality:
        model = intrinsics.model
        thresholds = self.INTRINSIC_THRESHOLDS[model]
        grade = self._grade_rms(intrinsics.rms_error, thresholds)

        per_frame = (list(intrinsics.per_frame_errors)
                     if intrinsics.per_frame_errors is not None else [])
        max_err = max(per_frame) if per_frame else 0.0
        mean_err = float(np.mean(per_frame)) if per_frame else 0.0

        details = {
            "模型": "针孔" if model == CameraModel.PINHOLE else "鱼眼",
            "RMS 重投影误差": f"{intrinsics.rms_error:.4f} px",
            "最大帧误差": f"{max_err:.4f} px" if per_frame else "未记录",
            "平均帧误差": f"{mean_err:.4f} px" if per_frame else "未记录",
            "逐帧误差口径": ("二维点距离 RMS（px）" if intrinsics.per_frame_error_metric == "euclidean_rms_px"
                         else "历史结果未记录；请勿直接跨版本比较"),
            "焦距 (fx, fy)": (
                f"({intrinsics.camera_matrix[0, 0]:.1f}, "
                f"{intrinsics.camera_matrix[1, 1]:.1f})"
            ),
            "主点 (cx, cy)": (
                f"({intrinsics.camera_matrix[0, 2]:.1f}, "
                f"{intrinsics.camera_matrix[1, 2]:.1f})"
            ),
        }

        suggestions = []
        if not np.isfinite(intrinsics.rms_error) or intrinsics.rms_error < 0:
            suggestions.append("RMS 重投影误差无效，需重新核对标定结果")
        if intrinsics.rms_error > thresholds[1]:
            suggestions.append("重投影误差偏高，建议检查标定板平整度和采集质量")
        if max_err > intrinsics.rms_error * 3:
            bad_frames = [
                i for i, e in enumerate(per_frame) if e > intrinsics.rms_error * 2.5
            ]
            has_ids = intrinsics.frame_ids is not None and len(intrinsics.frame_ids) == len(per_frame)
            labels = [intrinsics.frame_ids[i] if has_ids else str(i + 1) for i in bad_frames]
            prefix = "源编号" if has_ids else "参与序号（非原图编号）"
            suggestions.append(f"{prefix} {', '.join(labels)} 误差异常，建议检查后重新标定")

        fx = intrinsics.camera_matrix[0, 0]
        fy = intrinsics.camera_matrix[1, 1]
        if abs(fx - fy) / max(fx, fy) > 0.05:
            suggestions.append("fx 与 fy 差异较大，可能存在非正方形像素或标定异常")

        cx = intrinsics.camera_matrix[0, 2]
        cy = intrinsics.camera_matrix[1, 2]
        w, h = intrinsics.image_size
        if abs(cx - w / 2) > w * 0.15 or abs(cy - h / 2) > h * 0.15:
            suggestions.append("主点偏离图像中心较远，建议检查标定数据")

        if not suggestions:
            suggestions.append("标定质量良好")

        return CalibrationQuality(
            camera_or_pair_id="",
            model=model,
            rms_error=intrinsics.rms_error,
            grade=grade,
            score=self._rms_to_score(intrinsics.rms_error, thresholds),
            per_frame_errors=per_frame,
            max_frame_error=max_err,
            mean_frame_error=mean_err,
            details=details,
            suggestions=suggestions,
        )

    def evaluate_stereo(
        self,
        pair: StereoPairCalibration,
        left_images: Optional[List[np.ndarray]] = None,
        right_images: Optional[List[np.ndarray]] = None,
    ) -> CalibrationQuality:
        model = pair.stereo.model
        thresholds = self.STEREO_THRESHOLDS[model]
        grade = self._grade_rms(pair.stereo.rms_error, thresholds)

        details = {
            "模型": "针孔" if model == CameraModel.PINHOLE else "鱼眼",
            "RMS 重投影误差": f"{pair.stereo.rms_error:.4f} px",
            "基线距离": f"{np.linalg.norm(pair.stereo.T):.4f} m",
        }
        alignment = pair.calibration_info.get('rectification', {})
        health = pair.calibration_info.get('health', {})
        if health:
            details['有效性检查'] = '通过（数值采样）' if health['status'] == 'ok' else '失败，不可用于立体校正'
        selection = pair.calibration_info.get('model_selection', {})
        if selection:
            details['自动择模'] = selection['selected']
            chosen = selection['candidates'][selection['selected']]
            if 'validation' in chosen:
                details['留出跨相机投影 RMS'] = f"{chosen['validation']['rms_px']:.4f} px（同序列）"
        if alignment:
            coordinate = alignment.get('coordinate', '?')
            details['同 ID 校正检查'] = (
                f"训练配对 {alignment.get('frame_count', 0)} 对 / {alignment.get('point_count', 0)} 点；"
                f"校正像素 {coordinate} 坐标差，非独立验证"
            )
            if alignment.get('status') == 'ok':
                details['同 ID 校正残差 RMS'] = f"{alignment['rms_px']:.4f} px"
                details['同 ID 校正残差 P95'] = f"{alignment['p95_abs_px']:.4f} px"

        epipolar_errors = None
        if (
            left_images
            and right_images
            and pair.stereo.map1_left is not None
        ):
            epipolar_errors = self._compute_epipolar_errors(pair, left_images, right_images)
            mean_epi = float(np.mean(epipolar_errors)) if epipolar_errors else 0.0
            details["平均极线误差"] = f"{mean_epi:.4f} px"

        suggestions = []
        if alignment:
            if alignment.get('status') != 'ok':
                suggestions.append('同 ID 校正检查无效，请核对模型、校正矩阵和角点')
            elif alignment['rms_px'] > 1.:
                suggestions.append('同 ID 校正残差偏高，请目视检查校正图及异常配对；此值受校正像素尺度影响')
        if not np.isfinite(pair.stereo.rms_error) or pair.stereo.rms_error < 0:
            suggestions.append("双目 RMS 重投影误差无效，需重新核对标定结果")
        if pair.stereo.rms_error > thresholds[1]:
            suggestions.append("双目重投影误差偏高，建议增加标定帧或检查同步精度")

        baseline = np.linalg.norm(pair.stereo.T)
        if baseline < 0.01:
            suggestions.append("基线距离过短，可能导致深度精度不足")
        if baseline > 1.0:
            suggestions.append("基线距离较长，近距离物体可能出现遮挡")

        if not suggestions:
            suggestions.append("双目标定质量良好")

        return CalibrationQuality(
            camera_or_pair_id=pair.pair_name,
            model=model,
            rms_error=pair.stereo.rms_error,
            grade=grade,
            score=self._rms_to_score(pair.stereo.rms_error, thresholds),
            epipolar_errors=epipolar_errors,
            details=details,
            suggestions=suggestions,
        )

    def _compute_epipolar_errors(
        self,
        pair: StereoPairCalibration,
        left_images: List[np.ndarray],
        right_images: List[np.ndarray],
    ) -> List[float]:
        """Compute per-frame mean epipolar error on rectified image pairs."""
        stereo = pair.stereo
        errors = []

        for left, right in zip(left_images, right_images):
            rect_l = cv2.remap(left, stereo.map1_left, stereo.map2_left, cv2.INTER_LINEAR)
            rect_r = cv2.remap(right, stereo.map1_right, stereo.map2_right, cv2.INTER_LINEAR)

            gray_l = cv2.cvtColor(rect_l, cv2.COLOR_BGR2GRAY) if len(rect_l.shape) == 3 else rect_l
            gray_r = cv2.cvtColor(rect_r, cv2.COLOR_BGR2GRAY) if len(rect_r.shape) == 3 else rect_r

            sift = cv2.SIFT_create(nfeatures=200)
            kp_l, desc_l = sift.detectAndCompute(gray_l, None)
            kp_r, desc_r = sift.detectAndCompute(gray_r, None)

            if desc_l is None or desc_r is None or len(kp_l) < 5:
                continue

            bf = cv2.BFMatcher()
            matches = bf.knnMatch(desc_l, desc_r, k=2)

            y_diffs = []
            for m_pair in matches:
                if len(m_pair) < 2:
                    continue
                m, n = m_pair
                if m.distance < 0.7 * n.distance:
                    pt_l = kp_l[m.queryIdx].pt
                    pt_r = kp_r[m.trainIdx].pt
                    y_diffs.append(abs(pt_l[1] - pt_r[1]))

            if y_diffs:
                errors.append(float(np.mean(y_diffs)))

        return errors

    def generate_rectification_preview(
        self,
        left: np.ndarray,
        right: np.ndarray,
        stereo: StereoCalibResult,
        num_lines: int = 20,
    ) -> np.ndarray:
        """Generate a side-by-side rectified image with horizontal epipolar lines."""
        rect_l = cv2.remap(left, stereo.map1_left, stereo.map2_left, cv2.INTER_LINEAR)
        rect_r = cv2.remap(right, stereo.map1_right, stereo.map2_right, cv2.INTER_LINEAR)

        h, w = rect_l.shape[:2]
        canvas = np.hstack([rect_l, rect_r])

        step = h // (num_lines + 1)
        for i in range(1, num_lines + 1):
            y = i * step
            color = (0, 255, 0) if i % 2 == 0 else (0, 200, 255)
            cv2.line(canvas, (0, y), (2 * w, y), color, 1)

        return canvas

    @staticmethod
    def _grade_rms(rms: float, thresholds: Tuple[float, float, float]) -> Grade:
        if not np.isfinite(rms) or rms < 0:
            return Grade.POOR
        if rms <= thresholds[0]:
            return Grade.EXCELLENT
        if rms <= thresholds[1]:
            return Grade.GOOD
        if rms <= thresholds[2]:
            return Grade.FAIR
        return Grade.POOR

    @staticmethod
    def _rms_to_score(rms: float, thresholds: Tuple[float, float, float]) -> float:
        if not np.isfinite(rms) or rms < 0:
            return 0.0
        if rms <= thresholds[0]:
            return 90 + 10 * max(0, (thresholds[0] - rms) / thresholds[0])
        if rms <= thresholds[1]:
            return 70 + 20 * (thresholds[1] - rms) / (thresholds[1] - thresholds[0])
        if rms <= thresholds[2]:
            return 40 + 30 * (thresholds[2] - rms) / (thresholds[2] - thresholds[1])
        return max(0, 40 * (2 * thresholds[2] - rms) / thresholds[2])
