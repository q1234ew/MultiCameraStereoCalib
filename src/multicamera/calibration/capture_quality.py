"""Conservative, original-image capture admission (not a calibration accuracy score)."""

from dataclasses import dataclass

import cv2
import numpy as np

from .models import CameraModel
from .observations import prepare_observation


@dataclass(frozen=True)
class CaptureDecision:
    state: str
    reasons: tuple[str, ...]
    common_count: int = 0

    @property
    def accepted(self):
        return self.state == '可采纳'

    def text(self):
        return f'{self.state} · 共同角点 {self.common_count} · ' + '；'.join(self.reasons)


def assess_stereo_capture(board, model, left, right, det_l, det_r, *, strict=False,
                          synchronized=True, expected_size=None, previous_common_counts=(),
                          expected_intrinsic_sizes=(None, None)):
    """Allow identified ChArUco subsets; hold uncertain photos out of solver inputs."""
    observations = []
    for index, (side, image, detection) in enumerate((('左目', left, det_l), ('右目', right, det_r))):
        if not detection.valid or detection.num_corners < 6:
            return CaptureDecision('拒收', (f'{side}有效角点不足 6 个，请重新采图',))
        try:
            points, ids, size = prepare_observation(detection, board, expected_size)
        except (ValueError, TypeError) as exc:
            return CaptureDecision('拒收', (f'{side}观测无效：{exc}',))
        if image is None or image.shape[1::-1] != size:
            return CaptureDecision('拒收', (f'{side}检测尺寸与原图不一致',))
        if expected_intrinsic_sizes[index] is not None and size != expected_intrinsic_sizes[index]:
            return CaptureDecision('拒收', (f'{side}原图尺寸与已有内参观测不一致，请重新配置会话',))
        p = points.reshape(-1, 2)
        if np.any(p < 0) or np.any(p >= np.array(size)):
            return CaptureDecision('拒收', (f'{side}角点超出原图边界',))
        if strict and len(ids) != board.num_corners:
            return CaptureDecision('拒收', (f'{side}未达到完整角点 {board.num_corners}，请重拍',))
        observations.append((p, ids.ravel(), image))
    common = np.intersect1d(observations[0][1], observations[1][1])
    count = len(common)
    if count < 6:
        return CaptureDecision('拒收', ('左右共同角点不足 6 个，请保证双路共视标定板',), count)
    reasons = []
    if not synchronized:
        reasons.append('未满足软件时间配对，请重新采图（不等同硬件同步）')
    if observations[0][2].shape[:2] != observations[1][2].shape[:2]:
        return CaptureDecision('拒收', ('当前双目求解要求左右原图分辨率一致',), count)
    if model == CameraModel.FISHEYE and any(n != count for n in previous_common_counts):
        reasons.append('鱼眼求解要求跨帧共同点数量一致，请重拍或显式使用共同角点筛选')
    for side, (points, ids, image) in zip(('左目', '右目'), observations):
        mask = np.isin(ids, common)
        physical = np.asarray(board.object_points)[ids[mask], :2].astype(np.float32)
        full = np.asarray(board.object_points)[:, :2].astype(np.float32)
        area = cv2.contourArea(cv2.convexHull(physical))
        full_area = cv2.contourArea(cv2.convexHull(full))
        hull = cv2.convexHull(points[mask].astype(np.float32))
        if area <= 0 or cv2.contourArea(hull) < 16:
            return CaptureDecision('拒收', (f'{side}共同点退化或板面过小，请重新采图',), count)
        if full_area > 0 and area / full_area < .3:
            reasons.append(f'{side}共同点过于集中，请展示更多板面')
        x, y, w, h = cv2.boundingRect(hull)
        gray = (cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image)
        roi = gray[y:y + h, x:x + w]
        roi_mask = np.zeros(roi.shape, np.uint8)
        cv2.fillConvexPoly(roi_mask, np.rint(hull.reshape(-1, 2) - [x, y]).astype(np.int32), 1)
        values = roi[roi_mask.astype(bool)]
        # Percentiles preserve naturally black/white squares; saturation alone is not failure.
        low, high = np.percentile(values, [10, 90])
        if high < 60 or low > 200 or high - low < 35:
            reasons.append(f'{side}板面亮度或对比度不足，请检查曝光/照明后重拍')
        lap = cv2.Laplacian(roi, cv2.CV_64F)
        if float(lap[roi_mask.astype(bool)].var()) < 50:
            reasons.append(f'{side}板面清晰度不足，请对焦并保持静止后重拍')
    if reasons:
        return CaptureDecision('待复核', tuple(reasons), count)
    partial = count < board.num_corners
    return CaptureDecision('可采纳', (
        'ChArUco 部分角点可用，不要求满点；精度仍需标定后评估' if partial
        else '原图通过基础验收；精度仍需标定后评估',), count)
