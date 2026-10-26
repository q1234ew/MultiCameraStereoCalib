"""Conservative, display-only quality decisions using the existing evaluator."""

from dataclasses import dataclass

import numpy as np

from .assessment import CalibrationEvaluator, Grade
from .models import CameraModel


def _finite_array(value):
    """Return real finite numeric results without coercing malformed values."""
    try:
        array = np.asarray(value)
        if array.dtype.kind not in 'fiu' or not np.isfinite(array).all():
            return None
        return array
    except (TypeError, ValueError):
        return None


def _is_rotation(value):
    rotation = _finite_array(value)
    return (rotation is not None and rotation.shape == (3, 3)
            and np.allclose(rotation.T @ rotation, np.eye(3), rtol=0, atol=1e-6)
            and np.isclose(np.linalg.det(rotation), 1., rtol=0, atol=1e-6))


def _is_rigid_transform(value):
    transform = _finite_array(value)
    return (transform is not None and transform.shape == (4, 4)
            and _is_rotation(transform[:3, :3])
            and np.allclose(transform[3], [0., 0., 0., 1.], rtol=0, atol=1e-6))


def _valid_distortion(intr):
    distortion = _finite_array(intr.dist_coeffs)
    sizes = (4,) if intr.model == CameraModel.FISHEYE else (4, 5, 8, 12, 14)
    return (distortion is not None and distortion.size in sizes
            and (distortion.ndim == 1
                 or (distortion.ndim == 2 and 1 in distortion.shape)))


@dataclass
class QualitySummary:
    title: str
    level: str
    risks: list[str]
    details: list[str]
    valid: bool = True


def summarize_quality(rig):
    evaluator = CalibrationEvaluator()
    grades, risks, details = [], [], []
    invalid = False

    def intrinsic(name, intr):
        nonlocal invalid
        k = _finite_array(intr.camera_matrix)
        if (k is None or k.shape != (3, 3) or k[0, 0] <= 0 or k[1, 1] <= 0 or
                not np.allclose(k[2], [0., 0., 1.], rtol=0, atol=1e-9) or
                not np.isfinite(intr.rms_error) or intr.rms_error < 0):
            invalid = True
            risks.append(f'{name}：内参或误差指标无效')
            return
        if not _valid_distortion(intr):
            invalid = True
            risks.append(f'{name}：畸变系数无效，需检查有限值、向量形状及模型系数数量')
            return
        if intr.per_frame_errors is not None:
            errors = _finite_array(intr.per_frame_errors)
            if errors is None or errors.ndim != 1 or np.any(errors < 0):
                invalid = True
                risks.append(f'{name}：逐帧误差包含无效值')
                return
            if intr.frame_ids is not None and len(intr.frame_ids) != len(errors):
                invalid = True
                risks.append(f'{name}：源编号与逐帧误差数量不一致')
                return
        if (len(intr.image_size) != 2 or
                any(isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer)) or v <= 0
                    for v in intr.image_size)):
            invalid = True
            risks.append(f'{name}：图像分辨率无效')
            return
        quality(name, evaluator.evaluate_intrinsic(intr))
        count = len(intr.per_frame_errors) if intr.per_frame_errors is not None else None
        details.append(f'{name}参与帧：{count if count is not None else "未记录"}')
        if count is None or count == 0:
            risks.append(f'{name}：缺少逐帧误差，需人工检查异常帧')

    def quality(name, result):
        nonlocal invalid
        if not np.isfinite(result.rms_error) or result.rms_error < 0:
            invalid = True
            risks.append(f'{name}：误差指标无效')
            return
        grades.append(result.grade)
        details.append(f'{name}：{result.grade.label} · RMS {result.rms_error:.3f} px')
        for message in result.suggestions:
            if message not in {'标定质量良好', '双目标定质量良好'}:
                risks.append(f'{name}：{message}')

    for name, pair in rig.pairs.items():
        try:
            intrinsic(f'{name} 左目', pair.left_intrinsics)
            intrinsic(f'{name} 右目', pair.right_intrinsics)
            alignment = pair.calibration_info.get('rectification', {})
            if alignment and alignment.get('status') != 'ok':
                invalid = True
                risks.append(f'{name}：同 ID 校正检查无效')
            if (pair.left_intrinsics.model != pair.stereo.model or
                    pair.right_intrinsics.model != pair.stereo.model or
                    pair.left_intrinsics.image_size != pair.right_intrinsics.image_size):
                invalid = True
                risks.append(f'{name}：左右内参与双目模型或分辨率不一致')
            if not _is_rotation(pair.stereo.R):
                invalid = True
                risks.append(f'{name}：双目旋转 R 无效，必须为 SO(3) 的 3×3 旋转矩阵')
            translation = _finite_array(pair.stereo.T)
            if (translation is None or translation.shape not in {(3,), (3, 1), (1, 3)}):
                invalid = True
                risks.append(f'{name}：双目平移 T 无效，必须为有限三维向量')
            elif not np.any(translation != 0):
                invalid = True
                risks.append(f'{name}：双目平移 T 为零，基线无效')
            else:
                quality(f'{name} 双目', evaluator.evaluate_stereo(pair))
        except Exception:
            invalid = True
            risks.append(f'{name}：质量指标不完整，需重新核对')
    for aux in rig.aux_cameras.values():
        try:
            intrinsic(aux.camera_id, aux.intrinsics)
            details.append(f'{aux.camera_id} 融合 RMS {aux.rms_error:.3f} px · {aux.frame_count} 组')
            if not np.isfinite(aux.rms_error) or aux.rms_error < 0:
                invalid = True
                risks.append(f'{aux.camera_id}：融合误差指标无效')
            if not _is_rigid_transform(aux.T_rgb_left_aux):
                invalid = True
                risks.append(f'{aux.camera_id}：融合外参无效，必须为 4×4 刚体变换（SO(3) 旋转及齐次末行）')
            if aux.frame_count < 3:
                risks.append(f'{aux.camera_id}：参与帧不足或未记录')
            risks.append(f'{aux.camera_id}：需目视验证跨模态对齐，不能仅由 RMS 判定可用')
        except Exception:
            invalid = True
            risks.append(f'{aux.camera_id}：辅助质量指标不完整')
    risks = list(dict.fromkeys(risks))
    if invalid:
        title, level = '质量指标无效，需重新核对', 'poor'
    elif Grade.POOR in grades:
        title, level = '不建议直接使用', 'poor'
    elif Grade.FAIR in grades:
        title, level = '建议补拍后重新标定', 'fair'
    elif risks or not grades:
        title, level = '建议复核后使用', 'fair'
    else:
        title, level = '建议接受，仍需目视验证', 'good'
    return QualitySummary(title, level, risks, details, valid=not invalid)
