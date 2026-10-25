"""Conservative, display-only quality decisions using the existing evaluator."""

from dataclasses import dataclass

import numpy as np

from .assessment import CalibrationEvaluator, Grade


@dataclass
class QualitySummary:
    title: str
    level: str
    risks: list[str]
    details: list[str]


def summarize_quality(rig):
    evaluator = CalibrationEvaluator()
    grades, risks, details = [], [], []
    invalid = False

    def intrinsic(name, intr):
        nonlocal invalid
        k = np.asarray(intr.camera_matrix)
        if (k.shape != (3, 3) or not np.isfinite(k).all() or k[0, 0] <= 0 or k[1, 1] <= 0 or
                not np.isfinite(intr.rms_error) or intr.rms_error < 0):
            invalid = True
            risks.append(f'{name}：内参或误差指标无效')
            return
        quality(name, evaluator.evaluate_intrinsic(intr))
        count = len(intr.per_frame_errors) if intr.per_frame_errors is not None else None
        details.append(f'{name}参与帧：{count if count is not None else "未记录"}')
        if count is None or count == 0:
            risks.append(f'{name}：缺少逐帧误差，需人工检查异常帧')
        elif not np.isfinite(intr.per_frame_errors).all() or min(intr.per_frame_errors) < 0:
            invalid = True
            risks.append(f'{name}：逐帧误差包含无效值')

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
            quality(f'{name} 双目', evaluator.evaluate_stereo(pair))
            if not np.isfinite(pair.stereo.R).all() or not np.isfinite(pair.stereo.T).all():
                invalid = True
                risks.append(f'{name}：双目外参包含无效值')
        except Exception:
            invalid = True
            risks.append(f'{name}：质量指标不完整，需重新核对')
    for aux in rig.aux_cameras.values():
        try:
            intrinsic(aux.camera_id, aux.intrinsics)
            details.append(f'{aux.camera_id} 融合 RMS {aux.rms_error:.3f} px · {aux.frame_count} 组')
            if (not np.isfinite(aux.rms_error) or aux.rms_error < 0 or
                    np.shape(aux.T_rgb_left_aux) != (4, 4) or not np.isfinite(aux.T_rgb_left_aux).all()):
                invalid = True
                risks.append(f'{aux.camera_id}：融合外参或误差指标无效')
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
    return QualitySummary(title, level, risks, details)
