"""Validate declared AUX/RGB lineage without guessing missing legacy bindings."""

import numpy as np

from .auxiliary import _validate_intrinsics
from .models import CameraIntrinsics, CameraModel


def auxiliary_binding_issue(rig, aux):
    """Return a compatibility reason; malformed declared metadata raises first."""
    info = aux.calibration_info
    if not isinstance(info, dict):
        raise ValueError(f'{aux.camera_id}: calibration_info 必须是对象')
    if 'rgb_binding' not in info:
        return None  # Historical files did not record this optional lineage.
    binding = info['rgb_binding']
    if not isinstance(binding, dict):
        raise ValueError(f'{aux.camera_id}: rgb_binding 必须是对象，不能为 null 或其他类型')
    left_id = binding.get('left_id', rig.reference_camera)
    if not isinstance(left_id, str) or not left_id:
        raise ValueError(f'{aux.camera_id}: rgb_binding.left_id 必须是非空字符串')
    pair = next((pair for pair in rig.pairs.values() if pair.left_id == left_id), None)
    if pair is None or rig.reference_camera and left_id != rig.reference_camera:
        return 'RGB参考相机与辅助绑定不一致'
    intr = pair.left_intrinsics
    if 'right_id' in binding and binding['right_id'] != pair.right_id:
        return 'RGB右目与辅助绑定不一致'
    if 'model' in binding:
        try:
            model = CameraModel(binding['model'])
        except (ValueError, TypeError) as exc:
            raise ValueError(f'{aux.camera_id}: rgb_binding.model 无效') from exc
        if model != intr.model:
            return 'RGB模型与辅助绑定不一致'
    if 'image_size' in binding:
        size = binding['image_size']
        if (not isinstance(size, (list, tuple)) or len(size) != 2 or
                any(isinstance(n, bool) or not isinstance(n, int) or n <= 0 for n in size)):
            raise ValueError(f'{aux.camera_id}: rgb_binding.image_size 无效')
        if tuple(size) != intr.image_size:
            return 'RGB分辨率与辅助绑定不一致'
    if 'fixed_intrinsics' in binding:
        fixed = binding['fixed_intrinsics']
        if not isinstance(fixed, dict) or not all(key in fixed for key in (
                'camera_matrix', 'dist_coeffs', 'image_size', 'model')):
            raise ValueError(f'{aux.camera_id}: rgb_binding.fixed_intrinsics 缺少明确的K/D/尺寸/模型')
        try:
            recorded = CameraIntrinsics.from_dict(fixed)
            _validate_intrinsics(recorded, recorded.image_size)
        except (ValueError, TypeError, KeyError, IndexError) as exc:
            raise ValueError(f'{aux.camera_id}: rgb_binding.fixed_intrinsics 无效：{exc}') from exc
        if (recorded.model != intr.model or recorded.image_size != intr.image_size or
                not np.array_equal(recorded.camera_matrix, intr.camera_matrix) or
                not np.array_equal(recorded.dist_coeffs.ravel(), intr.dist_coeffs.ravel())):
            return '固定RGB内参与当前RGB结果不一致'
    return None


def validate_auxiliary_bindings(rig):
    for aux in rig.aux_cameras.values():
        reason = auxiliary_binding_issue(rig, aux)
        if reason:
            raise ValueError(f'{aux.camera_id}: 辅助参数不允许参与当前融合导出：{reason}')
