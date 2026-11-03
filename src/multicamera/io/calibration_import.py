"""Read complete RGB stereo parameters without inferring physical camera identity."""

import json
import hashlib
from pathlib import Path

import cv2
import numpy as np

from ..calibration.models import (
    CameraIntrinsics, CameraModel, MultiCameraRig, StereoCalibResult, StereoPairCalibration,
)
from .export import _validate_rig
from ..calibration.offline_aux import file_identity


def load_stereo_calibration(path, *, translation_unit=None, camera_model=None):
    path = Path(path)
    payload = path.read_bytes()
    if path.suffix.lower() == '.json':
        data = json.loads(payload.decode('utf-8-sig'))
        if not isinstance(data, dict) or not isinstance(data.get('pairs'), dict):
            raise ValueError('缺少完整双目结构 pairs；单相机内参不能代替双目标定')
        for name, pair in data['pairs'].items():
            for key in ('left_id', 'right_id', 'left_intrinsics', 'right_intrinsics', 'stereo'):
                if key not in pair:
                    raise ValueError(f'{name}: 缺少 {key}')
            for side in ('left_intrinsics', 'right_intrinsics'):
                absent = [key for key in ('camera_matrix', 'dist_coeffs', 'image_size', 'model')
                          if key not in pair[side]]
                if absent:
                    raise ValueError(f'{name}/{side}: 缺少 ' + '、'.join(absent))
            if any('rms_error' not in pair[key] for key in ('left_intrinsics', 'right_intrinsics', 'stereo')):
                pair.setdefault('calibration_info', {})['missing_imported_metrics'] = True
        rig = MultiCameraRig.from_dict(data)
        # This entry imports RGB stereo only; AUX observations must be checked separately.
        rig.aux_cameras.clear()
    else:
        if translation_unit not in ('m', 'mm'):
            raise ValueError('请明确文件中 T 的长度单位 m 或 mm，不根据基线大小猜测')
        rig = _read_opencv(payload, translation_unit, camera_model)
    if not rig.pairs:
        raise ValueError('文件未包含有效的双目标定结果')
    models = {p.stereo.model for p in rig.pairs.values()}
    if len(models) != 1:
        raise ValueError('当前工作区需要 RGB 各组使用同一投影模型')
    ids = [cid for p in rig.pairs.values() for cid in (p.left_id, p.right_id)]
    if any(not isinstance(name, str) or not name.strip() or name in ('.', '..') or
           '/' in name or '\\' in name for name in rig.pairs):
        raise ValueError('相机组名无效，不能包含目录分隔符')
    if any(not isinstance(cid, str) or not cid.strip() for cid in ids) or len(set(ids)) != len(ids):
        raise ValueError('左右相机 ID 为空或重复，请明确左右绑定')
    _validate_rig(rig)
    source = file_identity(path)
    if source['sha256'] != hashlib.sha256(payload).hexdigest():
        raise ValueError('读取期间标定文件发生变化，请重试')
    provenance = dict(kind='imported_stereo_file', source=source,
                      translation_unit='m' if path.suffix.lower() == '.json' else translation_unit,
                      transform_convention='X_right = R @ X_left + T')
    for pair in rig.pairs.values():
        pair.calibration_info['imported_calibration'] = provenance
    return rig, provenance


def _read_opencv(payload, unit, declared_model):
    fs = cv2.FileStorage(payload.decode('utf-8-sig'), cv2.FILE_STORAGE_READ | cv2.FILE_STORAGE_MEMORY)
    if not fs.isOpened():
        raise ValueError('无法读取 OpenCV YAML/XML 文件')
    rig = MultiCameraRig()
    try:
        prefixes = [key[:-len('left_id')] for key in fs.root().keys()
                    if key.startswith('pair_') and key.endswith('_left_id')]
        for prefix in prefixes or ['']:
            def node(*names):
                for name in names:
                    found = fs.getNode(prefix + name)
                    if not found.empty():
                        return found
                raise ValueError('标定文件缺少字段：' + '/'.join(prefix + n for n in names))

            def matrix(*names):
                result = node(*names).mat()
                if result is None:
                    raise ValueError('标定矩阵无效：' + '/'.join(names))
                return np.asarray(result, dtype=float)

            def text(default, *names):
                for key in names:
                    found = fs.getNode(prefix + key)
                    if not found.empty():
                        return found.string()
                return default

            model_name = text(declared_model, 'camera_model', 'left_model')
            if not model_name:
                raise ValueError('缺少 camera_model；请明确选择 pinhole 或 fisheye，不按畸变数量推断')
            model = CameraModel(model_name)
            if text(model_name, 'right_model', 'right_camera_model') != model_name:
                raise ValueError('左右相机投影模型不一致')
            width = fs.getNode(prefix + 'image_width')
            height = fs.getNode(prefix + 'image_height')
            if width.empty() or height.empty():
                raise ValueError('缺少图像分辨率 image_width/image_height，请补齐后重新读取')
            values = (width.real(), height.real())
            if any(not np.isfinite(v) or v <= 0 or v != int(v) for v in values):
                raise ValueError('图像宽高必须为正整数')
            size = tuple(int(v) for v in values)
            left = CameraIntrinsics(matrix('M1', 'K1', 'cameraMatrix1'), matrix('D1', 'distCoeffs1'), size, model)
            right = CameraIntrinsics(matrix('M2', 'K2', 'cameraMatrix2'), matrix('D2', 'distCoeffs2'), size, model)
            rotation, translation = matrix('R'), matrix('T') * (.001 if unit == 'mm' else 1.)
            t = translation.reshape(3)
            skew = np.array([[0., -t[2], t[1]], [t[2], 0., -t[0]], [-t[1], t[0], 0.]])
            essential = skew @ rotation
            fundamental = np.linalg.inv(right.camera_matrix).T @ essential @ np.linalg.inv(left.camera_matrix)
            stereo = StereoCalibResult(rotation, translation, essential, fundamental, model=model)
            name = text('imported_stereo', 'name', 'pair_name')
            if name in rig.pairs:
                raise ValueError('双目组名重复，不能可靠恢复')
            pair = StereoPairCalibration(name, text('imported_L', 'left_id', 'left_camera_id'),
                text('imported_R', 'right_id', 'right_camera_id'), left, right, stereo,
                calibration_info={'missing_imported_metrics': True})
            rig.pairs[name] = pair
        rig.reference_camera = next(iter(rig.pairs.values())).left_id
        return rig
    finally:
        fs.release()
