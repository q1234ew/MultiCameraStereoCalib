"""Compare projection models by held-out right-image prediction, not appearance."""

import cv2
import numpy as np

from .adaptive_guidance import GuidanceObservation, _geometry
from .models import CameraModel
from .health import check_stereo_health

MIN_MODEL_VIEWS = 9
MAX_MODEL_VIEWS = 24


def _validate_intrinsics(intr):
    k, d = intr.camera_matrix, intr.dist_coeffs
    if (not np.isfinite(k).all() or not np.isfinite(d).all() or
            min(k[0, 0], k[1, 1]) <= 0 or not np.isfinite(intr.rms_error)):
        raise ValueError('内参不满足有限值/正焦距检查')


def _predict_errors(cal, index, left, right, stereo):
    objects = cal._obj_points[index].astype(float).reshape(-1, 1, 3)
    points = cal._img_points_left[index].astype(float).reshape(-1, 1, 2)
    target = cal._img_points_right[index].reshape(-1, 2)
    fish = stereo.model == CameraModel.FISHEYE
    k, d = left.camera_matrix, left.dist_coeffs
    if fish:
        points = cv2.fisheye.undistortPoints(points, k, d)
    ok, rv, tv = cv2.solvePnP(objects, points, np.eye(3) if fish else k, None if fish else d)
    if not ok:
        raise ValueError('留出左图板姿不可解')
    rotation = cv2.Rodrigues(rv)[0]
    if np.any((objects.reshape(-1, 3) @ rotation.T + tv.ravel())[:, 2] <= 0):
        raise ValueError('留出板点位于左相机背面')
    rotation = stereo.R @ rotation
    translation = stereo.R @ tv + stereo.T.reshape(3, 1)
    if np.any((objects.reshape(-1, 3) @ rotation.T + translation.ravel())[:, 2] <= 0):
        raise ValueError('留出板点位于右相机背面')
    project = cv2.fisheye.projectPoints if fish else cv2.projectPoints
    predicted = project(objects, cv2.Rodrigues(rotation)[0], translation,
                        right.camera_matrix, right.dist_coeffs)[0].reshape(-1, 2)
    error = np.linalg.norm(predicted - target, axis=1)
    if not np.isfinite(error).all():
        raise ValueError('留出预测包含无效值')
    return error


def assess_camera_models(left, right, stereo, cancelled=lambda: False, progress=lambda _: None):
    """Read-only, deterministic 3-fold comparison on matched raw observations.

    Both models see the same input views and held-out points. Pinhole keeps
    per-pair common IDs; fisheye uses the cross-view common training subset.
    This is within-sequence validation, not external metrological evidence.
    """
    def check():
        if cancelled():
            raise RuntimeError('模型评估已取消')

    for cal in (left, right, stereo):
        if len(cal.frame_ids) != len(set(cal.frame_ids)):
            raise ValueError('帧编号重复，无法可靠进行模型对照')
    maps = [{key: i for i, key in enumerate(cal.frame_ids)} for cal in (left, right)]
    ids = [key for key in stereo.frame_ids if all(key in m for m in maps)]
    if len(ids) < MIN_MODEL_VIEWS:
        raise ValueError(f'模型评估至少需要 {MIN_MODEL_VIEWS} 对共同有效观测，当前 {len(ids)} 对')
    selected = np.linspace(0, len(ids) - 1, min(MAX_MODEL_VIEWS, len(ids)), dtype=int)
    ids = [ids[i] for i in selected]
    stereo_map = dict(zip(stereo.frame_ids, range(stereo.num_frames)))
    sizes = {left._image_size, right._image_size, stereo._image_size}
    if len(sizes) != 1 or None in sizes:
        raise ValueError('左右图像尺寸不一致，不能公平对比模型')
    for cal, mapping in zip((left, right), maps):
        geometries = []
        for key in ids:
            index = mapping[key]
            geometry = _geometry(cal.board, GuidanceObservation(
                cal._all_corners[index], cal._all_ids[index], cal._image_size, key))
            if geometry is None:
                raise ValueError(f'{key}: 标定板观测不足，请补充清晰且完整的观测')
            geometries.append(geometry)
        perspective = np.array([g[4][2, :2] for g in geometries])
        spread = np.linalg.svd(perspective - perspective.mean(0), compute_uv=False) / np.sqrt(len(ids))
        if spread[-1] < .012 or np.std(perspective, axis=0).max() < .035:
            raise ValueError('倾角变化不足，无法可靠区分模型；请补拍两个方向的倾斜观测')
    report = {'method': 'three_fold_left_pose_right_prediction', 'input_count': stereo.num_frames,
              'evaluated_ids': ids, 'max_views': MAX_MODEL_VIEWS, 'candidates': {},
              'recommended_model': None, 'requires_confirmation': True,
              'limits': '同一序列内留出；不代表真实镜头物理类型或外部测量精度。模型变更需重新标定。'}
    for model in CameraModel:
        folds, all_errors = [], []
        for fold in range(3):
            check()
            progress(f'{model.value} 模型 · 留出检查 {fold + 1}/3')
            train = [key for i, key in enumerate(ids) if i % 3 != fold]
            test = [key for i, key in enumerate(ids) if i % 3 == fold]
            detail = {'train_ids': train, 'test_ids': test}
            try:
                lcal, rcal = [cal.copy_with_indices([mapping[key] for key in train], keep_images=False)
                              for cal, mapping in zip((left, right), maps)]
                scal = stereo.copy_with_indices([stereo_map[key] for key in train])
                for cal in (lcal, rcal, scal):
                    cal.model, cal.min_frames = model, 6
                if model == CameraModel.FISHEYE:
                    scal = scal.copy_with_common_ids()
                detail.update({
                    'training_corner_policy': ('cross_view_common_ids'
                                               if model == CameraModel.FISHEYE
                                               else 'per_pair_common_ids'),
                    'training_corner_counts': scal.frame_common_corner_counts,
                    'shared_training_ids': (scal.frame_corner_ids[0].tolist()
                                            if model == CameraModel.FISHEYE else None),
                })
                lintr, rintr = lcal.calibrate(), rcal.calibrate()
                _validate_intrinsics(lintr)
                _validate_intrinsics(rintr)
                check()
                result = scal.calibrate(lintr, rintr)
                health = check_stereo_health(lintr, rintr, result)
                detail['health'] = health
                if health['status'] != 'ok':
                    raise ValueError('模型有效性检查失败：' + '；'.join(health['reasons']))
                if (not np.isfinite(result.R).all() or not np.isfinite(result.T).all()
                        or not np.isfinite(result.rms_error)):
                    raise ValueError('双目结果包含无效值')
                errors = np.concatenate([_predict_errors(stereo, stereo_map[key], lintr, rintr, result)
                                         for key in test])
                detail.update({'status': 'ok', 'rms_px': float(np.sqrt(np.mean(errors ** 2))),
                               'point_count': len(errors), 'stereo_train_rms_px': result.rms_error})
                all_errors.append(errors)
            except (cv2.error, ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
                check()
                detail.update({'status': 'failed', 'error': str(exc)})
            folds.append(detail)
        complete = all(f['status'] == 'ok' for f in folds)
        errors = np.concatenate(all_errors) if complete else None
        report['candidates'][model.value] = {
            'status': 'ok' if complete else 'failed', 'folds': folds,
            'rms_px': float(np.sqrt(np.mean(errors ** 2))) if complete else None,
            'point_count': len(errors) if complete else 0,
        }
    valid = sorted(((data['rms_px'], name) for name, data in report['candidates'].items()
                    if data['status'] == 'ok'))
    if not valid or valid[0][0] > 2.:
        report['reason'] = '两模型均未获得可靠的留出预测，请检查板规格、图像质量和观测多样性'
    elif len(valid) == 1:
        report['recommended_model'] = valid[0][1]
        report['reason'] = '仅一个候选完成全部留出检查；需确认另一个候选的失败原因'
    elif valid[0][0] <= .8 * valid[1][0] and valid[1][0] - valid[0][0] >= .15:
        report['recommended_model'] = valid[0][1]
        report['reason'] = '留出预测误差有明显优势；应用前请确认设备和板规格'
    else:
        report['reason'] = '两模型差异不足，不建议自动改变当前模型'
    check()
    return report
