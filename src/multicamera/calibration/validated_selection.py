"""Conservative candidate selection inside the explicitly chosen projection family."""

from dataclasses import replace

import cv2
import numpy as np

from .health import check_stereo_health
from .model_selection import MIN_MODEL_VIEWS, _predict_errors
from .models import CameraModel

MAX_UNVALIDATED_RMS_PX = .5


def _fit(left, right, stereo, candidate):
    fish = stereo.model == CameraModel.FISHEYE
    flags = (cv2.fisheye.CALIB_FIX_K3 | cv2.fisheye.CALIB_FIX_K4 if fish else
             cv2.CALIB_FIX_K3) if candidate == 'constrained' else 0
    lintr, rintr = left.calibrate(flags=flags), right.calibrate(flags=flags)
    if candidate == 'rational':
        lintr = left.calibrate(flags=cv2.CALIB_RATIONAL_MODEL, initial=lintr)
        rintr = right.calibrate(flags=cv2.CALIB_RATIONAL_MODEL, initial=rintr)
    result = stereo.calibrate(lintr, rintr)
    return lintr, rintr, result


def _fold_calibrators(left, right, stereo, ids):
    calibrators = []
    for cal in (left, right, stereo):
        mapping = {key: i for i, key in enumerate(cal.frame_ids)}
        if len(mapping) != len(cal.frame_ids):
            raise ValueError('观测帧编号重复，不能可靠择模')
        indices = [mapping[key] for key in ids]
        clone = (cal.copy_with_indices(indices, keep_images=False)
                 if hasattr(cal, '_all_corners') else cal.copy_with_indices(indices))
        clone.min_frames = 6
        calibrators.append(clone)
    calibrators[2] = calibrators[2].copy_with_common_ids()
    return calibrators


def _cross_validate(left, right, stereo, candidate, ids, log):
    folds = 5 if len(ids) >= 15 else 3
    frame_map = {key: i for i, key in enumerate(stereo.frame_ids)}
    details, all_errors = [], []
    reverse_cal = stereo.copy_with_indices(list(range(stereo.num_frames)))
    reverse_cal._img_points_left, reverse_cal._img_points_right = (
        reverse_cal._img_points_right, reverse_cal._img_points_left)
    for fold in range(folds):
        train = [key for i, key in enumerate(ids) if i % folds != fold]
        test = [key for i, key in enumerate(ids) if i % folds == fold]
        log(f'候选 {candidate}：留出验证 {fold + 1}/{folds}')
        lc, rc, sc = _fold_calibrators(left, right, stereo, train)
        lintr, rintr, result = _fit(lc, rc, sc, candidate)
        health = check_stereo_health(lintr, rintr, result)
        if health['status'] != 'ok':
            raise ValueError('留出训练模型有效性失败：' + '；'.join(health['reasons']))
        reverse = replace(result, R=result.R.T, T=-result.R.T @ result.T.reshape(3, 1))
        errors = np.concatenate([
            error for key in test for error in (
                _predict_errors(stereo, frame_map[key], lintr, rintr, result),
                _predict_errors(reverse_cal, frame_map[key], rintr, lintr, reverse))])
        all_errors.append(errors)
        details.append({'train_ids': train, 'test_ids': test,
                        'rms_px': float(np.sqrt(np.mean(errors**2))),
                        'point_count': len(errors), 'health': health})
    errors = np.concatenate(all_errors)
    return {'rms_px': float(np.sqrt(np.mean(errors**2))), 'folds': details,
            'point_count': len(errors)}


def calibrate_validated(left, right, stereo, log=lambda _: None):
    """Never change pinhole/fisheye family or treat failed candidates as usable.

    A healthy low-error standard solution is retained after available holdouts.
    Alternatives need matched-view holdouts, trained without final-fit seeds.
    """
    names = ['standard', 'constrained']
    if stereo.model == CameraModel.PINHOLE:
        names.append('rational')
    report = {'method': 'health_gated_within_family_selection', 'candidates': {},
              'selected': None, 'policy_version': 2,
              'validation_min_views': MIN_MODEL_VIEWS,
              'limits': '同序列留出及全视场数值采样，非独立测量精度验收'}
    fitted = {}
    for cal in (left, right, stereo):
        if len(cal.frame_ids) != len(set(cal.frame_ids)):
            raise ValueError('观测帧编号重复，不能可靠择模')
    paired_ids = set(left.frame_ids) & set(right.frame_ids)
    ids = [key for key in stereo.frame_ids
           if key in paired_ids]
    if len(ids) != len(set(ids)):
        raise ValueError('观测帧编号重复，不能可靠择模')
    ids = [ids[i] for i in np.linspace(0, len(ids)-1, min(30, len(ids)), dtype=int)]
    report['evaluated_ids'] = ids
    for name in names:
        row = report['candidates'][name] = {'status': 'failed'}
        try:
            if len(ids) < MIN_MODEL_VIEWS and name != 'standard':
                raise ValueError(f'自动更换模型至少需要{MIN_MODEL_VIEWS}对共同观测')
            log(f'校验候选模型：{name}')
            lintr, rintr, result = _fit(left, right, stereo, name)
            health = check_stereo_health(lintr, rintr, result)
            row.update({'health': health, 'train_rms_px': float(result.rms_error)})
            if not np.isfinite(result.rms_error) or result.rms_error < 0:
                raise ValueError('训练双目误差无效')
            if health['status'] != 'ok':
                raise ValueError('；'.join(health['reasons']))
            fitted[name] = lintr, rintr, result
            if (name == 'standard' and len(ids) < MIN_MODEL_VIEWS
                    and result.rms_error <= MAX_UNVALIDATED_RMS_PX):
                report.update(selected=name, validation_status='insufficient_views',
                    reason=f'仅{len(ids)}对共同观测，不足{MIN_MODEL_VIEWS}对；'
                           '标准模型仅通过基础有效性及训练误差检查，未完成留出验证')
                row['status'] = 'ok'
                row['validation_status'] = 'insufficient_views'
                log(report['reason'])
                return (*fitted[name], report)
            if len(ids) < MIN_MODEL_VIEWS:
                raise ValueError(f'仅{len(ids)}对共同观测，标准双目RMS={result.rms_error:.3f} px'
                                 f'超过未验证解门槛{MAX_UNVALIDATED_RMS_PX:g} px；'
                                 f'请补足{MIN_MODEL_VIEWS}对观测并核对左右配对')
            row['validation'] = _cross_validate(left, right, stereo, name, ids, log)
            metric = row['validation']['rms_px']
            if not np.isfinite(metric) or metric < 0:
                raise ValueError('留出跨相机投影误差无效')
            if metric > 2.:
                raise ValueError('留出跨相机投影误差超过2 px')
            row['status'] = 'ok'
            row['validation_status'] = 'passed'
            if name == 'standard' and result.rms_error <= MAX_UNVALIDATED_RMS_PX:
                report.update(selected=name, validation_status='passed',
                    reason='标准模型通过有效性及留出验证，且训练误差较低，优先保留标准模型')
                log(f'已保留标准模型，留出投影 RMS={metric:.3f} px')
                return (*fitted[name], report)
        except (cv2.error, ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
            row['error'] = str(exc)
            log(f'候选 {name} 不可用：{exc}')
    valid = [(row['validation']['rms_px'], names.index(name), name)
             for name, row in report['candidates'].items() if row['status'] == 'ok']
    if not valid:
        raise ValueError('没有通过有效性及留出验证的候选模型；' + '；'.join(
            f'{name}: {row.get("error", "检查失败")}' for name, row in report['candidates'].items()))
    score, _, selected = min(valid)
    standard = report['candidates']['standard']
    if standard['status'] == 'ok' and standard['validation']['rms_px'] - score < .03:
        selected = 'standard'
    report.update(selected=selected, validation_status='passed',
                  reason='仅在有效性和全部留出检查通过的候选中比较；差异不足0.03 px时保留标准模型')
    log(f'已选择 {selected}，留出投影 RMS={report["candidates"][selected]["validation"]["rms_px"]:.3f} px')
    return (*fitted[selected], report)
