"""Offline cross-modal refinement with fixed RGB optics and disjoint-view checks."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import logging
import time

import cv2
import numpy as np

from .auxiliary import (AuxIntrinsicCalibrator, CrossModalExtrinsicCalibrator,
                        CrossModalExtrinsicResult, _solve_board_pose, _validated_detection,
                        _validate_intrinsics, _reprojection_error)
from .health import check_intrinsics
from .models import CameraModel


AUX_MODELS = {
    'pinhole_constrained': ('针孔 · k1/k2/p1/p2', CameraModel.PINHOLE, cv2.CALIB_FIX_K3, 4),
    'pinhole': ('针孔 · 五参数', CameraModel.PINHOLE, 0, 5),
    'fisheye_2': ('鱼眼投影 · 两参数', CameraModel.FISHEYE,
                  cv2.fisheye.CALIB_FIX_K3 | cv2.fisheye.CALIB_FIX_K4, 2),
}
MIN_OFFLINE_PAIRS = 6
LIMITS = {'rgb_transfer_rms_px': 2., 'aux_transfer_rms_px': 1.,
          'baseline_fold_relative': .2, 'baseline_fold_absolute_m': .005}
logger = logging.getLogger(__name__)


def _rigid_grid_orders(board):
    from .grid_order import allowed_orders
    from .planar import PlanarPatternDetector

    return allowed_orders(PlanarPatternDetector._make_object_points(board), board.rows, board.cols)


def _pose_order_candidates(records, board, rgb_intrinsics, aux_intrinsics):
    from .offline_aux import reordered_detection

    def pose(det, intrinsics):
        try:
            result = _solve_board_pose(det, intrinsics)
            if result is not None:
                error = _reprojection_error(det, *result, intrinsics)
                if np.isfinite(error) and error <= 1.5:
                    return result
        except cv2.error:
            return None
        return None

    candidates = []
    orders = _rigid_grid_orders(board)
    for key, rgb, aux in records:
        rgb_pose = pose(rgb, rgb_intrinsics)
        if rgb_pose is None:
            continue
        rgb_rotation = cv2.Rodrigues(rgb_pose[0])[0]
        for order in orders:
            aux_pose = pose(reordered_detection(aux, board, order), aux_intrinsics)
            if aux_pose is not None:
                rotation = rgb_rotation @ cv2.Rodrigues(aux_pose[0])[0].T
                translation = rgb_pose[1].ravel() - rotation @ aux_pose[1].ravel()
                candidates.append((str(key), order, rotation, translation))
    return candidates


def _correspondence_report(records, board, rgb_intrinsics, aux_intrinsics, *, rotation_only=False):
    """Suggest grid symmetries from a rigid-rig consensus, never relabel observations."""
    candidates = _pose_order_candidates(records, board, rgb_intrinsics, aux_intrinsics)
    report = dict(status='ambiguous', physical_correspondence_verified=False,
                  consensus_views=0, alternative_views=0, frames=[], suspect_ids=[],
                  checked_views=0, unresolved_ids=[str(row[0]) for row in records])
    if not candidates:
        return report
    rotations = np.array([row[2] for row in candidates]).reshape(-1, 9)
    angles = np.degrees(np.arccos(np.clip((rotations @ rotations.T - 1.) / 2., -1., 1.)))
    translations = np.array([row[3] for row in candidates])
    distances = np.linalg.norm(translations[:, None] - translations[None], axis=-1)
    tolerance = max(.005, board.square_size * .5)
    # Each grid symmetry differs by 180 degrees, so a view has at most one vote.
    votes = np.sum((angles <= 8.) & (True if rotation_only else distances <= tolerance), axis=1)
    best = int(np.argmax(votes))
    separate = ((angles[best] > 16.) if rotation_only else
                (angles[best] > 16.) | (distances[best] > 2 * tolerance))
    runner_up = int(np.max(votes[separate])) if np.any(separate) else 0
    report.update(consensus_views=int(votes[best]), alternative_views=runner_up)
    if votes[best] < max(3, np.ceil(.7 * len(records))) or (
            votes[best] - runner_up < max(1, np.ceil(.2 * len(records)))):
        return report
    report['status'] = 'consistent'
    for key, *_ in records:
        indices = [i for i, row in enumerate(candidates) if row[0] == str(key)]
        chosen = min(indices, key=lambda i: angles[best, i] / 8. +
                     (0. if rotation_only else distances[best, i] / tolerance)) if indices else None
        current = next((i for i in indices if candidates[i][1] == 'identity'), None)
        if (chosen is None or current is None or angles[best, chosen] > 8. or
                (not rotation_only and distances[best, chosen] > tolerance)):
            continue
        suspect = bool(angles[best, current] > 90.)
        report['frames'].append(dict(source_id=str(key),
            aux_order=candidates[chosen][1], current_rotation_delta_deg=float(angles[best, current]),
            suggested_rotation_delta_deg=float(angles[best, chosen]), suspect=suspect))
        if suspect:
            report['suspect_ids'].append(str(key))
    report['checked_views'] = len(report['frames'])
    checked = {row['source_id'] for row in report['frames']}
    report['unresolved_ids'] = [str(row[0]) for row in records if str(row[0]) not in checked]
    if report['suspect_ids']:
        report['status'] = 'inconsistent'
    elif report['unresolved_ids']:
        report['status'] = 'partial'
    return report


def _ordered_records(records, board, orders):
    from .offline_aux import reordered_detection

    return [(key, rgb, reordered_detection(aux, board, orders[str(key)]))
            for key, rgb, aux in records]


def _rotation_hypotheses(records, board, rgb_intrinsics, aux_intrinsics):
    """Bounded initial assignments only; rotation votes alone never authorize a result."""
    candidates = _pose_order_candidates(records, board, rgb_intrinsics, aux_intrinsics)
    if not candidates:
        return []
    rotations = np.asarray([row[2] for row in candidates]).reshape(-1, 9)
    angles = np.degrees(np.arccos(np.clip((rotations @ rotations.T - 1.) / 2., -1., 1.)))
    votes = np.sum(angles <= 8., axis=1)
    hypotheses, seen = [], set()
    for index in np.argsort(-votes, kind='stable'):
        if votes[index] < max(3, np.ceil(.7 * len(records))):
            continue
        assignment = {}
        for key, *_ in records:
            choices = [i for i, row in enumerate(candidates) if row[0] == str(key)]
            if not choices:
                break
            chosen = min(choices, key=lambda i: angles[index, i])
            if angles[index, chosen] > 8.:
                break
            assignment[str(key)] = candidates[chosen][1]
        identity = tuple(assignment.items())
        if len(assignment) != len(records) or identity in seen:
            continue
        seen.add(identity)
        hypotheses.append(assignment)
        if len(hypotheses) == 4:
            break
    return hypotheses


def _resolve_for_transform(records, board, rgb_intrinsics, aux_intrinsics, transform):
    """Resolve discrete labels with a frozen rig; never fit held-out camera parameters."""
    from .offline_aux import reordered_detection

    aligned, frames = [], []
    rotation, translation = transform[:3, :3], transform[:3, 3]
    legal = _rigid_grid_orders(board)
    for key, rgb, aux in records:
        xyz = (_pose(rgb, rgb_intrinsics) - translation) @ rotation
        if not np.isfinite(xyz).all() or np.min(xyz[:, 2]) <= 0:
            raise ValueError(f'{key}: 跨模态对应预测深度无效')
        predicted = project(xyz, aux_intrinsics)
        candidates = []
        for order in legal:
            detection = reordered_detection(aux, board, order)
            candidates.append((_rms(predicted - detection.image_points), order, detection))
        candidates.sort(key=lambda row: row[0])
        best, runner = candidates[:2]
        if (not np.isfinite(best[0]) or not np.isfinite(runner[0])
                or runner[0] - best[0] < 2. or runner[0] < 1.5 * max(best[0], 1e-9)):
            raise ValueError(f'{key}: 跨模态编号候选差距不足，不能自动确定对应')
        aligned.append((key, rgb, best[2]))
        frames.append(dict(source_id=str(key), aux_order=best[1],
                           best_aux_transfer_rms_px=best[0], runner_aux_transfer_rms_px=runner[0],
                           physical_correspondence_verified=False))
    return aligned, frames


def inspect_aux_correspondences(records, board, rgb_intrinsics, model_key):
    """Read-only annotation aid; fitted diagnostic optics are not reused by any fold."""
    if model_key not in AUX_MODELS or len(records) < 3:
        raise ValueError('对应检查需要至少3组完整观测及明确的辅助模型')
    if len({str(row[0]) for row in records}) != len(records):
        raise ValueError('对应检查的观测编号重复')
    _, model, flags, _ = AUX_MODELS[model_key]
    mono = AuxIntrinsicCalibrator(board, model, min_frames=3)
    validated = []
    for key, rgb, aux in records:
        rgb, aux = _validated_detection(rgb, board), _validated_detection(aux, board)
        if rgb is None or aux is None or not mono.add_detection(aux):
            raise ValueError(f'{key}: 对应检查的网格或分辨率无效')
        _validate_intrinsics(rgb_intrinsics, rgb.image_size)
        validated.append((str(key), rgb, aux))
    report = _correspondence_report(validated, board, rgb_intrinsics, mono.calibrate(flags))
    if report['status'] == 'ambiguous' or report['unresolved_ids']:
        try:
            refined, _, fit = _fit(validated, board, rgb_intrinsics, model_key)
            report = _correspondence_report(validated, board, rgb_intrinsics, refined)
            resolution = fit['numbering_resolution']
            report['diagnostic_refinement'] = dict(
                method=resolution['method'], candidate_limit=resolution['candidate_limit'],
                candidate_count=len(resolution['attempts']), converged=fit['converged'],
                source_ids=resolution['source_ids'], physical_correspondence_verified=False,
                parameter_scope='diagnostic_only_not_reused_by_training_folds')
        except (ValueError, cv2.error) as exc:
            report['refinement_error'] = str(exc)
    report.update(aux_model=model_key, source_ids=[row[0] for row in validated],
                  board=board.to_dict(), fixed_rgb_intrinsics=rgb_intrinsics.to_dict(),
                  annotation_scope='full_review_dataset')
    return report


def project(points, intrinsics):
    fun = cv2.fisheye.projectPoints if intrinsics.model == CameraModel.FISHEYE else cv2.projectPoints
    return fun(np.asarray(points, dtype=float).reshape(-1, 1, 3), np.zeros(3), np.zeros(3),
               intrinsics.camera_matrix, intrinsics.dist_coeffs)[0].reshape(-1, 2)


def _rms(residual):
    return float(np.sqrt(np.mean(np.sum(np.asarray(residual) ** 2, axis=-1))))


def _pose(det, intr):
    pose = _solve_board_pose(det, intr)
    if pose is None:
        raise ValueError('板位姿求解失败或板点位于相机后方')
    rvec, tvec = pose
    return det.object_points @ cv2.Rodrigues(rvec)[0].T + tvec.reshape(3)


def transfer_metrics(records, rgb_intrinsics, aux_intrinsics, transform):
    """Predict the OTHER sensor from one sensor's PnP, in native pixels."""
    rgb_residuals, aux_residuals, frames = [], [], []
    rotation, translation = transform[:3, :3], transform[:3, 3]
    for key, rgb, aux in records:
        rgb_xyz = _pose(rgb, rgb_intrinsics)
        aux_xyz = _pose(aux, aux_intrinsics)
        predicted_rgb = aux_xyz @ rotation.T + translation
        predicted_aux = (rgb_xyz - translation) @ rotation
        if min(np.min(predicted_rgb[:, 2]), np.min(predicted_aux[:, 2])) <= 0:
            raise ValueError(f'{key}: 跨相机投影出现负深度')
        rgb_error = project(predicted_rgb, rgb_intrinsics) - rgb.image_points
        aux_error = project(predicted_aux, aux_intrinsics) - aux.image_points
        if not np.isfinite(rgb_error).all() or not np.isfinite(aux_error).all():
            raise ValueError(f'{key}: 跨相机投影无效')
        frames.append({'source_id': str(key), 'rgb_rms_px': _rms(rgb_error),
                       'aux_rms_px': _rms(aux_error)})
        rgb_residuals.extend(rgb_error)
        aux_residuals.extend(aux_error)
    return {'rgb_transfer_rms_px': _rms(rgb_residuals),
            'aux_transfer_rms_px': _rms(aux_residuals), 'frames': frames}


def _fit_fixed_orders(records, board, rgb_intrinsics, model_key, *, seed=None,
                      check_numbering=True):
    from scipy.optimize import least_squares
    from scipy.sparse import lil_matrix
    from scipy.spatial.transform import Rotation

    _, model, flags, parameter_count = AUX_MODELS[model_key]
    mono = AuxIntrinsicCalibrator(board, model, min_frames=3)
    cross = CrossModalExtrinsicCalibrator(board, min_frames=3)
    for _, rgb, aux in records:
        if not mono.add_detection(aux) or not cross.add_detection_pair(rgb, aux):
            raise ValueError('观测网格或分辨率不一致')
    seed = mono.calibrate(flags) if seed is None else deepcopy(seed)
    correspondence = _correspondence_report(records, board, rgb_intrinsics, seed)
    if check_numbering and correspondence['status'] == 'inconsistent':
        raise ValueError('角点编号方向待核对（不代表RGB/IR图像配错），图像对 ' +
            '、'.join(correspondence['suspect_ids']) +
            ' 的相对旋转偏差超过90°；请在离线逐对复核中检查对应候选、0点和X/Y方向，'
            '核对后重新明确纳入；未自动翻转或剔除原图')
    initial = cross.calibrate(rgb_intrinsics, seed).T_rgb_left_aux
    poses = [_solve_board_pose(rgb, rgb_intrinsics) for _, rgb, _ in records]
    if any(pose is None for pose in poses):
        raise ValueError('RGB 初始板位姿无效')
    shared = np.r_[Rotation.from_matrix(initial[:3, :3]).as_rotvec(), initial[:3, 3]]
    optics = np.r_[np.log(seed.camera_matrix[0, 0]), np.log(seed.camera_matrix[1, 1]),
                   seed.camera_matrix[:2, 2], seed.dist_coeffs.ravel()[:parameter_count]]
    offset = 6 + len(optics)
    initial_x = np.r_[shared, optics, np.array([np.r_[r.ravel(), t.ravel()]
                                               for r, t in poses]).ravel()]
    targets = np.asarray([[rgb.image_points, aux.image_points] for _, rgb, aux in records])
    objects = records[0][1].object_points

    def unpack(x):
        k, d = seed.camera_matrix.copy(), seed.dist_coeffs.copy().ravel()
        k[0, 0], k[1, 1] = np.exp(x[6]), np.exp(x[7])
        k[:2, 2] = x[8:10]
        d[:parameter_count] = x[10:10 + parameter_count]
        return replace(seed, camera_matrix=k, dist_coeffs=d)

    def residual(x):
        intr = unpack(x)
        poses = x[offset:].reshape(-1, 6)
        rotations = Rotation.from_rotvec(poses[:, :3]).as_matrix()
        xyz_rgb = np.einsum('nij,pj->npi', rotations, objects) + poses[:, None, 3:]
        rot = Rotation.from_rotvec(x[:3]).as_matrix()
        xyz_aux = (xyz_rgb - x[None, None, 3:6]) @ rot
        predictions = np.stack((
            project(xyz_rgb.reshape(-1, 3), rgb_intrinsics).reshape(targets[:, 0].shape),
            project(xyz_aux.reshape(-1, 3), intr).reshape(targets[:, 1].shape)), axis=1)
        return (predictions - targets).ravel()

    rows_per_view = len(objects) * 4
    sparse = lil_matrix((len(records) * rows_per_view, len(initial_x)), dtype=int)
    for i in range(len(records)):
        sparse[i*rows_per_view:(i+1)*rows_per_view, :offset] = 1
        sparse[i*rows_per_view:(i+1)*rows_per_view, offset+i*6:offset+(i+1)*6] = 1
    started = time.monotonic()
    initial_residual = residual(initial_x)
    result = least_squares(residual, initial_x, jac_sparsity=sparse.tocsr(), x_scale='jac',
                           loss='soft_l1', f_scale=.5, max_nfev=300,
                           ftol=1e-9, xtol=1e-9, gtol=1e-9)
    finite = bool(np.isfinite(result.x).all())
    diagnostics = dict(converged=bool(result.success and finite), status=int(result.status),
        nfev=int(result.nfev), njev=int(result.njev or 0), cost=float(result.cost),
        optimality=float(result.optimality), termination=str(result.message),
        seconds=time.monotonic() - started, source_ids=[str(row[0]) for row in records],
        initial_pixel_rms=float(np.sqrt(np.mean(initial_residual**2))))
    logger.info('Offline joint solve: model=%s diagnostics=%s', model_key, diagnostics)
    if not result.success or not finite:
        raw = residual(result.x if finite else initial_x).reshape(len(records), 2, -1, 2)
        worst = sorted(((str(row[0]), _rms(errors[0]), _rms(errors[1]))
                        for row, errors in zip(records, raw)), key=lambda row: -max(row[1:]))[:5]
        log = logger.error if check_numbering else logger.warning
        log('Offline joint failure: worst RGB/AUX RMS px=%s', worst)
        raise ValueError('联合优化未收敛：' + str(result.message) +
            f'；nfev={result.nfev}，cost={result.cost:.6g}，optimality={result.optimality:.6g}' +
            '；请复核图像对 ' + '、'.join(row[0] for row in worst))
    intr = unpack(result.x)
    transform = np.eye(4)
    transform[:3, :3] = Rotation.from_rotvec(result.x[:3]).as_matrix()
    transform[:3, 3] = result.x[3:6]
    final_poses = result.x[offset:].reshape(-1, 6)
    depths = []
    intr.rvecs, intr.tvecs = [], []
    for pose in final_poses:
        rotation = cv2.Rodrigues(pose[:3])[0]
        xyz_rgb = objects @ rotation.T + pose[3:]
        xyz_aux = (xyz_rgb - transform[:3, 3]) @ transform[:3, :3]
        depths.extend((float(np.min(xyz_rgb[:, 2])), float(np.min(xyz_aux[:, 2]))))
        aux_rotation = transform[:3, :3].T @ rotation
        intr.rvecs.append(cv2.Rodrigues(aux_rotation)[0])
        intr.tvecs.append((transform[:3, :3].T @ (pose[3:] - transform[:3, 3])).reshape(3, 1))
    if not np.isfinite(depths).all() or min(depths) <= 0:
        raise ValueError('联合优化后的板点深度无效')
    health = check_intrinsics(intr)
    if health['status'] != 'ok':
        raise ValueError('辅助模型全视场有效性检查失败：' + '；'.join(health['reasons']) +
                         '；请对比其他独立辅助模型，RGB内参保持不变')
    raw = residual(result.x).reshape(len(records), 2, -1, 2)
    intr.rms_error = _rms(raw[:, 1])
    intr.per_frame_errors = [_rms(row[1]) for row in raw]
    intr.per_frame_error_metric = 'joint_training_euclidean_rms_px'
    intr.frame_ids = [str(key) for key, _, _ in records]
    return intr, transform, {**diagnostics, 'health': health,
        'correspondence_check': correspondence,
        'minimum_joint_depth_m': min(depths),
        'joint_training_rgb_rms_px': _rms(raw[:, 0]),
        'joint_training_aux_rms_px': _rms(raw[:, 1])}


def _fit(records, board, rgb_intrinsics, model_key, *, resolve_correspondence=True):
    """Fit only this training subset, including its own discrete label initialization."""
    from .offline_aux import detection_fingerprint

    if not resolve_correspondence:
        return _fit_fixed_orders(records, board, rgb_intrinsics, model_key)
    points = np.concatenate([_pose(rgb, rgb_intrinsics) for _, rgb, _ in records])
    if np.linalg.matrix_rank(points - points.mean(axis=0), tol=board.square_size * 1e-4) < 3:
        raise ValueError('跨模态编号共识不足：重复或共面观测不能唯一确定对应，请补充距离和倾角变化')
    _, model, flags, _ = AUX_MODELS[model_key]
    mono = AuxIntrinsicCalibrator(board, model, min_frames=3)
    for _, _, aux in records:
        if not mono.add_detection(aux):
            raise ValueError('辅助观测网格或分辨率不一致')
    seed = mono.calibrate(flags)
    strict = _correspondence_report(records, board, rgb_intrinsics, seed)
    complete = strict['status'] != 'ambiguous' and not strict['unresolved_ids']
    if complete:
        hypotheses = [{row['source_id']: row['aux_order'] for row in strict['frames']}]
    else:
        hypotheses = _rotation_hypotheses(records, board, rgb_intrinsics, seed)
    if not hypotheses:
        raise ValueError('跨模态编号共识不足，未获得覆盖全部观测的初始化；请复核图像和角点')
    attempts, valid = [], []
    for assignment in hypotheses:
        attempt = dict(orders=assignment)
        attempts.append(attempt)
        try:
            aligned = _ordered_records(records, board, assignment)
            intr, transform, fit = _fit_fixed_orders(aligned, board, rgb_intrinsics, model_key,
                                                    seed=seed, check_numbering=False)
            post = _correspondence_report(records, board, rgb_intrinsics, intr)
            post_orders = {row['source_id']: row['aux_order'] for row in post['frames']}
            # Low-resolution planar PnP can have unstable normals. The frozen rig must
            # support every label with a clear projection margin, not just a pose vote.
            _, verified = _resolve_for_transform(records, board, rgb_intrinsics, intr, transform)
            if {row['source_id']: row['aux_order'] for row in verified} != assignment:
                raise ValueError('跨相机投影不支持当前编号解')
            metrics = transfer_metrics(aligned, rgb_intrinsics, intr, transform)
            score = max(metrics['rgb_transfer_rms_px'] / LIMITS['rgb_transfer_rms_px'],
                        metrics['aux_transfer_rms_px'] / LIMITS['aux_transfer_rms_px'])
            if not np.isfinite(score) or score > 1.:
                raise ValueError('编号候选跨相机误差未通过既有像素门槛')
            pose_disagreement = [key for key, order in assignment.items()
                                 if post_orders.get(key) != order]
            attempt.update(status='valid', score=float(score), nfev=fit['nfev'],
                           pose_diagnostic_disagreement_ids=pose_disagreement)
            valid.append((float(score), intr, transform, fit, post, verified))
        except (ValueError, cv2.error) as exc:
            attempt.update(status='rejected', reason=str(exc))
    if not valid:
        raise ValueError('跨模态对应求解失败：' + '；'.join(row['reason'] for row in attempts))
    valid.sort(key=lambda row: row[0])
    if len(valid) > 1 and (valid[1][0] - valid[0][0] < .25
                           or valid[1][0] < 2. * max(valid[0][0], 1e-9)):
        raise ValueError('跨模态编号存在多个近似等价解，未自动采用；请核对物理标记或补充不同观测')
    _, intr, transform, fit, post, verified = valid[0]
    post_orders = {row['source_id']: row['aux_order'] for row in post['frames']}
    disagreement = [row['source_id'] for row in verified
                    if post_orders.get(row['source_id']) != row['aux_order']]
    fit['numbering_resolution'] = dict(
        method='rigid_consensus' if complete else 'bounded_rotation_hypotheses_joint_refinement',
        source_ids=[str(row[0]) for row in records], candidate_limit=4, attempts=attempts,
        physical_correspondence_verified=False, final_pose_consensus=post,
        pose_consensus_role='diagnostic_only_after_joint_fit',
        pose_diagnostic_disagreement_ids=disagreement,
        acceptance_basis='converged_healthy_rig_unique_projection_and_cross_camera_error',
        frames=[dict(row, rgb_input_fingerprint=detection_fingerprint(rgb),
                     aux_input_fingerprint=detection_fingerprint(aux),
                     aux_solver_fingerprint=detection_fingerprint(
                         _ordered_records([(key, rgb, aux)], board, {str(key): row['aux_order']})[0][2]))
                for row, (key, rgb, aux) in zip(verified, records)])
    if disagreement:
        logger.info('Single-view PnP diagnostic differs for %s; frozen-rig projection margins '
                    'and cross-camera pixel limits passed; held-out validation remains required',
                    disagreement)
    return intr, transform, fit


def calibrate_offline_aux(records, board, rgb_intrinsics, model_key, log=lambda text: None):
    if model_key not in AUX_MODELS:
        raise ValueError('请选择独立的辅助相机模型')
    if len(records) < MIN_OFFLINE_PAIRS:
        raise ValueError(f'离线联合验证至少需要 {MIN_OFFLINE_PAIRS} 组（数量不代表精度）')
    if len({str(row[0]) for row in records}) != len(records):
        raise ValueError('离线观测编号重复')
    validated = []
    for key, rgb, aux in records:
        rgb, aux = _validated_detection(rgb, board), _validated_detection(aux, board)
        if rgb is None or aux is None:
            raise ValueError(f'{key}: 观测网格无效')
        _validate_intrinsics(rgb_intrinsics, rgb.image_size)
        validated.append((str(key), rgb, aux))
    if len({aux.image_size for _, _, aux in validated}) != 1:
        raise ValueError('辅助相机分辨率不一致')
    rgb_health = check_intrinsics(rgb_intrinsics)
    if rgb_health['status'] != 'ok':
        raise ValueError('固定 RGB 内参未通过全视场检查')
    rgb_intrinsics = deepcopy(rgb_intrinsics)
    report = dict(schema_version=1, status='offline_candidate_pending_physical_validation',
        method='fixed_rgb_joint_aux_intrinsics_and_extrinsics', aux_model=model_key,
        transform_direction='aux_to_rgb_left', translation_unit='metre',
        metric='cross_camera_transfer_native_pixels', rgb_intrinsics_fixed=True,
        loss='soft_l1', loss_scale_px=.5, pixel_sigma={'rgb': 1., 'aux': 1.},
        limits=dict(LIMITS), source_ids=[key for key, _, _ in validated], folds=[],
        temporal_stability='not_verified_offline', synchronization='not_verified_offline',
        qualification='Internal cross-validation only; not independent physical accuracy.')
    report['numbering_policy'] = 'automatic_legal_grid_correspondence_per_training_fold'
    fold_frames, baselines = [], []
    log(f'离线辅助模型：{model_key}；RGB内参固定；三折独立训练/留出验证')
    for fold in range(3):
        training = [row for i, row in enumerate(validated) if i % 3 != fold]
        heldout = [row for i, row in enumerate(validated) if i % 3 == fold]
        log(f'离线联合验证 {fold+1}/3：训练 {len(training)}，留出 {len(heldout)}')
        log('训练编号：' + '、'.join(row[0] for row in training) +
            '；留出编号：' + '、'.join(row[0] for row in heldout))
        intr, transform, fit = _fit(training, board, rgb_intrinsics, model_key)
        inferred = fit.get('numbering_resolution')
        if inferred:
            changed = [row['source_id'] for row in inferred['frames'] if row['aux_order'] != 'identity']
            log('训练折自动编号对应：' + ('辅助相对重排 ' + '、'.join(changed) if changed else
                '当前规范编号已对应') + '；仅本折初始化，未旋转原图或确认物理0点')
        aligned_heldout, heldout_orders = _resolve_for_transform(
            heldout, board, rgb_intrinsics, intr, transform)
        metrics = transfer_metrics(aligned_heldout, rgb_intrinsics, intr, transform)
        metrics['numbering_resolution'] = dict(
            method='frozen_training_model_discrete_assignment_only', frames=heldout_orders,
            parameter_refit=False, physical_correspondence_verified=False)
        baselines.append(float(np.linalg.norm(transform[:3, 3])))
        report['folds'].append(dict(train_ids=[row[0] for row in training],
            heldout_ids=[row[0] for row in heldout], baseline_m=baselines[-1],
            T_rgb_left_aux=transform.tolist(), fit=fit, **metrics))
        fold_frames.extend(metrics['frames'])
    holdout = {name: float(np.sqrt(np.mean([frame[field]**2 for frame in fold_frames])))
               for name, field in [('rgb_transfer_rms_px', 'rgb_rms_px'),
                                   ('aux_transfer_rms_px', 'aux_rms_px')]}
    for metric, value in holdout.items():
        if not np.isfinite(value) or value > LIMITS[metric]:
            raise ValueError(f'离线留出检查失败：{metric}={value:.3f} px，'
                             f'上限 {LIMITS[metric]:g} px；请复核角点对应、图像和模型')
    log('留出检查完成，使用全部已确认观测进行最终联合细化')
    intr, transform, fit = _fit(validated, board, rgb_intrinsics, model_key)
    inferred = fit.get('numbering_resolution')
    if inferred:
        for row in inferred['frames']:
            log(f'最终自动对应 {row["source_id"]}: AUX={row["aux_order"]}；'
                f'输入={row["aux_input_fingerprint"][:12]}；'
                f'求解={row["aux_solver_fingerprint"][:12]}；待实物验证')
    baseline = float(np.linalg.norm(transform[:3, 3]))
    translation_deltas = [float(np.linalg.norm(np.asarray(fold['T_rgb_left_aux'])[:3, 3]
                                               - transform[:3, 3])) for fold in report['folds']]
    if baseline <= 1e-6 or max(translation_deltas) > max(
            LIMITS['baseline_fold_absolute_m'], baseline*LIMITS['baseline_fold_relative']):
        raise ValueError('跨折基线不稳定，请补充不同位置和倾角的观测并复核对应关系')
    aligned_all, _ = _resolve_for_transform(validated, board, rgb_intrinsics, intr, transform)
    training_metrics = transfer_metrics(aligned_all, rgb_intrinsics, intr, transform)
    report.update(heldout=holdout, fit=fit, training=training_metrics, baseline_m=baseline,
                  fold_translation_max_delta_m=max(translation_deltas))
    # Legacy scalar is explicitly AUX transfer error, never pooled RGB/IR pixels.
    extr = CrossModalExtrinsicResult(transform, training_metrics['aux_transfer_rms_px'],
        len(validated), [row['aux_rms_px'] for row in training_metrics['frames']])
    extr.calibration_info = report
    return intr, extr
