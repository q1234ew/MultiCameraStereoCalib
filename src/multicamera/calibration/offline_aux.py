"""Explicit, derived offline reviews; never mutate historical capture records."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
from pathlib import Path

import numpy as np

from .aux_capture_quality import assess_aux_frame, quality_reason_text
from .health import check_stereo_health
from .models import CameraModel


ORDER_OPTIONS = {'identity': '保持检测顺序', 'reverse_rows': '翻转行顺序',
                 'reverse_cols': '翻转列顺序', 'reverse_both': '反向编号（180°）'}


def review_order_check(decisions, report, board):
    """Compare selected relative numbering, not raw suspect flags or physical identity."""
    from .joint_aux import _rigid_grid_orders

    allowed = _rigid_grid_orders(board)
    candidates = {str(row['source_id']): row.get('aux_order')
                  for row in (report or {}).get('frames', [])}
    reliable = (report or {}).get('status') in ('consistent', 'inconsistent', 'partial')
    result = dict(conflict_ids=[], matched_ids=[], unchecked_ids=[], invalid_order_ids=[])
    flips = {'identity': (False, False), 'reverse_rows': (True, False),
             'reverse_cols': (False, True), 'reverse_both': (True, True)}
    for key, decision in decisions.items():
        orders = decision.get('corner_orders', {})
        rgb, aux = orders.get('rgb'), orders.get('aux')
        if rgb not in flips or aux not in flips:
            raise ValueError(f'{key}: 未知角点排列，请重新逐对确认')
        if rgb not in allowed or aux not in allowed:
            result['invalid_order_ids'].append(str(key))
            result['conflict_ids'].append(str(key))
            continue
        candidate = candidates.get(str(key)) if reliable else None
        if candidate not in allowed:
            result['unchecked_ids'].append(str(key))
            continue
        # Applying the same grid symmetry to both sensors preserves correspondence.
        relative = tuple(a != b for a, b in zip(flips[rgb], flips[aux]))
        state = 'matched_ids' if relative == flips[candidate] else 'conflict_ids'
        result[state].append(str(key))
    return result


def detection_fingerprint(detection):
    """Portable digest of the exact numbered solver observation, including geometry."""
    digest = hashlib.sha256(str(tuple(detection.image_size)).encode('ascii'))
    for points in (detection.object_points, detection.image_points):
        array = np.asarray(points, dtype='<f8')
        digest.update(str(array.shape).encode('ascii'))
        digest.update(array.tobytes(order='C'))
    return digest.hexdigest()


def numbering_provenance_summary(calibration_info: dict) -> dict:
    """Keep input attestations separate from the final algorithm's relative orders."""
    reviews = calibration_info.get('offline_reviews', [])
    reviews = {str(row['source_id']): row for row in reviews
               if isinstance(row, dict) and 'source_id' in row} if isinstance(reviews, list) else {}
    fit = calibration_info.get('fit', {})
    resolution = fit.get('numbering_resolution', {}) if isinstance(fit, dict) else {}
    rows = resolution.get('frames', []) if isinstance(resolution, dict) else []
    frames = []
    for row in rows if isinstance(rows, list) else []:
        if not isinstance(row, dict) or 'source_id' not in row:
            continue
        key = str(row['source_id'])
        review = reviews.get(key, {})
        orders = review.get('corner_orders', {})
        order = row.get('aux_order')
        if not isinstance(order, str) or order not in ORDER_OPTIONS:
            order = 'not_recorded'
        frames.append(dict(
            source_id=key,
            input_corner_orders=deepcopy(orders) if isinstance(orders, dict) else {},
            input_review_declaration=review.get('physical_corner_order', 'not_recorded'),
            algorithm_aux_order_relative_to_input=order,
            input_rgb_fingerprint=row.get('rgb_input_fingerprint'),
            input_aux_fingerprint=row.get('aux_input_fingerprint'),
            final_aux_fingerprint=row.get('aux_solver_fingerprint')))
    batches, invalid_batches = [], []
    has_batch_metadata = False
    for name, container in (('report', calibration_info), ('fit', fit),
                            ('resolution', resolution)):
        if not isinstance(container, dict) or 'source_ids' not in container:
            continue
        ids = container['source_ids']
        if (not isinstance(ids, list) or not ids or
                any(not isinstance(key, str) or not key for key in ids) or
                len(ids) != len(set(ids))):
            invalid_batches.append(name)
            continue
        batches.append(ids)
        has_batch_metadata |= name in ('report', 'fit')
    expected = list(dict.fromkeys(key for batch in batches for key in batch))
    valid_ids = [row['source_id'] for row in frames
                 if row['algorithm_aux_order_relative_to_input'] in ORDER_OPTIONS]
    metadata_consistent = (has_batch_metadata and not invalid_batches
                           and all(set(batch) == set(expected) for batch in batches))
    complete = (metadata_consistent and len(valid_ids) == len(frames) == len(expected)
                and len(valid_ids) == len(set(valid_ids)) and set(valid_ids) == set(expected)
                and len(frames) == len(rows))
    status = 'recorded' if complete else 'partial' if valid_ids else 'not_recorded'
    return dict(
        schema_version=1, input_review_scope='user_attestation_of_input_numbering_only',
        final_correspondence_scope='algorithm_candidate_not_physical_verification',
        physical_correspondence_verified=False,
        input_attested_source_ids=[key for key, review in reviews.items()
                                   if review.get('physical_corner_order') ==
                                   'manually_confirmed_per_pair'],
        final_numbering_status=status, expected_source_ids=expected,
        batch_metadata_consistent=bool(metadata_consistent),
        invalid_batch_metadata=invalid_batches,
        missing_source_ids=[key for key in expected if key not in valid_ids],
        unexpected_source_ids=[key for key in dict.fromkeys(valid_ids)
                               if expected and key not in expected],
        automatically_reordered_source_ids=[row['source_id'] for row in frames
            if row['algorithm_aux_order_relative_to_input'] in ORDER_OPTIONS
            and row['algorithm_aux_order_relative_to_input'] != 'identity'],
        frames=frames)


def reordered_detection(detection, board, order):
    if order not in ORDER_OPTIONS:
        raise ValueError('未知角点排列')
    if not detection.valid or detection.num_points != board.cols * board.rows:
        raise ValueError('未检测到完整网格，不能确认角点顺序')
    grid = detection.image_points.reshape(board.rows, board.cols, 2)
    if order in ('reverse_rows', 'reverse_both'):
        grid = grid[::-1]
    if order in ('reverse_cols', 'reverse_both'):
        grid = grid[:, ::-1]
    return replace(detection, image_points=grid.reshape(-1, 2).copy(),
                   detection_info=deepcopy(detection.detection_info))


def file_identity(path):
    source = Path(path).resolve()
    return {'path': str(source), 'sha256': hashlib.sha256(source.read_bytes()).hexdigest()}


def offline_pair_review(key, images, detections, board, *, modality='ir', paths=None,
                        capture_review=None):
    qualities = [assess_aux_frame(image, detection, board,
                 modality='rgb' if side == 0 or modality == 'acoustic_rgb' else 'ir')
                 for side, (image, detection) in enumerate(zip(images, detections))]
    reasons = [f'{name}：{quality_reason_text(reason)}'
               for name, quality in zip(('RGB', 'IR/辅助'), qualities)
               for reason in quality.reasons]
    return dict(schema_version=1, source_id=str(key), kind='offline_image_review',
        created=datetime.now(timezone.utc).isoformat(), board=board.to_dict(),
        image_accepted=not reasons, image_reasons=reasons,
        metrics=[quality.metrics for quality in qualities],
        detections=[deepcopy(det.detection_info) for det in detections],
        sources=[file_identity(path) for path in paths] if paths else [],
        capture_review=deepcopy(capture_review), temporal_stability='not_verified_offline',
        synchronization='not_verified_offline', physical_corner_order='unconfirmed',
        included=False)


def confirm_offline_pair(review, rgb_order, aux_order):
    if not review.get('image_accepted'):
        raise ValueError('原图质量检查未通过，不能纳入离线求解')
    if rgb_order not in ORDER_OPTIONS or aux_order not in ORDER_OPTIONS:
        raise ValueError('未知角点排列')
    result = deepcopy(review)
    result.update(included=True, physical_corner_order='manually_confirmed_per_pair',
                  corner_orders={'rgb': rgb_order, 'aux': aux_order},
                  confirmed=datetime.now(timezone.utc).isoformat())
    return result


def is_legacy_offline_pair(name, pair, recorded):
    return (name == 'offline' and name not in recorded and
            (pair.left_id, pair.right_id) == ('offline_L', 'offline_R'))


def validate_session_rig(session, configured_pairs=None, *, allow_legacy_offline=False):
    """Preflight before replacing current UI state. No network or source writes."""
    rig = session.load_rig()
    if rig is None or not rig.pairs:
        raise ValueError('会话缺少已保存的 RGB 双目标定结果；请先完成 RGB 标定')
    metadata = session.metadata
    recorded = {pair['name']: pair for pair in metadata.get('pairs', [])}
    if not recorded and not allow_legacy_offline:
        raise ValueError('会话缺少左右相机绑定信息，不能恢复 RGB 内参')
    configured = ({pair.name: pair for pair in configured_pairs}
                  if configured_pairs is not None else None)
    for name, pair in rig.pairs.items():
        binding = recorded.get(name)
        legacy = allow_legacy_offline and is_legacy_offline_pair(name, pair, recorded)
        if not legacy and (binding is None or (pair.left_id, pair.right_id) != (
                binding.get('left_id'), binding.get('right_id')) or pair.left_id == pair.right_id):
            stored = binding or {}
            raise ValueError(f'{name}: RGB 结果与会话左右相机 ID 不一致；'
                f'结果 左={pair.left_id} / 右={pair.right_id}；'
                f'会话 左={stored.get("left_id", "未记录")} / 右={stored.get("right_id", "未记录")}')
        binding = binding or {}
        current = configured.get(name) if configured is not None else None
        if configured is not None and (current is None or
                (current.left.camera_id, current.right.camera_id) != (pair.left_id, pair.right_id)):
            raise ValueError(f'{name}: 当前相机配置与会话左右绑定不同，请先配置相同相机组')
        model = CameraModel(metadata.get('camera_model', 'pinhole'))
        intrinsics = (pair.left_intrinsics, pair.right_intrinsics)
        if any(intr.model != model for intr in intrinsics) or pair.stereo.model != model:
            raise ValueError(f'{name}: 会话与 RGB 标定模型不一致')
        if intrinsics[0].image_size != intrinsics[1].image_size:
            raise ValueError(f'{name}: RGB 左右内参分辨率不一致')
        # Declared dimensions, when present in newer sessions, must also agree.
        for side, intr in zip(('left', 'right'), intrinsics):
            size = binding.get(f'{side}_image_size')
            if size is not None and tuple(size) != tuple(intr.image_size):
                raise ValueError(f'{name}: {side} 会话分辨率与内参不同')
        health = check_stereo_health(*intrinsics, pair.stereo, require_rectification=False)
        if health['status'] != 'ok':
            raise ValueError(f'{name}: RGB 结果有效性检查未通过：' + '；'.join(health['reasons']))
    return rig


def verify_review_sources(reviews):
    for review in reviews:
        if not review.get('included') or review.get('physical_corner_order') != 'manually_confirmed_per_pair':
            raise ValueError('存在未逐对确认的离线观测')
        for source in review.get('sources', []):
            if file_identity(source['path'])['sha256'] != source['sha256']:
                raise ValueError(f"原图已变化，请重新导入复核：{source['path']}")
    return True
