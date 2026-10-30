"""Preview-first stereo capture workbench; measurements are never synthetic."""

from __future__ import annotations

import time
import copy
import math
import logging
from dataclasses import replace

import cv2
import numpy as np
from PySide6.QtCore import QEvent, Qt, QTimer, QThreadPool, Slot
from PySide6.QtWidgets import (
    QComboBox, QGridLayout, QHBoxLayout, QLabel, QScrollArea, QSizePolicy, QToolButton,
    QVBoxLayout, QWidget,
)

from ...calibration.models import CameraModel
from ...calibration.adaptive_guidance import (
    DISTANCE_LABELS, GuidanceObservation, evaluate_distance, measure_pose, measure_composition,
    _candidate_feasible, reliable_distance_pose, MAX_DISTANCE_REPAIR_REQUESTS, SCALE_AREA_RANGES,
)
from ...calibration.observations import similar_composition
from ..threads.guidance_worker import GuidanceWorker
from ..theme import ACCENT, BORDER, SUCCESS, TEXT_DIM, WARNING, app_icon
from .capture_guide import BoardTilt, TILT_LABELS, CaptureTarget, CaptureSequence
from .stereo_schematic import StereoSchematic, TILT_ACTIONS
from .stable_text import StableTextLabel

logger = logging.getLogger(__name__)


class StereoCaptureWorkspace(QWidget):
    """Hosts existing controls while preserving their ownership and signals."""

    MAX_SAMPLE_AGE = 1.5
    MIN_STABLE_SECONDS = .6
    MIN_STABLE_SAMPLES = 3

    def __init__(self, panel, stream_view, board):
        super().__init__()
        self.panel, self.view, self.board = panel, stream_view, board
        self.samples = {}
        self._clock = time.monotonic
        self._context = None
        self._coverage_key = None
        self._coverage_summary = []
        self._compact = False
        self._analysis = None
        self._analysis_key = None
        self._requested_key = None
        self._guidance_worker = None
        self._guidance_error = ''
        self._distance_override = None
        self._distance_scope = None
        self._distance_session = None
        self._instruction_stage = 'waiting'
        self._adaptive_sequence = CaptureSequence([CaptureTarget(1, 1, BoardTilt.FRONT)])
        self.banner = QWidget()
        banner = QHBoxLayout(self.banner)
        banner.setContentsMargins(14, 8, 14, 8)
        self.banner.setObjectName('stereoCaptureBanner')
        self.banner.setStyleSheet(
            f"QWidget#stereoCaptureBanner {{background:#202830; border-bottom:1px solid {BORDER};}}"
            "QWidget#stereoCaptureBanner QLabel {background:transparent; border:none;}"
        )
        self.instruction = StableTextLabel("连接左右相机并放入标定板", lines=2)
        self.instruction.setWordWrap(True)
        self.instruction.setStyleSheet(
            f"color:{ACCENT}; font-size:18px; font-weight:bold; background:transparent;"
        )
        banner.addWidget(self.instruction, 2)
        self.reason = StableTextLabel("固定采图计划 · 等待实时检测", lines=2)
        self.reason.setWordWrap(True)
        self.reason.setStyleSheet('background:transparent;')
        banner.addWidget(self.reason, 1)
        self.guide_mode = QComboBox()
        self.guide_mode.addItem('自适应引导', 'adaptive')
        self.guide_mode.addItem('固定采图计划', 'fixed')
        self.guide_mode.setAccessibleName('采图引导策略')
        self.guide_mode.currentIndexChanged.connect(self.refresh)
        banner.addWidget(self.guide_mode)
        self.pair = QComboBox()
        self.pair.setMaximumWidth(220)
        self.pair.setAccessibleName("双目采集目标")
        self.pair.currentTextChanged.connect(self._select_pair)
        banner.addWidget(self.pair)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(6, 0, 6, 0)
        outer.setSpacing(6)
        outer.setAlignment(Qt.AlignTop)
        body = QHBoxLayout()
        schematic_column = QVBoxLayout()
        self.schematic_title = QLabel("相机与目标示意")
        self.schematic_title.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        schematic_column.addWidget(self.schematic_title)
        self.schematic = StereoSchematic()
        schematic_column.addWidget(self.schematic)
        body.addLayout(schematic_column, 3)
        self.pose = QLabel("当前姿态（相对左相机）\n\n距离  待内参\n左右倾  待内参\n上下倾  待内参")
        self.pose.setWordWrap(True)
        self.pose.setMinimumWidth(165)
        pose_column = QVBoxLayout()
        pose_column.addWidget(self.pose)
        self.pose_status = QLabel("姿态：待人工确认")
        self.pose_status.setWordWrap(True)
        self.pose_status.setStyleSheet(f"color:{WARNING};")
        pose_column.addWidget(self.pose_status)
        body.addLayout(pose_column, 1)
        self.status = QLabel("采集状态\n\n双路可见：等待检测\n清晰度：待检测\n运动稳定：待检测\n目标位置：待检测")
        self.status.setWordWrap(True)
        self.status.setMinimumWidth(205)
        body.addWidget(self.status, 1)
        outer.addLayout(body)
        self.compact_summary = StableTextLabel(lines=7)
        self.compact_summary.setWordWrap(True)
        self.compact_summary.hide()
        outer.addWidget(self.compact_summary)
        self.advice = StableTextLabel("建议补拍不同位置和倾角", lines=3)
        self.advice.setWordWrap(True)
        self.advice.setStyleSheet(f"color:{TEXT_DIM}; font-size:12px;")
        outer.addWidget(self.advice)
        self.receipt = StableTextLabel('本次原图：尚未采集', lines=2)
        self.receipt.setAccessibleName('最近原图采集结果')
        outer.addWidget(self.receipt)

        self.distance_tools = QWidget()
        distance_row = QGridLayout(self.distance_tools)
        self._distance_layout = distance_row
        self._distance_wrapped = None
        self._distance_viewport = None
        distance_row.setContentsMargins(0, 0, 0, 0)
        self.btn_near = QToolButton()
        self.btn_far = QToolButton()
        for column, (button, text, scale) in enumerate((
                (self.btn_near, '补近距离', 'large'), (self.btn_far, '补远距离', 'small'))):
            button.setText(text)
            button.setIcon(app_icon('camera'))
            button.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
            button.setAccessibleName(text)
            button.setToolTip(f'在中央正面补采{DISTANCE_LABELS[scale]}；保留全部原图验收和自动采集门禁')
            button.clicked.connect(lambda checked=False, s=scale: self._start_distance_capture(s))
            distance_row.addWidget(button, 0, column)
        self.btn_resume_guidance = QToolButton()
        self.btn_resume_guidance.setText('返回推荐')
        self.btn_resume_guidance.setIcon(app_icon('rotate-ccw'))
        self.btn_resume_guidance.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.btn_resume_guidance.setToolTip('结束指定距离补采，返回基于现有数据的自适应推荐')
        self.btn_resume_guidance.clicked.connect(self._resume_guidance)
        self.btn_resume_guidance.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        policy = self.btn_resume_guidance.sizePolicy()
        policy.setRetainSizeWhenHidden(True)
        self.btn_resume_guidance.setSizePolicy(policy)
        distance_row.addWidget(self.btn_resume_guidance, 0, 2)
        distance_row.setColumnStretch(3, 1)
        self.distance_tools.installEventFilter(self)
        outer.insertWidget(0, self.distance_tools)

        self.coverage_button = QToolButton()
        self.coverage_button.setCheckable(True)
        self.coverage_button.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.coverage_button.setIcon(app_icon('chevron-right'))
        self.coverage_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.coverage_button.setAccessibleName("展开覆盖分析与采集设置")
        outer.addWidget(self.coverage_button)
        self.details = QScrollArea()
        self.details.setWidgetResizable(True)
        self.details.setMaximumHeight(220)
        self.details.hide()
        self.coverage_button.toggled.connect(self._toggle_details)
        outer.addWidget(self.details)
        self.timer = QTimer(self)
        self.timer.setInterval(250)
        self.timer.timeout.connect(self.refresh)
        stream_view.preview_detected.connect(self.observe)
        stream_view.preview_detection_expired.connect(self._expire_sample)
        panel.data_source_changed.connect(self.refresh)
        panel._seq_widget.user_navigated.connect(self._manual_navigation)
        self.refresh()

    @property
    def adaptive(self):
        return self.guide_mode.currentData() == 'adaptive' and not self.panel.offline_selection

    def eventFilter(self, watched, event):
        if (watched in (self.distance_tools, self._distance_viewport)
                and event.type() in (QEvent.Resize, QEvent.LayoutRequest)):
            self._layout_distance_controls()
        return super().eventFilter(watched, event)

    def _layout_distance_controls(self):
        buttons = (self.btn_near, self.btn_far, self.btn_resume_guidance)
        visible = [button for button in buttons if not button.isHidden()]
        required = sum(button.sizeHint().width() for button in visible)
        required += max(0, len(visible) - 1) * self._distance_layout.horizontalSpacing()
        margins = self.layout().contentsMargins()
        parent_width = self.parentWidget().width() if self.parentWidget() else self.width()
        available = min(self.distance_tools.width(), parent_width - margins.left() - margins.right())
        wrapped = not self.btn_resume_guidance.isHidden() and required > available
        if wrapped == self._distance_wrapped:
            return
        self._distance_wrapped = wrapped
        self._distance_layout.removeWidget(self.btn_resume_guidance)
        if wrapped:
            self._distance_layout.addWidget(self.btn_resume_guidance, 1, 0, 1, 2)
        else:
            self._distance_layout.addWidget(self.btn_resume_guidance, 0, 2)
        for column, stretch in enumerate((1, 1, 0, 0) if wrapped else (0, 0, 0, 1)):
            self._distance_layout.setColumnStretch(column, stretch)

    def _manual_navigation(self):
        self.guide_mode.setCurrentIndex(1)

    def _distance_sizes(self):
        sizes = []
        for pair in self.panel._selected_pairs().values():
            for camera in (pair.left, pair.right):
                frame = self.panel._sm.get_latest_frame(camera.camera_id)
                cal = self.panel._intr_cals.get(camera.camera_id)
                intr = self.panel._intrinsics.get(camera.camera_id)
                size = (frame[0].shape[1::-1] if frame is not None else
                        cal._image_size if cal is not None and cal._image_size else
                        intr.image_size if intr is not None else None)
                if size:
                    sizes.append(size)
        return sizes

    def _distance_available(self, scale):
        return _candidate_feasible(self.board, (1, 1, 'front', scale), self._distance_sizes())

    def _apply_guidance_target(self):
        if self._distance_override:
            target = CaptureTarget(1, 1, BoardTilt.FRONT, scale=self._distance_override)
        elif self._analysis is not None:
            rec = self._analysis.recommendation
            tilts = dict(zip(('front', 'left', 'right', 'up', 'down'), BoardTilt))
            target = CaptureTarget(rec.zone_row, rec.zone_col, tilts[rec.tilt], scale=rec.scale)
        else:
            return
        self._adaptive_sequence = CaptureSequence([target])

    def _start_distance_capture(self, scale):
        button = self.btn_near if scale == 'large' else self.btn_far
        if not button.isEnabled() or not self.distance_tools.isVisible():
            return
        self._distance_override = scale
        self._apply_guidance_target()
        logger.info('Explicit distance capture selected: scale=%s', scale)
        self.refresh()

    def _resume_guidance(self):
        self._distance_override = None
        self._apply_guidance_target()
        self.refresh()

    def _guidance_context(self):
        names = list(self.panel._selected_pairs())
        # A recommendation belongs to one rig pair; all-pairs capture keeps the fixed plan.
        if len(names) != 1 or names[0] not in self.panel._sm.stereo_pairs:
            return None, None
        name = names[0]
        pair = self.panel._sm.stereo_pairs[name]
        ids = (pair.left.camera_id, pair.right.camera_id)
        cals = [self.panel._intr_cals.get(cid) for cid in ids]
        intr_key = tuple((cid, self.panel._intrinsics[cid].image_size,
                          self.panel._intrinsics[cid].model,
                          self.panel._intrinsics[cid].camera_matrix.tobytes(),
                          self.panel._intrinsics[cid].dist_coeffs.tobytes())
                         for cid in ids if cid in self.panel._intrinsics)
        key = (id(self.panel._board), id(self.panel._session), name, self.panel._camera_model,
               tuple((id(c), c.data_revision if c else 0) for c in cals),
               tuple(self.panel._live_preview_indices.get(name, [])), intr_key,
               tuple(sorted(self.panel._guidance_failed_saves)))
        return key, (name, ids, cals)

    def _request_guidance(self):
        key, context = self._guidance_context()
        if key is None:
            return None
        if (self._guidance_worker is not None or self._requested_key == key
                or self.panel._pending_save_tokens or self.panel._calib_worker is not None
                or self.panel._stereo_import_worker is not None):
            return key
        name, ids, cals = context
        observations = {cid: [] for cid in ids}
        failed = self.panel._guidance_failed_saves
        for li, ri, source in self.panel._live_preview_indices.get(name, []):
            if (id(self.panel._session), name, str(source)) in failed:
                continue
            if any(cal is None or i >= cal.num_frames for cal, i in zip(cals, (li, ri))):
                continue
            for cid, cal, i in zip(ids, cals, (li, ri)):
                observations[cid].append(GuidanceObservation(
                    corners=cal._all_corners[i].copy(), ids=cal._all_ids[i].copy(),
                    image_size=cal._image_size, frame_id=str(source),
                ))
        worker = GuidanceWorker(
            key, self.panel._board, self.panel._camera_model, observations,
            copy.deepcopy({cid: self.panel._intrinsics[cid] for cid in ids
                           if cid in self.panel._intrinsics}),
        )
        worker.signals.completed.connect(self._guidance_completed)
        self._requested_key = key
        self._guidance_worker = worker
        self._guidance_error = ''
        QThreadPool.globalInstance().start(worker)
        return key

    @Slot(object, object, str)
    def _guidance_completed(self, key, result, error):
        self._guidance_worker = None
        current_key, context = self._guidance_context()
        if current_key == key:
            self._analysis, self._analysis_key = result, key
            self._guidance_error = error
            if result is not None:
                rec = result.recommendation
                self._apply_guidance_target()
                logger.info('Capture guidance updated: pair=%s samples=%s target=%s/%s/%s/%s '
                            'pose_available=%s feasible=%s phase=%s initial_total=%s '
                            'repairs=%s distance_counts=%s missing=%s', context[0], rec.sample_count,
                            rec.zone_row, rec.zone_col, rec.tilt, rec.scale,
                            rec.pose_available, rec.feasible, rec.phase, rec.initial_total,
                            rec.repair_count, result.distance_counts, rec.missing_distances)
        self.refresh()

    def _target(self):
        key, _ = self._guidance_context()
        if self.adaptive and key is not None and self._distance_override:
            return self._adaptive_sequence.current
        if self.adaptive and key is not None and self._analysis_key == key and self._analysis:
            return self._adaptive_sequence.current
        return self.panel._capture_sequence.current

    def _pose_measurement(self, camera_id, sample):
        if not sample or not sample['valid']:
            return None
        key, _ = self._guidance_context()
        intr = self.panel._intrinsics.get(camera_id)
        if intr is None and self._analysis and self._analysis_key == key:
            intr = self._analysis.intrinsics.get(camera_id)
        sample['pose_expected'] = intr is not None
        if intr is None or intr.model != self.panel._camera_model:
            return None
        det = sample['detection']
        cache_key = (id(intr), id(det))
        if sample.get('pose_key') == cache_key:
            return sample.get('pose')
        pose = measure_pose(self.board, GuidanceObservation(
            corners=det.charuco_corners, ids=det.charuco_ids,
            image_size=det.image_size, frame_id='preview',
        ), intr)
        sample['pose_key'], sample['pose'] = cache_key, pose
        return pose

    def _distance_status(self, pairs, samples, target):
        if not target or not samples or not all(s['valid'] for s in samples):
            return False, '等待双路距离检测'
        cameras = [camera for pair in pairs for camera in (pair.left, pair.right)]
        poses = [self._pose_measurement(camera.camera_id, sample)
                 for camera, sample in zip(cameras, samples)]
        if len(poses) != len(samples):
            return False, '等待双路距离检测'
        ready, text, _, _ = evaluate_distance(
            target.scale, [dict(sample, pose=pose) for sample, pose in zip(samples, poses)])
        return ready, text

    def _adaptive_aligned(self, pairs, samples, target):
        feasible = (self._distance_available(self._distance_override) if self._distance_override else
                    self._analysis is None or self._analysis.recommendation.feasible)
        if not feasible:
            return False, '无可用构图，请调整分辨率或标定板'
        if not target or not samples or not all(s['valid'] for s in samples):
            return False, '等待双路检测'
        distance_ready, distance_message = self._distance_status(pairs, samples, target)
        if not distance_ready:
            return False, distance_message
        if not all(np.linalg.norm(s['center'] - (target.norm_cx, target.norm_cy)) < .16
                   for s in samples):
            return False, '位置待对齐'
        poses = [self._pose_measurement(camera.camera_id, sample)
                 for pair, offset in zip(pairs, range(0, len(samples), 2))
                 for camera, sample in zip((pair.left, pair.right), samples[offset:offset + 2])]
        if not poses or any(p is None for p in poses):
            return False, '倾角待内参；可手动采集基础样本'
        def matches(p):
            yaw, pitch = p['yaw'], p['pitch']
            if target.tilt == BoardTilt.FRONT:
                return abs(yaw) < 15 and abs(pitch) < 15
            primary, secondary, sign = {
                BoardTilt.TILT_LEFT: (yaw, pitch, -1),
                BoardTilt.TILT_RIGHT: (yaw, pitch, 1),
                BoardTilt.TILT_UP: (pitch, yaw, -1),
                BoardTilt.TILT_DOWN: (pitch, yaw, 1),
            }[target.tilt]
            return 15 <= sign * primary <= 40 and abs(secondary) < 20
        if all(matches(p) for p in poses):
            return True, '姿态接近建议范围'
        angles = ' / '.join(f'左右 {p["yaw"]:+.0f}°、上下 {p["pitch"]:+.0f}°' for p in poses)
        return False, f'倾角待调整：{angles}；倾斜目标15–40°，正面小于15°'

    def _select_pair(self, text):
        if text:
            self.panel._combo_pair.setCurrentText(text)

    def _toggle_details(self, expanded):
        self.details.setVisible(expanded)
        self.coverage_button.setIcon(app_icon('chevron-down' if expanded else 'chevron-right'))

    def showEvent(self, event):
        super().showEvent(event)
        if self.parentWidget() is not self._distance_viewport:
            if self._distance_viewport is not None:
                self._distance_viewport.removeEventFilter(self)
            self._distance_viewport = self.parentWidget()
            if self._distance_viewport is not None:
                self._distance_viewport.installEventFilter(self)
        self._layout_distance_controls()
        self.samples.clear()
        self.timer.start()
        self.refresh()

    def hideEvent(self, event):
        self.timer.stop()
        self.samples.clear()
        super().hideEvent(event)

    def observe(self, camera_id, detection, frame):
        if not self.isVisible() or self.view.review_key:
            return
        self._refresh_sample_context()
        previous = self.samples.get(camera_id)
        now = self._clock()
        sample = self._detection_sample(detection, frame)
        sample['time'] = now
        if sample['valid']:
            pts = detection.charuco_corners.reshape(-1, 2)
            w, h = detection.image_size
            anchor = {'ids': detection.charuco_ids.copy(), 'corners': pts.copy(), 'size': (w, h)}
            since, count = now, 1
            if (previous and previous['valid'] and
                    0 < now - previous['time'] < self.MAX_SAMPLE_AGE):
                old = previous['anchor']
                common, old_idx, new_idx = np.intersect1d(
                    old['ids'].ravel(), detection.charuco_ids.ravel(), return_indices=True,
                )
                if (len(common) >= max(6, .6 * max(len(old['ids']), len(pts)))
                        and old['size'] == detection.image_size):
                    delta = old['corners'][old_idx] - pts[new_idx]
                    if float(np.max(np.linalg.norm(delta, axis=1))) / max(w, h) < .008:
                        anchor, since, count = old, previous['stable_since'], previous['stable_count'] + 1
            sample.update(anchor=anchor, stable_since=since, stable_count=count,
                          stable=count >= self.MIN_STABLE_SAMPLES and now - since >= self.MIN_STABLE_SECONDS)
        self.samples[camera_id] = sample
        self.refresh()

    def _expire_sample(self, camera_id):
        self.samples.pop(camera_id, None)
        if self.isVisible():
            self.refresh()

    def _detection_sample(self, detection, frame):
        result = {'valid': False, 'detection': detection, 'sharp': False, 'stable': False}
        if not detection.valid or frame.shape[1::-1] != tuple(detection.image_size):
            return result
        geometry = measure_composition(self.panel._board, GuidanceObservation(
            detection.charuco_corners, detection.charuco_ids, detection.image_size, 'preview'))
        if geometry is None:
            return result
        result.update(geometry, valid=True)
        pts = detection.charuco_corners.reshape(-1, 2)
        w, h = detection.image_size
        x, y, rw, rh = cv2.boundingRect(pts.astype(np.float32))
        roi = frame[max(y, 0):min(y + rh, h), max(x, 0):min(x + rw, w)]
        if roi.size:
            scale = min(1., 320 / max(roi.shape[:2]))
            small = cv2.resize(roi, None, fx=scale, fy=scale) if scale < 1 else roi
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY) if small.ndim == 3 else small
            result['sharp'] = float(cv2.Laplacian(gray, cv2.CV_64F).var()) >= 50
        return result

    def _current_samples(self):
        names = self.panel._selected_pairs()
        pairs = [self.panel._sm.stereo_pairs[n] for n in names if n in self.panel._sm.stereo_pairs]
        samples = []
        now = self._clock()
        for pair in pairs:
            for camera in (pair.left, pair.right):
                sample = self.samples.get(camera.camera_id)
                entry = self.panel._sm.get_latest_frame(camera.camera_id)
                if (sample is None or entry is None or
                        not math.isfinite(now - sample['time']) or
                        not 0 <= now - sample['time'] <= self.MAX_SAMPLE_AGE or
                        not math.isfinite(time.time() - entry[1]) or
                        not 0 <= time.time() - entry[1] <= self.MAX_SAMPLE_AGE):
                    return pairs, []
                samples.append(sample)
        return pairs, samples

    def _duplicate_source(self, pair, detections):
        if not all(d is not None and d.valid for d in detections):
            return None
        cals = [self.panel._intr_cals.get(c.camera_id) for c in (pair.left, pair.right)]
        if any(c is None for c in cals):
            return None
        for li, ri, source in reversed(self.panel._live_preview_indices.get(pair.name, [])):
            if (id(self.panel._session), pair.name, str(source)) in self.panel._guidance_failed_saves:
                continue
            if any(i >= cal.num_frames for i, cal in zip((li, ri), cals)):
                continue
            if all(similar_composition(cal._all_corners[i], cal._all_ids[i], cal._image_size,
                                       d.charuco_corners, d.charuco_ids, d.image_size)
                   for i, cal, d in zip((li, ri), cals, detections)):
                return str(source)
        return None

    def _duplicate_matches(self, pairs, samples):
        return [(pair.name, source) for offset, pair in enumerate(pairs)
                if len(samples) >= 2 * offset + 2 and (source := self._duplicate_source(
                    pair, [s['detection'] for s in samples[2 * offset:2 * offset + 2]])) is not None]

    def auto_ready(self):
        if (self.panel._pending_save_tokens or self.panel._failed_save_jobs or self.panel._save_workers
                or self.panel._save_queue or self.panel._model_worker or self.panel._calib_worker
                or self.panel._stereo_import_worker or self.panel._aux_dataset_busy()):
            return False
        _, samples = self._current_samples()
        target = self._target()
        if self.adaptive:
            key, _ = self._guidance_context()
            if key is not None:
                pairs, samples = self._current_samples()
                return bool(
                    self.isVisible() and not self.view.review_key and self._analysis
                    and self._analysis_key == key and not self.panel._pending_save_tokens
                    and (self._distance_override is not None
                         or self._analysis.recommendation.phase != 'review')
                    and self._guidance_worker is None
                    and all(s['valid'] and s.get('sharp') and s.get('stable') for s in samples)
                    and self._adaptive_aligned(pairs, samples, target)[0]
                    and not self._duplicate_matches(pairs, samples)
                    and all(self.panel._sm.get_sync_pair(name) is not None
                            for name in self.panel._selected_pairs())
                )
        return bool(
            self.isVisible() and not self.view.review_key and target and samples
            and target.tilt == BoardTilt.FRONT
            and all(s["valid"] and s.get("sharp") and s.get("stable") for s in samples)
            and all(np.linalg.norm(s["center"] - (target.norm_cx, target.norm_cy)) < .16
                    for s in samples)
            and all(self.panel._sm.get_sync_pair(name) is not None
                    for name in self.panel._selected_pairs())
        )

    def validate_capture(self, name, left, right, det_left, det_right):
        """Recheck already-detected raw images before any observation or save mutation."""
        if not self.auto_ready():
            return False, '实时采集条件已变化'
        pair = self.panel._sm.stereo_pairs.get(name)
        if pair is None or name not in self.panel._selected_pairs():
            return False, '采集相机组已变化'
        samples = [self._detection_sample(detection, image) for detection, image in
                   ((det_left, left), (det_right, right))]
        if not all(s['valid'] for s in samples):
            return False, '原图板点不足或几何无效'
        if det_left.image_size != det_right.image_size:
            return False, '左右原图分辨率不一致'
        if not all(s['sharp'] for s in samples):
            return False, '原图清晰度不足'
        for camera, detection in zip((pair.left, pair.right), (det_left, det_right)):
            previous = self.samples.get(camera.camera_id)
            if previous is None:
                return False, '缺少当前稳定检测'
            old = previous['detection']
            common, a, b = np.intersect1d(old.charuco_ids.ravel(), detection.charuco_ids.ravel(),
                                        return_indices=True)
            if (old.image_size != detection.image_size or
                    len(common) < max(6, .6 * max(old.num_corners, detection.num_corners))):
                return False, '原图与预览可见板点不一致'
            delta = old.charuco_corners.reshape(-1, 2)[a] - detection.charuco_corners.reshape(-1, 2)[b]
            if np.max(np.linalg.norm(delta, axis=1)) / max(detection.image_size) >= .008:
                return False, '原图相对稳定预览已移动'
        target = self._target()
        if self.adaptive:
            accepted, reason = self._adaptive_aligned([pair], samples, target)
            if not accepted:
                return False, '原图' + reason
        elif not target or target.tilt != BoardTilt.FRONT or not all(
                np.linalg.norm(s['center'] - (target.norm_cx, target.norm_cy)) < .16 for s in samples):
            return False, '原图位置偏离建议区域'
        for camera, detection in zip((pair.left, pair.right), (det_left, det_right)):
            cal = self.panel._intr_cals.get(camera.camera_id)
            if cal is not None and cal._image_size not in (None, detection.image_size):
                return False, '原图分辨率与已采数据不一致'
        stereo = self.panel._stereo_cals.get(name)
        if stereo is not None and stereo._image_size not in (None, det_left.image_size):
            return False, '原图分辨率与双目观测不一致'
        source = self._duplicate_source(pair, (det_left, det_right))
        if source is not None:
            return False, f'原图接近已采配对 {source}，请改变位置、大小或倾角'
        return True, ''

    def _pose_text(self, camera_id, sample):
        intr = self.panel._intrinsics.get(camera_id)
        key, _ = self._guidance_context()
        temporary = False
        if intr is None and self._analysis and self._analysis_key == key:
            intr = self._analysis.intrinsics.get(camera_id)
            temporary = intr is not None
        if intr is None:
            return "距离  待内参\n左右倾  待内参\n上下倾  待内参"
        if not sample or not sample["valid"]:
            return "距离  待实时检测\n左右倾  待实时检测\n上下倾  待实时检测"
        det = sample["detection"]
        if det.image_size != intr.image_size:
            return "图像尺寸与内参不匹配\n请重新标定或恢复分辨率"
        pose = self._pose_measurement(camera_id, sample)
        if pose is None:
            return "姿态估计不可用\n请检查内参、可见角点或调整姿态"
        prefix = '临时内参估计\n' if temporary else ''
        distance = (f"板中心深度  {pose['depth']:.2f} m（可信）" if reliable_distance_pose(pose)
                    else '距离待确认（不显示米数）')
        return prefix + (f"{distance}\n"
                         f"左右倾  {pose['yaw']:.1f}°\n上下倾  {pose['pitch']:.1f}°")

    def _refresh_sample_context(self):
        scope = (id(self.panel._board), self.panel._camera_model,
                 tuple((name, pair.left.camera_id, pair.right.camera_id,
                        pair.left.url, pair.right.url, id(pair))
                       for name, pair in self.panel._selected_pairs().items()))
        if (scope != self._distance_scope or not self.adaptive or self.panel._active_step != 0
                or self.panel._workflow_phase != 'capture'
                or (self._distance_session is not None
                    and self._distance_session is not self.panel._session)):
            had_override = self._distance_override is not None
            self._distance_override = None
            if had_override:
                self._apply_guidance_target()
        self._distance_scope = scope
        self._distance_session = self.panel._session
        context = (id(self.panel._board), id(self.panel._session), self.panel._camera_model,
                   tuple((name, pair.left.url, pair.right.url, id(pair))
                         for name, pair in self.panel._selected_pairs().items()), self.view.review_key)
        if context != self._context:
            self.samples.clear()
            self._context = context
        self.board = self.panel._board

    def _primary_action(self, pairs, samples, target, duplicate):
        """Order one actionable instruction without relaxing any capture gate."""
        if self.panel._pending_save_tokens or self.panel._save_workers or self.panel._save_queue:
            return 'saving', '保持当前位置，等待原图保存完成', 'hold'
        if self.panel._failed_save_jobs:
            return 'saving', '先重试保存失败的原图，或撤销该次采集', 'hold'
        if self.panel._white_balance_guard.busy:
            return 'preparation', '保持当前位置，等待相机参数稳定', 'hold'
        if (self.panel._model_worker or self.panel._calib_worker or self.panel._stereo_import_worker
                or self.panel._aux_dataset_busy()):
            return 'preparation', '保持当前位置，等待当前任务完成', 'hold'
        if not self._distance_override and self._analysis.recommendation.phase == 'review':
            return 'review', '本轮引导结束，可前往标定评估；仍可手动补拍', 'hold'
        feasible = (self._distance_available(self._distance_override) if self._distance_override else
                    self._analysis.recommendation.feasible)
        if not feasible:
            return 'blocked', '当前距离构图不可用，请检查分辨率和标定板', 'hold'
        if not samples or not all(s['valid'] for s in samples):
            return 'visibility', '让标定板同时进入左右画面', 'hold'
        ready, message = self._distance_status(pairs, samples, target)
        if not ready:
            if '距离临界' in message:
                return 'distance', '前后小幅微调至目标距离档位中部，再保持静止', 'hold'
            if '距离待确认' in message:
                return 'distance', '保持板面清晰、朝向稳定，等待距离确认', 'hold'
            if '判断冲突' in message:
                return 'distance', '先检查左右共视，暂停前后移动', 'hold'
            if '靠近' in message:
                return 'distance', '板面先正对相机，再靠近镜头；保持当前高度', 'approach'
            if '远离' in message:
                return 'distance', '板面先正对相机，再远离镜头；保持当前高度', 'retreat'
            return 'distance', '保持板面朝向，等待双路距离检测', 'hold'
        if not all(np.linalg.norm(s['center'] - (target.norm_cx, target.norm_cy)) < .16
                   for s in samples):
            positions = (('左上', '上方', '右上'), ('左侧', '中央', '右侧'), ('左下', '下方', '右下'))
            return 'position', f'平移至画面{positions[target.zone_row][target.zone_col]}，保持前后距离', 'hold'
        aligned, message = self._adaptive_aligned(pairs, samples, target)
        if not aligned:
            tilt_motion = 'plan' if target.tilt != BoardTilt.FRONT else 'hold'
            if '待内参' in message:
                action = TILT_ACTIONS.get(target.tilt)
                move = action[0] if action else '将板面正对相机'
                return 'tilt', move + '，然后手动点击采集', tilt_motion
            action = TILT_ACTIONS.get(target.tilt)
            return 'tilt', action[0] if action else '将板面转正，保持前后距离', tilt_motion
        if not all(s.get('sharp') for s in samples):
            return 'quality', '检查对焦与光照，保持板面位置', 'hold'
        if not all(s.get('stable') for s in samples):
            return 'stability', '保持静止至少 0.6 秒', 'hold'
        if duplicate:
            return 'duplicate', '稍微改变板面朝向，避免重复构图', 'hold'
        if any(self.panel._sm.get_sync_pair(name) is None for name in self.panel._selected_pairs()):
            return 'pairing', '等待左右帧满足软件时间配对', 'hold'
        return 'ready', '保持静止，可以采集', 'hold'

    def refresh(self):
        options = [self.panel._combo_pair.itemText(i) for i in range(self.panel._combo_pair.count())]
        self.pair.blockSignals(True)
        if options != [self.pair.itemText(i) for i in range(self.pair.count())]:
            self.pair.clear()
            self.pair.addItems(options)
        self.pair.setCurrentText(self.panel._combo_pair.currentText())
        self.pair.blockSignals(False)
        self._refresh_sample_context()
        self.reason.setToolTip('')
        offline = self.panel.offline_selection
        self.guide_mode.setEnabled(not offline)
        key = self._request_guidance() if self.adaptive and not self.view.review_key else None
        using_adaptive = bool(self.adaptive and key is not None and self._analysis
                              and self._analysis_key == key)
        using_override = bool(self.adaptive and key is not None and self._distance_override)
        self._instruction_stage = ('review' if self.view.review_key else
                                   'offline' if offline else 'fixed')
        motion = 'hold'
        live = (self.adaptive and not offline and not self.view.review_key and key is not None
                and self.panel._active_step == 0 and self.panel._workflow_phase == 'capture')
        self.distance_tools.setVisible(live)
        busy = bool(self.panel._pending_save_tokens or self.panel._failed_save_jobs
                    or self.panel._save_workers or self.panel._save_queue
                    or self.panel._white_balance_guard.busy or self.panel._model_worker
                    or self.panel._calib_worker or self.panel._stereo_import_worker
                    or self.panel._aux_dataset_busy() or self._guidance_worker is not None)
        for button, scale in ((self.btn_near, 'large'), (self.btn_far, 'small')):
            available = live and using_adaptive and not busy and self._distance_available(scale)
            button.setEnabled(available)
            button.setToolTip(
                f'中央正面补采{DISTANCE_LABELS[scale]}；保持原图验收，完成后可返回推荐' if available else
                '仅单组实时自适应采集可用；请结束回看、等待任务完成，并确认分辨率能满足角点间距')
        self.btn_resume_guidance.setVisible(self._distance_override is not None)
        self._layout_distance_controls()
        target = self._target()
        if not offline and self.panel._active_step == 0:
            self.view.set_capture_sequence(self._adaptive_sequence if using_adaptive or using_override
                                           else self.panel._capture_sequence)
        self.panel._seq_widget.setVisible(not offline and not using_adaptive)
        self.schematic.setVisible(not offline)
        self.schematic_title.setText('离线数据检查' if offline else '相机与目标示意')
        loaded = len(self.panel.preview_pairs()) if offline else 0
        if self.view.review_key:
            self.instruction.setText(f"离线回看 · {self.view.review_key}" if offline else "正在回看已采图像")
            self.reason.setText("结束回看后恢复实时采图引导")
            if offline:
                self.reason.setText('原文件不修改 · 左右角色按当前设置')
        elif offline:
            self.instruction.setText(f'离线双目图像 · 已加载 {loaded} 对')
            model = '针孔' if self.panel._camera_model == CameraModel.PINHOLE else '鱼眼'
            self.reason.setText(f'{model}模型 · 姿态覆盖未验证')
        elif target:
            positions = (("左上", "上方", "右上"), ("左侧", "中央", "右侧"), ("左下", "下方", "右下"))
            action = TILT_ACTIONS.get(target.tilt)
            suffix = f"，{action[0]}" if action else "，板面正对镜头"
            self.instruction.setText(f"将标定板移至画面{positions[target.zone_row][target.zone_col]}{suffix}")
            self.reason.setText(f"固定计划建议 · {TILT_LABELS[target.tilt]} · 保持双路可见")
        else:
            self.instruction.setText("计划已浏览完，可继续补拍")
            self.reason.setText("浏览结束不代表标定质量达标")
        if not offline and not self.view.review_key and self.adaptive:
            if using_adaptive:
                self.instruction.setText(self.instruction.text() + f'；{DISTANCE_LABELS[target.scale]}')
                if self._analysis.recommendation.repairing_distance:
                    self.instruction.setText(self.instruction.text() + ' · 距离补采')
                self.reason.setText(
                    f'指定补采：中央正面 · {DISTANCE_LABELS[target.scale]}'
                    if self._distance_override else self._analysis.recommendation.reason)
            elif key is None:
                self.reason.setText('多组同时采集沿用固定计划；选择单组可用自适应引导')
            elif self._guidance_error:
                self.reason.setText('自适应分析不可用，已回退固定计划')
                self.reason.setToolTip(self._guidance_error)
            else:
                self.reason.setText('正在分析已采配对，暂不自动采集')
        pairs, samples = self._current_samples()
        if using_adaptive and not offline and not self.view.review_key:
            _, distance_message = self._distance_status(pairs, samples, target)
            self.reason.setToolTip(self.reason.text() + '\n' + distance_message)
        visible = bool(samples) and all(s["valid"] for s in samples)
        sharp = visible and all(s.get("sharp") for s in samples)
        stable = visible and all(s.get("stable") for s in samples)
        aligned = bool(visible and target) and all(
            np.linalg.norm(s["center"] - (target.norm_cx, target.norm_cy)) < .16 for s in samples
        )
        if using_adaptive and aligned and not self.view.review_key:
            self.instruction.setText(self.instruction.text().replace('移至', '保持在'))
        self.status.setText(
            "基础画面检查（不含姿态）\n\n"
            f"双路可见：{'已检测' if visible else '等待检测'}\n"
            f"清晰度：{'良好' if sharp else '待检查'}\n"
            f"运动稳定：{'稳定' if stable else '请保持静止至少 0.6 秒'}\n"
            f"目标位置：{'接近建议区域' if aligned else '待对齐'}"
        )
        self.status.setStyleSheet(f"color:{SUCCESS if visible and sharp and stable and aligned else TEXT_DIM};")
        self.status.setToolTip("稳定要求至少3次检测并持续0.6秒，且相对窗口首帧无明显累计位移。预览阈值不是硬件同步或精度证明；基础位置检查不验证倾角。")
        if not using_adaptive and not offline and not self.view.review_key and target and target.tilt != BoardTilt.FRONT:
            self.reason.setText(f"固定计划建议 · {TILT_LABELS[target.tilt]} · 倾角步骤请手动采集")
        self.pose_status.setText("姿态：待人工确认" if visible and not self.view.review_key else "姿态：待实时检测")
        self.pose_status.setToolTip("PnP 数值仅供观察；本轮不自动验收角度、方向或姿态多样性")
        if using_adaptive and not self.view.review_key:
            reached, message = self._adaptive_aligned(pairs, samples, target)
            # Keep the compact status readable; full numeric diagnostics stay in tooltips.
            short_message = message.split(' · 等效板面占比')[0].split('；')[0]
            if '非实测距离' in message and '距离待确认' not in message:
                short_message += '（占比近似，待内参）'
            self.pose_status.setText(short_message)
            self.pose_status.setStyleSheet(f'color:{SUCCESS if reached else WARNING};')
            self.pose_status.setToolTip(message + '\n经验验收范围，不代表精度合格；'
                                        '双路位置、距离可信度及倾角共同检查')
        matches = self._duplicate_matches(pairs, samples) if visible else []
        duplicate = bool(matches)
        self.advice.setText(
            '近似已采配对：' + '、'.join(f'{name} {source}' for name, source in matches)
            + '；建议明显改变倾角或占比，手动采集仍可保留'
            if duplicate else "姿态分布尚未自动评估；建议补拍不同位置、大小和倾角"
        )
        if using_adaptive and not duplicate:
            self.advice.setText(self._analysis.summary)
        if using_adaptive and not self.view.review_key and not offline:
            self._instruction_stage, action, motion = self._primary_action(pairs, samples, target, duplicate)
            self.instruction.setText('操作：' + action)
            evidence = ('距离与倾角可检测' if self._analysis.recommendation.pose_available
                        else '建立初始模型，请手动采图')
            complete = (self._distance_override is not None and self._distance_override
                        not in self._analysis.recommendation.missing_distances)
            subtitle = '本档数量已补足，可返回推荐' if complete else evidence
            tilts = {BoardTilt.FRONT: '正面', BoardTilt.TILT_LEFT: '左边近',
                     BoardTilt.TILT_RIGHT: '右边近', BoardTilt.TILT_UP: '上边近',
                     BoardTilt.TILT_DOWN: '下边近'}
            rec = self._analysis.recommendation
            phase = (f'首轮 {min(rec.sample_count + 1, rec.initial_total)}/{rec.initial_total}'
                     if rec.phase == 'initial' else '复核' if rec.phase == 'review' else
                     f'补采 {min(rec.repair_count + 1, MAX_DISTANCE_REPAIR_REQUESTS)}'
                     f'/{MAX_DISTANCE_REPAIR_REQUESTS}')
            # The diagram and visible goal describe this action, not a later tilt step.
            current_tilt = ('先调距离，暂不倾斜' if self._instruction_stage == 'distance' else
                            '先平移，暂不倾斜' if self._instruction_stage == 'position' else
                            tilts[target.tilt])
            goal = f'目标：{DISTANCE_LABELS[target.scale]} · {current_tilt}'
            goal = f'{"指定补采" if self._distance_override else phase} · {goal}'
            if not rec.feasible and not self._distance_override:
                goal = '当前构图受限 · 请检查分辨率和标定板'
            if self._instruction_stage == 'review':
                goal = '本轮引导结束 · 可进入标定评估'
                missing = '、'.join(DISTANCE_LABELS[s] for s in rec.missing_distances)
                subtitle = (f'待补：{missing}；可自主补拍' if missing else '距离数量已满足')
                subtitle += '；不代表精度达标'
            self.reason.setText(f'{goal}\n状态：{subtitle}')
            poses = [self._pose_measurement(camera.camera_id, sample)
                     for pair, offset in zip(pairs, range(0, len(samples), 2))
                     for camera, sample in zip((pair.left, pair.right), samples[offset:offset + 2])]
            if (not complete and self._instruction_stage != 'review'
                    and poses and all(reliable_distance_pose(p) for p in poses)):
                values = ' / '.join(f"{p['depth']:.2f}" for p in poses)
                low, high = SCALE_AREA_RANGES[target.scale]
                lower = max(math.sqrt(p['distance_area'] * p['depth'] ** 2 / high) for p in poses)
                upper = min(math.sqrt(p['distance_area'] * p['depth'] ** 2 / low) for p in poses)
                distance_goal = (f'目标 {lower:.2f}–{upper:.2f} m' if lower < upper else
                                 '左右目标范围冲突，请检查配置')
                self.reason.setText(f'{goal}\n'
                                    f'板中心深度 {values} m · {distance_goal}')
            busy_state = {'saving': '保存失败，请重试' if self.panel._failed_save_jobs else '等待原图保存',
                          'preparation': '等待当前准备完成',
                          'analysis': '更新采图建议中'}.get(self._instruction_stage)
            if busy_state:
                self.reason.setText(f'{goal}\n状态：{busy_state}')
            if self._distance_override:
                label = DISTANCE_LABELS[self._distance_override]
                self.advice.setText(f'指定补采：{label} · '
                                    + ('本档数量已补足，可返回推荐' if complete else '继续补拍，或返回推荐')
                                    + '；仍需原图质量及标定后评估。\n' + self._analysis.summary)
        elif live and not using_adaptive:
            self._instruction_stage = 'analysis'
            self.instruction.setText('正在更新采图建议，请保持当前位置' if not self._guidance_error else
                                     '自适应分析不可用，请检查固定计划')
        adaptive_action = using_adaptive and not self.view.review_key and not offline
        self.schematic.set_target(
            replace(target, tilt=BoardTilt.FRONT)
            if adaptive_action and self._instruction_stage in ('distance', 'position') else target,
            active=adaptive_action or self._instruction_stage == 'fixed',
            instruction=self.instruction.text() if adaptive_action or live else None,
            motion=motion if adaptive_action else 'plan' if self._instruction_stage == 'fixed' else 'hold',
        )
        self.instruction.setToolTip(self.instruction.text())
        self._refresh_receipt(pairs, offline)
        self.advice.setStyleSheet(f"color:{WARNING if duplicate else TEXT_DIM}; font-size:12px;")
        self.advice.setVisible(not self._compact)
        left = pairs[0].left.camera_id if pairs else None
        self.pose.setText("当前姿态（相对左相机）\n\n" + self._pose_text(left, samples[0] if samples else None))
        self.pose.setToolTip("深度为整板角点中心的相机 Z 值；左右倾正值为右边近，上下倾正值为下边近；不包含画面内旋转")
        if offline:
            self.pose.setText('离线观测\n\n不进行实时运动检查')
            self.pose_status.setText('姿态覆盖：未验证')
            self.status.setText(f'已加载 {loaded} 对\n\n' + self.panel._lbl_capture_count.text())
            self.status.setStyleSheet(f'color:{TEXT_DIM};')
            self.advice.hide()
        cameras = (("左目", pairs[0].left), ("右目", pairs[0].right)) if pairs else []
        cals = [(label, self.panel._intr_cals.get(camera.camera_id)) for label, camera in cameras]
        coverage_key = tuple((label, id(cal), cal.data_revision if cal else 0)
                             for label, cal in cals)
        if coverage_key != self._coverage_key:
            self._coverage_key = coverage_key
            self._coverage_summary = [
                f"{label} {cal.compute_coverage() if cal else 0.:.0%}/"
                f"{cal.compute_image_coverage() if cal else 0.:.0%}"
                for label, cal in cals
            ]
        self.compact_summary.setText(
            f"可见 {'✓' if visible else '待检测'} · 清晰 {'✓' if sharp else '待检查'} · "
            f"稳定 {'✓' if stable else '待检查'}\n{self.pose_status.text()}\n"
            "数据集累计：板角点 ID / 视场网格\n" + '  '.join(self._coverage_summary)
        )
        self.compact_summary.setToolTip(self.pose.text() + '\n' + self.status.text()
                                       + '\n' + self.pose_status.toolTip())
        if offline:
            counts = self.view.displayed_corner_counts()
            origin = self.view.review_detection_origin if self.view.review_key else '缩放检测'
            if origin == '预览重检':
                origin += '，缩放检测'
            current = '  '.join(
                f'{label} {counts.get(camera.camera_id) if counts.get(camera.camera_id) is not None else "未显示"}'
                f'/{self.board.num_corners}' for label, camera in cameras
            )
            self.compact_summary.setText(
                f'离线观测 {loaded} 对\n{self.panel._lbl_capture_count.text()}\n'
                f'姿态覆盖未验证\n当前预览角点（{origin}）\n' + current +
                '\n数据集累计：板角点 ID / 视场网格\n' + '  '.join(self._coverage_summary)
            )
            self.status.setText(self.compact_summary.text())
        self.coverage_button.setToolTip(
            self.panel._lbl_capture_count.text() + '\n' + '  '.join(self._coverage_summary) +
            "\n数据集累计：所有已采图出现过的板角点 ID 比例 / 4×4 视场网格占比。"
            "不是当前帧完整度或姿态覆盖率；当前预览角点可能因缩放检测少于原图检测。",
        )
        self.coverage_button.setText(
            "数据分析" if self._compact else
            "累计板角点 ID / 视场网格  ·  " + "  |  ".join(
                [self.panel._lbl_capture_count.text(), *self._coverage_summary],
            )
        )

    def _refresh_receipt(self, pairs, offline):
        self.receipt.setVisible(not offline)
        if self.view.review_key:
            self.receipt.setText(f'正在回看：{self.view.review_key}\n结束回看后继续实时采图')
            self.receipt.setToolTip(self.receipt.text())
            return
        reviews = self.panel._capture_reviews.get(pairs[0].name, []) if pairs else []
        if not reviews:
            self.receipt.setText('本次原图：尚未采集\n预览角点与采图原图分别检测')
            self.receipt.setToolTip(self.receipt.text())
            return
        item = reviews[-1]
        token = item.save_token
        state = ('保存失败' if token in self.panel._failed_save_jobs else
                 '保存中' if token in self.panel._pending_save_tokens else
                 '已保存' if token in self.panel._saved_capture_tokens else item.save_state)
        left = getattr(item.left_detection, 'num_corners', 0)
        right = getattr(item.right_detection, 'num_corners', 0)
        accepted = item.quality_state == '可采纳'
        quality = ('完整角点可采纳' if min(left, right) == self.board.num_corners else
                   '部分角点可采纳') if accepted else item.quality_state or '验收待确认'
        self.receipt.setText(f'本次原图 {item.key.rsplit(" ", 1)[-1]} · {state}\n'
                             f'左{left} / 右{right} · {quality}')
        next_step = ('未计入有效样本，请查看验收详情后重拍。' if not accepted else
                     '保存失败，请重试保存原图，不要重复采集。' if state == '保存失败' else
                     '下一步：按上方操作调整后继续采图；精度待标定评估。' if state == '已保存' else
                     '请等待原图保存确认，再按上方操作继续。')
        self.receipt.setToolTip(item.result_text() + '\n' + next_step)

    def set_compact(self, compact):
        self._compact = compact
        self.schematic.set_compact(compact)
        self.pose.setMinimumWidth(145 if compact else 165)
        self.status.setMinimumWidth(185 if compact else 205)
        self.pose.setVisible(not compact)
        self.pose_status.setVisible(not compact)
        self.status.setVisible(not compact)
        self.compact_summary.setVisible(compact)
        self.setMaximumWidth(440 if compact else 16777215)
        self.refresh()
