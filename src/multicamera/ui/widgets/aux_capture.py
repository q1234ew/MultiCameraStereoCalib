"""Bounded RGB-left/AUX guidance and persistent, independently counted evidence."""

from __future__ import annotations

import time
from copy import deepcopy
from dataclasses import replace

import numpy as np
from PySide6.QtCore import QThreadPool, Slot, Qt
from PySide6.QtWidgets import QGridLayout, QLabel, QVBoxLayout, QWidget

from ...calibration.aux_guidance import (
    AREA_RANGES, DISTANCE_NAMES, MAX_REPAIRS, guidance_action, initial_plan,
)
from ...calibration.aux_capture_quality import quality_reason_text
from ..threads.aux_capture_worker import (
    AuxCaptureWorker, AuxGuidanceProgressWorker, AuxGuidanceWorker,
)
from ..theme import ACCENT, BORDER, SUCCESS, TEXT, TEXT_DIM, WARNING
from .capture_guide import BoardTilt, CaptureTarget, build_multimodal_sequence
from .stable_text import StableTextLabel


TILTS = {'front': BoardTilt.FRONT, 'left': BoardTilt.TILT_LEFT, 'right': BoardTilt.TILT_RIGHT,
         'up': BoardTilt.TILT_UP, 'down': BoardTilt.TILT_DOWN}


class AuxCaptureGuide(QWidget):
    MAX_AGE = 1.5
    RECEIPT_HOLD_SECONDS = 5.
    INSTRUCTION_MIN_SECONDS = 2.
    INSTRUCTION_CONFIRM_SECONDS = .8

    def __init__(self, panel):
        super().__init__(panel)
        self.panel = panel
        self.analysis = None
        self._key = None
        self._requested = None
        self._worker = None
        self._refined_key = None
        self._refined_intrinsics = (None, None)
        self._progress_worker = None
        self._progress_requested = None
        self._progress_pool = QThreadPool(self)
        self._progress_pool.setMaxThreadCount(1)
        self._preview_worker = None
        self._preview_started = 0.
        self._preview_result_at = 0.
        self._preview_invalidated_at = 0.
        self._revision = 0
        self._samples = {}
        self._sample_timestamps = {}
        self._view = None
        self._clock = time.monotonic
        self._manual = False
        self._scope = None
        self._analysis_error = ''
        self._receipt_record = None
        self._receipt_override = None
        self._receipt_hold_until = 0.
        self._receipt_scope = None
        self._instruction = None
        self._instruction_candidate = None
        self._last_progress_log = None
        self._recommended_action = ''
        self.last_qualities = []
        self._preview_detections = []
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        self.heading = QLabel('当前采集任务')
        self.heading.setStyleSheet(f'color:{TEXT}; font-weight:bold;')
        self.heading.setContentsMargins(4, 0, 0, 0)
        self.phase = StableTextLabel('基础采集 · 等待图像', lines=2)
        self._set_phase_color(ACCENT)
        self.action = StableTextLabel('让同一块标定板完整进入RGB左目和辅助相机画面', lines=2)
        self.action.setStyleSheet(
            f'color:{TEXT}; font-size:16px; font-weight:bold; border-left:3px solid {ACCENT};')
        self.action.setContentsMargins(8, 0, 0, 0)
        self.action.setAccessibleName('当前采集动作')
        self.next_step = StableTextLabel('下一步：连接两路实时图像，或导入配对原图', lines=2)
        self.next_step.setStyleSheet(f'color:{TEXT_DIM}; font-size:12px;')
        self.next_step.setAccessibleName('下一步采集安排')
        self.target_info = StableTextLabel('目标：等待采图建议', lines=2)
        self.target_info.setStyleSheet(f'color:{TEXT_DIM}; font-size:12px;')
        self.target_info.setAccessibleName('当前构图目标与标定板')
        self.counts = StableTextLabel('原图已保存 0 组 · 源目录 0 组\n验收通过 0 组 · 引导采用 待更新', lines=2)
        self.counts.setStyleSheet(f'color:{TEXT_DIM}; font-size:12px;')
        self.receipt = StableTextLabel('最近原图：尚未采集', lines=2)
        self.next_title = QLabel('下一步安排')
        self.next_title.setStyleSheet(f'color:{TEXT}; font-size:12px; font-weight:bold;')
        self.next_title.setContentsMargins(4, 0, 0, 0)
        for label in (self.heading, self.phase, self.action, self.target_info,
                      self.next_title, self.next_step):
            label.setAlignment(Qt.AlignLeft | Qt.AlignTop)
            layout.addWidget(label)
        self.details = QWidget(panel)
        details = QVBoxLayout(self.details)
        details.setContentsMargins(0, 0, 0, 0)
        details.setSpacing(6)
        self.checks = QWidget(self.details)
        checks = QGridLayout(self.checks)
        checks.setContentsMargins(0, 0, 0, 0)
        checks.setHorizontalSpacing(12)
        checks.setVerticalSpacing(2)
        checks.setColumnStretch(1, 1)
        self.checks_heading = QLabel('实时检查')
        self.checks_heading.setStyleSheet(f'color:{TEXT}; font-weight:bold;')
        checks.addWidget(self.checks_heading, 0, 0, 1, 2)
        self.check_values = {}
        for row, (key, title) in enumerate((('board', '双路板点'), ('quality', '原图预检'),
                                            ('stability', '稳定性'), ('pairing', '视频配对')), 1):
            label = QLabel(title)
            label.setStyleSheet(f'color:{TEXT_DIM}; font-size:12px;')
            value = StableTextLabel('等待检测', lines=1)
            value.setStyleSheet(f'color:{TEXT_DIM}; font-size:12px;')
            checks.addWidget(label, row, 0)
            checks.addWidget(value, row, 1)
            self.check_values[key] = value
        details.addWidget(self.checks)
        self.data_heading = QLabel('数据概况')
        self.data_heading.setStyleSheet(
            f'color:{TEXT}; font-weight:bold; border-top:1px solid {BORDER}; padding-top:6px;')
        details.addWidget(self.data_heading)
        details.addWidget(self.counts)
        details.addWidget(self.receipt)

    def _set_phase_color(self, color):
        self.phase.setStyleSheet(f'color:{color}; font-size:12px; font-weight:normal;')

    @property
    def auxiliary_label(self):
        return '声像仪RGB' if self.panel._aux_modality() == 'acoustic_rgb' else '红外'

    @property
    def camera_labels(self):
        return ('RGB左目', self.auxiliary_label)

    def bind_preview(self, view):
        self._view = view
        # Temporal evidence comes only from the paired original-frame sampler.
        # Tile detections can refer to an older frame and optional tiles can be closed.
        view.review_changed.connect(self._on_review_changed)

    def invalidate_live(self):
        self._samples.clear()
        self._sample_timestamps.clear()
        self.last_qualities = []
        self._preview_detections = []
        self._preview_invalidated_at = self._clock()

    def capture_problem(self):
        if (len(self.last_qualities) != 2 or len(self._preview_detections) != 2
                or not 0 <= self._clock() - self._preview_result_at <= self.MAX_AGE):
            return '等待双路原图预检，请让整板同时可见'
        reasons = [f'{name}：{quality_reason_text(reason)}'
                   for name, quality in zip(self.camera_labels, self.last_qualities)
                   for reason in quality.reasons]
        reasons += self.stability_reasons(self._preview_detections)
        return '；'.join(reasons)

    @property
    def reviewing(self):
        return bool(self._view is not None and self._view.review_key)

    @Slot(str, bool)
    def _on_review_changed(self, key, active):
        if self.panel._active_step != 2:
            return
        self._samples.clear()
        self._sample_timestamps.clear()
        self.last_qualities = []
        self._preview_detections = []
        self._preview_invalidated_at = self._clock()
        self.panel._refresh_aux_readiness()
        if not active and self.analysis is not None and self._key == self.context_key():
            self.render()

    def context_key(self):
        panel = self.panel
        intrinsics = [panel._intrinsics.get(panel._rgb_left_camera_id()), panel._aux_intrinsics]
        return (id(panel._session), tuple(sorted(panel._aux_board_config().to_dict().items())),
                panel._camera_model, panel._rgb_left_camera_id(), panel._aux_camera_id(),
                tuple(panel._sm.configured_camera_urls().items()), self._revision,
                tuple(None if intr is None else (intr.model, intr.image_size,
                      intr.camera_matrix.tobytes(), intr.dist_coeffs.tobytes()) for intr in intrinsics))

    def reset(self):
        self._revision += 1
        self._key = self._requested = None
        self._refined_key = self._progress_requested = None
        self._refined_intrinsics = (None, None)
        self.analysis = None
        self._samples.clear()
        self._sample_timestamps.clear()
        self.last_qualities = []
        self._preview_detections = []
        self._manual = False
        self._analysis_error = ''
        self._receipt_record = None
        self._receipt_override = None
        self._receipt_hold_until = 0.
        self._instruction = self._instruction_candidate = None
        self._last_progress_log = None
        self.receipt.setText('最近原图：尚未采集')
        self.refresh()

    def data_changed(self):
        self._revision += 1
        self._key = None
        self._manual = False
        self.refresh()

    def manual_navigation(self):
        self._manual = True
        self._show_browsing_feedback()
        self._tooltips()

    def _show_browsing_feedback(self):
        if self.reviewing:
            self._show_review_feedback()
            return
        seq = self.panel._aux_capture_sequence
        self.phase.setText(f'浏览参考动作 {min(seq.current_index + 1, seq.total)}/{seq.total}\n非采集进度')
        self.action.setText('浏览不增加采集记录；手动采图仍按原图验收，采图后恢复推荐')
        self._set_phase_color(ACCENT)
        self._reference_copy()

    def _show_review_feedback(self):
        self.phase.setText(f'正在回看原图\n{self._view.review_key}')
        self._set_phase_color(WARNING)
        self.action.setText('实时指引已暂停；请先结束回看，再采集实时图像')

    @Slot(str)
    def expire(self, camera_id):
        self._samples.pop(camera_id, None)
        self._sample_timestamps.pop(camera_id, None)
        self._preview_invalidated_at = self._clock()
        self.last_qualities = []
        self._preview_detections = []
        if self.analysis is not None and not self._manual:
            self.render()

    @Slot(str, object, object)
    def observe(self, camera_id, detection, image, *, timestamp=None, observed_at=None):
        panel = self.panel
        if (panel._active_step != 2 or panel._workflow_phase != 'capture'
                or (self._view is not None and self._view._review_images is not None)
                or camera_id not in (panel._rgb_left_camera_id(), panel._aux_camera_id())):
            return
        if not detection.valid or detection.num_corners != np.prod(
                (panel._aux_board_config().cols, panel._aux_board_config().rows)):
            self.expire(camera_id)
            return
        now = self._clock() if observed_at is None else observed_at
        entry = panel._sm.get_latest_frame(camera_id) if timestamp is None else None
        stamp = (entry[1] if entry is not None else now) if timestamp is None else timestamp
        if self._sample_timestamps.get(camera_id) == stamp:
            return
        if (camera_id in self._sample_timestamps
                and stamp < self._sample_timestamps[camera_id]):
            return
        self._sample_timestamps[camera_id] = stamp
        points = np.array(detection.charuco_corners, copy=True).reshape(-1, 2)
        size = detection.image_size
        history = self._samples.setdefault(camera_id, [])
        if history and (history[-1][2] != size or now - history[-1][0] > self.MAX_AGE):
            history.clear()
        if history and not self._close(points, history[0][1], size):
            history.clear()
        history.append((now, points, size))
        while len(history) > 6:
            history.pop(0)

    def _request_preview(self):
        panel = self.panel
        if self.last_qualities and self._clock() - self._preview_result_at > self.MAX_AGE:
            self.last_qualities = []
            self._preview_detections = []
            if not self._manual:
                self.render()
        if (panel._active_step != 2 or panel._workflow_phase != 'capture'
                or self._preview_worker or panel._aux_operation_busy(include_live_wait=False)
                or self._clock() - self._preview_started < 1.
                or self._view is not None and self._view._review_images is not None):
            return
        frames, _ = panel._aux_live_frames()
        if frames is None:
            self._samples.clear()
            self._sample_timestamps.clear()
            self.last_qualities = []
            self._preview_detections = []
            if not self._manual:
                self.render()
            return
        entries = [panel._sm.get_latest_frame(cid)
                   for cid in (panel._rgb_left_camera_id(), panel._aux_camera_id())]
        if any(entry is None or entry[0] is not frame
               for entry, frame in zip(entries, frames[:2])):
            return
        self._preview_started = self._clock()
        worker = AuxCaptureWorker((self.context_key(), self._preview_started,
                                   tuple(entry[1] for entry in entries)),
            tuple(frame.copy() for frame in frames[:2]), deepcopy(panel._aux_board_config()),
            self.intrinsics(), panel._aux_modality(), enhanced=True)
        self._preview_worker = worker
        worker.signals.completed.connect(self._preview_ready)
        QThreadPool.globalInstance().start(worker)

    @Slot(object, object, str)
    def _preview_ready(self, key, result, error):
        self._preview_worker = None
        if (key[0] != self.context_key()
                or self.panel._active_step != 2
                or self._view is not None and self._view._review_images is not None):
            return
        if (error or self._clock() - key[1] > self.MAX_AGE
                or key[1] < self._preview_invalidated_at):
            self.last_qualities = []
            self._preview_detections = []
            if not self._manual:
                self.render()
            return
        if len(key) > 2 and any(not 0 <= time.time() - stamp <= self.MAX_AGE
                                for stamp in key[2]):
            self.invalidate_live()
            return
        detections, self.last_qualities = result
        self._preview_detections = detections
        self._preview_result_at = key[1]
        # This independent sampler also works when an optional preview tile is closed.
        from types import SimpleNamespace
        for i, (camera_id, det) in enumerate(zip(
                (self.panel._rgb_left_camera_id(), self.panel._aux_camera_id()), detections)):
            self.observe(camera_id, SimpleNamespace(valid=det.valid, num_corners=det.num_points,
                         charuco_corners=det.image_points, image_size=det.image_size), None,
                         timestamp=key[2][i] if len(key) > 2 else key[1], observed_at=key[1])
        if not self._manual:
            self.render()

    @staticmethod
    def _close(a, b, size):
        if a.shape != b.shape:
            return False
        # Accept a detector's 180-degree enumeration change only for motion testing;
        # the original order is retained for all calibration and correspondence checks.
        rms = min(np.sqrt(np.mean(np.sum((a - points) ** 2, axis=1))) for points in (b, b[::-1]))
        return bool(np.isfinite(rms) and rms <= max(.6, .002 * max(size)))

    def stability_reasons(self, detections, snapshot=None):
        reasons = []
        now, histories = snapshot if snapshot is not None else (self._clock(), self._samples)
        if len(detections) != 2:
            detections = (None, None)
        for name, camera_id, det in zip(self.camera_labels,
                (self.panel._rgb_left_camera_id(), self.panel._aux_camera_id()), detections):
            history = histories.get(camera_id, [])
            stable = (det is not None and det.valid and len(history) >= 3
                      and 0 <= now - history[-1][0] <= self.MAX_AGE
                      and history[-1][0] - history[0][0] >= .6
                      and det.image_size == history[-1][2]
                      and self._close(det.image_points.reshape(-1, 2), history[0][1], det.image_size))
            if not stable:
                reasons.append(f'{name}稳定性未通过，请保持整板可见并等待连续3次稳定检测')
        return reasons

    def intrinsics(self):
        if self.analysis is not None and self._key == self.context_key():
            return deepcopy(self.analysis.intrinsics)
        return deepcopy((self.panel._intrinsics.get(self.panel._rgb_left_camera_id()),
                         self.panel._aux_intrinsics))

    def refresh(self):
        if not hasattr(self.panel, '_aux_seq_widget'):
            return
        key = self.context_key()
        if self._scope != key[:6]:
            self._scope = key[:6]
            self._refined_intrinsics = (None, None)
            self._samples.clear()
            self._sample_timestamps.clear()
            self.last_qualities = []
            self._preview_detections = []
        intrinsics = (self.panel._intrinsics.get(self.panel._rgb_left_camera_id()),
                      self.panel._aux_intrinsics)
        if (self._key != key and self._progress_requested != key
                and self._progress_worker is None):
            progress = AuxGuidanceProgressWorker(key, deepcopy(self.panel._aux_board_config()),
                deepcopy(self.panel._aux_pairs), deepcopy(tuple(
                    supplied if supplied is not None else previous
                    for supplied, previous in zip(intrinsics, self._refined_intrinsics))))
            self._progress_worker, self._progress_requested = progress, key
            progress.signals.completed.connect(self._progressed)
            self._progress_pool.start(progress)
        if self._refined_key != key and self._worker is None:
            worker = AuxGuidanceWorker(key, deepcopy(self.panel._aux_board_config()),
                self.panel._camera_model, deepcopy(self.panel._aux_pairs),
                deepcopy(intrinsics))
            self._worker, self._requested = worker, key
            self._analysis_error = ''
            worker.signals.completed.connect(self._analyzed)
            QThreadPool.globalInstance().start(worker)
        self._update_counts()
        self._request_preview()
        if self.reviewing:
            self._show_review_feedback()
        elif self._manual:
            self._show_browsing_feedback()
        elif self._analysis_error and self._key == key:
            self._set_phase_color(WARNING)
            self.phase.setText('引导分析失败 · 原图仍保留')
            self.action.setText('请查看日志；重新采集或导入后会重新分析，原图仍需验收')
        elif self.analysis is None or self._key != key:
            self._set_phase_color(ACCENT)
            self.phase.setText('正在分析采图数据；当前原图记录保留')
            self.action.setText('正在更新采图建议；本次原图仍需验收')
        self._tooltips()

    @Slot(object, object, str)
    def _progressed(self, key, analysis, error):
        self._progress_worker = None
        if key != self.context_key():
            self.refresh()
            return
        if self._refined_key == key:
            return
        if error:
            self.panel._log('融合轻量进度更新失败，等待完整分析：' + error)
            return
        self.analysis, self._key = analysis, key
        self._analysis_error = ''
        self.render()
        self._announce_progress(analysis)

    @Slot(object, object, str)
    def _analyzed(self, key, analysis, error):
        self._worker = None
        if key != self.context_key():
            self.refresh()
            return
        self._key = key
        self._refined_key = key
        if error:
            self.analysis = None
            self._analysis_error = error
            self.refresh()
            self.panel._log('融合采图分析失败：' + error)
            return
        self.analysis = analysis
        self._refined_intrinsics = deepcopy(analysis.intrinsics)
        self._analysis_error = ''
        if analysis.phase == 'complete':
            self.panel._log('融合本轮引导结束；满足标定条件后可计算，仍可补拍，不代表精度合格')
        self.render()
        self._announce_progress(analysis)

    def _announce_progress(self, analysis):
        record = self._receipt_record
        if record is None or self.panel._aux_offline_records:
            return
        decision = analysis.sample_decisions.get(record['key'])
        if decision not in ('included', 'duplicate'):
            return
        identity = (self.context_key(), record['key'], decision, analysis.unique_count)
        if identity == self._last_progress_log:
            return
        self._last_progress_log = identity
        adopted = f'引导采用 {analysis.unique_count}/{analysis.initial_total} 组'
        if analysis.phase == 'repair':
            next_step = f'补采建议 {analysis.repair_count+1}/{MAX_REPAIRS}'
        elif analysis.target is not None:
            next_step = f'建议参考 {min(analysis.unique_count+1, analysis.initial_total)}/{analysis.initial_total}'
        else:
            next_step = '正在评估距离与倾角分布' if analysis.provisional else '请查看本轮引导结论'
        reason = '本组已计入' if decision == 'included' else '重复构图，参考不推进'
        self.panel._log(f'{record["key"]}: {reason}；{adopted}；{next_step}')

    def render(self):
        if self.panel._aux_offline_records:
            self._set_phase_color(ACCENT)
            self.phase.setText(f'离线候选复核 · 已确认 {len(self.panel._aux_pairs)} 组')
            self.action.setText('逐对确认原图质量与物理角点对应；历史稳定性和同步不由离线复核证明')
            self.panel._aux_seq_widget.hide()
            self._update_counts()
            self._tooltips()
            return
        if self.reviewing or self._manual:
            self._show_browsing_feedback()
            self._update_counts()
            self._tooltips()
            return
        analysis = self.analysis
        if analysis is None:
            return
        seq = self.panel._aux_capture_sequence
        seq.targets = build_multimodal_sequence()
        if analysis.phase == 'refining':
            seq._index = seq.total
            self._set_phase_color(ACCENT)
            self.phase.setText(f'基础图像已记录 · 引导采用 {analysis.unique_count} 组\n正在评估距离与倾角分布')
            self.action.setText('基础进度已更新，正在检查分布；尚未结束引导或决定补采')
            self.panel._aux_seq_widget.set_completion_message('基础图像已记录',
                '正在评估距离与倾角分布，尚未结束引导')
        elif analysis.target is None:
            seq._index = seq.total
            self.phase.setText('当前构图不可用 · 请调整板型或分辨率' if analysis.phase == 'blocked'
                               else '本轮引导结束 · 可继续补拍')
            missing = '、'.join(DISTANCE_NAMES[s] for s in analysis.missing)
            self.action.setText('当前板型与分辨率无法安全放入建议区域，请调整配置或手动选择构图'
                if analysis.phase == 'blocked' else
                ('仍待验证距离：' + missing + '；可继续补拍\n' if missing else '')
                + (self.panel._aux_calibration_problem() or '已满足计算条件，可前往标定评估；不代表精度合格'))
            self._set_phase_color(WARNING if analysis.phase == 'blocked' else SUCCESS)
            self.panel._aux_seq_widget.set_completion_message(
                '当前构图不可用' if analysis.phase == 'blocked' else '本轮引导结束',
                '请检查板型和分辨率' if analysis.phase == 'blocked' else '满足标定条件后可计算，仍可补拍')
        else:
            self.panel._aux_seq_widget.set_completion_message('', '')
            target = analysis.target
            capture = CaptureTarget(target.row, target.col, TILTS[target.tilt], scale=target.scale,
                                    target_area=target.target_area)
            if analysis.phase == 'repair':
                seq.targets.append(capture)
                seq._index = len(initial_plan())
                stage = f'补采建议 {analysis.repair_count + 1}/{MAX_REPAIRS}'
            else:
                seq._index = analysis.unique_count
                seq.targets[seq._index] = capture
                stage = f'基础采集 · 引导采用 {analysis.unique_count}/{analysis.initial_total}组'
            self._set_phase_color(ACCENT)
            self.phase.setText(stage + ' · ' + DISTANCE_NAMES[target.scale])
            ir_pose = self.last_qualities[1].pose if len(self.last_qualities) == 2 else None
            if ir_pose is not None and ir_pose.get('distance_reliable', False):
                low, high = AREA_RANGES[target.scale]
                factor = ir_pose['distance_area'] * ir_pose['depth'] ** 2
                self.phase.setText(self.phase.text() + f"\n{self.auxiliary_label}深度 {ir_pose['depth']:.2f} m"
                    f' · 目标 {np.sqrt(factor / high):.2f}–{np.sqrt(factor / low):.2f} m')
            else:
                self.phase.setText(self.phase.text() + f'\n{self.auxiliary_label}距离按画面占比估计，非实测')
            instruction_stage, action = guidance_action(
                self.panel._aux_board_config(), target, self.last_qualities,
                auxiliary_label=self.auxiliary_label)
            if instruction_stage == 'ready' and (
                    not 0 <= self._clock() - self._preview_result_at <= self.MAX_AGE
                    or self.stability_reasons(self._preview_detections)):
                instruction_stage, action = 'stability', '保持整板可见，等待双路连续3次稳定检测'
            if instruction_stage == 'tilt_unverified':
                action += '；可手动采图，仍需原图验收'
            instruction_stage, action = self._stable_instruction(instruction_stage, action, target)
            if instruction_stage in ('distance', 'position', 'quality'):
                seq.targets[seq._index] = replace(capture, tilt=BoardTilt.FRONT)
            self.action.setText(action)
        self._recommended_action = self.action.text()
        self.panel._aux_seq_widget.refresh()
        self.panel._aux_seq_widget._large_guide.show_distance = True
        self._reference_copy()
        if self.panel._active_step == 2:
            self.panel.sequence_updated.emit(seq)
        self._update_counts()
        self._tooltips()

    def _stable_instruction(self, stage, text, target):
        """Damp directional chatter only; never keep a stale permission to capture."""
        now = self._clock()
        scope = (self.context_key()[:6], target.row, target.col, target.tilt, target.scale)
        directional = {'distance', 'position', 'tilt', 'tilt_unverified'}
        previous = self._instruction
        if (previous is None or previous[0] != scope or stage not in directional
                or previous[1] not in directional):
            self._instruction = (scope, stage, text, now)
            self._instruction_candidate = None
            return stage, text
        if (stage, text) == previous[1:3]:
            self._instruction_candidate = None
            return stage, text
        candidate = self._instruction_candidate
        if candidate is None or (stage, text) != candidate[:2]:
            self._instruction_candidate = (stage, text, now)
        elif (now-candidate[2] >= self.INSTRUCTION_CONFIRM_SECONDS
                and now-previous[3] >= self.INSTRUCTION_MIN_SECONDS):
            self._instruction = (scope, stage, text, now)
            self._instruction_candidate = None
            return stage, text
        return previous[1:3]

    def _reference_copy(self):
        widget = self.panel._aux_seq_widget
        widget._title_lbl.setText(f'参考动作 · 位置以{self.auxiliary_label}画面为准')
        widget._title_lbl.setToolTip(
            '图中序号是参考动作，不是已采数量；浏览不增加采集记录。'
            '上下/左右倾以RGB左目为参考，手动采图仍按本次原图验收。')

    def record_receipt(self, key, state, reasons, detections, *, saved=True):
        self._receipt_override = None
        self._receipt_record = dict(key=key, state=state, reasons=list(reasons), saved=saved,
                                    points=tuple(det.num_points for det in detections))
        self._receipt_hold_until = self._clock() + self.RECEIPT_HOLD_SECONDS
        self._receipt_scope = self.context_key()[:6]
        self._render_receipt()
        self.receipt.setStyleSheet(f'color:{SUCCESS if state == "可采纳" else WARNING};')
        self._tooltips()

    def show_receipt_message(self, text):
        self._receipt_override = text
        self.receipt.setStyleSheet(f'color:{WARNING};')
        self._render_receipt()
        self._tooltips()

    def _render_receipt(self):
        if self._receipt_override:
            self.receipt.setText(self._receipt_override)
            self.receipt.setToolTip(self._receipt_override)
            return
        record = self._receipt_record
        if record is None:
            if self.panel._aux_imported_count:
                self.receipt.setText(f'离线导入 {self.panel._aux_imported_count} 组 · 原图保留在源目录\n'
                                     + ('逐对复核后纳入；历史实时验收不变' if self.panel._aux_offline_records else
                                        '验收通过与引导采用见统计，导入不等于姿态已验证'))
            self.receipt.setToolTip(self.receipt.text())
            return
        accepted = record['state'] == '可采纳'
        decision = (self.analysis.sample_decisions.get(record['key'])
                    if self.analysis and self._key == self.context_key() else None)
        guidance = ('引导未采用' if not accepted else '引导已采用' if decision == 'included' else
                    '引导未采用（重复构图）' if decision == 'duplicate' else
                    '引导未采用' if decision else '引导分析失败' if self._analysis_error else '引导待更新')
        points = ' / '.join(f'{name} {n}' for name, n in zip(self.camera_labels, record['points']))
        storage = '原图已保存' if record['saved'] else '原图仅在内存'
        state = '验收通过' if accepted else '验收待复核'
        reasons = record['reasons'][:]
        if decision in ('invalid_detection', 'size_mismatch'):
            reasons.append('引导要求检测有效且图像尺寸一致；请检查本组原图')
        detail = f'{points} · {guidance}'
        if not accepted and reasons:
            reason = reasons[0].split('，', 1)[0]
            if any('验收记录写入失败' in value for value in reasons):
                reason = '验收记录写入失败'
            elif '稳定性未通过' in reason:
                reason = '等待连续3次稳定检测'
            detail = f'{guidance} · {reason}'
        self.receipt.setText(f'{record["key"]} · {storage} · {state}\n{detail}')
        self.receipt.setToolTip(self.receipt.text() + '\n' + points + '\n' + '；'.join(reasons)
                               + '\n验收通过不代表物理角点对应已确认或标定精度合格。')

    def _update_counts(self):
        records = getattr(self.panel, '_aux_quality_records', [])
        detected = sum(r.get('detected', False) for r in records)
        unique = self.analysis.unique_count if self.analysis and self._key == self.context_key() else '待更新'
        pose_count = self.analysis.pose_count if self.analysis and self._key == self.context_key() else '待更新'
        if self.analysis and self.analysis.provisional:
            pose_count = '待更新'
        if self._analysis_error and self._key == self.context_key():
            unique = pose_count = '分析失败'
        unique_text = f'{unique} 组' if isinstance(unique, int) else unique
        label = '离线确认' if self.panel._aux_offline_records else '验收通过'
        self.counts.setText(f'原图已保存 {self.panel._aux_saved_count} 组 · 源目录 {self.panel._aux_imported_count} 组\n'
                            f'{label} {len(self.panel._aux_pairs)} 组 · 引导采用 {unique_text}')
        self.counts.setToolTip(
            self.counts.text() + f'\n双路检测有效 {detected} 组 · 双路姿态可解 {pose_count} 组\n'
            '引导采用按检测有效、尺寸一致且非重复的配对计数，不是参考动作完成数。'
            '验收通过不代表物理角点对应已确认；姿态可解不等于推荐姿态已到位或精度合格。'
            '\n稳定要求至少3次检测且跨度至少0.6秒；通常需静止约1秒或更久。'
            + (self.analysis.summary if self.analysis and self._key == self.context_key() else ''))
        self.counts.setAccessibleName(self.counts.text())
        total = self.analysis.initial_total if self.analysis and self._key == self.context_key() else len(initial_plan())
        self.panel._aux_cap_progress.setMaximum(total)
        self.panel._aux_cap_progress.setValue(min(unique, total) if isinstance(unique, int) else 0)
        self.panel._aux_cap_progress.setFormat(f'引导采用 {unique_text} · 基础目标 {total} 组')
        self.panel._aux_cap_progress.setToolTip(self.counts.toolTip())
        self._render_receipt()

    def _tooltips(self):
        self.refresh_workbench()
        for label in (self.phase, self.action, self.next_step):
            label.setToolTip(label.text())
            label.setAccessibleDescription(label.text())

    def refresh_workbench(self):
        """Refresh presentation only; never start work or change capture eligibility."""
        panel = self.panel
        frames, pairing = panel._aux_live_frames()
        fresh = (frames is not None and len(self.last_qualities) == 2
                 and 0 <= self._clock() - self._preview_result_at <= self.MAX_AGE)
        next_text, override = self._workbench_next_step(frames, fresh)
        self.next_step.setText('下一步：' + next_text)
        if override:
            self.action.setText(override)
        elif (not self.reviewing and not self._manual and self._recommended_action
              and self.analysis is not None and self._key == self.context_key()):
            self.action.setText(self._recommended_action)
        self._update_target_info()
        paused = self.reviewing or panel._aux_operation_busy(include_live_wait=False)
        statuses = {
            'board': ('回看中，实时检查暂停' if self.reviewing else '等待检测', TEXT_DIM),
            'quality': ('回看中，实时检查暂停' if self.reviewing else '等待检测', TEXT_DIM),
            'stability': ('回看中，实时检查暂停' if self.reviewing else '等待连续3次检测', TEXT_DIM),
            'pairing': ('回看中，非实时原图' if self.reviewing else
                        '处理中' if paused else '软件已配对' if frames is not None else '未配对',
                        TEXT_DIM if paused or frames is not None else WARNING),
        }
        reasons = []
        if fresh and not paused:
            detections = self._preview_detections
            board = panel._aux_board_config()
            complete = (len(detections) == 2 and all(
                det.valid and det.num_points == board.cols * board.rows for det in detections))
            statuses['board'] = ('双路完整可见' if complete else '未检测到双路完整板点',
                                 SUCCESS if complete else WARNING)
            reasons = [f'{name}：{quality_reason_text(reason)}'
                       for name, quality in zip(self.camera_labels, self.last_qualities)
                       for reason in quality.reasons]
            passed = all(quality.accepted for quality in self.last_qualities)
            statuses['quality'] = ('预检通过' if passed else '待调整，查看详情',
                                   SUCCESS if passed else WARNING)
            stable = not self.stability_reasons(detections)
            statuses['stability'] = ('连续检测稳定' if stable else '保持静止，等待稳定',
                                     SUCCESS if stable else WARNING)
        if panel._aux_offline_records:
            statuses = {'board': ('离线原图检测', TEXT_DIM),
                        'quality': (f'逐对确认 {len(panel._aux_pairs)} 组', TEXT_DIM),
                        'stability': ('离线未验证', WARNING), 'pairing': ('离线未验证同步', WARNING)}
        for key, (text, color) in statuses.items():
            value = self.check_values[key]
            value.setText(text)
            value.setStyleSheet(f'color:{color}; font-size:12px;')
            detail = (pairing if key == 'pairing' else '；'.join(reasons) if key == 'quality'
                      else '稳定需连续3次检测、跨度至少0.6秒' if key == 'stability' else '')
            value.setToolTip(text + ('\n' + detail if detail else '')
                             + '\n实时预检不是本次采图验收；保存、验收与引导采用分别统计。')
        for label in (self.action, self.next_step):
            label.setToolTip(label.text())
            label.setAccessibleDescription(label.text())

    def _update_target_info(self):
        board = self.panel._aux_board_config()
        target = self.panel._aux_capture_sequence.current
        if not self._manual and self.analysis is not None and self.analysis.target is not None:
            planned = self.analysis.target
            target = CaptureTarget(planned.row, planned.col, TILTS[planned.tilt],
                                   scale=planned.scale)
        if self.panel._aux_offline_records:
            goal = '离线原图 · 不计为实时引导姿态完成'
        elif self.reviewing:
            goal = '目标：回看原图，实时目标暂停'
        elif self._analysis_error and self._key == self.context_key() and not self._manual:
            goal = '目标：引导分析失败，原图仍保留'
        elif not self._manual and (self.analysis is None or self._key != self.context_key()):
            goal = '目标：等待采图建议更新'
        elif target is not None:
            prefix = '浏览参考' if self._manual else '目标'
            goal = f'{prefix}：{DISTANCE_NAMES[target.scale]} · {target.label}'
        elif self.analysis is not None and self.analysis.phase == 'blocked':
            goal = '目标：当前构图不可用'
        elif self.analysis is not None and self.analysis.phase == 'refining':
            goal = '目标：正在评估距离和倾角分布，尚未决定补采'
        elif self._manual:
            goal = '浏览参考：已浏览完，非采集进度'
        else:
            goal = '目标：本轮引导结束，可继续补拍'
        pattern = '棋盘格' if board.pattern_type.value == 'chessboard' else '圆点阵'
        self.target_info.setText(
            f'{goal}\n{pattern} · 检测点 {board.cols}×{board.rows} · '
            f'{board.square_size * 1000:g} mm')
        self.target_info.setToolTip(self.target_info.text()
            + f'\n位置以{self.auxiliary_label}画面为准，倾角以RGB左目为参考；距离占比不是实测距离。')
        self.target_info.setAccessibleDescription(self.target_info.toolTip())

    def _workbench_next_step(self, frames, fresh):
        panel = self.panel
        if panel._aux_live_gate.waiting:
            return ('已请求采集；保持整板可见并静止，双路稳定后自动取本次原图；可取消',
                    panel._aux_live_gate.reason or '双路稳定，正在取图')
        if panel._aux_offline_records and not panel._aux_operation_busy():
            problem = panel._aux_calibration_problem()
            if panel._rgb_left_camera_id() not in panel._intrinsics:
                return '加载匹配的 RGB 标定结果，或先完成 RGB 双目标定', None
            if len(panel._aux_pairs) < 6:
                return '打开“离线逐对复核”，确认物理对应后纳入至少 6 组', None
            return (problem if problem else '前往标定，选择辅助独立模型并验证离线候选'), None
        if self.reviewing:
            return '结束回看，返回实时图像后再采集', None
        if panel._aux_capture_worker is not None:
            return ('等待本次原图检查与保存结果，暂勿重复采集',
                    '正在检查本次原图，尚未确认保存')
        if panel._white_balance_guard.busy:
            return '等待白平衡准备完成，再采集配对图像', '正在准备白平衡'
        if panel._aux_operation_busy():
            return '等待当前后台任务完成，再继续采集或标定', '后台任务进行中，请稍候'
        if self._receipt_override:
            if '失败' in self._receipt_override or '未保存' in self._receipt_override:
                return ('查看最近原图详情，处理失败原因后重新采图',
                        self._receipt_override.split('\n', 1)[0])
            if '尚未确认保存' in self._receipt_override:
                return '等待本次原图检查与保存结果，暂勿重复采集', '本次原图尚未确认保存'
        receipt = self._receipt_record
        if (receipt and not self._manual and not self._receipt_override
                and self._receipt_scope == self.context_key()[:6]
                and self._clock() < self._receipt_hold_until):
            stored = '已保存' if receipt['saved'] else '仅在内存'
            if receipt['state'] != '可采纳':
                reason = receipt['reasons'][0] if receipt['reasons'] else '请查看原图验收详情'
                return (f'本组未计入引导：{reason}；调整后重拍当前参考动作',
                        f'{receipt["key"]} · {stored} · 验收待复核')
            decision = (self.analysis.sample_decisions.get(receipt['key'])
                        if self.analysis and self._key == self.context_key() else None)
            if decision == 'duplicate':
                return ('本组为重复构图，原图保留但引导不推进；请调整板的位置、距离或倾角',
                        f'{receipt["key"]} · {stored} · 重复构图未推进')
            if decision == 'included':
                return ('本组已计入引导；接着按下个参考动作调整，可继续采图或前往标定复核',
                        f'{receipt["key"]} · {stored} · 验收通过')
            return ('本组验收通过，正在更新引导进度；原图和验收计数已保留',
                    f'{receipt["key"]} · {stored} · 引导分析中')
        if self._manual:
            text = ('当前仅浏览参考；可手动采图，采图后恢复推荐' if frames is not None else
                    '当前仅浏览参考；连接两路实时图像后可手动采图，或导入原图')
            return text, None
        current = self._key == self.context_key()
        if self._analysis_error and current:
            return '查看日志；重新采集或导入后将重新分析引导', None
        if self.analysis is None or not current:
            return '等待采图建议更新；已有原图和独立计数保留', None
        if self.analysis.phase == 'blocked':
            return '调整标定板或分辨率配置，或手动选择可用构图', None
        if self.analysis.phase == 'refining':
            return '等待距离与倾角分析，再按分析结果继续；当前尚未结束引导', None
        if self.analysis.target is None:
            problem = panel._aux_calibration_problem()
            return ('前往标定评估；' + problem if problem else
                    '前往标定评估，计算后检查质量；仍可继续补拍'), None
        if frames is None:
            problem = panel._aux_calibration_problem()
            return ('连接两路实时图像或导入配对原图' if problem else
                    '可前往标定评估；继续实时补拍需连接两路图像'), None
        stage, _ = guidance_action(panel._aux_board_config(), self.analysis.target,
                                   self.last_qualities, auxiliary_label=self.auxiliary_label)
        if stage == 'ready' and (not fresh or self.stability_reasons(self._preview_detections)):
            return ('等待双路连续稳定检测，再采集配对图像',
                    '保持整板可见，等待双路连续3次稳定检测')
        if stage == 'ready':
            return '采集配对图像；保存与验收后自动更新推荐', None
        if stage == 'tilt_unverified':
            return '人工确认倾角后可手动采图；本次原图仍需验收', None
        return '完成当前动作后保持静止，再采集配对图像；本次原图仍需验收', None
