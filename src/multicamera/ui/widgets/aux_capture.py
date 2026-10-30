"""Bounded RGB-left/AUX guidance and persistent, independently counted evidence."""

from __future__ import annotations

import time
from copy import deepcopy
from dataclasses import replace

import numpy as np
from PySide6.QtCore import QThreadPool, Slot
from PySide6.QtWidgets import QVBoxLayout, QWidget

from ...calibration.aux_guidance import (
    AREA_RANGES, DISTANCE_NAMES, MAX_REPAIRS, guidance_action, initial_plan,
)
from ..threads.aux_capture_worker import AuxCaptureWorker, AuxGuidanceWorker
from ..theme import ACCENT, SUCCESS, TEXT_DIM, WARNING
from .capture_guide import BoardTilt, CaptureTarget, build_multimodal_sequence
from .stable_text import StableTextLabel


TILTS = {'front': BoardTilt.FRONT, 'left': BoardTilt.TILT_LEFT, 'right': BoardTilt.TILT_RIGHT,
         'up': BoardTilt.TILT_UP, 'down': BoardTilt.TILT_DOWN}


class AuxCaptureGuide(QWidget):
    MAX_AGE = 1.5

    def __init__(self, panel):
        super().__init__(panel)
        self.panel = panel
        self.analysis = None
        self._key = None
        self._requested = None
        self._worker = None
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
        self.last_qualities = []
        self._preview_detections = []
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        self.phase = StableTextLabel('基础采集 · 等待图像', lines=2)
        self.phase.setStyleSheet(f'color:{ACCENT}; font-weight:bold;')
        self.action = StableTextLabel('让同一块标定板完整进入 RGB 左目和 IR 画面', lines=2)
        self.counts = StableTextLabel('已保存 0 · 检测有效 0\n质量可用 0 · 非重复 0 · 双路姿态可解 0', lines=2)
        self.counts.setStyleSheet(f'color:{TEXT_DIM}; font-size:12px;')
        self.receipt = StableTextLabel('最近原图：尚未采集', lines=2)
        for label in (self.phase, self.action):
            layout.addWidget(label)
        self.details = QWidget(panel)
        details = QVBoxLayout(self.details)
        details.setContentsMargins(0, 0, 0, 0)
        details.setSpacing(4)
        details.addWidget(self.counts)
        details.addWidget(self.receipt)

    def bind_preview(self, view):
        self._view = view
        view.preview_detected.connect(self.observe)
        view.preview_detection_expired.connect(self.expire)

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
        self.analysis = None
        self._samples.clear()
        self._sample_timestamps.clear()
        self.last_qualities = []
        self._preview_detections = []
        self._manual = False
        self.receipt.setText('最近原图：尚未采集')
        self.refresh()

    def data_changed(self):
        self._revision += 1
        self._key = None
        self.refresh()

    def manual_navigation(self):
        self._manual = True
        self.phase.setText('手动查看计划 · 不计入引导进度')
        self.action.setText('手动浏览计划，不计为采集或姿态验证；采图后恢复推荐')

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
    def observe(self, camera_id, detection, image):
        panel = self.panel
        if (panel._active_step != 2 or panel._workflow_phase != 'capture'
                or (self._view is not None and self._view._review_images is not None)
                or camera_id not in (panel._rgb_left_camera_id(), panel._aux_camera_id())):
            return
        if not detection.valid or detection.num_corners != np.prod(
                (panel._aux_board_config().cols, panel._aux_board_config().rows)):
            self.expire(camera_id)
            return
        now = self._clock()
        entry = panel._sm.get_latest_frame(camera_id)
        stamp = entry[1] if entry is not None else now
        if self._sample_timestamps.get(camera_id) == stamp:
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
                or self._preview_worker or panel._aux_operation_busy()
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
        self._preview_started = self._clock()
        worker = AuxCaptureWorker((self.context_key(), self._preview_started),
            tuple(frame.copy() for frame in frames[:2]), deepcopy(panel._aux_board_config()),
            self.intrinsics(), panel._aux_modality())
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
        detections, self.last_qualities = result
        self._preview_detections = detections
        self._preview_result_at = key[1]
        # This independent sampler also works when an optional preview tile is closed.
        from types import SimpleNamespace
        for camera_id, det in zip((self.panel._rgb_left_camera_id(), self.panel._aux_camera_id()),
                                   detections):
            self.observe(camera_id, SimpleNamespace(valid=det.valid, num_corners=det.num_points,
                         charuco_corners=det.image_points, image_size=det.image_size), None)
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
        for name, camera_id, det in zip(('RGB', 'IR/辅助'),
                (self.panel._rgb_left_camera_id(), self.panel._aux_camera_id()), detections):
            history = histories.get(camera_id, [])
            stable = (det is not None and det.valid and len(history) >= 3
                      and 0 <= now - history[-1][0] <= self.MAX_AGE
                      and history[-1][0] - history[0][0] >= .6
                      and det.image_size == history[-1][2]
                      and self._close(det.image_points.reshape(-1, 2), history[0][1], det.image_size))
            if not stable:
                reasons.append(f'{name}稳定性未通过，请双路完整可见并静止至少0.6秒')
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
            self._samples.clear()
            self._sample_timestamps.clear()
            self.last_qualities = []
            self._preview_detections = []
        if self._key != key and self._worker is None:
            worker = AuxGuidanceWorker(key, deepcopy(self.panel._aux_board_config()),
                self.panel._camera_model, deepcopy(self.panel._aux_pairs),
                deepcopy((self.panel._intrinsics.get(self.panel._rgb_left_camera_id()),
                          self.panel._aux_intrinsics)))
            self._worker, self._requested = worker, key
            worker.signals.completed.connect(self._analyzed)
            QThreadPool.globalInstance().start(worker)
        self._update_counts()
        self._request_preview()
        if self.analysis is None or self._key != key:
            self.phase.setText('正在分析采图数据；当前原图记录保留')
        self._tooltips()

    @Slot(object, object, str)
    def _analyzed(self, key, analysis, error):
        self._worker = None
        if key != self.context_key():
            self.refresh()
            return
        self._key = key
        if error:
            self.analysis = None
            self.phase.setText('采图分析失败；可检查原图或重新导入')
            self.action.setText(error)
            self.panel._log('融合采图分析失败：' + error)
            return
        self.analysis = analysis
        self._manual = False
        if analysis.phase == 'complete':
            self.panel._log('融合本轮采集完成，可以前往标定评估；仍可补拍，未宣称精度合格')
        self.render()

    def render(self):
        analysis = self.analysis
        if analysis is None:
            return
        seq = self.panel._aux_capture_sequence
        seq.targets = build_multimodal_sequence()
        if analysis.target is None:
            seq._index = seq.total
            self.phase.setText('当前构图不可用 · 请调整板型或分辨率' if analysis.phase == 'blocked'
                               else '本轮采集完成 · 可以前往标定评估')
            missing = '、'.join(DISTANCE_NAMES[s] for s in analysis.missing)
            self.action.setText('当前板型与分辨率无法安全放入建议区域，请调整配置或手动选择构图'
                if analysis.phase == 'blocked' else
                ('仍待验证距离：' + missing + '；可继续补拍\n' if missing else '')
                + '采图结束不代表标定精度合格')
            self.phase.setStyleSheet(f'color:{WARNING if analysis.phase == "blocked" else SUCCESS}; font-weight:bold;')
            self.panel._aux_seq_widget.set_completion_message(
                '当前构图不可用' if analysis.phase == 'blocked' else '本轮采集完成',
                '请检查板型和分辨率' if analysis.phase == 'blocked' else '可前往标定评估，或继续补拍')
        else:
            self.panel._aux_seq_widget.set_completion_message('', '')
            target = analysis.target
            capture = CaptureTarget(target.row, target.col, TILTS[target.tilt], scale=target.scale,
                                    target_area=target.target_area)
            if analysis.phase == 'repair':
                seq.targets.append(capture)
                seq._index = len(initial_plan())
                stage = f'有限补采 {analysis.repair_count + 1}/{MAX_REPAIRS}'
            else:
                seq._index = analysis.unique_count
                seq.targets[seq._index] = capture
                stage = f'基础采集 {analysis.unique_count + 1}/{analysis.initial_total}'
            self.phase.setStyleSheet(f'color:{ACCENT}; font-weight:bold;')
            self.phase.setText(stage + ' · ' + DISTANCE_NAMES[target.scale])
            ir_pose = self.last_qualities[1].pose if len(self.last_qualities) == 2 else None
            if ir_pose is not None and ir_pose.get('distance_reliable', False):
                low, high = AREA_RANGES[target.scale]
                factor = ir_pose['distance_area'] * ir_pose['depth'] ** 2
                self.phase.setText(self.phase.text() + f"\nIR 深度 {ir_pose['depth']:.2f} m"
                    f' · 目标 {np.sqrt(factor / high):.2f}–{np.sqrt(factor / low):.2f} m')
            else:
                self.phase.setText(self.phase.text() + '\nIR 距离按占比近似，非实测距离')
            instruction_stage, action = guidance_action(
                self.panel._aux_board_config(), target, self.last_qualities)
            if instruction_stage == 'ready' and (
                    not 0 <= self._clock() - self._preview_result_at <= self.MAX_AGE
                    or self.stability_reasons(self._preview_detections)):
                instruction_stage, action = 'stability', '双路保持静止至少0.6秒，再采集原图'
            if instruction_stage in ('distance', 'position', 'quality'):
                seq.targets[seq._index] = replace(capture, tilt=BoardTilt.FRONT)
            self.action.setText(action)
        self.panel._aux_seq_widget.refresh()
        self.panel._aux_seq_widget._large_guide.show_distance = True
        self.panel._aux_seq_widget._title_lbl.setText('建议构图 · 位置以 IR/辅助画面为准')
        self.panel._aux_seq_widget._title_lbl.setToolTip(
            '计划步骤不等于已验证姿态；当前帧以原图验收为准。上下/左右倾以RGB左目为参考。')
        self.panel._aux_cap_progress.setMaximum(analysis.initial_total)
        self.panel._aux_cap_progress.setValue(min(analysis.unique_count, analysis.initial_total))
        self.panel._aux_cap_progress.setFormat(
            f'非重复 {analysis.unique_count} 组 · 基础 {analysis.initial_total} 组')
        if self.panel._active_step == 2:
            self.panel.sequence_updated.emit(seq)
        self._update_counts()
        self._tooltips()

    def record_receipt(self, key, state, reasons, detections):
        self.receipt.setText(f'{state} · {key} · RGB {detections[0].num_points}'
                             f' / IR {detections[1].num_points}\n'
                             + ('；'.join(reasons) if reasons else '原图已保存，已纳入质量可用数据'))
        self.receipt.setStyleSheet(f'color:{SUCCESS if state == "可采纳" else WARNING};')
        self._tooltips()

    def _update_counts(self):
        records = getattr(self.panel, '_aux_quality_records', [])
        detected = sum(r.get('detected', False) for r in records)
        unique = self.analysis.unique_count if self.analysis and self._key == self.context_key() else '待更新'
        pose_count = self.analysis.pose_count if self.analysis and self._key == self.context_key() else '待更新'
        self.counts.setText(f'已保存 {self.panel._aux_saved_count} · 源目录 {self.panel._aux_imported_count}'
                            f' · 检测有效 {detected}\n质量可用 {len(self.panel._aux_pairs)}'
                            f' · 非重复 {unique} · 双路姿态可解 {pose_count}')
        if self.analysis:
            self.counts.setToolTip(self.analysis.summary + '\n姿态可解不等于推荐姿态已到位；临时内参不写入结果。')
        self.counts.setAccessibleName(self.counts.text())

    def _tooltips(self):
        for label in (self.phase, self.action, self.receipt):
            label.setToolTip(label.text())
