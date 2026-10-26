"""Preview-first stereo capture workbench; measurements are never synthetic."""

from __future__ import annotations

import time
import copy

import cv2
import numpy as np
from PySide6.QtCore import Qt, QTimer, QThreadPool, Slot
from PySide6.QtWidgets import (
    QComboBox, QHBoxLayout, QLabel, QScrollArea, QSizePolicy, QToolButton,
    QVBoxLayout, QWidget,
)

from ...calibration.models import CameraModel
from ...calibration.adaptive_guidance import GuidanceObservation, measure_pose
from ..threads.guidance_worker import GuidanceWorker
from ..theme import ACCENT, BORDER, SUCCESS, TEXT_DIM, WARNING, app_icon
from .capture_guide import BoardTilt, TILT_LABELS, CaptureTarget, CaptureSequence
from .stereo_schematic import StereoSchematic, TILT_ACTIONS


class StereoCaptureWorkspace(QWidget):
    """Hosts existing controls while preserving their ownership and signals."""

    def __init__(self, panel, stream_view, board):
        super().__init__()
        self.panel, self.view, self.board = panel, stream_view, board
        self.samples = {}
        self._context = None
        self._coverage_key = None
        self._coverage_summary = []
        self._compact = False
        self._analysis = None
        self._analysis_key = None
        self._requested_key = None
        self._guidance_worker = None
        self._guidance_error = ''
        self._adaptive_sequence = CaptureSequence([CaptureTarget(1, 1, BoardTilt.FRONT)])
        self.banner = QWidget()
        banner = QHBoxLayout(self.banner)
        banner.setContentsMargins(14, 8, 14, 8)
        self.banner.setObjectName('stereoCaptureBanner')
        self.banner.setStyleSheet(
            f"QWidget#stereoCaptureBanner {{background:#202830; border-bottom:1px solid {BORDER};}}"
            "QWidget#stereoCaptureBanner QLabel {background:transparent; border:none;}"
        )
        self.instruction = QLabel("连接左右相机并放入标定板")
        self.instruction.setWordWrap(True)
        self.instruction.setStyleSheet(
            f"color:{ACCENT}; font-size:18px; font-weight:bold; background:transparent;"
        )
        banner.addWidget(self.instruction, 2)
        self.reason = QLabel("固定采图计划 · 等待实时检测")
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
        self.compact_summary = QLabel()
        self.compact_summary.setWordWrap(True)
        self.compact_summary.hide()
        outer.addWidget(self.compact_summary)
        self.advice = QLabel("建议补拍不同位置和倾角")
        self.advice.setWordWrap(True)
        self.advice.setStyleSheet(f"color:{TEXT_DIM}; font-size:12px;")
        outer.addWidget(self.advice)

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
        panel.data_source_changed.connect(self.refresh)
        panel._seq_widget.user_navigated.connect(self._manual_navigation)
        self.refresh()

    @property
    def adaptive(self):
        return self.guide_mode.currentData() == 'adaptive' and not self.panel.offline_selection

    def _manual_navigation(self):
        self.guide_mode.setCurrentIndex(1)

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
        current_key, _ = self._guidance_context()
        if current_key == key:
            self._analysis, self._analysis_key = result, key
            self._guidance_error = error
            if result is not None:
                rec = result.recommendation
                tilts = dict(zip(('front', 'left', 'right', 'up', 'down'), BoardTilt))
                self._adaptive_sequence = CaptureSequence([
                    CaptureTarget(rec.zone_row, rec.zone_col, tilts[rec.tilt], scale=rec.scale),
                ])
        self.refresh()

    def _target(self):
        key, _ = self._guidance_context()
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
        if intr is None:
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

    def _adaptive_aligned(self, pairs, samples, target):
        if not target or not samples or not all(s['valid'] for s in samples):
            return False, '等待双路检测'
        if not all(np.linalg.norm(s['center'] - (target.norm_cx, target.norm_cy)) < .16
                   for s in samples):
            return False, '位置待对齐'
        low, high = {'small': (.012, .04), 'medium': (.04, .12), 'large': (.12, .45)}[target.scale]
        if not all(low <= s.get('area', 0) <= high for s in samples):
            return False, '画面占比待调整'
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
        return (True, '姿态接近建议范围') if all(matches(p) for p in poses) else (False, '倾角或方向待调整')

    def _select_pair(self, text):
        if text:
            self.panel._combo_pair.setCurrentText(text)

    def _toggle_details(self, expanded):
        self.details.setVisible(expanded)
        self.coverage_button.setIcon(app_icon('chevron-down' if expanded else 'chevron-right'))

    def showEvent(self, event):
        super().showEvent(event)
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
        previous = self.samples.get(camera_id)
        sample = {"time": time.monotonic(), "valid": detection.valid, "detection": detection}
        if detection.valid:
            pts = detection.charuco_corners.reshape(-1, 2)
            w, h = detection.image_size
            sample["center"] = pts.mean(axis=0) / (w, h)
            sample['area'] = float(cv2.contourArea(cv2.convexHull(pts.astype(np.float32)))) / (w * h)
            x, y, rw, rh = cv2.boundingRect(pts.astype(np.float32))
            roi = frame[max(y, 0):min(y + rh, h), max(x, 0):min(x + rw, w)]
            if roi.size:
                scale = min(1., 320 / max(roi.shape[:2]))
                small = cv2.resize(roi, None, fx=scale, fy=scale) if scale < 1 else roi
                gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY) if small.ndim == 3 else small
                sample["sharp"] = float(cv2.Laplacian(gray, cv2.CV_64F).var()) >= 50
            stable = False
            if previous and previous["valid"] and sample["time"] - previous["time"] < 1.5:
                old = previous["detection"]
                common, old_idx, new_idx = np.intersect1d(
                    old.charuco_ids.ravel(), detection.charuco_ids.ravel(), return_indices=True,
                )
                if len(common) >= 6 and old.image_size == detection.image_size:
                    delta = old.charuco_corners.reshape(-1, 2)[old_idx] - pts[new_idx]
                    stable = float(np.max(np.linalg.norm(delta, axis=1))) / max(w, h) < .008
            sample["stable"] = stable
            cal = self.panel._intr_cals.get(camera_id)
            sample['similar'] = cal.find_similar_detection(detection) if cal is not None else None
        self.samples[camera_id] = sample
        self.refresh()

    def _current_samples(self):
        names = self.panel._selected_pairs()
        pairs = [self.panel._sm.stereo_pairs[n] for n in names if n in self.panel._sm.stereo_pairs]
        samples = []
        now = time.monotonic()
        for pair in pairs:
            for camera in (pair.left, pair.right):
                sample = self.samples.get(camera.camera_id)
                entry = self.panel._sm.get_latest_frame(camera.camera_id)
                if (sample is None or now - sample["time"] > 1.5 or entry is None
                        or time.time() - entry[1] > 1.5):
                    return pairs, []
                samples.append(sample)
        return pairs, samples

    def auto_ready(self):
        _, samples = self._current_samples()
        target = self._target()
        if self.adaptive:
            key, _ = self._guidance_context()
            if key is not None:
                pairs, samples = self._current_samples()
                return bool(
                    self.isVisible() and not self.view.review_key and self._analysis
                    and self._analysis_key == key and not self.panel._pending_save_tokens
                    and self._guidance_worker is None
                    and all(s['valid'] and s.get('sharp') and s.get('stable') for s in samples)
                    and self._adaptive_aligned(pairs, samples, target)[0]
                    and not all(s.get('similar') is not None for s in samples)
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
        if self.adaptive and self._pose_measurement(camera_id, sample) is None:
            return "姿态估计不可用\n请检查内参、可见角点或调整姿态"
        try:
            ids = det.charuco_ids.ravel()
            objects = self.board.object_points[ids]
            points = det.charuco_corners
            K, dist = intr.camera_matrix, intr.dist_coeffs
            if intr.model == CameraModel.FISHEYE:
                points = cv2.fisheye.undistortPoints(points, K, dist)
                K, dist = np.eye(3), None
            ok, rvec, tvec = cv2.solvePnP(objects, points, K, dist)
            if not ok or not np.isfinite(tvec).all() or tvec[2, 0] <= 0:
                raise ValueError("Invalid pose")
            normal = cv2.Rodrigues(rvec)[0][:, 2]
            if normal[2] < 0:
                normal = -normal
            yaw = np.degrees(np.arctan2(normal[0], normal[2]))
            pitch = np.degrees(np.arctan2(normal[1], normal[2]))
            prefix = '临时内参估计\n' if temporary else ''
            return prefix + f"板原点深度  {tvec[2, 0]:.2f} m\n左右倾  {yaw:.1f}°\n上下倾  {pitch:.1f}°"
        except (cv2.error, ValueError, IndexError):
            return "姿态估计不可用\n请增加可见角点或调整姿态"

    def refresh(self):
        options = [self.panel._combo_pair.itemText(i) for i in range(self.panel._combo_pair.count())]
        self.pair.blockSignals(True)
        if options != [self.pair.itemText(i) for i in range(self.pair.count())]:
            self.pair.clear()
            self.pair.addItems(options)
        self.pair.setCurrentText(self.panel._combo_pair.currentText())
        self.pair.blockSignals(False)
        context = (id(self.panel._board), tuple(self.panel._selected_pairs()), self.view.review_key)
        if context != self._context:
            self.samples.clear()
            self._context = context
        self.board = self.panel._board
        offline = self.panel.offline_selection
        self.guide_mode.setEnabled(not offline)
        key = self._request_guidance() if self.adaptive and not self.view.review_key else None
        using_adaptive = bool(self.adaptive and key is not None and self._analysis
                              and self._analysis_key == key)
        target = self._target()
        if not offline and self.panel._active_step == 0:
            self.view.set_capture_sequence(self._adaptive_sequence if using_adaptive
                                           else self.panel._capture_sequence)
        self.panel._seq_widget.setVisible(not offline and not using_adaptive)
        self.schematic.set_target(target, active=not bool(self.view.review_key) and not offline)
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
                scale_text = {'small': '缩小画面占比', 'medium': '保持适中大小',
                              'large': '增大画面占比'}[target.scale]
                self.instruction.setText(self.instruction.text() + f'；{scale_text}')
                self.reason.setText(self._analysis.recommendation.reason)
            elif key is None:
                self.reason.setText('多组同时采集沿用固定计划；选择单组可用自适应引导')
            elif self._guidance_error:
                self.reason.setText('自适应分析不可用，已回退固定计划')
                self.reason.setToolTip(self._guidance_error)
            else:
                self.reason.setText('正在分析已采配对，暂不自动采集')
        pairs, samples = self._current_samples()
        visible = bool(samples) and all(s["valid"] for s in samples)
        sharp = visible and all(s.get("sharp") for s in samples)
        stable = visible and all(s.get("stable") for s in samples)
        aligned = bool(visible and target) and all(
            np.linalg.norm(s["center"] - (target.norm_cx, target.norm_cy)) < .16 for s in samples
        )
        self.status.setText(
            "基础画面检查（不含姿态）\n\n"
            f"双路可见：{'已检测' if visible else '等待检测'}\n"
            f"清晰度：{'良好' if sharp else '待检查'}\n"
            f"运动稳定：{'稳定' if stable else '请保持静止'}\n"
            f"目标位置：{'接近建议区域' if aligned else '待对齐'}"
        )
        self.status.setStyleSheet(f"color:{SUCCESS if visible and sharp and stable and aligned else TEXT_DIM};")
        self.status.setToolTip("预览阈值仅辅助采图，不是硬件同步或标定精度证明；目标位置不验证倾角")
        if not using_adaptive and not offline and not self.view.review_key and target and target.tilt != BoardTilt.FRONT:
            self.reason.setText(f"固定计划建议 · {TILT_LABELS[target.tilt]} · 倾角步骤请手动采集")
        self.pose_status.setText("姿态：待人工确认" if visible and not self.view.review_key else "姿态：待实时检测")
        self.pose_status.setToolTip("PnP 数值仅供观察；本轮不自动验收角度、方向或姿态多样性")
        if using_adaptive and not self.view.review_key:
            reached, message = self._adaptive_aligned(pairs, samples, target)
            self.pose_status.setText(message)
            self.pose_status.setStyleSheet(f'color:{SUCCESS if reached else WARNING};')
            self.pose_status.setToolTip('经验验收范围，不代表精度合格；双路位置、角点凸包占比及倾角共同检查')
        duplicate = visible and all(s.get('similar') is not None for s in samples)
        self.advice.setText(
            "重复构图提醒：当前双路画面接近已有样本，建议改变位置、大小或倾角"
            if duplicate else "姿态分布尚未自动评估；建议补拍不同位置、大小和倾角"
        )
        if using_adaptive and not duplicate:
            self.advice.setText(self._analysis.summary)
        self.advice.setStyleSheet(f"color:{WARNING if duplicate else TEXT_DIM}; font-size:12px;")
        self.advice.setVisible(duplicate or not self._compact)
        left = pairs[0].left.camera_id if pairs else None
        self.pose.setText("当前姿态（相对左相机）\n\n" + self._pose_text(left, samples[0] if samples else None))
        self.pose.setToolTip("深度为板坐标原点的相机 Z 值；左右倾正值为右边近，上下倾正值为下边近；不包含画面内旋转")
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
            "角点完整度/视场网格\n" + '  '.join(self._coverage_summary)
        )
        self.compact_summary.setToolTip(self.pose.text() + '\n' + self.status.text())
        if offline:
            self.compact_summary.setText(
                f'离线观测 {loaded} 对\n{self.panel._lbl_capture_count.text()}\n'
                '姿态覆盖未验证\n角点完整度/视场网格\n' + '  '.join(self._coverage_summary)
            )
        self.coverage_button.setToolTip(
            '  '.join(self._coverage_summary) +
            "\n角点完整度：出现过的板角点 ID 比例；视场网格：4×4 图像网格有角点的占比，不是姿态覆盖率",
        )
        self.coverage_button.setText(
            "数据分析 · " + self.panel._lbl_capture_count.text() if self._compact else
            "角点完整度/视场网格  ·  " + "  |  ".join(
                [self.panel._lbl_capture_count.text(), *self._coverage_summary],
            )
        )

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
