"""Preview-first stereo capture workbench; measurements are never synthetic."""

from __future__ import annotations

import time

import cv2
import numpy as np
from PySide6.QtCore import QPointF, Qt, QTimer
from PySide6.QtGui import QColor, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import (
    QComboBox, QHBoxLayout, QLabel, QScrollArea, QSizePolicy, QToolButton,
    QVBoxLayout, QWidget,
)

from ...calibration.models import CameraModel
from ..theme import ACCENT, BORDER, SUCCESS, TEXT_DIM, WARNING, app_icon
from .capture_guide import BoardTilt, TILT_LABELS, board_corners_f


class StereoSchematic(QWidget):
    """Qualitative line drawing, not a metric reconstruction of the rig."""

    def __init__(self):
        super().__init__()
        self.target = None
        self.setMinimumSize(210, 125)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setAccessibleName("双目相机与建议板姿态线稿")
        self.setToolTip("定性示意，不表示实际基线、距离或已测量姿态")

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.scale(self.width() / 320, self.height() / 125)
        painter.setPen(QPen(QColor(TEXT_DIM), 1.5))
        painter.drawLine(30, 100, 115, 110)
        for x, y, label in ((26, 91, "左"), (110, 101, "右")):
            painter.drawRoundedRect(x, y, 24, 17, 2, 2)
            painter.drawEllipse(QPointF(x + 12, y + 8), 5, 5)
            painter.drawText(x + 4, y - 6, label)
        row = self.target.zone_row if self.target else 1
        col = self.target.zone_col if self.target else 1
        tx, ty = 225 + (col - 1) * 24, 35 + (row - 1) * 15
        painter.setPen(QPen(QColor(TEXT_DIM), 1, Qt.DashLine))
        painter.drawLine(QPointF(50, 95), QPointF(tx, ty))
        painter.drawLine(QPointF(135, 105), QPointF(tx + 45, ty + 40))
        for x, y, color, dashed, label in (
            (170, 60, WARNING, False, "位置示意"),
            (tx, ty, ACCENT, True, "建议"),
        ):
            painter.setPen(QPen(QColor(color), 2, Qt.DashLine if dashed else Qt.SolidLine))
            tilt = self.target.tilt if dashed and self.target else BoardTilt.FRONT
            corners = board_corners_f(x + 22, y + 21, 22, 21, tilt)
            painter.drawPolygon(QPolygonF([QPointF(*point) for point in corners]))
            for a, b, c, d in ((0, 1, 3, 2), (0, 3, 1, 2)):
                start = (corners[a] + corners[b]) / 2
                end = (corners[c] + corners[d]) / 2
                painter.drawLine(QPointF(*start), QPointF(*end))
            painter.drawText(QPointF(x - 10, y - 6), label)


class StereoCaptureWorkspace(QWidget):
    """Hosts existing controls while preserving their ownership and signals."""

    def __init__(self, panel, stream_view, board):
        super().__init__()
        self.panel, self.view, self.board = panel, stream_view, board
        self.samples = {}
        self._context = None
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
        self.pair = QComboBox()
        self.pair.setMaximumWidth(220)
        self.pair.setAccessibleName("双目采集目标")
        self.pair.currentTextChanged.connect(self._select_pair)
        banner.addWidget(self.pair)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(6, 0, 6, 0)
        outer.setSpacing(6)
        body = QHBoxLayout()
        schematic_column = QVBoxLayout()
        schematic_column.addWidget(QLabel("相机与目标示意（非比例线稿）"))
        self.schematic = StereoSchematic()
        schematic_column.addWidget(self.schematic)
        body.addLayout(schematic_column, 1)
        self.pose = QLabel("当前姿态（相对左相机）\n\n距离  待内参\n左右倾  待内参\n上下倾  待内参")
        self.pose.setWordWrap(True)
        self.pose.setMinimumWidth(165)
        body.addWidget(self.pose, 1)
        self.status = QLabel("采集状态\n\n双路可见：等待检测\n清晰度：待检测\n运动稳定：待检测\n目标位置：待检测")
        self.status.setWordWrap(True)
        self.status.setMinimumWidth(205)
        body.addWidget(self.status, 1)
        outer.addLayout(body)

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
        self.refresh()

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
        target = self.panel._capture_sequence.current
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
        if intr is None:
            return "距离  待内参\n左右倾  待内参\n上下倾  待内参"
        if not sample or not sample["valid"]:
            return "距离  待实时检测\n左右倾  待实时检测\n上下倾  待实时检测"
        det = sample["detection"]
        if det.image_size != intr.image_size:
            return "图像尺寸与内参不匹配\n请重新标定或恢复分辨率"
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
            pitch = np.degrees(np.arctan2(-normal[1], np.hypot(normal[0], normal[2])))
            return f"板原点深度  {tvec[2, 0]:.2f} m\n左右倾  {yaw:.1f}°\n上下倾  {pitch:.1f}°"
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
        target = self.panel._capture_sequence.current
        self.schematic.target = target
        self.schematic.update()
        if self.view.review_key:
            self.instruction.setText("正在回看已采图像")
            self.reason.setText("结束回看后恢复实时采图引导")
        elif target:
            positions = (("左上", "上方", "右上"), ("左侧", "中央", "右侧"), ("左下", "下方", "右下"))
            self.instruction.setText(f"将标定板移至画面{positions[target.zone_row][target.zone_col]}")
            self.reason.setText(f"固定计划建议 · {TILT_LABELS[target.tilt]} · 保持双路可见")
        else:
            self.instruction.setText("计划已浏览完，可继续补拍")
            self.reason.setText("浏览结束不代表标定质量达标")
        pairs, samples = self._current_samples()
        visible = bool(samples) and all(s["valid"] for s in samples)
        sharp = visible and all(s.get("sharp") for s in samples)
        stable = visible and all(s.get("stable") for s in samples)
        aligned = bool(visible and target) and all(
            np.linalg.norm(s["center"] - (target.norm_cx, target.norm_cy)) < .16 for s in samples
        )
        self.status.setText(
            "采集状态（预览估计）\n\n"
            f"双路可见：{'已检测' if visible else '等待检测'}\n"
            f"清晰度：{'良好' if sharp else '待检查'}\n"
            f"运动稳定：{'稳定' if stable else '请保持静止'}\n"
            f"目标位置：{'接近建议区域' if aligned else '待对齐'}"
        )
        self.status.setStyleSheet(f"color:{SUCCESS if visible and sharp and stable and aligned else TEXT_DIM};")
        self.status.setToolTip("预览阈值仅辅助采图，不是硬件同步或标定精度证明；目标位置不验证倾角")
        if target and target.tilt != BoardTilt.FRONT:
            self.reason.setText(f"固定计划建议 · {TILT_LABELS[target.tilt]} · 倾角步骤请手动采集")
        left = pairs[0].left.camera_id if pairs else None
        self.pose.setText("当前姿态（相对左相机）\n\n" + self._pose_text(left, samples[0] if samples else None))
        self.pose.setToolTip("深度为板坐标原点的相机 Z 值；倾角由板面法向估计，不包含画面内旋转")
        coverage = []
        if pairs:
            for label, camera in (("左目", pairs[0].left), ("右目", pairs[0].right)):
                cal = self.panel._intr_cals.get(camera.camera_id)
                value = cal.compute_coverage() if cal is not None and cal.num_frames else 0.
                coverage.append(f"{label}覆盖 {value:.0%}")
        self.coverage_button.setText("覆盖分析与采集设置  ·  " + "  |  ".join(
            [self.panel._lbl_capture_count.text(), *coverage],
        ))
