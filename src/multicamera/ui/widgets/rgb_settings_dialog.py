"""RGB camera setting dialog for FusionSensor-style HTTP camera services."""

from __future__ import annotations

import json
import logging
from typing import Any
from urllib.parse import urljoin
from urllib.request import Request, urlopen

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ..theme import ACCENT, DANGER, SUCCESS, TEXT_DIM

logger = logging.getLogger(__name__)

DEFAULT_RGB_SETTINGS_URL = "http://192.168.31.55:8080/"

_QUICK_KEYS = [
    "auto_exposure",
    "exposure_time_absolute",
    "brightness",
    "gain",
    "white_balance_automatic",
    "white_balance_temperature",
    "power_line_frequency",
]

_LABEL_FALLBACK = {
    "auto_exposure": "曝光 / Exposure",
    "exposure_time_absolute": "曝光时间 / Time",
    "brightness": "亮度 / Brightness",
    "gain": "增益 / Gain",
    "white_balance_automatic": "白平衡 / WB",
    "white_balance_temperature": "色温 / Temp",
    "power_line_frequency": "防闪烁 / Flicker",
    "contrast": "对比度 / Contrast",
    "saturation": "饱和度 / Saturation",
    "gamma": "伽马 / Gamma",
    "sharpness": "锐度 / Sharpness",
    "hue": "色调 / Hue",
    "backlight_compensation": "背光 / Backlight",
    "exposure_dynamic_framerate": "动态帧率 / Dynamic FPS",
}


class _RgbConfigWorker(QThread):
    loaded = Signal(dict)
    failed = Signal(str)

    def __init__(self, base_url: str, payload: dict[str, int] | None = None, parent=None):
        super().__init__(parent)
        self.base_url = base_url
        self.payload = payload

    def run(self):
        try:
            endpoint = urljoin(_normalise_base_url(self.base_url), "api/rgb/config")
            if self.payload is None:
                req = Request(endpoint, method="GET")
            else:
                data = json.dumps(self.payload).encode("utf-8")
                req = Request(
                    endpoint,
                    data=data,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
            with urlopen(req, timeout=5) as resp:
                body = resp.read().decode("utf-8", errors="replace")
            data = json.loads(body)
            if not data.get("ok"):
                raise RuntimeError(data.get("error") or "RGB settings API returned ok=false")
            self.loaded.emit(data)
        except Exception as exc:
            self.failed.emit(str(exc))


class RgbSettingsDialog(QDialog):
    """Edit RGB camera controls exposed by /api/rgb/config."""

    def __init__(self, base_url: str = DEFAULT_RGB_SETTINGS_URL, parent=None):
        super().__init__(parent)
        self.setWindowTitle("RGB 参数 / RGB Settings")
        self.resize(620, 760)
        self._controls_meta: dict[str, dict[str, Any]] = {}
        self._control_widgets: dict[str, dict[str, Any]] = {}
        self._worker: _RgbConfigWorker | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        url_row = QHBoxLayout()
        self._edit_url = QLineEdit(_normalise_base_url(base_url))
        self._edit_url.setPlaceholderText(DEFAULT_RGB_SETTINGS_URL)
        url_row.addWidget(QLabel("目标相机:"), 0)
        url_row.addWidget(self._edit_url, 1)
        self._btn_refresh = QPushButton("刷新")
        self._btn_refresh.clicked.connect(self.refresh_config)
        url_row.addWidget(self._btn_refresh)
        layout.addLayout(url_row)

        self._lbl_device = QLabel("设备: --")
        self._lbl_device.setStyleSheet(f"color: {TEXT_DIM};")
        layout.addWidget(self._lbl_device)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        self._body = QWidget()
        self._body_layout = QVBoxLayout(self._body)
        self._body_layout.setContentsMargins(0, 0, 0, 0)
        self._body_layout.setSpacing(8)
        scroll.setWidget(self._body)
        layout.addWidget(scroll, 1)

        bottom = QHBoxLayout()
        self._lbl_status = QLabel("点击刷新读取相机参数")
        self._lbl_status.setStyleSheet(f"color: {TEXT_DIM};")
        bottom.addWidget(self._lbl_status, 1)
        self._btn_apply = QPushButton("应用 / Apply")
        self._btn_apply.clicked.connect(self.apply_config)
        self._btn_apply.setEnabled(False)
        bottom.addWidget(self._btn_apply)
        self._btn_close = QPushButton("关闭")
        self._btn_close.clicked.connect(self.accept)
        bottom.addWidget(self._btn_close)
        layout.addLayout(bottom)

        self.refresh_config()

    def refresh_config(self):
        self._start_worker(None, "读取 RGB 参数...")

    def apply_config(self):
        payload = self._collect_payload()
        if not payload:
            self._set_status("没有可应用的参数", ok=False)
            return
        self._start_worker(payload, "应用 RGB 参数...")

    def accept(self):
        self._wait_for_worker()
        super().accept()

    def reject(self):
        self._wait_for_worker()
        super().reject()

    def closeEvent(self, event):
        self._wait_for_worker()
        super().closeEvent(event)

    def _start_worker(self, payload: dict[str, int] | None, status: str):
        if self._worker is not None and self._worker.isRunning():
            return
        self._set_status(status)
        self._set_busy(True)
        self._worker = _RgbConfigWorker(self._edit_url.text().strip(), payload, self)
        self._worker.loaded.connect(self._on_loaded)
        self._worker.failed.connect(self._on_failed)
        self._worker.finished.connect(self._on_worker_finished)
        self._worker.start()

    def _on_loaded(self, data: dict):
        self._edit_url.setText(_normalise_base_url(self._edit_url.text()))
        self._lbl_device.setText(f"设备: {data.get('device') or '--'}")
        self._controls_meta = data.get("controls") or {}
        self._rebuild_controls()
        self._btn_apply.setEnabled(bool(self._controls_meta))
        self._set_status("RGB 参数已就绪", ok=True)
        logger.info("RGB settings loaded: url=%s device=%s", self._edit_url.text(), data.get("device"))

    def _on_failed(self, reason: str):
        self._set_status(f"RGB 参数接口不可用: {reason}", ok=False)
        logger.warning("RGB settings request failed: url=%s reason=%s", self._edit_url.text(), reason)
        QMessageBox.warning(
            self,
            "RGB 参数",
            "无法读取或应用 RGB 参数。\n\n"
            f"目标: {self._edit_url.text()}\n"
            f"原因: {reason}\n\n"
            "请确认相机服务已启动，并支持 /api/rgb/config。",
        )

    def _on_worker_finished(self):
        if self._worker is not None:
            self._worker.deleteLater()
            self._worker = None
        self._set_busy(False)

    def _rebuild_controls(self):
        _clear_layout(self._body_layout)
        self._control_widgets.clear()

        quick = QGroupBox("常用参数")
        quick_layout = QVBoxLayout(quick)
        quick_layout.setSpacing(8)
        more = QGroupBox("更多参数")
        more_layout = QVBoxLayout(more)
        more_layout.setSpacing(8)

        handled: set[str] = set()
        for key in _QUICK_KEYS:
            ctl = self._controls_meta.get(key)
            if ctl:
                self._add_control(quick_layout, key, ctl)
                handled.add(key)

        for key, ctl in self._controls_meta.items():
            if key not in handled:
                self._add_control(more_layout, key, ctl)

        self._body_layout.addWidget(quick)
        self._body_layout.addWidget(more)
        self._body_layout.addStretch(1)
        self._update_dependencies()

    def _add_control(self, parent_layout: QVBoxLayout, key: str, ctl: dict[str, Any]):
        row = QWidget()
        row_layout = QVBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(3)

        head = QHBoxLayout()
        label = QLabel(_control_label(key, ctl))
        label.setStyleSheet(f"color: {TEXT_DIM};")
        head.addWidget(label, 1)
        meta = QLabel(_control_meta(ctl))
        meta.setStyleSheet(f"color: {TEXT_DIM}; font-size: 11px;")
        head.addWidget(meta)
        row_layout.addLayout(head)

        input_row = QHBoxLayout()
        kind = ctl.get("kind")
        widgets: dict[str, Any] = {"row": row, "kind": kind, "meta": ctl}
        if kind in {"menu", "bool"}:
            combo = QComboBox()
            for value, text in _combo_items(key, ctl):
                combo.addItem(text, value)
            idx = combo.findData(int(ctl.get("value", 0)))
            if idx >= 0:
                combo.setCurrentIndex(idx)
            combo.currentIndexChanged.connect(self._update_dependencies)
            input_row.addWidget(combo, 1)
            widgets["combo"] = combo
        else:
            slider = QSlider(Qt.Horizontal)
            spin = QSpinBox()
            min_v = int(ctl.get("min", 0))
            max_v = int(ctl.get("max", 100))
            step = max(1, int(ctl.get("step", 1) or 1))
            value = int(ctl.get("value", min_v))
            slider.setRange(min_v, max_v)
            slider.setSingleStep(step)
            slider.setPageStep(step * 5)
            spin.setRange(min_v, max_v)
            spin.setSingleStep(step)
            slider.setValue(value)
            spin.setValue(value)
            slider.valueChanged.connect(spin.setValue)
            spin.valueChanged.connect(slider.setValue)
            input_row.addWidget(slider, 1)
            input_row.addWidget(spin, 0)
            widgets["slider"] = slider
            widgets["spin"] = spin
        row_layout.addLayout(input_row)
        parent_layout.addWidget(row)
        self._control_widgets[key] = widgets

    def _update_dependencies(self):
        for key, widgets in self._control_widgets.items():
            ctl = widgets["meta"]
            disabled = bool(ctl.get("inactive"))
            if key == "exposure_time_absolute":
                disabled = self._control_value("auto_exposure") != 1
            elif key == "white_balance_temperature":
                disabled = self._control_value("white_balance_automatic") != 0
            for name in ("combo", "slider", "spin"):
                widget = widgets.get(name)
                if widget is not None:
                    widget.setDisabled(disabled)

    def _control_value(self, key: str) -> int | None:
        widgets = self._control_widgets.get(key)
        if not widgets:
            return None
        combo = widgets.get("combo")
        if combo is not None:
            return int(combo.currentData())
        spin = widgets.get("spin")
        if spin is not None:
            return int(spin.value())
        return None

    def _collect_payload(self) -> dict[str, int]:
        payload: dict[str, int] = {}
        for key, widgets in self._control_widgets.items():
            disabled = False
            for name in ("combo", "slider", "spin"):
                widget = widgets.get(name)
                if widget is not None and not widget.isEnabled():
                    disabled = True
                    break
            if disabled:
                continue
            value = self._control_value(key)
            if value is not None:
                payload[key] = value
        return payload

    def _set_busy(self, busy: bool):
        self._btn_refresh.setDisabled(busy)
        self._btn_apply.setDisabled(busy or not bool(self._controls_meta))
        self._edit_url.setDisabled(busy)

    def _set_status(self, text: str, ok: bool | None = None):
        color = SUCCESS if ok is True else DANGER if ok is False else ACCENT
        self._lbl_status.setStyleSheet(f"color: {color};")
        self._lbl_status.setText(text)

    def _wait_for_worker(self):
        if self._worker is not None and self._worker.isRunning():
            self._worker.wait(6000)


def _normalise_base_url(url: str) -> str:
    text = (url or DEFAULT_RGB_SETTINGS_URL).strip()
    if not text.startswith(("http://", "https://")):
        text = "http://" + text
    for suffix in ("/stream", "/snapshot", "/api/rgb/config"):
        if text.endswith(suffix):
            text = text[: -len(suffix)]
    return text.rstrip("/") + "/"


def _control_label(key: str, ctl: dict[str, Any]) -> str:
    zh = ctl.get("label_zh")
    en = ctl.get("label_en")
    if zh and en:
        return f"{zh} / {en}"
    return _LABEL_FALLBACK.get(key, key)


def _control_meta(ctl: dict[str, Any]) -> str:
    if ctl.get("kind") in {"menu", "bool"}:
        return ""
    return f"{ctl.get('min', '')}-{ctl.get('max', '')}"


def _combo_items(key: str, ctl: dict[str, Any]) -> list[tuple[int, str]]:
    if isinstance(ctl.get("menu"), list):
        items = []
        for item in ctl["menu"]:
            value = int(item.get("value", 0))
            zh = item.get("label_zh") or str(value)
            en = item.get("label_en") or str(value)
            items.append((value, f"{zh} / {en}"))
        return items
    if key == "white_balance_automatic":
        return [(1, "自动 / Auto"), (0, "手动 / Manual")]
    return [(1, "开启 / On"), (0, "关闭 / Off")]


def _clear_layout(layout):
    while layout.count():
        item = layout.takeAt(0)
        widget = item.widget()
        if widget is not None:
            widget.deleteLater()
