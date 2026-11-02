"""RGB camera setting dialog for FusionSensor-style HTTP camera services."""

from __future__ import annotations

import json
import logging
from typing import Any
from urllib.parse import urljoin, urlsplit, urlunsplit

import aiohttp
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ..theme import ACCENT, DANGER, SUCCESS, TEXT_DIM, app_icon

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

    def __init__(
        self,
        base_url: str,
        payload: dict[str, int] | None = None,
        apply_urls: list[str] | None = None,
        parent=None,
    ):
        super().__init__(parent)
        self.base_url = base_url
        self.payload = payload
        self.apply_urls = apply_urls or [base_url]

    def run(self):
        try:
            data = self._request_config()
            if not data.get("ok"):
                raise RuntimeError(data.get("error") or "RGB settings API returned ok=false")
            self.loaded.emit(data)
        except Exception as exc:
            self.failed.emit(str(exc))

    def _request_config(self) -> dict[str, Any]:
        import asyncio

        return asyncio.run(self._request_config_async())

    async def _request_config_async(self) -> dict[str, Any]:
        timeout = aiohttp.ClientTimeout(total=5.0, connect=3.0, sock_connect=3.0, sock_read=5.0)
        connector = aiohttp.TCPConnector(family=0, force_close=True)
        async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
            if self.payload is None:
                endpoint = urljoin(_normalise_base_url(self.base_url), "api/rgb/config")
                async with session.get(endpoint) as resp:
                    body = await resp.text()
                    if resp.status >= 400:
                        raise RuntimeError(f"HTTP {resp.status}")
                return json.loads(body)
            else:
                results: list[dict[str, Any]] = []
                targets = _unique_urls(self.apply_urls)
                for url in targets:
                    endpoint = urljoin(_normalise_base_url(url), "api/rgb/config")
                    async with session.post(endpoint, json=self.payload) as resp:
                        body = await resp.text()
                        if resp.status >= 400:
                            raise RuntimeError(f"{_normalise_base_url(url)} HTTP {resp.status}: {body[:200]}")
                    data = json.loads(body)
                    if not data.get("ok"):
                        raise RuntimeError(
                            f"{_normalise_base_url(url)}: {data.get('error') or 'ok=false'}"
                        )
                    results.append(data)
                data = dict(results[0]) if results else {"ok": True}
                data["applied_urls"] = targets
                return data


class RgbSettingsDialog(QDialog):
    """Edit RGB camera controls exposed by /api/rgb/config."""

    settings_applied = Signal(list)

    def __init__(
        self,
        base_url: str = DEFAULT_RGB_SETTINGS_URL,
        sync_urls: list[str] | None = None,
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("RGB 相机参数")
        self.setMinimumSize(520, 360)
        self.resize(620, 760)
        self._sync_urls = _unique_urls([base_url, *(sync_urls or [])])
        self._last_applied_urls: list[str] = []
        self._controls_meta: dict[str, dict[str, Any]] = {}
        self._control_widgets: dict[str, dict[str, Any]] = {}
        self._worker: _RgbConfigWorker | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QFrame.NoFrame)
        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(16, 16, 16, 16)
        content_layout.setSpacing(16)
        connection = QGroupBox("目标与同步")
        connection_layout = QVBoxLayout(connection)
        connection_layout.setContentsMargins(12, 12, 12, 12)
        connection_layout.setSpacing(8)

        url_row = QHBoxLayout()
        self._edit_url = QLineEdit(_normalise_base_url(base_url))
        self._edit_url.setPlaceholderText(DEFAULT_RGB_SETTINGS_URL)
        url_row.addWidget(QLabel("目标相机:"), 0)
        url_row.addWidget(self._edit_url, 1)
        self._btn_refresh = QPushButton("刷新")
        self._btn_refresh.setIcon(app_icon("rotate-ccw"))
        self._btn_refresh.setAutoDefault(False)
        self._btn_refresh.clicked.connect(self.refresh_config)
        url_row.addWidget(self._btn_refresh)
        connection_layout.addLayout(url_row)

        self._chk_sync = QCheckBox("同步应用到已配置 RGB 左/右")
        self._chk_sync.setChecked(len(self._sync_urls) > 1)
        self._chk_sync.setVisible(len(self._sync_urls) > 1)
        connection_layout.addWidget(self._chk_sync)

        self._lbl_targets = QLabel(self._format_targets())
        self._lbl_targets.setWordWrap(True)
        self._lbl_targets.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self._lbl_targets.setTextFormat(Qt.PlainText)
        self._lbl_targets.setStyleSheet(f"color: {TEXT_DIM};")
        self._lbl_targets.setVisible(len(self._sync_urls) > 1)
        connection_layout.addWidget(self._lbl_targets)

        self._lbl_device = QLabel("设备: --")
        self._lbl_device.setWordWrap(True)
        self._lbl_device.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self._lbl_device.setTextFormat(Qt.PlainText)
        self._lbl_device.setStyleSheet(f"color: {TEXT_DIM};")
        connection_layout.addWidget(self._lbl_device)
        content_layout.addWidget(connection)

        self._body = QWidget()
        self._body_layout = QVBoxLayout(self._body)
        self._body_layout.setContentsMargins(0, 0, 0, 0)
        self._body_layout.setSpacing(16)
        content_layout.addWidget(self._body, 1)
        self._scroll.setWidget(content)
        layout.addWidget(self._scroll, 1)

        footer = QFrame()
        footer.setObjectName("actionFooter")
        actions = QVBoxLayout(footer)
        actions.setContentsMargins(16, 12, 16, 12)
        actions.setSpacing(8)
        self._lbl_status = QLabel("点击刷新读取相机参数")
        self._lbl_status.setWordWrap(True)
        self._lbl_status.setTextFormat(Qt.PlainText)
        self._lbl_status.setStyleSheet(f"color: {TEXT_DIM}; background:transparent;")
        actions.addWidget(self._lbl_status)
        self._lbl_next_step = QLabel("下一步：应用到相机后返回采集；关闭不撤销已应用的参数。")
        self._lbl_next_step.setWordWrap(True)
        self._lbl_next_step.setStyleSheet(f"color: {TEXT_DIM}; background:transparent;")
        actions.addWidget(self._lbl_next_step)
        bottom = QHBoxLayout()
        bottom.addStretch()
        self._btn_apply = QPushButton("应用到相机")
        self._btn_apply.setIcon(app_icon("check"))
        self._btn_apply.setProperty("class", "primary")
        self._btn_apply.setDefault(True)
        self._btn_apply.clicked.connect(self.apply_config)
        self._btn_apply.setEnabled(False)
        self._btn_close = QPushButton("关闭")
        self._btn_close.setIcon(app_icon("x"))
        self._btn_close.setAutoDefault(False)
        self._btn_close.setToolTip("关闭窗口，放弃尚未应用的修改；已应用的相机参数保持不变")
        self._btn_close.clicked.connect(self.accept)
        bottom.addWidget(self._btn_close)
        bottom.addWidget(self._btn_apply)
        actions.addLayout(bottom)
        layout.addWidget(footer)

        self.refresh_config()

    def refresh_config(self):
        self._start_worker(None, "读取 RGB 参数...")

    def apply_config(self):
        payload = self._collect_payload()
        if not payload:
            self._set_status("没有可应用的参数", ok=False)
            return
        urls = self._apply_urls()
        if len(urls) > 1:
            self._start_worker(payload, f"同步应用 RGB 参数到 {len(urls)} 路...", urls)
        else:
            self._start_worker(payload, "应用 RGB 参数...", urls)

    def accept(self):
        self._wait_for_worker()
        super().accept()

    def reject(self):
        self._wait_for_worker()
        super().reject()

    def closeEvent(self, event):
        self._wait_for_worker()
        super().closeEvent(event)

    def _start_worker(
        self,
        payload: dict[str, int] | None,
        status: str,
        apply_urls: list[str] | None = None,
    ):
        if self._worker is not None and self._worker.isRunning():
            return
        self._set_status(status)
        self._set_busy(True)
        self._worker = _RgbConfigWorker(
            self._edit_url.text().strip(),
            payload,
            apply_urls=apply_urls,
            parent=self,
        )
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
        applied_urls = data.get("applied_urls") or []
        if applied_urls:
            self._last_applied_urls = [_normalise_base_url(u) for u in applied_urls]
            self._set_status(f"RGB 参数已同步应用到 {len(self._last_applied_urls)} 路", ok=True)
        else:
            self._set_status("RGB 参数已就绪", ok=True)
        if self._worker is not None and self._worker.payload is not None:
            self.settings_applied.emit(self._last_applied_urls or [self._edit_url.text()])
        logger.info(
            "RGB settings loaded: url=%s device=%s applied_urls=%s",
            self._edit_url.text(),
            data.get("device"),
            ",".join(self._last_applied_urls),
        )

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
        quick_layout.setContentsMargins(12, 12, 12, 12)
        quick_layout.setSpacing(12)
        more = QGroupBox("更多参数")
        more_layout = QVBoxLayout(more)
        more_layout.setContentsMargins(12, 12, 12, 12)
        more_layout.setSpacing(12)

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
        quick.setVisible(bool(handled))
        more.setVisible(bool(set(self._controls_meta) - handled))
        self._body_layout.addStretch(1)
        self._update_dependencies()

    def _add_control(self, parent_layout: QVBoxLayout, key: str, ctl: dict[str, Any]):
        row = QWidget()
        row_layout = QVBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(3)

        head = QHBoxLayout()
        label = QLabel(_control_label(key, ctl))
        label.setWordWrap(True)
        label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        label.setTextFormat(Qt.PlainText)
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
        self._chk_sync.setDisabled(busy)

    def _set_status(self, text: str, ok: bool | None = None):
        color = SUCCESS if ok is True else DANGER if ok is False else ACCENT
        self._lbl_status.setStyleSheet(f"color: {color}; background:transparent;")
        self._lbl_status.setText(text)

    def _wait_for_worker(self):
        if self._worker is not None and self._worker.isRunning():
            self._worker.wait(6000)

    def _apply_urls(self) -> list[str]:
        edited = _normalise_base_url(self._edit_url.text())
        if self._chk_sync.isVisible() and self._chk_sync.isChecked():
            return _unique_urls([edited, *self._sync_urls])
        return [edited]

    def _format_targets(self) -> str:
        if len(self._sync_urls) <= 1:
            return ""
        return "同步目标: " + "，".join(self._sync_urls)


def _normalise_base_url(url: str) -> str:
    text = (url or DEFAULT_RGB_SETTINGS_URL).strip()
    if not text.startswith(("http://", "https://")):
        text = "http://" + text
    parts = urlsplit(text)
    path = parts.path.rstrip("/")
    for suffix in ("/stream", "/snapshot", "/api/rgb/config"):
        if path.endswith(suffix):
            path = path[: -len(suffix)]
            break
    return urlunsplit((parts.scheme, parts.netloc, path.rstrip("/") + "/", "", ""))


def _unique_urls(urls: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for url in urls:
        normalised = _normalise_base_url(url)
        if normalised in seen:
            continue
        seen.add(normalised)
        result.append(normalised)
    return result


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
