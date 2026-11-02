"""Settings layout and action regressions without camera or network access."""

import json
import os
from pathlib import Path

import numpy as np
import pytest
from PySide6.QtCore import QPoint
from PySide6.QtWidgets import QDialog, QDialogButtonBox, QFrame, QGroupBox

from multicamera.board.charuco_board import CharucoBoard, CharucoBoardConfig
from multicamera.calibration.models import (
    CameraExtrinsics,
    CameraIntrinsics,
    MultiCameraRig,
    StereoCalibResult,
    StereoPairCalibration,
)
from multicamera.streaming.stream_manager import StreamManager
from multicamera.streaming.discovery import DiscoveryWorker
from multicamera.ui.theme import configure_application_style
from multicamera.ui.widgets.config_dialog import ConfigDialog
from multicamera.ui.widgets.export_dialog import ExportDialog
from multicamera.ui.widgets.rgb_settings_dialog import RgbSettingsDialog, _RgbConfigWorker


@pytest.fixture(autouse=True)
def offline_settings(app, monkeypatch):
    configure_application_style(app)

    def forbid_request(*args, **kwargs):
        pytest.fail("Settings layout tests must not start a camera request")

    monkeypatch.setattr(_RgbConfigWorker, "start", forbid_request)
    monkeypatch.setattr(DiscoveryWorker, "start", forbid_request)
    monkeypatch.setattr(RgbSettingsDialog, "refresh_config", lambda self: None)


@pytest.fixture
def config():
    return ConfigDialog(StreamManager(), CharucoBoard(CharucoBoardConfig()))


@pytest.fixture
def export():
    intrinsics = CameraIntrinsics(
        np.array([[800., 0., 320.], [0., 800., 240.], [0., 0., 1.]]),
        np.zeros(5), (640, 480),
    )
    rotation, translation = np.eye(3), np.array([[-.1], [0.], [0.]])
    pair = StereoPairCalibration(
        "rgb", "left", "right", intrinsics, intrinsics,
        StereoCalibResult(rotation, translation, np.eye(3), np.eye(3)),
    )
    rig = MultiCameraRig(
        pairs={"rgb": pair}, reference_camera="left",
        extrinsics={
            "left": CameraExtrinsics(np.eye(3), np.zeros((3, 1))),
            "right": CameraExtrinsics(rotation, translation),
        },
    )
    return ExportDialog(rig)


@pytest.fixture
def rgb():
    dialog = RgbSettingsDialog("http://camera.test/", ["http://other.test/"])
    dialog._on_loaded({"ok": True, "device": "RGB camera", "controls": {
        "auto_exposure": {"kind": "menu", "value": 1, "menu": [
            {"value": 1, "label_zh": "手动"}, {"value": 3, "label_zh": "自动"},
        ]},
        "exposure_time_absolute": {"min": 1, "max": 10000, "value": 100},
        "white_balance_automatic": {"kind": "bool", "value": 0},
        "white_balance_temperature": {"min": 2000, "max": 10000, "value": 4500},
        "brightness": {"min": -64, "max": 64, "value": 0},
        "gain": {"min": 0, "max": 255, "value": 20},
        "contrast": {"min": 0, "max": 100, "value": 50},
    }})
    return dialog


def assert_footer(dialog, scroll, buttons):
    footer = dialog.findChild(QFrame, "actionFooter")
    assert footer is not None and footer.isVisible()
    assert not scroll.isAncestorOf(footer)
    assert scroll.mapTo(dialog, scroll.rect().bottomLeft()).y() < footer.y()
    for button in buttons:
        assert button.isVisible()
        assert footer.isAncestorOf(button)
        top_left = button.mapTo(dialog, QPoint(0, 0))
        bottom_right = button.mapTo(dialog, button.rect().bottomRight())
        assert dialog.rect().contains(top_left)
        assert dialog.rect().contains(bottom_right)
    assert scroll.horizontalScrollBar().maximum() == 0


def screenshot(dialog, tmp_path, name):
    directory = Path(os.environ.get("MULTICAMERA_UI_QA_DIR", tmp_path))
    directory.mkdir(parents=True, exist_ok=True)
    assert dialog.grab().save(str(directory / f"{name}.png"))


@pytest.mark.parametrize("size", [(1080, 760), (900, 480)])
def test_config_short_pages_keep_actions_and_next_step(config, app, size, tmp_path):
    config._add_row()
    config.focus_auxiliary()
    config.resize(*size)
    config.show()
    for page in range(3):
        config._tabs.setCurrentIndex(page)
        app.processEvents()
        scroll = config._tabs.widget(page)
        assert_footer(config, scroll, [
            config._buttons.button(QDialogButtonBox.Save),
            config._buttons.button(QDialogButtonBox.Cancel),
        ])
        assert "下一步" in config._lbl_next_step.text()
        assert config._btn_next_page.isVisible() == (page != 2)
        if size[1] == 480 and page == 2:
            assert scroll.verticalScrollBar().maximum() > 0
        screenshot(config, tmp_path, f"settings-config-{page}-{size[1]}")


def test_config_next_only_navigates_and_cancel_discards(config, app):
    original = config._proposed_configuration()
    config.show()
    app.processEvents()
    config._btn_next_page.click()
    assert config._tabs.currentIndex() == 2
    assert config._proposed_configuration() == original
    config._spin_sq_len.setValue(50)
    assert config._buttons.button(QDialogButtonBox.Save).text() == "应用配置"
    config._buttons.button(QDialogButtonBox.Cancel).click()
    assert config.result() == QDialog.Rejected
    assert config.board.config.square_length == .04
    assert not config._stream_manager.camera_ids


def test_config_discovery_toolbar_stays_compact(config, app):
    config.show()
    app.processEvents()
    assert config._btn_scan.width() < 200
    assert config._btn_scan.width() < config.width() / 4


@pytest.mark.parametrize("size", [(680, 540), (560, 360)])
def test_export_short_window_keeps_save_visible(export, app, size, tmp_path):
    export.resize(*size)
    export.show()
    app.processEvents()
    assert_footer(export, export._scroll, [
        export._buttons.button(QDialogButtonBox.Save),
        export._buttons.button(QDialogButtonBox.Cancel),
    ])
    assert "选择保存位置" in export._lbl_next_step.text()
    assert not export._error.isVisible()
    if size[1] == 360:
        assert export._scroll.verticalScrollBar().maximum() > 0
    screenshot(export, tmp_path, f"settings-export-{size[1]}")


def test_export_scope_format_validation_and_protocol_unchanged(export, tmp_path):
    standard = export._format.findData("standard")
    assert not export._format.model().item(standard).isEnabled()
    export._scope.setCurrentIndex(next(
        index for index in range(export._scope.count())
        if export._scope.itemData(index) == ("pair", "rgb")
    ))
    assert export._format.model().item(standard).isEnabled()
    export._format.setCurrentIndex(export._format.findData("json"))
    output = export.export_to(tmp_path / export.suggested_filename)
    assert set(json.loads(output.read_text())["pairs"]) == {"rgb"}
    export._rig.pairs["rgb"].left_intrinsics.dist_coeffs = np.array([0., 0., 0., 0., .1])
    export._scope.setCurrentIndex(0)
    export._format.setCurrentIndex(export._format.findData("kalibr"))
    assert "高阶畸变" in export._error.text()
    assert not export._buttons.button(QDialogButtonBox.Save).isEnabled()


@pytest.mark.parametrize("size", [(620, 760), (520, 360)])
def test_rgb_short_window_scrolls_with_fixed_actions(rgb, app, size, tmp_path):
    rgb.resize(*size)
    rgb.show()
    app.processEvents()
    assert_footer(rgb, rgb._scroll, [rgb._btn_apply, rgb._btn_close])
    assert rgb._scroll.verticalScrollBar().maximum() > 0
    assert "不撤销已应用" in rgb._lbl_next_step.text()
    assert rgb._btn_apply.property("class") == "primary"
    for widget in rgb._control_widgets.values():
        assert rgb._scroll.isAncestorOf(widget["row"])
    screenshot(rgb, tmp_path, f"settings-rgb-{size[1]}-top")
    before = rgb.findChild(QFrame, "actionFooter").geometry()
    rgb._scroll.verticalScrollBar().setValue(rgb._scroll.verticalScrollBar().maximum())
    app.processEvents()
    assert rgb.findChild(QFrame, "actionFooter").geometry() == before
    screenshot(rgb, tmp_path, f"settings-rgb-{size[1]}")


def test_rgb_apply_preserves_payload_dependencies_and_sync(rgb, app, monkeypatch):
    rgb.show()
    app.processEvents()
    requests = []
    monkeypatch.setattr(rgb, "_start_worker", lambda *args: requests.append(args))
    rgb._control_widgets["exposure_time_absolute"]["spin"].setValue(200)
    rgb._btn_apply.click()
    assert requests[-1][0]["exposure_time_absolute"] == 200
    assert requests[-1][2] == ["http://camera.test/", "http://other.test/"]
    rgb._control_widgets["auto_exposure"]["combo"].setCurrentIndex(1)
    rgb._control_widgets["white_balance_automatic"]["combo"].setCurrentIndex(0)
    rgb._chk_sync.setChecked(False)
    rgb._btn_apply.click()
    payload, _, urls = requests[-1]
    assert "exposure_time_absolute" not in payload
    assert "white_balance_temperature" not in payload
    assert urls == ["http://camera.test/"]
    rgb._set_busy(True)
    assert not rgb._btn_apply.isEnabled() and not rgb._btn_refresh.isEnabled()
    rgb._set_busy(False)
    assert rgb._btn_apply.isEnabled() and rgb._btn_refresh.isEnabled()
    rgb._btn_close.click()
    assert len(requests) == 2
    assert rgb.result() == QDialog.Accepted


def test_rgb_empty_groups_hidden_and_long_status_wraps(rgb, app):
    rgb._controls_meta = {"brightness": {"min": 0, "max": 100, "value": 50}}
    rgb._rebuild_controls()
    rgb._set_status("读取失败，请检查目标相机服务。" * 8, ok=False)
    rgb.resize(520, 360)
    rgb.show()
    app.processEvents()
    groups = {group.title(): group for group in rgb._body.findChildren(QGroupBox)}
    assert not groups["更多参数"].isVisible()
    assert rgb._lbl_status.wordWrap()
    assert_footer(rgb, rgb._scroll, [rgb._btn_apply, rgb._btn_close])


@pytest.mark.parametrize("surface", ["config", "export", "rgb"])
def test_large_font_and_long_names_fit_short_windows(surface, request, app, tmp_path):
    dialog = request.getfixturevalue(surface)
    dialog.setStyleSheet("QWidget {font-size:16px;} QComboBox {font-size:17px;}")
    if surface == "config":
        dialog.resize(900, 480)
        dialog.focus_auxiliary()
        scroll = dialog._tabs.currentWidget()
        buttons = [dialog._buttons.button(QDialogButtonBox.Save), dialog._btn_next_page]
    elif surface == "export":
        dialog.resize(560, 360)
        dialog._scope.setItemText(0, "全部标定结果：" + "RGB_reference_camera_" * 10)
        scroll = dialog._scroll
        buttons = [dialog._buttons.button(QDialogButtonBox.Save)]
    else:
        dialog.resize(520, 360)
        dialog._lbl_targets.setText("同步目标：http://camera.test/" + "long_camera_path/" * 12)
        dialog._lbl_device.setText("设备：" + "RGB_camera_" * 12)
        scroll = dialog._scroll
        buttons = [dialog._btn_apply, dialog._btn_close]
    dialog.show()
    app.processEvents()
    assert dialog.height() == (480 if surface == "config" else 360)
    assert_footer(dialog, scroll, buttons)
    screenshot(dialog, tmp_path, f"settings-{surface}-large-font")
