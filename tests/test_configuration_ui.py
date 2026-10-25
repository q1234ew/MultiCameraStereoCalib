"""Configuration layout, unit conversion, validation and discovery selection."""

import os
from pathlib import Path

import pytest
from PySide6.QtWidgets import QDialogButtonBox, QMessageBox, QFormLayout

from multicamera.board.charuco_board import CharucoBoard, CharucoBoardConfig, ArucoDictType
from multicamera.calibration.planar import PatternBoardConfig, PatternType
from multicamera.streaming.discovery import DiscoveredService
from multicamera.streaming.stream_manager import StreamManager, CameraConfig, StereoPairConfig
from multicamera.ui.theme import configure_application_style
from multicamera.ui.widgets.config_dialog import ConfigDialog


@pytest.fixture
def dialog(app):
    configure_application_style(app)
    manager = StreamManager()
    dlg = ConfigDialog(manager, CharucoBoard(CharucoBoardConfig()))
    yield dlg
    dlg.close()
    manager.stop_all()


def service(host, eye, kind="rgb", port=8080):
    return DiscoveredService(host, port, f"/{eye}", f"{kind}_{eye}", stream_type=kind, eye=eye)


def test_default_tab_and_unchanged_configuration(dialog):
    assert [dialog._tabs.tabText(i) for i in range(3)] == ["相机连接", "融合相机", "标定板"]
    assert dialog._tabs.currentIndex() == 0
    assert "未修改" in dialog._lbl_config_status.text()
    assert dialog._buttons.button(QDialogButtonBox.Save).isEnabled()
    assert not dialog._btn_remove.isEnabled()
    assert not dialog._btn_swap_sources.isEnabled()
    assert not dialog._btn_add_selected_pair.isEnabled()
    assert not dialog._btn_use_aux_service.isEnabled()


def test_fusion_discovery_distinguishes_unscanned_and_no_matches(dialog):
    dialog._combo_fusion_mode.setCurrentIndex(1)
    assert '尚未扫描' in dialog._lbl_aux_discovery.text()
    assert dialog._btn_discover_aux.isEnabled()
    dialog._on_scan_finished([])
    assert '扫描完成' in dialog._lbl_aux_discovery.text()
    assert '未发现' in dialog._lbl_aux_discovery.text()


def test_board_millimetres_roundtrip_and_cancel_does_not_apply(dialog):
    assert dialog._spin_sq_len.value() == 40
    assert dialog._spin_mk_len.value() == 30
    dialog._spin_sq_len.setValue(50)
    cfg, _, _, _ = dialog._proposed_configuration()
    assert cfg.square_length == .05
    assert cfg.marker_length == .03
    assert "有未应用修改" in dialog._lbl_config_status.text()
    dialog.reject()
    assert dialog.board.config.square_length == .04


def test_save_reopen_preserves_units_and_noop_stream_state(dialog, monkeypatch):
    calls = []
    monkeypatch.setattr(dialog._stream_manager, "stop_all", lambda: calls.append(True))
    dialog._spin_sq_len.setValue(45)
    dialog._on_accept()
    assert dialog.board.config.square_length == .045
    assert not calls
    reopened = ConfigDialog(dialog._stream_manager, dialog.board)
    try:
        assert reopened._spin_sq_len.value() == 45
        assert "未修改" in reopened._lbl_config_status.text()
    finally:
        reopened.close()


def test_auxiliary_physical_squares_and_point_modes_are_equivalent(dialog):
    dialog.focus_auxiliary()
    assert dialog._tabs.currentIndex() == 1
    assert dialog._spin_aux_cols.value() == 7
    assert dialog._spin_aux_rows.value() == 5
    expected = PatternBoardConfig.ir_chessboard()
    assert dialog._aux_board_values() == expected
    dialog._combo_aux_count_mode.setCurrentIndex(1)
    assert (dialog._spin_aux_cols.value(), dialog._spin_aux_rows.value()) == (6, 4)
    assert dialog._aux_board_values() == expected
    dialog._combo_aux_count_mode.setCurrentIndex(0)
    assert dialog._aux_board_values() == expected
    dialog._combo_aux_board.setCurrentIndex(1)
    assert not dialog._combo_aux_count_mode.isEnabled()
    assert dialog._aux_board_values().cols == 6
    dialog._combo_aux_board.setCurrentIndex(0)
    assert dialog._aux_board_values() == expected


def test_aux_import_values_preserve_detected_counts_in_either_mode(dialog):
    dialog.focus_auxiliary()
    for mode in (0, 1):
        dialog._combo_aux_count_mode.setCurrentIndex(mode)
        board = PatternBoardConfig(8, 5, .025, PatternType.CIRCLES_GRID, True)
        dialog._set_aux_board_values(board)
        assert dialog._aux_board_values() == board
        assert (dialog._spin_aux_cols.value(), dialog._spin_aux_rows.value()) == (8, 5)
        dialog._btn_aux_board_preset.click()
        assert dialog._aux_board_values() == PatternBoardConfig.ir_chessboard()


@pytest.mark.parametrize("problem", ["marker", "dictionary", "url", "same_source", "half_offline", "duplicate_name"])
def test_validation_blocks_apply_and_focuses_relevant_page(dialog, monkeypatch, problem):
    dialog._add_row()
    if problem == "marker":
        dialog._spin_mk_len.setValue(50)
    elif problem == "dictionary":
        dialog._combo_dict.setCurrentIndex(dialog._combo_dict.findData(ArucoDictType.DICT_4X4_50))
    elif problem in {"url", "same_source", "half_offline"}:
        dialog._table.item(0, 3).setText("http://device/left")
        dialog._table.item(0, 5).setText({"url": "rtsp://device/right", "same_source": "http://device/left",
                                        "half_offline": ""}[problem])
    else:
        dialog._add_row()
        dialog._table.item(1, 0).setText(dialog._table.item(0, 0).text())
    assert not dialog._buttons.button(QDialogButtonBox.Save).isEnabled()
    errors, confirmations = [], []
    monkeypatch.setattr(QMessageBox, "warning", lambda *args: errors.append(args))
    dialog.before_apply = lambda *args: confirmations.append(args) or True
    dialog._on_accept()
    assert errors and not confirmations
    assert dialog._tabs.currentIndex() == (2 if problem in {"marker", "dictionary"} else 0)
    assert not dialog._stream_manager.stereo_pairs


def test_offline_add_delete_uses_unique_names_and_selected_actions(dialog):
    dialog._add_row()
    first = dialog._table.item(0, 0).text()
    dialog._add_row()
    second = dialog._table.item(1, 0).text()
    dialog._table.selectRow(0)
    dialog._remove_row()
    dialog._add_row()
    assert dialog._table.item(1, 0).text() != second
    assert first != second
    assert dialog._btn_remove.isEnabled()
    assert dialog._btn_swap_sources.isEnabled()
    dialog._on_accept()
    assert len(dialog._stream_manager.stereo_pairs) == 2
    assert not dialog._stream_manager.camera_ids


def test_auto_pair_never_crosses_hosts_or_guesses_unknown_eyes(dialog):
    dialog._discovered = [service("a.test", "left"), service("b.test", "right"),
                          service("c.test", "unknown", port=8081), service("c.test", "unknown", port=8082)]
    dialog._auto_pair_discovered()
    assert dialog._table.rowCount() == 0
    dialog._discovered.extend([service("d.test", "left"), service("d.test", "right")])
    dialog._auto_pair_discovered()
    assert dialog._table.rowCount() == 1
    assert "d.test" in dialog._table.item(0, 3).text()
    dialog._auto_pair_discovered()
    assert dialog._table.rowCount() == 1


def test_auto_pair_leaves_ambiguous_same_host_services_for_manual_selection(dialog):
    dialog._discovered = [service("a.test", "left", port=8080), service("a.test", "left", port=8081),
                          service("a.test", "right", port=8082)]
    dialog._auto_pair_discovered()
    assert dialog._table.rowCount() == 0


def test_new_services_preserve_user_selection_and_auxiliary_id(dialog):
    dialog._discovered = [service("a.test", "left"), service("b.test", "left"),
                          service("b.test", "right"), service("a.test", "unknown", "ir")]
    dialog._refresh_camera_dropdowns()
    dialog._combo_left_camera.setCurrentIndex(1)
    selected = dialog._combo_left_camera.currentData().key
    dialog._on_service_found(service("c.test", "left"))
    assert dialog._combo_left_camera.currentData().key == selected
    dialog.focus_auxiliary()
    dialog._edit_aux_id.setText("stable_ir_id")
    dialog._use_selected_aux_service()
    assert dialog._edit_aux_id.text() == "stable_ir_id"
    assert dialog._edit_aux_url.text() == "http://a.test:8080/unknown"


def test_invalid_existing_url_can_be_repaired_without_constructor_failure(app):
    manager = StreamManager()
    manager.add_virtual_stereo_pair(StereoPairConfig("old", CameraConfig("L", "http://"), CameraConfig("R", "http://")))
    dlg = ConfigDialog(manager, CharucoBoard(CharucoBoardConfig()))
    try:
        assert not dlg._buttons.button(QDialogButtonBox.Save).isEnabled()
        dlg._table.item(0, 3).setText("")
        dlg._table.item(0, 5).setText("")
        assert dlg._buttons.button(QDialogButtonBox.Save).isEnabled()
    finally:
        dlg.close()


@pytest.mark.parametrize("size", [(1080, 760), (900, 640)])
def test_configuration_pages_fit_and_footer_stays_visible(dialog, app, size, tmp_path):
    dialog.focus_auxiliary()
    dialog._add_row()
    dialog.resize(*size)
    dialog.show()
    directory = Path(os.environ.get("MULTICAMERA_UI_QA_DIR", tmp_path))
    directory.mkdir(parents=True, exist_ok=True)
    for page in range(3):
        dialog._tabs.setCurrentIndex(page)
        app.processEvents()
        scroll = dialog._tabs.widget(page)
        assert scroll.horizontalScrollBar().maximum() == 0
        button = dialog._buttons.button(QDialogButtonBox.Save)
        assert button.isVisible()
        assert button.mapTo(dialog, button.rect().bottomRight()).y() <= dialog.height()
        dialog.grab().save(str(directory / f"config-{page}-{size[0]}x{size[1]}.png"))


@pytest.mark.parametrize("size", [(1080, 760), (900, 640)])
def test_fusion_fields_expand_without_reference_overlap(dialog, app, size, tmp_path):
    dialog._add_row()
    reference = "rgb_left_" + "reference_camera_" * 10
    dialog._table.item(0, 2).setText(reference)
    dialog.focus_auxiliary()
    dialog._edit_aux_id.setText("fusionsensor_188_infrared")
    dialog._edit_aux_url.setText("http://192.168.31.188:8081/stream")
    dialog.resize(*size)
    dialog.show()
    app.processEvents()
    fields = (dialog._combo_fusion_mode, dialog._edit_aux_id, dialog._edit_aux_url, dialog._lbl_aux_reference)
    for form in dialog.findChildren(QFormLayout):
        assert form.fieldGrowthPolicy() == QFormLayout.AllNonFixedFieldsGrow
    for field in fields:
        assert field.width() > size[0] * .6
        assert abs(field.geometry().left() - dialog._edit_aux_url.geometry().left()) <= 1
        assert abs(field.geometry().right() - dialog._edit_aux_url.geometry().right()) <= 1
    reference_field = dialog._lbl_aux_reference
    assert reference_field.isReadOnly()
    assert reference in reference_field.toolTip()
    assert reference_field.cursorPosition() == 0
    assert reference_field.height() >= reference_field.fontMetrics().height() + 6
    assert reference_field.geometry().bottom() < dialog._btn_aux_board_page.geometry().top()
    assert dialog._edit_aux_url.geometry().bottom() < reference_field.geometry().top()
    directory = Path(os.environ.get("MULTICAMERA_UI_QA_DIR", tmp_path))
    directory.mkdir(parents=True, exist_ok=True)
    dialog.grab().save(str(directory / f"fusion-fields-{size[0]}x{size[1]}.png"))
