"""Presentation-only workbench changes preserve calibration and storage contracts."""

import numpy as np
import pytest
from PySide6.QtTest import QSignalSpy
from PySide6.QtWidgets import QFrame, QLabel, QVBoxLayout

from multicamera.board.charuco_board import CharucoBoard, CharucoBoardConfig
from multicamera.calibration.models import (
    CameraIntrinsics, MultiCameraRig, StereoCalibResult, StereoPairCalibration,
)
from multicamera.io.session import CalibrationSession
from multicamera.streaming.stream_manager import CameraConfig, StereoPairConfig, StreamManager
from multicamera.ui.theme import configure_application_style
from multicamera.ui.widgets.calib_panel import CalibrationPanel


@pytest.fixture
def panel(app, tmp_path):
    configure_application_style(app)
    manager = StreamManager()
    manager.add_stereo_pair(StereoPairConfig(
        'rgb', CameraConfig('left', ''), CameraConfig('right', '')))
    board = CharucoBoard(CharucoBoardConfig())
    widget = CalibrationPanel(manager, board)
    widget.refresh_pairs()
    widget.set_session(CalibrationSession.create(
        tmp_path, board, list(manager.stereo_pairs.values())))
    widget.resize(1280, 800)
    widget.show()
    app.processEvents()
    yield widget
    widget._auto_timer.stop()
    widget._readiness_timer.stop()
    widget._aux_readiness_timer.stop()
    manager.stop_all()
    widget.close()


def make_rig(rms=.15):
    intrinsics = CameraIntrinsics(
        np.array([[800., 0., 320.], [0., 800., 240.], [0., 0., 1.]]),
        np.zeros((1, 5)), (640, 480), rms_error=rms)
    stereo = StereoCalibResult(
        np.eye(3), np.array([[-.1], [0.], [0.]]), np.eye(3), np.eye(3), rms_error=rms)
    pair = StereoPairCalibration('rgb', 'left', 'right', intrinsics, intrinsics, stereo)
    return MultiCameraRig(pairs={'rgb': pair}, reference_camera='left')


def test_next_step_distinguishes_readiness_from_calibration_success(panel, monkeypatch):
    counts = dict(panel._pair_capture_press_count)
    panel.show_workflow('calibration')
    assert '继续采集或导入' in panel._lbl_workflow_next.text()
    assert '数据就绪 0 / 1' in panel._lbl_data_readiness.text()
    assert not panel._btn_calib_all.isEnabled()
    monkeypatch.setattr(panel, '_pair_data_ready', lambda name: name == 'rgb')
    panel._refresh_workflow_guidance()
    assert '数据就绪 1 / 1' in panel._lbl_data_readiness.text()
    assert '开始标定就绪组' in panel._lbl_workflow_next.text()
    panel.show_workflow('capture')
    assert '前往标定' in panel._lbl_workflow_next.text()
    assert '不代表精度通过' in panel._lbl_workflow_next.text()
    assert panel._pair_capture_press_count == counts
    assert panel._rig is None


def test_next_step_prioritizes_real_save_and_worker_blockers(panel, monkeypatch):
    monkeypatch.setattr(panel, '_pair_data_ready', lambda name: True)
    panel._pending_save_tokens.add('pending')
    panel._refresh_workflow_guidance()
    assert '等待原图保存完成' in panel._lbl_workflow_next.text()
    panel._pending_save_tokens.clear()
    panel._failed_save_jobs['failed'] = object()
    panel._refresh_workflow_guidance()
    assert '重试原图保存' in panel._lbl_workflow_next.text()
    panel._failed_save_jobs.clear()
    panel._white_balance_guard.busy = True
    panel._refresh_workflow_guidance()
    assert '等待白平衡' in panel._lbl_workflow_next.text()
    panel._white_balance_guard.busy = False
    panel._calib_worker = object()
    panel._refresh_workflow_guidance()
    assert '等待标定计算完成' in panel._lbl_workflow_next.text()
    panel._calib_worker = None


def test_result_next_step_and_navigation_preserve_native_signals(panel, tmp_path):
    panel.show_workflow('results')
    assert not panel._btn_export_results.isEnabled()
    assert '前往标定' in panel._lbl_workflow_next.text()
    panel.set_result_model(make_rig())
    assert '再导出' in panel._lbl_workflow_next.text()
    assert not panel._btn_open_results.isEnabled()
    directory = tmp_path / 'results'
    panel.set_result_storage(directory, [directory / 'rig.json'])
    assert '已保存 1 个' in panel._lbl_result_storage.text()
    assert '打开结果目录' in panel._lbl_workflow_next.text()
    export = QSignalSpy(panel.export_requested)
    opened = QSignalSpy(panel.open_results_requested)
    panel._btn_export_results.click()
    panel._btn_open_results.click()
    assert export.count() == 1
    assert opened.count() == 1
    panel.set_result_storage(directory, [directory / 'rig.json'], ['rig_kalibr.yaml: unsupported'])
    assert '文件生成失败原因' in panel._lbl_workflow_next.text()
    assert '部分格式未生成' in panel._lbl_result_storage.text()
    assert panel._btn_export_results.isEnabled()
    panel._btn_back_to_calibration.click()
    assert panel.workflow_phase == 'calibration'
    panel._btn_back_to_capture.click()
    assert panel.workflow_phase == 'capture'
    assert panel._active_step == 0


def test_result_risks_and_outdated_data_do_not_imply_success(panel):
    panel.set_result_model(make_rig(rms=8.))
    panel.show_workflow('results')
    assert '补采后重新标定' in panel._lbl_workflow_next.text()
    assert panel._btn_export_results.isEnabled()
    panel._clear_calibration_results()
    panel._results_outdated = True
    panel._refresh_result_summary()
    assert '重新计算' in panel._lbl_workflow_next.text()
    assert not panel._btn_export_results.isEnabled()


def test_auxiliary_next_step_uses_existing_prerequisite_gate(panel):
    panel.show_workflow('calibration', True)
    assert panel._aux_calibration_problem() in panel._lbl_workflow_next.text()
    assert '先完成 RGB' in panel._lbl_workflow_next.text()
    assert not panel._btn_calib_aux.isEnabled()
    panel._btn_back_to_aux_capture.click()
    assert panel.workflow_phase == 'capture'
    assert panel._active_step == 2


def test_compact_import_proxy_keeps_signal_and_enabled_state(panel):
    panel._btn_import.clicked.disconnect(panel._on_import_images)
    clicked = QSignalSpy(panel._btn_import.clicked)
    panel.set_capture_footer_compact(True)
    assert not panel._btn_import.isVisible()
    assert panel._btn_import.isEnabled()
    panel._btn_import.click()
    assert clicked.count() == 1
    panel._btn_import.setEnabled(False)
    panel.set_capture_footer_compact(False)
    assert panel._btn_import.isVisible()
    assert not panel._btn_import.isEnabled()


def test_receipt_can_be_moved_out_of_footer_without_being_reclaimed(panel, app):
    receipt = QLabel('原图待验收')
    panel.set_stereo_receipt(receipt)
    preview_host = QFrame()
    QVBoxLayout(preview_host).addWidget(receipt)
    for phase in ('capture', 'calibration', 'results', 'capture'):
        panel.set_capture_footer_compact(phase == 'capture')
        panel.show_workflow(phase)
        app.processEvents()
        assert receipt.parentWidget() is preview_host
        assert receipt.text() == '原图待验收'


@pytest.mark.parametrize('width', [360, 960, 1000, 1280, 1440])
def test_footer_reflows_without_clipping_required_controls(panel, app, width, tmp_path):
    panel.resize(width, 680)
    panel.set_capture_footer_compact(width >= 900)
    for _ in range(4):
        app.processEvents()
    footer = panel._action_footer
    for widget in (panel._lbl_raw_save, panel._check_wb, panel._combo_wb_mode,
                   panel._btn_capture, panel._btn_auto, panel._btn_undo,
                   panel._btn_to_calibration):
        assert widget.isVisible()
        assert footer.rect().contains(widget.mapTo(footer, widget.rect().topLeft()))
        assert footer.rect().contains(widget.mapTo(footer, widget.rect().bottomRight()))
        assert widget.height() >= widget.minimumHeight()
        if widget is not panel._lbl_raw_save:
            assert widget.height() >= widget.minimumSizeHint().height()
    assert panel._lbl_raw_save.visible_text().replace('\n', '') == panel._lbl_raw_save.text()
    assert panel._btn_import.isVisible() == (width < 900)
    assert panel._inspector_scroll.horizontalScrollBar().maximum() == 0
    panel.grab().save(str(tmp_path / f'capture-workbench-{width}.png'))
    if width >= 900:
        assert footer.height() <= 145, [
            (item.geometry(), item.sizeHint()) for item in
            (footer, panel._footer_controls, panel._footer_info, panel._footer_pages)]
        assert panel._capture_primary_row.indexOf(panel._btn_to_calibration) >= 0
        assert panel._capture_primary_row.indexOf(panel._btn_undo) >= 0
    for phase in ('calibration', 'results', 'capture'):
        panel.show_workflow(phase)
        app.processEvents()
        assert panel._lbl_workflow_next.isVisible() == (phase != 'capture')
        assert panel._footer_info.isVisible() == (phase == 'capture')
        panel.grab().save(str(tmp_path / f'{phase}-workbench-{width}.png'))


@pytest.mark.parametrize('phase', ['capture', 'calibration', 'results'])
def test_auxiliary_footer_also_fits_full_width_short_window(panel, app, phase):
    panel.resize(1000, 680)
    panel.set_capture_footer_compact(True)
    panel.show_workflow(phase, True)
    for _ in range(4):
        app.processEvents()
    assert panel._action_footer.height() <= 145
    assert panel._inspector_scroll.horizontalScrollBar().maximum() == 0
