"""Stereo workbench hierarchy, persistent feedback and rectification navigation."""

import os
import time
from pathlib import Path

import numpy as np
import pytest
from PySide6.QtCore import QPoint

from multicamera.calibration.models import (
    CameraIntrinsics, MultiCameraRig, StereoCalibResult, StereoPairCalibration,
)
from multicamera.ui import main_window
from multicamera.io.session import CalibrationSession
from multicamera.ui.theme import configure_application_style
from test_calibration_workflow import seed_observations, wait_until
from test_multimodal_preview import LEFT, RIGHT, configure
from test_stereo_capture_ui import publish_stable, sample


@pytest.fixture
def window(app, monkeypatch, tmp_path):
    configure_application_style(app)
    monkeypatch.setattr(main_window, 'sessions_dir', lambda: tmp_path / 'sessions')
    win = main_window.MainWindow()
    yield win
    win._stream_manager.stop_all()
    win.close()


@pytest.fixture
def stereo_bench(window):
    configure(window, auxiliary=False)
    window._stereo_capture.guide_mode.setCurrentIndex(1)
    window.show()
    return window


@pytest.fixture
def rig():
    intr = CameraIntrinsics(np.array([[800., 0., 320.], [0., 800., 240.], [0., 0., 1.]]),
                            np.zeros(5), (640, 480))
    stereo = StereoCalibResult(np.eye(3), np.array([[-.1], [0.], [0.]]), np.eye(3), np.eye(3))
    pair = StereoPairCalibration('rgb', 'left', 'right', intr, intr, stereo)
    return MultiCameraRig(pairs={'rgb': pair}, reference_camera='left')


@pytest.mark.parametrize('size', [(1000, 680), (1280, 800), (1440, 1024)])
def test_acceptance_feedback_precedes_optional_schematic(stereo_bench, app, tmp_path, size):
    win = stereo_bench
    guide = win._stereo_capture
    det, frame, _ = sample(win)
    publish_stable(win, det, frame)
    for cid in (LEFT, RIGHT):
        win._stream_view._on_frame(cid, frame, time.time())
    win.resize(*size)
    for _ in range(5):
        app.processEvents()
    viewport = win._guide_scroll.viewport()
    for label in (guide.compact_summary,):
        top = label.mapTo(viewport, QPoint(0, 0))
        assert top.y() >= 0
        assert top.y() + label.height() <= viewport.height()
    assert win._calib_panel._action_footer.isAncestorOf(guide.receipt)
    assert win.rect().contains(guide.receipt.mapTo(win, guide.receipt.rect().bottomRight()))
    baseline = win._stream_view.geometry()
    guide.schematic_toggle.click()
    app.processEvents()
    assert guide.schematic_body.isHidden()
    assert win._stream_view.geometry() == baseline
    guide.schematic_toggle.click()
    app.processEvents()
    assert not guide.schematic_body.isHidden()
    directory = Path(os.environ.get('MULTICAMERA_DESIGN_QA_DIR', tmp_path))
    directory.mkdir(parents=True, exist_ok=True)
    win.grab().save(str(directory / f'stereo-design-{size[0]}.png'))


def test_import_selects_matching_model_assessment_and_honest_count(window):
    image, (left, right) = seed_observations(window)
    panel = window._calib_panel
    panel._spin_min.setValue(20)
    panel._on_import_images_ready([
        (str(i), image, image.copy(), left, right) for i in range(10)
    ])
    assert panel._combo_model_pair.currentText() == 'offline'
    assert panel._combo_pair.currentText() == 'offline'
    assert panel._spin_min.value() == 10
    assert '计算门槛' in panel._lbl_capture_count.text()
    assert '建议数量' not in panel._lbl_capture_count.text()


def test_calibration_settings_and_primary_action_hierarchy(window, app):
    seed_observations(window)
    panel = window._calib_panel
    panel.show_workflow('calibration', False)
    window.show()
    app.processEvents()
    assert panel._calibration_options.isHidden()
    panel._btn_calibration_options.click()
    assert not panel._calibration_options.isHidden()
    assert panel._pair_rows['rgb'].btn.property('class') == 'secondary'
    assert panel._btn_calib_all.property('class') == 'primary'
    assert panel._btn_calib_all.text() == '开始标定当前组'
    assert 'rgb' in panel._btn_calib_all.toolTip()


def test_calibration_summary_tracks_options_without_widening_button(window, app):
    seed_observations(window)
    panel = window._calib_panel
    panel.show_workflow('calibration', False)
    window.resize(1000, 680)
    window.show()
    app.processEvents()
    button = panel._btn_calibration_options
    width = button.minimumSizeHint().width()
    summary = panel._lbl_calibration_options_summary
    for strategy in ('standard', 'joint_refine', 'auto_validated'):
        panel._combo_calib_strategy.setCurrentIndex(panel._combo_calib_strategy.findData(strategy))
        for filtering in ('quality', 'fisheye_common', 'all'):
            panel._combo_data_filter.setCurrentIndex(panel._combo_data_filter.findData(filtering))
            app.processEvents()
            assert button.text() == '标定设置'
            assert button.minimumSizeHint().width() == width
            assert panel._calib_strategy_label().split('：')[0] in summary.text()
            assert '数据：' in summary.text()
            assert summary.isVisible() and summary.wordWrap()
            assert summary.height() >= summary.heightForWidth(summary.width())
            assert panel._inspector_scroll.horizontalScrollBar().maximum() == 0
            button.click()
            app.processEvents()
            assert not panel._calibration_options.isHidden()
            assert panel._inspector_scroll.horizontalScrollBar().maximum() == 0
            button.click()
            app.processEvents()
            assert panel._calibration_options.isHidden()
            assert summary.isVisible()


def test_busy_and_failed_rows_never_claim_ready(window):
    seed_observations(window)
    panel = window._calib_panel
    panel._active_calibration_pair = 'rgb'
    panel._sync_pair_states()
    row = panel._pair_rows['rgb']
    assert '正在计算' in row._lbl_status.text()
    assert '正在标定' in panel._btn_calib_all.text()
    panel._on_pair_calibration_failed('rgb', '校正视野未通过；需要检查模型')
    assert '标定未通过' in row._lbl_status.text()
    assert '已就绪' not in row._lbl_status.text()
    assert row.btn.text() == '重试标定'
    assert '需要检查模型' in row._lbl_status.toolTip()
    assert panel._btn_calib_all.isEnabled()
    panel._combo_calib_strategy.setCurrentIndex(1)
    assert not panel._pair_failures
    assert '标定未通过' not in row._lbl_status.text()


def test_failure_survives_output_session_binding_but_not_changed_observations(window, tmp_path):
    image, (left, _) = seed_observations(window)
    panel = window._calib_panel
    panel._on_pair_calibration_failed('rgb', '校正视野未通过')
    panel.set_session(CalibrationSession.create(tmp_path, panel._board, []))
    panel._sync_pair_states()
    assert '标定未通过' in panel._pair_rows['rgb']._lbl_status.text()
    panel._intr_cals['left'].add_detection(left, image=image)
    panel._sync_pair_states()
    assert '标定未通过' not in panel._pair_rows['rgb']._lbl_status.text()


def test_selected_frame_updates_rectification_and_zoom_viewer(window, rig, app, monkeypatch):
    seed_observations(window)
    panel = window._calib_panel
    panel._pair_calibs = rig.pairs
    monkeypatch.setattr(panel._evaluator, 'generate_rectification_preview',
                        lambda left, right, *args, **kwargs: np.hstack((left, right)))
    images = {'left': np.full((480, 640, 3), 50, np.uint8),
              'right': np.full((480, 640, 3), 100, np.uint8)}
    window._on_gallery_pair_selected('rgb 0001', images)
    assert '原图 rgb 0001' in panel._lbl_rect_context.text()
    assert panel._rect_preview_image.pixelColor(0, 0).red() == 50
    panel._btn_rect_expand.click()
    app.processEvents()
    viewer = panel._rectification_viewer
    assert viewer.isVisible()
    viewer.actual_size()
    assert viewer.image.size() == panel._rect_preview_image.size()
    viewer.zoom(1.25)
    assert viewer._scale == 1.25
    viewer.fit()
    assert viewer._fit
    images['left'] = np.full((480, 640, 3), 150, np.uint8)
    window._on_gallery_pair_selected('rgb 0002', images)
    assert viewer.isVisible() and 'rgb 0002' in viewer.context.text()
    assert panel._rect_preview_image.pixelColor(0, 0).red() == 150
    panel._clear_rect_preview()
    assert not viewer.isVisible()
    assert panel._rect_preview_image is None
    assert panel._btn_rect_expand.isHidden()
    assert viewer._pixmap.isNull()


def test_wrong_size_selection_does_not_leave_old_rectification(window, rig, monkeypatch):
    seed_observations(window)
    panel = window._calib_panel
    panel._pair_calibs = rig.pairs
    monkeypatch.setattr(panel._evaluator, 'generate_rectification_preview',
                        lambda left, right, *args, **kwargs: np.hstack((left, right)))
    image = np.zeros((480, 640, 3), np.uint8)
    panel.show_review_rectification('rgb 1', {'left': image, 'right': image})
    panel.show_review_rectification('rgb 2', {'left': image[:100], 'right': image})
    assert panel._rect_preview_image is None
    assert panel._lbl_rect_context.isHidden()
    assert panel._btn_rect_expand.isHidden()


def test_calibration_result_and_diagnostics_remain_accessible(window, app, tmp_path):
    image, _ = seed_observations(window)
    panel = window._calib_panel
    panel._imported_pair_frames = {'rgb': [(image, image.copy())]}
    panel.images_imported.emit({'left': image, 'right': image.copy()})
    panel.show_workflow('calibration', False)
    window.resize(1280, 800)
    window.show()
    app.processEvents()
    panel._btn_calib_all.click()
    wait_until(app, lambda: panel._calib_worker is None)
    assert panel._pair_calibs and panel._btn_view_results.isEnabled()
    assert panel._rect_preview_image is not None
    assert panel._btn_rect_expand.isVisible()
    directory = Path(os.environ.get('MULTICAMERA_DESIGN_QA_DIR', tmp_path))
    directory.mkdir(parents=True, exist_ok=True)
    window.grab().save(str(directory / 'stereo-calibration-result.png'))
    panel._btn_rect_expand.click()
    app.processEvents()
    panel._rectification_viewer.grab().save(str(directory / 'stereo-rectification-viewer.png'))
    panel._rectification_viewer.close()
    panel._btn_view_results.click()
    app.processEvents()
    window.grab().save(str(directory / 'stereo-result-page.png'))
    assert panel._btn_export_results.isVisible()


@pytest.mark.parametrize('size', [(1000, 680), (1280, 800), (1440, 1024)])
def test_model_evidence_and_local_maximum_are_visible_in_results(window, rig, app, size, tmp_path):
    panel = window._calib_panel
    pair = rig.pairs['rgb']
    pair.calibration_info = {
        'model_selection': {'selected': 'rational', 'reason': '标准模型边界检查失败',
            'candidates': {'rational': {'status': 'ok', 'validation': {
                'rms_px': .424, 'folds': [{}] * 5}}}},
        'rectification': {'status': 'ok', 'rms_px': .383, 'p95_abs_px': .571,
            'max_abs_px': 4.19, 'per_frame': [
                {'frame_id': '0010', 'status': 'ok', 'rms_px': .8, 'max_abs_px': 2.14},
                {'frame_id': '0013', 'status': 'ok', 'rms_px': .64, 'max_abs_px': 4.19}]},
    }
    panel.set_result_model(rig)
    panel.show_workflow('results', False)
    window.resize(*size)
    window.show()
    app.processEvents()
    label = panel._lbl_result_validation
    assert label.isVisible()
    text = label.text()
    for expected in ('有理模型', '标准模型边界检查失败', '0.424', '5 折', '4.190', '0013'):
        assert expected in text
    assert panel._inspector_scroll.horizontalScrollBar().maximum() == 0
    assert window.rect().contains(panel._btn_export_results.mapTo(
        window, panel._btn_export_results.rect().bottomRight()))
    directory = Path(os.environ.get('MULTICAMERA_DESIGN_QA_DIR', tmp_path))
    directory.mkdir(parents=True, exist_ok=True)
    window.grab().save(str(directory / f'model-evidence-{size[0]}.png'))
    panel.set_result_model(None)
    assert label.isHidden() and not label.text()


@pytest.mark.parametrize('status,expected', [('insufficient_views', '未完成留出验证'),
                                           (None, '留出验证未记录')])
def test_missing_holdouts_never_display_as_passed(window, rig, status, expected):
    pair = rig.pairs['rgb']
    pair.calibration_info['model_selection'] = {'selected': 'standard', 'candidates': {
        'standard': {'status': 'ok', 'validation_status': status}}}
    window._calib_panel.set_result_model(rig)
    text = window._calib_panel._lbl_result_validation.text()
    assert expected in text
    assert '验证通过' not in text


def test_invalid_local_errors_are_not_reported_as_zero(window, rig):
    pair = rig.pairs['rgb']
    pair.calibration_info['rectification'] = {
        'status': 'ok', 'rms_px': None, 'p95_abs_px': float('nan'), 'max_abs_px': float('inf'),
        'per_frame': [{'frame_id': 'invalid', 'status': 'ok', 'max_abs_px': float('nan')}]}
    window._calib_panel.set_result_model(rig)
    text = window._calib_panel._lbl_result_validation.text()
    assert '未记录' in text and '源帧未记录' in text
    assert '0.000' not in text and '帧 invalid' not in text
