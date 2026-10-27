"""Opt-in real fisheye calibration through the same Qt controls as the user."""

import os
from pathlib import Path

import cv2
import pytest
from PySide6.QtCore import QCoreApplication, QEvent, Qt
from PySide6.QtWidgets import QFileDialog

from multicamera.calibration.models import CameraModel
from multicamera.ui import main_window
from multicamera.ui.theme import configure_application_style
from test_real_pairs_ui import wait_for


@pytest.mark.skipif(not os.environ.get('MULTICAMERA_REAL_PAIR_DIR'), reason='Real-image opt-in test')
def test_real_fisheye_explicit_compatibility_and_diagnostic_navigation(app, monkeypatch, tmp_path):
    source = Path(os.environ['MULTICAMERA_REAL_PAIR_DIR']).resolve()
    directory = Path(os.environ.get('MULTICAMERA_UI_QA_DIR', tmp_path / 'screenshots'))
    directory.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(main_window, 'sessions_dir', lambda: tmp_path / 'sessions')
    monkeypatch.setattr(QFileDialog, 'getExistingDirectory', lambda *args: str(source))
    configure_application_style(app)
    win = main_window.MainWindow()
    panel = win._calib_panel
    monkeypatch.setattr(panel._assessor, 'assess_frame',
                        lambda *args: pytest.fail('Imported image quality must be computed by the worker'))
    win._camera_model = CameraModel.FISHEYE
    panel.set_camera_model(CameraModel.FISHEYE)
    win.resize(1280, 800)
    win.show()
    try:
        panel._btn_import.click()
        wait_for(app, lambda: not panel.has_active_background_tasks())
        original_counts = panel._stereo_cals['offline'].frame_common_corner_counts.copy()
        panel._btn_to_calibration.click()
        panel._btn_calib_all.click()
        wait_for(app, lambda: not panel.has_active_background_tasks())
        assert win._rig is None
        assert '鱼眼兼容：跨帧共同角点' in panel._log_text.toPlainText()
        assert 'matrix_wrap.cpp' not in panel._log_text.toPlainText()
        panel._combo_data_filter.setCurrentIndex(panel._combo_data_filter.findData('fisheye_common'))
        for strategy in ('standard', 'joint_refine'):
            panel._combo_calib_strategy.setCurrentIndex(panel._combo_calib_strategy.findData(strategy))
            panel._btn_calib_all.click()
            wait_for(app, lambda: not panel.has_active_background_tasks())
            assert win._rig is not None, panel._log_text.toPlainText()
            pair = win._rig.pairs['offline']
            assert pair.stereo.rms_error < 1.
            assert pair.left_intrinsics.rms_error < 1. and pair.right_intrinsics.rms_error < 1.
            selection = pair.calibration_info['selection']['stereo']
            assert len(selection['used_ids']) == len(panel._offline_import_records)
            assert not selection['excluded_ids']
            assert len(set(selection['used_corner_counts'])) == 1
            assert selection['input_corner_counts'] == original_counts
            assert panel._stereo_cals['offline'].frame_common_corner_counts == original_counts
            assert pair.calibration_info['rectification']['status'] == 'ok'
            assert pair.calibration_info['rectification']['rms_px'] < 1.
            app.processEvents()
            assert '鱼眼模型' in panel._lbl_rect_context.text()
            assert not panel._lbl_rect_preview.isHidden()
            assert panel._inspector_scroll.horizontalScrollBar().maximum() == 0
            panel._btn_frame_diagnostics.click()
            dialog = panel._diagnostic_dialog
            dialog._table.sortItems(1, Qt.DescendingOrder)
            dialog._table.setCurrentCell(0, 0)
            chosen_id = dialog._table.item(0, 0).text()
            assert dialog._btn_preview.isEnabled()
            dialog._btn_preview.click()
            assert win._stream_view.review_key == f'offline {chosen_id}'
            app.processEvents()
            dialog.grab().save(str(directory / f'10-fisheye-diagnostics-{strategy}.png'))
            dialog.close()
            win.grab().save(str(directory / f'11-fisheye-calibrated-{strategy}.png'))
            win._stream_view.end_review()
            saved = win._current_session.load_rig().pairs['offline']
            assert saved.calibration_info['selection'] == pair.calibration_info['selection']
            path = win._current_session.result_directory / 'rig_opencv_standard.yml'
            storage = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
            try:
                assert storage.getNode('camera_model').string() == 'fisheye'
                assert storage.getNode('M1').mat().shape == (3, 3)
            finally:
                storage.release()
    finally:
        for worker in (panel._stereo_import_worker, panel._calib_worker):
            if worker is not None:
                worker.wait(90000)
        wait_for(app, lambda: not panel.has_active_background_tasks())
        win.close()
        win.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
        app.processEvents()
