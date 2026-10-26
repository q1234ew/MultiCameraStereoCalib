"""Offline source semantics and live/offline navigation without solving a rig."""

from pathlib import Path

import pytest
from PySide6.QtCore import QCoreApplication, QEvent

from multicamera.ui import main_window
from multicamera.ui.theme import configure_application_style, SUCCESS
from test_multimodal_preview import configure
from test_stereo_capture_ui import sample


@pytest.fixture
def offline_window(app, monkeypatch, tmp_path):
    configure_application_style(app)
    monkeypatch.setattr(main_window, 'sessions_dir', lambda: tmp_path / 'sessions')
    win = main_window.MainWindow()
    configure(win)
    win.resize(1280, 800)
    win.show()
    app.processEvents()
    det, image, _ = sample(win)
    win._calib_panel._pending_import_folder = tmp_path / 'source'
    win._calib_panel._capture_sequence.advance()
    win._calib_panel._on_import_images_ready([
        ('0001', image, image.copy(), det, det), ('0011', image.copy(), image.copy(), det, det),
    ])
    app.processEvents()
    yield win
    win._stream_manager.stop_all()
    win.close()
    win.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
    app.processEvents()


def test_offline_import_preserves_guide_and_selects_correct_source(offline_window):
    win = offline_window
    panel, guide = win._calib_panel, win._stereo_capture
    assert panel._combo_pair.currentText() == 'offline' and panel.offline_selection
    assert panel._capture_sequence.done_count == 1
    assert not panel._btn_capture.isVisible() and not panel._check_wb.isVisible()
    assert not panel._btn_auto.isVisible() and not panel._btn_skip_stereo.isVisible()
    assert not guide.schematic.isVisible() and not guide.schematic.timer.isActive()
    assert '离线' in guide.instruction.text() and '实时' not in guide.compact_summary.text()
    assert '源目录' in panel._lbl_raw_save.text()
    assert panel._btn_to_calibration.property('class') == 'primary'
    assert [pair.key for pair in panel.preview_pairs()] == ['offline 0001', 'offline 0011']
    for view in win._stream_view._views.values():
        assert '离线' in view._header.toolTip()
        assert SUCCESS not in view._status_dot.styleSheet()
        assert '离线图像' in view._info.text()
    assert panel._stereo_cals['offline'].min_frames == panel._spin_min.minimum()
    assert not panel._stereo_cals['offline'].ready


def test_swap_failure_and_live_return_do_not_destroy_prior_state(offline_window):
    win = offline_window
    panel = win._calib_panel
    before = panel._capture_sequence.done_count
    folder = panel._offline_source_folder
    panel._btn_swap_offline.click()
    assert panel._offline_lr_swapped and panel._capture_sequence.done_count == before
    panel._pending_import_folder = Path('/missing/photos')
    panel._on_import_images_failed('未找到离线左右图')
    assert panel._offline_source_folder == folder and len(panel._offline_import_records) == 2
    assert win._toggle_logs.isChecked() and '离线导入失败' in panel._log_text.toPlainText()
    panel._combo_pair.setCurrentText('rgb')
    assert not panel.offline_selection
    assert panel._btn_capture.isVisible() and panel._check_wb.isVisible()
    assert win._stereo_capture.schematic.isVisible()
    assert panel._capture_sequence.done_count == before


def test_gallery_choice_and_rgb_toolbar_stay_coherent(offline_window, app):
    win = offline_window
    win._gallery._toggle.setChecked(True)
    win._gallery._toggle.setChecked(False)
    win._calib_panel._btn_to_calibration.click()
    app.processEvents()
    assert not win._gallery._toggle.isChecked()
    assert not win._view_aux.isVisible() and not win._view_split.isVisible()
    assert win._calib_panel._lbl_calib_hint.isVisible()
    assert '针孔' in win._calib_panel._lbl_calib_hint.text()
    button = win._calib_panel._pair_rows['offline'].btn
    button.setText('重新标定')
    app.processEvents()
    assert button.width() >= button.sizeHint().width()


def test_minimum_pair_setting_updates_readiness_and_never_drops_below_five(offline_window):
    panel = offline_window._calib_panel
    records = panel._offline_import_records
    image_l, image_r, det_l, det_r = records[0][1:]
    panel._on_import_images_ready([
        (f'{index:04d}', image_l, image_r, det_l, det_r) for index in range(5)
    ])
    assert panel._stereo_cals['offline'].ready
    panel._spin_min.setValue(7)
    assert panel._stereo_cals['offline'].min_frames == 7
    assert not panel._pair_rows['offline'].btn.isEnabled()
    panel._spin_min.setValue(5)
    assert panel._pair_rows['offline'].btn.isEnabled()
