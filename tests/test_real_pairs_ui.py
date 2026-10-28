"""Opt-in real-image Qt workflow; never modifies the input photographs."""

import json
import os
import hashlib
from pathlib import Path

import cv2
import pytest
from PySide6.QtCore import QCoreApplication, QEvent, QEventLoop, QTimer, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QFileDialog, QDialog, QDialogButtonBox

from multicamera.ui import main_window
from multicamera.ui.theme import configure_application_style
from multicamera.ui.widgets.export_dialog import ExportDialog
from multicamera.ui.threads.import_worker import _discover_stereo_pairs


def source_image_hashes(source):
    paths = {path for _, left, right in _discover_stereo_pairs(str(source))
             for path in (left, right)}
    assert paths, 'No paired source images were included in the integrity check'
    return {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def wait_for(app, predicate, timeout=90000):
    loop = QEventLoop()
    poll, deadline = QTimer(), QTimer()
    poll.setInterval(20)
    poll.timeout.connect(lambda: loop.quit() if predicate() else None)
    deadline.setSingleShot(True)
    deadline.timeout.connect(loop.quit)
    poll.start()
    deadline.start(timeout)
    if not predicate():
        loop.exec()
    poll.stop()
    deadline.stop()
    app.processEvents()
    assert predicate(), 'Real-image background workflow timed out'


@pytest.mark.skipif(not os.environ.get('MULTICAMERA_REAL_PAIR_DIR'), reason='Real-image opt-in test')
def test_real_pairs_import_review_calibrate_export(app, monkeypatch, tmp_path):
    source = Path(os.environ['MULTICAMERA_REAL_PAIR_DIR']).resolve()
    assert source.is_dir()
    original_hashes = source_image_hashes(source)
    configure_application_style(app)
    monkeypatch.setattr(main_window, 'sessions_dir', lambda: tmp_path / 'sessions')
    monkeypatch.setattr(QFileDialog, 'getExistingDirectory', lambda *args: str(source))
    directory = Path(os.environ.get('MULTICAMERA_UI_QA_DIR', tmp_path / 'screenshots'))
    directory.mkdir(parents=True, exist_ok=True)
    win = main_window.MainWindow()
    panel = win._calib_panel
    dialog = None
    snapshots = {}

    def shot(name):
        app.processEvents()
        app.processEvents()
        win.grab().save(str(directory / f'{name}.png'))
        snapshots[name] = {
            'phase': panel._workflow_phase, 'guide_done': panel._capture_sequence.done_count,
            'banner': win._stereo_capture.instruction.text(),
            'raw_state': panel._lbl_raw_save.text(), 'capture_button': panel._btn_capture.text(),
            'review_key': win._stream_view.review_key,
        }

    try:
        win.resize(1280, 800)
        win.show()
        shot('01-empty')
        panel._btn_import.click()
        wait_for(app, lambda: panel._stereo_import_worker is None)
        assert panel._stereo_cals.get('offline') is not None, panel._log_text.toPlainText()
        assert panel._stereo_cals['offline'].num_frames >= 5, panel._log_text.toPlainText()
        assert panel.offline_selection and panel._capture_sequence.done_count == 0
        assert not panel._btn_capture.isVisible() and not panel._check_wb.isVisible()
        assert not win._stereo_capture.schematic.timer.isActive()
        assert [pair.key for pair in panel.preview_pairs()] == [
            f'offline {record[0]}' for record in panel._offline_import_records
        ]
        shot('02-imported')
        for width, height in ((1000, 680), (1440, 1024), (1280, 800)):
            win.resize(width, height)
            app.processEvents()
            app.processEvents()
            for button in (panel._btn_import, panel._btn_swap_offline, panel._btn_to_calibration):
                assert button.isVisible()
                assert win.rect().contains(button.mapTo(win, button.rect().topLeft()))
                assert win.rect().contains(button.mapTo(win, button.rect().bottomRight()))
            assert win._preview_scroll.horizontalScrollBar().maximum() == 0
            shot(f'02-imported-{width}')
        win._gallery._toggle.setChecked(True)
        app.processEvents()
        assert win._gallery._start == 0
        win._gallery._next.click()
        win._gallery._next.click()
        app.processEvents()
        item = win._gallery._list.item(0)
        assert item is not None
        QTest.mouseClick(win._gallery._list.viewport(), Qt.LeftButton,
                         pos=win._gallery._list.visualItemRect(item).center())
        assert win._stream_view.review_key
        selected_index = panel._import_view_idx
        assert win._stream_view.review_key == f'offline {panel._offline_import_records[selected_index][0]}'
        summary = win._stereo_capture.compact_summary.text()
        assert '导入原图' in summary
        for label, detection in zip(('左目', '右目'), panel._offline_review_detections(selected_index)):
            assert f'{label} {detection.num_corners}/{win._board.num_corners}' in summary
        shot('03-review')
        win._btn_end_review.click()
        panel._btn_swap_offline.click()
        assert panel._import_view_idx == selected_index
        assert win._stream_view._static_images[False]['offline_L'] is panel._offline_import_records[selected_index][2]
        shot('04-swapped')
        panel._btn_swap_offline.click()
        win._gallery._toggle.setChecked(False)
        panel._btn_to_calibration.click()
        assert not win._view_aux.isVisible()
        assert not win._gallery._toggle.isChecked()
        shot('05-calibration')

        def inspect_config(dialog):
            dialog._tabs.setCurrentIndex(2)
            dialog.show()
            app.processEvents()
            dialog.grab().save(str(directory / '05-board-configuration.png'))
            dialog._buttons.button(QDialogButtonBox.Cancel).click()
            return QDialog.Rejected

        monkeypatch.setattr(main_window.ConfigDialog, 'exec', inspect_config)
        panel._btn_configure_stereo.click()
        assert panel._stereo_cals['offline'].num_frames >= 5
        panel._btn_calib_all.click()
        wait_for(app, lambda: panel._calib_worker is None)
        assert win._rig is not None, panel._log_text.toPlainText()
        if win._rig.pairs['offline'].stereo.rms_error > 1.2:
            assert '不建议直接使用' in panel._lbl_result_decision.text()
            assert '已计算' in panel._pair_rows['offline']._lbl_status.text()
        assert win._current_session.load_rig().pairs['offline'].stereo.rms_error >= 0
        shot('06-calibrated')
        panel._btn_view_results.click()
        shot('07-results')
        dialog = ExportDialog(win._rig, win)
        dialog.show()
        dialog._scope.setCurrentIndex(1)
        assert dialog._scope.currentData()[1] == 'offline'
        dialog._format.setCurrentIndex(dialog._format.findData('standard'))
        assert 'offline' in dialog.suggested_filename
        app.processEvents()
        dialog.grab().save(str(directory / '08-export.png'))
        output = tmp_path / 'offline-standard.yml'
        dialog.export_to(output)
        storage = cv2.FileStorage(str(output), cv2.FILE_STORAGE_READ)
        try:
            assert storage.isOpened()
            assert storage.getNode('M1').mat().shape == (3, 3)
            assert storage.getNode('T').mat().size == 3
        finally:
            storage.release()
        snapshots['output'] = {'directory': str(win._current_session.session_dir),
                               'pairs': panel._stereo_cals['offline'].num_frames,
                               'rms': win._rig.pairs['offline'].stereo.rms_error}
        panel.show_workflow('capture')
        previous = win._rig
        empty = tmp_path / 'empty'
        empty.mkdir()
        monkeypatch.setattr(QFileDialog, 'getExistingDirectory', lambda *args: str(empty))
        panel._btn_import.click()
        wait_for(app, lambda: panel._stereo_import_worker is None)
        assert win._rig is previous and panel._offline_source_folder == source
        assert win._toggle_logs.isChecked() and '离线导入失败' in panel._log_text.toPlainText()
        shot('09-import-error-preserves-data')
        assert original_hashes == source_image_hashes(source)
        (directory / 'observations.json').write_text(json.dumps(snapshots, ensure_ascii=False, indent=2))
    finally:
        if dialog is not None:
            dialog.close()
        for worker in (panel._stereo_import_worker, panel._calib_worker):
            if worker is not None:
                worker.wait(90000)
        win._stream_manager.stop_all()
        win.close()
        win.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
        app.processEvents()
