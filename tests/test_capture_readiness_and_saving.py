"""Connection freshness and recoverable raw-photo persistence contracts."""

from pathlib import Path
import time

import numpy as np
import pytest

from multicamera.board.charuco_board import CharucoBoard, CharucoBoardConfig
from multicamera.io.session import CalibrationSession
from multicamera.streaming.stream_manager import CameraConfig, FrameEntry, StereoPairConfig, StreamManager
from multicamera.ui.widgets.calib_panel import CalibrationPanel, _UndoCaptureEntry


@pytest.fixture
def capture_panel(app, tmp_path, monkeypatch):
    manager = StreamManager()
    board = CharucoBoard(CharucoBoardConfig())
    manager.add_stereo_pair(StereoPairConfig('rgb', CameraConfig('left', 'http://camera:8080/stream'),
                                           CameraConfig('right', 'http://camera:8082/stream')))
    panel = CalibrationPanel(manager, board)
    session = CalibrationSession.create(tmp_path, board, list(manager.stereo_pairs.values()))
    panel.set_session(session)
    monkeypatch.setattr(panel, '_start_next_frame_save', lambda: None)
    yield panel
    panel._failed_save_jobs.clear()
    panel._pending_save_tokens.clear()
    panel._save_queue.clear()
    manager.stop_all()


def test_pair_readiness_rejects_stale_frames(capture_panel):
    manager = capture_panel._sm
    image = np.zeros((20, 30, 3), np.uint8)
    assert manager.pair_capture_status('rgb')[0] == 'waiting'
    now = time.time()
    manager._latest_frames = {'left': FrameEntry(image, now), 'right': FrameEntry(image, now)}
    assert manager.pair_capture_status('rgb')[0] == 'ready'
    assert manager.get_sync_pair('rgb') is not None
    manager._latest_frames['right'] = FrameEntry(image, now - .5)
    assert manager.pair_capture_status('rgb')[0] == 'unpaired'
    assert manager.get_sync_pair('rgb') is None
    assert manager.get_latest_pair_relaxed('rgb') is not None
    manager._latest_frames['right'] = FrameEntry(image, now - 3)
    assert manager.pair_capture_status('rgb')[0] == 'stale'
    assert manager.get_latest_pair_relaxed('rgb') is None
    manager._latest_frames['right'] = FrameEntry(image, float('nan'))
    assert manager.get_sync_pair('rgb') is None
    manager._on_connection_lost('left', 'timeout')
    assert manager.get_latest_frame('left') is None


def test_preflight_lists_board_and_output(capture_panel):
    capture_panel._refresh_capture_readiness()
    text = capture_panel._lbl_capture_readiness.toolTip()
    assert 'ChArUco' in text and '自动稳定' in text
    assert str(capture_panel._session.resolved_images_base()) in text


def test_failed_save_retries_same_frame_without_duplicate_observations(capture_panel):
    panel = capture_panel
    image = np.full((20, 30, 3), 120, np.uint8)
    index, token = panel._start_frame_save('rgb', image, image)
    panel._save_queue.clear()  # Simulate the worker having consumed this job.
    assert '保存中 1' in panel._lbl_raw_save.text()
    assert not panel._saved_capture_tokens
    panel._on_frame_save_failed(token, 'rgb', index, 'disk full')
    assert panel.has_failed_saves and not panel._pending_save_tokens
    assert '失败 1' in panel._lbl_raw_save.text()
    assert not panel._btn_retry_save.isHidden()
    with pytest.raises(RuntimeError, match='保存失败'):
        panel.set_session(CalibrationSession(Path('unused')))
    panel._retry_failed_saves()
    assert panel._save_queue[0][0] == token
    assert panel._save_queue[0][-1] == index
    assert not panel.has_failed_saves
    assert panel._next_session_frame_idx == index + 1
    panel._save_queue.clear()
    panel._on_frame_save_finished(token, 'rgb', index)
    assert '本次已落盘 1' in panel._lbl_raw_save.text()
    assert not panel._save_jobs
    assert not panel._pending_save_tokens


def test_failed_save_undo_releases_retry_images(capture_panel):
    panel = capture_panel
    image = np.zeros((20, 30, 3), np.uint8)
    index, token = panel._start_frame_save('rgb', image, image)
    panel._save_queue.clear()
    panel._on_frame_save_failed(token, 'rgb', index, 'disk full')
    panel._undo_apply_entry(_UndoCaptureEntry('rgb', 'left', 'right', index, token,
                                            False, False, False, panel._session))
    assert not panel.has_failed_saves
    assert not panel._save_jobs


def test_canceled_worker_failure_refreshes_calibration_state(capture_panel, monkeypatch):
    panel = capture_panel
    image = np.zeros((20, 30, 3), np.uint8)
    index, token = panel._start_frame_save('rgb', image, image)
    panel._save_queue.clear()
    panel._canceled_save_tokens.add(token)
    refreshed = []
    monkeypatch.setattr(panel, '_sync_pair_states', lambda: refreshed.append(True))
    panel._on_frame_save_failed(token, 'rgb', index, 'disk full')
    assert refreshed == [True]
    assert not panel.has_failed_saves
    assert not panel._pending_save_tokens
    assert not panel._save_jobs


def test_calibration_blocks_unpersisted_data(capture_panel):
    panel = capture_panel
    panel._pending_save_tokens.add('pending')
    panel._calibrate_pair('rgb')
    panel._on_calibrate_all()
    assert panel._calib_worker is None
    assert '原图尚未全部保存' in panel._log_text.toPlainText()


def test_no_frames_does_not_prepare_camera_parameters(capture_panel, monkeypatch):
    monkeypatch.setattr(capture_panel, '_prepare_white_balance',
                        lambda *args: pytest.fail('No frames must not trigger parameter preparation'))
    capture_panel._capture_impl()
    assert '采集未开始' in capture_panel._log_text.toPlainText()


def test_real_worker_save_failure_keeps_retry_job(capture_panel, monkeypatch, app):
    from multicamera.ui.widgets.calib_panel import CalibrationPanel

    panel = capture_panel
    monkeypatch.setattr(panel, '_start_next_frame_save',
                        CalibrationPanel._start_next_frame_save.__get__(panel))

    def fail(*args, **kwargs):
        raise OSError('disk full')

    monkeypatch.setattr(panel._session, 'save_frame_pair', fail)
    image = np.zeros((20, 30, 3), np.uint8)
    _, token = panel._start_frame_save('rgb', image, image)
    deadline = time.monotonic() + 3
    while panel.has_active_background_tasks() and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(.01)
    assert not panel.has_active_background_tasks()
    assert token in panel._failed_save_jobs
    assert panel._session.frame_count == 0
    assert '失败 1' in panel._lbl_raw_save.text()


def test_aux_save_failure_does_not_advance_data(capture_panel, monkeypatch):
    from multicamera.streaming.stream_manager import AuxiliaryCameraConfig

    panel = capture_panel
    panel._sm.set_auxiliary_camera(AuxiliaryCameraConfig('ir', 'http://camera:8081/stream'))
    image = np.zeros((240, 320, 3), np.uint8)
    monkeypatch.setattr(panel, '_aux_live_frames', lambda: ((image, image, 0.), 'ready'))
    monkeypatch.setattr(panel, '_prepare_white_balance', lambda *args: True)

    def fail(*args, **kwargs):
        raise OSError('disk full')

    monkeypatch.setattr(panel._session, 'save_multimodal_pair', fail)
    panel._on_capture_aux()
    assert panel._aux_capture_count == 0
    assert not panel._aux_images and not panel._aux_pairs
    assert '未增加采集记录' in panel._log_text.toPlainText()


def test_outdated_results_have_explicit_status(capture_panel, tmp_path):
    panel = capture_panel
    panel._results_outdated = True
    panel._previous_result_directory = tmp_path / 'results'
    panel.set_result_model(None)
    assert '重新标定' in panel._lbl_result_summary.text()
    assert '旧结果已过期' in panel._lbl_result_storage.text()
    assert str(tmp_path / 'results') in panel._lbl_result_storage.text()
    assert not panel._btn_export_results.isEnabled()


def test_partial_export_keeps_full_directory_visible_and_copyable(capture_panel, tmp_path, app):
    directory = tmp_path / 'results'
    capture_panel.set_result_storage(directory, [directory / 'rig.json'], ['Kalibr: incompatible distortion'])
    text = capture_panel._lbl_result_storage.text()
    assert str(directory.resolve()) in text
    assert '部分格式未生成' in text and 'incompatible' in text
    assert capture_panel._btn_open_results.isEnabled()
    capture_panel._btn_copy_results.click()
    assert app.clipboard().text() == str(directory.resolve())
