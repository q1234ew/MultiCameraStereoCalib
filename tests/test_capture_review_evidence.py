"""A recorded capture must never be replaced by a fresh preview detection."""

from copy import deepcopy
import os
from pathlib import Path
import time

import cv2
import numpy as np
import pytest

from multicamera.ui.widgets import stream_view
from multicamera.ui.threads.import_worker import StereoImageImportWorker
from test_adaptive_workflow_followup import prepare_raw_capture
from test_multimodal_preview import LEFT, RIGHT
from test_stereo_capture_ui import sample
from test_real_pairs_ui import wait_for

pytest_plugins = ['test_stereo_capture_ui', 'test_offline_ui_state']


def capture_with_counts(win, monkeypatch, left_count=114, right_count=122):
    panel, detection, frame = prepare_raw_capture(win, monkeypatch)
    results = []
    for cid, count in ((LEFT, left_count), (RIGHT, right_count)):
        result = deepcopy(detection)
        result.charuco_ids = result.charuco_ids[:count]
        result.charuco_corners = result.charuco_corners[:count]
        results.append(result)
        monkeypatch.setattr(panel._intr_cals[cid].detector, 'detect', lambda image, d=result: d)
    panel._capture_impl()
    return panel, results, frame


def test_capture_review_matches_log_and_saved_image_without_detection(stereo_bench, monkeypatch, app):
    win = stereo_bench
    panel, results, frame = capture_with_counts(win, monkeypatch)
    assert not win._stream_view.review_key
    wait_for(app, lambda: not panel._pending_save_tokens)
    pair = panel.preview_pairs()[-1]
    assert pair.detection_origin == '采图原图' and pair.save_state == '已保存'
    assert pair.key == 'rgb 0000'
    assert pair.left_detection.num_corners == 114 and pair.right_detection.num_corners == 122
    assert '[0000]: L=114' in panel._log_text.toPlainText()
    assert 'R=122' in panel._log_text.toPlainText()
    saved = panel._session.session_dir / 'images/rgb/left_0000.png'
    np.testing.assert_array_equal(cv2.imread(str(saved)), pair.left)
    np.testing.assert_array_equal(pair.left, frame)
    expected_corners = pair.left_detection.charuco_corners.copy()
    results[0].charuco_corners[:] = 0
    np.testing.assert_array_equal(pair.left_detection.charuco_corners, expected_corners)
    preview_detector = win._stream_view._detect_for_preview
    monkeypatch.setattr(win._stream_view, '_detect_for_preview',
                        lambda image: pytest.fail('Recorded capture must not be re-detected'))
    win._gallery._latest_capture.click()
    assert win._stream_view.review_key == pair.key
    assert win._stream_view.displayed_corner_counts() == {LEFT: 114, RIGHT: 122}
    assert '采图原图角点 114/126' in win._stream_view._views[LEFT]._info.text()
    assert '已保存' in win._stream_view._views[LEFT]._info.text()
    np.testing.assert_array_equal(pair.left, frame)
    win._stream_view._last_detect_ts[LEFT] = time.monotonic() - 2
    win._stream_view._expire_preview_detections()
    win._stream_view._on_disconnected(LEFT, 'injected disconnect')
    assert win._stream_view.displayed_corner_counts()[LEFT] == 114
    assert '采图原图角点 114/126' in win._stream_view._views[LEFT]._info.text()
    monkeypatch.setattr(win._stream_view, '_detect_for_preview', preview_detector)
    panel._on_undo_capture()
    assert not win._stream_view.review_key and not panel.preview_pairs()
    assert not saved.exists()


def test_failed_save_review_retry_updates_without_new_observations(stereo_bench, monkeypatch, app):
    win = stereo_bench
    panel = win._calib_panel
    assert panel._ensure_capture_session()
    save = panel._session.save_frame_pair
    def fail(*args, **kwargs):
        raise OSError('injected disk full')
    monkeypatch.setattr(panel._session, 'save_frame_pair', fail)
    capture_with_counts(win, monkeypatch)
    wait_for(app, lambda: not panel._pending_save_tokens)
    assert panel.preview_pairs()[-1].save_state == '保存失败'
    monkeypatch.setattr(win._stream_view, '_detect_for_preview',
                        lambda image: pytest.fail('Save status update must not re-detect'))
    win._gallery._latest_capture.click()
    assert '保存失败' in win._stream_view._views[LEFT]._info.text()
    count = panel._stereo_cals['rgb'].num_frames
    monkeypatch.setattr(panel._session, 'save_frame_pair', save)
    panel._retry_failed_saves()
    assert panel.preview_pairs()[-1].save_state == '保存中'
    wait_for(app, lambda: not panel._pending_save_tokens)
    assert panel._stereo_cals['rgb'].num_frames == count
    assert '已保存' in win._stream_view._views[LEFT]._info.text()


def test_failed_detection_still_has_original_review_and_can_be_undone(stereo_bench, monkeypatch, app):
    panel, _, frame = capture_with_counts(stereo_bench, monkeypatch, 0, 0)
    wait_for(app, lambda: not panel._pending_save_tokens)
    pair = panel.preview_pairs()[-1]
    assert pair.left_detection.num_corners == 0
    assert panel._stereo_cals['rgb'].num_frames == 0
    np.testing.assert_array_equal(pair.left, frame)
    stereo_bench._gallery._latest_capture.click()
    assert stereo_bench._stream_view.displayed_corner_counts()[LEFT] == 0
    panel._on_undo_capture()
    assert not panel.preview_pairs() and not stereo_bench._stream_view.review_key


def test_cached_preview_corners_never_overlay_new_image_and_expire(stereo_bench, monkeypatch):
    win = stereo_bench
    view = win._stream_view
    detection, frame, _ = sample(win)
    clock = [time.monotonic()]
    monkeypatch.setattr(stream_view.time, 'monotonic', lambda: clock[0])
    calls, overlays = [], []
    def detect(image):
        calls.append(image)
        return detection
    def draw(image, result, **kwargs):
        overlays.append(result)
        return image
    monkeypatch.setattr(view, '_detect_for_preview', detect)
    monkeypatch.setattr(view, '_draw_detection', draw)
    view._on_frame(LEFT, frame, time.time())
    assert overlays[-1].valid
    clock[0] += .1
    newer = np.zeros_like(frame)
    view._on_frame(LEFT, newer, time.time())
    assert len(calls) == 1 and not overlays[-1].valid
    assert '上次预览角点' in view._views[LEFT]._info.text()
    assert '非本帧' in view._views[LEFT]._info.text()
    clock[0] += 1
    view._expire_preview_detections()
    assert '已过期' in view._views[LEFT]._info.text()
    assert view._views[LEFT].corner_count is None
    assert LEFT not in view._last_detection
    assert LEFT not in win._stereo_capture.samples
    assert not overlays[-1].valid


def test_offline_review_uses_import_detection_and_swap_role(offline_window, monkeypatch):
    win = offline_window
    panel = win._calib_panel
    records = panel._offline_import_records
    left = deepcopy(records[0][3])
    left.charuco_ids = left.charuco_ids[:114]
    left.charuco_corners = left.charuco_corners[:114]
    records[0] = (*records[0][:3], left, records[0][4])
    pair = panel.preview_pairs()[0]
    monkeypatch.setattr(win._stream_view, '_detect_for_preview',
                        lambda image: pytest.fail('Imported review must reuse imported detections'))
    win._on_gallery_pair_selected(pair.key, {pair.left_id: pair.left, pair.right_id: pair.right})
    assert win._stream_view.displayed_corner_counts()[pair.left_id] == 114
    assert '导入原图' in win._stream_view._views[pair.left_id]._info.text()
    panel._offline_lr_swapped = True
    swapped = panel.preview_pairs()[0]
    assert swapped.left_detection.num_corners == 126
    assert swapped.right_detection.num_corners == 114


@pytest.mark.skipif(not os.environ.get('MULTICAMERA_CORNER_IMAGES'), reason='Field image opt-in')
def test_real_114_corner_capture_reuses_exact_import_result(stereo_bench, monkeypatch, app, tmp_path):
    win = stereo_bench
    panel = win._calib_panel
    folder = Path(os.environ['MULTICAMERA_CORNER_IMAGES'])
    worker = StereoImageImportWorker(str(folder), panel._board, panel)
    panel._stereo_import_worker = worker
    panel._pending_import_folder = folder
    worker.finished_ok.connect(panel._on_import_images_ready)
    failures = []
    worker.failed.connect(failures.append)
    worker.run()
    assert not failures
    pair = next(p for p in panel.preview_pairs() if p.key == 'offline 0016')
    assert pair.left_detection.num_corners == 114
    assert pair.right_detection.num_corners == 122
    monkeypatch.setattr(win._stream_view, '_detect_for_preview',
                        lambda image: pytest.fail('Review must not run scaled detection'))
    win._on_gallery_pair_selected(pair.key, {pair.left_id: pair.left, pair.right_id: pair.right})
    assert win._stream_view.displayed_corner_counts() == {'offline_L': 114, 'offline_R': 122}
    directory = Path(os.environ.get('MULTICAMERA_CAPTURE_REVIEW_QA_DIR', tmp_path))
    directory.mkdir(parents=True, exist_ok=True)
    for width, height in ((1000, 680), (1280, 800), (1440, 1024)):
        win.resize(width, height)
        app.processEvents()
        assert win._preview_scroll.horizontalScrollBar().maximum() == 0
        assert panel._import_view_idx == 7
        win.grab().save(str(directory / f'capture-review-{width}.png'))
