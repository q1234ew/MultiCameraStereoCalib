"""Live measurements must share historical geometry and fail closed on stale evidence."""

from copy import deepcopy
import time

import numpy as np
import pytest
import cv2

from multicamera.calibration import adaptive_guidance as ag
from multicamera.calibration.intrinsic import IntrinsicCalibrator
from multicamera.calibration.stereo import StereoCalibrator
from multicamera.streaming.stream_manager import FrameEntry
from test_multimodal_preview import LEFT, RIGHT
from test_stereo_capture_ui import sample, publish, publish_stable
from test_real_pairs_ui import wait_for

pytest_plugins = ['test_stereo_capture_ui']


def test_partial_preview_and_recommender_use_same_board_center(stereo_bench):
    win = stereo_bench
    det, frame, _ = sample(win)
    columns = win._board.config.cols - 1
    keep = det.charuco_ids.ravel() % columns < int(columns * .75)
    det.charuco_corners = det.charuco_corners[keep].copy()
    det.charuco_ids = det.charuco_ids[keep].copy()
    expected = ag._geometry(win._board, ag.GuidanceObservation(
        det.charuco_corners, det.charuco_ids, det.image_size, 'preview'))
    assert expected is not None
    publish(win, det, frame)
    measured = win._stereo_capture.samples[LEFT]
    np.testing.assert_allclose(measured['center'], expected[2], atol=1e-7)


@pytest.mark.parametrize('stamp', [float('nan'), float('inf'), 'future'])
def test_invalid_detection_timestamp_is_not_current_evidence(stereo_bench, stamp):
    win = stereo_bench
    det, frame, _ = sample(win)
    publish(win, det, frame)
    guide = win._stereo_capture
    guide.samples[LEFT]['time'] = time.monotonic() + 5 if stamp == 'future' else stamp
    assert not guide._current_samples()[1]
    assert not guide.auto_ready()


def test_fixed_plan_pose_display_rejects_unreliable_intrinsics(stereo_bench):
    win = stereo_bench
    det, frame, intr = sample(win)
    intr.rms_error = float('nan')
    win._calib_panel._intrinsics[LEFT] = intr
    publish(win, det, frame)
    assert '不可用' in win._stereo_capture._pose_text(LEFT, win._stereo_capture.samples[LEFT])


def test_immediate_repeated_detections_are_not_a_stability_window(stereo_bench):
    win = stereo_bench
    det, frame, _ = sample(win)
    publish(win, det, frame)
    publish(win, det, frame)
    assert not win._stereo_capture.auto_ready()


@pytest.mark.parametrize('stamp', [float('nan'), float('inf'), 'future'])
def test_invalid_received_frame_timestamp_is_not_current_evidence(stereo_bench, stamp):
    win = stereo_bench
    det, frame, _ = sample(win)
    publish(win, det, frame)
    stamp = time.time() + 5 if stamp == 'future' else stamp
    win._stream_manager._latest_frames[RIGHT] = FrameEntry(frame, stamp)
    assert not win._stereo_capture._current_samples()[1]


def test_slow_accumulated_motion_resets_the_stability_window(stereo_bench):
    win = stereo_bench
    guide = win._stereo_capture
    det, frame, _ = sample(win)
    clock = [time.monotonic()]
    guide._clock = lambda: clock[0]
    for dx in (0, 6, 12):
        shifted = deepcopy(det)
        shifted.charuco_corners[:, :, 0] += dx
        clock[0] += .35
        publish(win, shifted, frame)
    assert not guide.samples[LEFT]['stable']
    assert not guide.auto_ready()
    for _ in range(3):
        clock[0] += .35
        publish(win, shifted, frame)
    assert guide.samples[LEFT]['stable']
    assert guide.auto_ready()


def test_invalid_partial_board_and_source_change_clear_stability(stereo_bench):
    win = stereo_bench
    guide = win._stereo_capture
    det, frame, _ = sample(win)
    publish_stable(win, det, frame)
    assert guide.auto_ready()
    partial = deepcopy(det)
    partial.charuco_corners = partial.charuco_corners[:10]
    partial.charuco_ids = partial.charuco_ids[:10]
    publish(win, partial, frame)
    assert not guide.samples[LEFT]['valid']
    publish_stable(win, det, frame)
    win._stream_manager.stereo_pairs['rgb'].left.url = 'http://changed.test/left'
    guide.refresh()
    assert not guide.samples and not guide.auto_ready()


def test_pending_failures_block_auto_but_its_timer_does_not(stereo_bench):
    win = stereo_bench
    panel, guide = win._calib_panel, win._stereo_capture
    det, frame, _ = sample(win)
    publish_stable(win, det, frame)
    panel._auto_timer.start(10000)
    try:
        assert guide.auto_ready()
        panel._failed_save_jobs['failed'] = object()
        assert not guide.auto_ready()
    finally:
        panel._auto_timer.stop()
        panel._failed_save_jobs.clear()


def prepare_raw_capture(win, monkeypatch):
    panel = win._calib_panel
    assert panel._ensure_capture_session()
    for cid in (LEFT, RIGHT):
        panel._intr_cals[cid] = IntrinsicCalibrator(panel._board)
        panel._fqs[cid] = []
    panel._stereo_cals['rgb'] = StereoCalibrator(panel._board)
    monkeypatch.setattr(panel, '_prepare_white_balance', lambda *args: True)
    det, frame, _ = sample(win)
    publish_stable(win, det, frame)
    assert win._stereo_capture.auto_ready()
    return panel, det, frame


@pytest.mark.parametrize('problem', ['moved', 'blur', 'missing', 'exception'])
def test_auto_raw_rejection_does_not_save_or_advance(stereo_bench, monkeypatch, problem):
    win = stereo_bench
    panel, det, frame = prepare_raw_capture(win, monkeypatch)
    original_index = panel._next_session_frame_idx
    right = deepcopy(det)
    if problem == 'moved':
        right.charuco_corners[:, :, 0] += 40
    elif problem == 'missing':
        right.charuco_corners = right.charuco_ids = None
    if problem == 'blur':
        win._stream_manager._latest_frames[RIGHT] = FrameEntry(np.zeros_like(frame), time.time())
    monkeypatch.setattr(panel._intr_cals[LEFT].detector, 'detect', lambda image: det)
    def detect_right(image):
        if problem == 'exception':
            raise cv2.error('injected raw detection failure')
        return right
    monkeypatch.setattr(panel._intr_cals[RIGHT].detector, 'detect', detect_right)
    # Preview still passes: rejecting happens on the actual raw pair before mutation.
    assert win._stereo_capture.auto_ready()
    panel._on_auto_capture()
    assert not panel._auto_capture_request
    assert panel._next_session_frame_idx == original_index
    assert panel._session.frame_count == 0
    assert all(cal.num_frames == 0 for cal in panel._intr_cals.values())
    assert panel._stereo_cals['rgb'].num_frames == 0
    assert not panel._pending_save_tokens and not panel._undo_stack
    assert not list(panel._session.session_dir.rglob('*.png'))
    assert '原图' in panel._log_text.toPlainText()


def test_successful_auto_capture_detects_once_saves_then_undoes(stereo_bench, monkeypatch, app):
    win = stereo_bench
    panel, det, frame = prepare_raw_capture(win, monkeypatch)
    calls = []
    def detect(image):
        calls.append(id(image))
        return deepcopy(det)
    for cid in (LEFT, RIGHT):
        monkeypatch.setattr(panel._intr_cals[cid].detector, 'detect', detect)
    panel._on_auto_capture()
    wait_for(app, lambda: not panel._pending_save_tokens)
    assert len(calls) == 2
    assert panel._session.frame_count == 1
    assert panel._stereo_cals['rgb'].num_frames == 1
    path = panel._session.session_dir / 'images/rgb/left_0000.png'
    np.testing.assert_array_equal(cv2.imread(str(path)), frame)
    panel._on_undo_capture()
    assert panel._stereo_cals['rgb'].num_frames == 0
    assert not path.exists()


def test_auto_pair_race_never_uses_relaxed_frames(stereo_bench, monkeypatch):
    win = stereo_bench
    panel, _, frame = prepare_raw_capture(win, monkeypatch)
    original = win._stream_manager.get_sync_pair
    calls = []
    def synchronized(name):
        calls.append(name)
        return original(name) if len(calls) == 1 else None
    monkeypatch.setattr(win._stream_manager, 'get_sync_pair', synchronized)
    monkeypatch.setattr(win._stream_manager, 'get_latest_pair_relaxed',
                        lambda name: pytest.fail('Automatic capture must never relax pairing'))
    panel._on_auto_capture()
    assert panel._session.frame_count == 0
    assert not list(panel._session.session_dir.rglob('*.png'))


def test_failed_capture_retry_and_undo_reanalyze_saved_samples(stereo_bench, monkeypatch, app):
    from test_adaptive_capture_ui import activate, settle

    win = stereo_bench
    panel, det, _ = prepare_raw_capture(win, monkeypatch)
    calls = []
    def detect(image):
        calls.append(id(image))
        return deepcopy(det)
    for cid in (LEFT, RIGHT):
        monkeypatch.setattr(panel._intr_cals[cid].detector, 'detect', detect)
    save = panel._session.save_frame_pair
    def fail(*args, **kwargs):
        raise OSError('injected disk full')
    monkeypatch.setattr(panel._session, 'save_frame_pair', fail)
    panel._on_auto_capture()
    wait_for(app, lambda: not panel._pending_save_tokens)
    assert panel.has_failed_saves and panel._session.frame_count == 0
    assert panel._stereo_cals['rgb'].num_frames == 1
    guide = activate(win, app)
    assert guide._analysis.recommendation.sample_count == 0
    assert not guide.auto_ready()

    monkeypatch.setattr(panel._session, 'save_frame_pair', save)
    panel._retry_failed_saves()
    wait_for(app, lambda: not panel._pending_save_tokens)
    assert not panel.has_failed_saves and panel._session.frame_count == 1
    assert len(calls) == 2
    assert panel._stereo_cals['rgb'].num_frames == 1
    guide.refresh()
    settle(guide, app)
    assert guide._analysis.recommendation.sample_count == 1

    panel._on_undo_capture()
    guide.refresh()
    settle(guide, app)
    assert guide._analysis.recommendation.sample_count == 0
    assert panel._session.frame_count == 0
    assert not list(panel._session.session_dir.rglob('*.png'))


def test_multiple_pairs_validate_before_any_save_or_observation(stereo_bench, monkeypatch, app):
    from multicamera.streaming.stream_manager import CameraConfig, StereoPairConfig

    win = stereo_bench
    panel, det, frame = prepare_raw_capture(win, monkeypatch)
    win._stream_manager.add_stereo_pair(StereoPairConfig(
        'second', CameraConfig('second_L', 'http://second.test/left', 'left', 'second'),
        CameraConfig('second_R', 'http://second.test/right', 'right', 'second')))
    panel.refresh_pairs()
    panel._stereo_cals['second'] = StereoCalibrator(panel._board)
    ids = (LEFT, RIGHT, 'second_L', 'second_R')
    for cid in ids:
        panel._intr_cals[cid] = IntrinsicCalibrator(panel._board)
        panel._fqs[cid] = []
        monkeypatch.setattr(panel._intr_cals[cid].detector, 'detect', lambda image: deepcopy(det))
    clock = [time.monotonic()]
    guide = win._stereo_capture
    guide._clock = lambda: clock[0]
    guide.samples.clear()
    for _ in range(3):
        clock[0] += .35
        for cid in ids:
            win._stream_manager._latest_frames[cid] = FrameEntry(frame, time.time())
            guide.observe(cid, det, frame)
    assert guide.auto_ready()
    panel._on_auto_capture()
    wait_for(app, lambda: not panel._pending_save_tokens)
    assert panel._session.frame_count == 2
    assert all(cal.num_frames == 1 for cal in panel._stereo_cals.values())
    panel._on_undo_capture()
    assert panel._session.frame_count == 0
    assert all(cal.num_frames == 0 for cal in panel._stereo_cals.values())
