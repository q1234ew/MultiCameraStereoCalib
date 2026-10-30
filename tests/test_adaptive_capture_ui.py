"""Adaptive recommendations stay advisory, revision-bound and reversible."""

import time
import os
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

from PySide6.QtCore import QThreadPool
from multicamera.calibration.intrinsic import IntrinsicCalibrator

from test_stereo_capture_ui import sample, publish, publish_stable
from test_multimodal_preview import LEFT, RIGHT

pytest_plugins = ['test_stereo_capture_ui']


def settle(guide, app):
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        app.processEvents()
        if guide._guidance_worker is None:
            guide.refresh()
            if guide._guidance_worker is None:
                return
        time.sleep(.01)
    raise AssertionError('Guidance worker did not finish')


def activate(window, app):
    guide = window._stereo_capture
    guide.guide_mode.setCurrentIndex(0)
    guide.refresh()
    settle(guide, app)
    assert guide._analysis is not None
    return guide


def add_pair(window, source='0000'):
    panel = window._calib_panel
    det, frame, intr = sample(window)
    for cid in (LEFT, RIGHT):
        if cid not in panel._intr_cals:
            panel._intr_cals[cid] = IntrinsicCalibrator(panel._board)
        panel._intr_cals[cid].add_detection(det, frame_id=source)
    panel._live_preview_indices.setdefault('rgb', []).append((
        panel._intr_cals[LEFT].num_frames - 1,
        panel._intr_cals[RIGHT].num_frames - 1, source,
    ))
    return det, frame, intr


def test_default_adaptive_worker_updates_overlay_without_final_intrinsics(stereo_bench, app):
    win = stereo_bench
    guide = activate(win, app)
    panel = win._calib_panel
    assert guide._analysis.recommendation.sample_count == 0
    assert win._stream_view._capture_sequence is guide._adaptive_sequence
    assert not panel._seq_widget.isVisible()
    assert '建立初始模型' in guide.reason.text()
    initial = guide._target()
    add_pair(win)
    guide.refresh()
    settle(guide, app)
    assert guide._analysis.recommendation.sample_count == 1
    assert (guide._target().zone_row, guide._target().zone_col, guide._target().tilt) != (
        initial.zone_row, initial.zone_col, initial.tilt,
    )
    assert not panel._intrinsics
    key = guide._analysis_key
    guide.refresh()
    assert guide._analysis_key == key and guide._guidance_worker is None


def test_pending_failed_saves_and_stale_results_do_not_advance(stereo_bench, app):
    win = stereo_bench
    guide = activate(win, app)
    panel = win._calib_panel
    old_key, old_result = guide._analysis_key, deepcopy(guide._analysis)
    add_pair(win)
    panel._pending_save_tokens.add('pending')
    guide.refresh()
    assert guide._guidance_worker is None
    assert not guide.auto_ready()
    guide._guidance_completed(old_key, old_result, '')
    assert guide._analysis_key == old_key
    panel._guidance_failed_saves.add((id(panel._session), 'rgb', '0000'))
    panel._pending_save_tokens.clear()
    guide.refresh()
    settle(guide, app)
    assert guide._analysis.recommendation.sample_count == 0


def test_undo_and_manual_navigation_restore_consistent_target(stereo_bench, app):
    win = stereo_bench
    guide = activate(win, app)
    add_pair(win)
    guide.refresh()
    settle(guide, app)
    for cal in win._calib_panel._intr_cals.values():
        cal.pop_last_frame()
    win._calib_panel._live_preview_indices['rgb'].clear()
    guide.refresh()
    settle(guide, app)
    assert guide._analysis.recommendation.sample_count == 0
    win._calib_panel._btn_skip_stereo.click()
    assert not guide.adaptive
    guide.refresh()
    assert win._stream_view._capture_sequence is win._calib_panel._capture_sequence


def test_cold_start_never_automatically_accepts_unknown_tilt(stereo_bench, app):
    win = stereo_bench
    guide = activate(win, app)
    det, frame, _ = sample(win)
    publish(win, det, frame)
    publish(win, det, frame)
    settle(guide, app)  # First live dimensions invalidate the initial unknown-size snapshot.
    assert not guide.auto_ready()
    assert '待内参' in guide.pose_status.text() or '占比' in guide.pose_status.text()
    QThreadPool.globalInstance().waitForDone(15000)


def test_adaptive_layout_and_background_failure_fallback(stereo_bench, app, tmp_path):
    win = stereo_bench
    guide = activate(win, app)
    det, frame, _ = add_pair(win)
    for cid in (LEFT, RIGHT):
        win._stream_manager.camera_connected.emit(cid)
        win._stream_view._on_frame(cid, frame, time.time())
    guide.refresh()
    settle(guide, app)
    publish(win, det, frame)
    directory = Path(os.environ.get('MULTICAMERA_ADAPTIVE_QA_DIR', tmp_path))
    directory.mkdir(parents=True, exist_ok=True)
    for width, height in ((1440, 1024), (1280, 800), (1000, 680)):
        win.resize(width, height)
        app.processEvents()
        settle(guide, app)
        assert win._preview_scroll.horizontalScrollBar().maximum() == 0
        assert guide.instruction.height() >= guide.instruction.fontMetrics().height()
        button = win._calib_panel._btn_capture
        assert win.rect().contains(button.mapTo(win, button.rect().bottomRight()))
        win.grab().save(str(directory / f'adaptive-capture-{width}.png'))
    key, _ = guide._guidance_context()
    guide._guidance_completed(key, None, 'synthetic analysis failure')
    assert '回退固定计划' in guide.reason.text()
    assert win._stream_view._capture_sequence is win._calib_panel._capture_sequence
    assert not guide.auto_ready()


def test_calibrated_adaptive_auto_requires_both_camera_poses(stereo_bench, app):
    win = stereo_bench
    det, frame, intr = sample(win)
    pts = det.charuco_corners.reshape(-1, 2)
    area = cv2.contourArea(cv2.convexHull(pts)) / (1280 * 720)
    pts[:] = (pts - pts.mean(axis=0)) * np.sqrt(.07 / area) + (640, 360)
    win._calib_panel._intrinsics.update({LEFT: deepcopy(intr), RIGHT: deepcopy(intr)})
    guide = activate(win, app)
    publish_stable(win, det, frame)
    settle(guide, app)
    for s in guide.samples.values():
        s['sharp'] = True
    assert guide.auto_ready()
    guide.samples[RIGHT]['detection'] = deepcopy(det)
    guide.samples[RIGHT]['detection'].image_size = (640, 480)
    assert not guide.auto_ready()


def test_dynamic_target_center_is_shared_by_preview_raw_gate_and_schematic(stereo_bench, app, monkeypatch):
    from multicamera.calibration.guidance_replay import _matches
    from multicamera.ui.widgets.capture_guide import BoardTilt

    win = stereo_bench
    guide = activate(win, app)
    det, frame, _ = sample(win)
    pts = det.charuco_corners.reshape(-1, 2)
    pts[:] += np.array([.32 * 1280, .32 * 720]) - pts.mean(axis=0)
    publish_stable(win, det, frame)
    settle(guide, app)
    rec = replace(guide._analysis.recommendation, zone_row=0, zone_col=0, tilt='front',
                  target_center=(.32, .32), target_area=.055)
    guide._analysis.recommendation = rec
    guide._apply_guidance_target()
    pose = dict(yaw=0., pitch=0., depth=1., distance_area=.055, distance_reliable=True)
    monkeypatch.setattr(guide, '_pose_measurement', lambda *args: pose)
    original = guide._detection_sample

    def sharp_sample(detection, image):
        return dict(original(detection, image), sharp=True)

    monkeypatch.setattr(guide, '_detection_sample', sharp_sample)
    for s in guide.samples.values():
        s['sharp'] = True
    guide.refresh()
    target = guide._target()
    assert target.tilt == BoardTilt.FRONT
    assert (target.norm_cx, target.norm_cy) == (.32, .32)
    assert guide.schematic.target.target_center == rec.target_center
    assert guide.auto_ready()
    assert guide.validate_capture('rgb', frame, frame, det, det)[0]
    measurements = [dict(s, geometry_valid=True, pose=pose) for s in guide.samples.values()]
    assert _matches(rec, measurements)['composition_match']
    guide._analysis.recommendation = replace(rec, target_center=(1 / 6, 1 / 6))
    guide._apply_guidance_target()
    assert not guide.auto_ready()


def test_camera_address_or_dimensions_invalidate_pending_guidance(stereo_bench, app):
    from multicamera.streaming.stream_manager import FrameEntry

    win = stereo_bench
    guide = activate(win, app)
    old_key, old_result = guide._analysis_key, guide._analysis
    win._stream_manager.stereo_pairs['rgb'].left.url = 'http://new-device.test:8080/stream'
    assert guide._guidance_context()[0] != old_key
    guide._guidance_completed(old_key, old_result, '')
    settle(guide, app)
    assert guide._analysis_key != old_key
    key = guide._analysis_key
    win._stream_manager._latest_frames[LEFT] = FrameEntry(np.zeros((480, 640, 3), np.uint8), time.time())
    assert guide._guidance_context()[0] != key
    guide.refresh()
    settle(guide, app)
    assert guide._analysis_key != key
