"""Explicit distance capture and a single prioritized primary instruction."""

import os
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from multicamera.calibration.models import CameraModel
from multicamera.io.session import CalibrationSession
from multicamera.streaming.stream_manager import FrameEntry
from multicamera.ui.widgets.capture_guide import BoardTilt
from test_adaptive_capture_ui import activate, add_pair, settle
from test_multimodal_preview import LEFT, RIGHT
from test_stereo_capture_ui import publish_stable, sample

pytest_plugins = ['test_stereo_capture_ui']


def prepare(win, app):
    guide = activate(win, app)
    detection, frame, intr = sample(win)
    win._calib_panel._intrinsics.update({LEFT: intr, RIGHT: intr})
    publish_stable(win, detection, frame)
    settle(guide, app)
    assert guide.btn_near.isEnabled() and guide.btn_far.isEnabled()
    return guide, frame


def test_direct_distance_entry_survives_new_data_and_returns_to_recommendation(stereo_bench, app):
    win = stereo_bench
    guide, _ = prepare(win, app)
    guide.btn_near.click()
    assert guide._distance_override == 'large'
    assert (guide._target().zone_row, guide._target().zone_col, guide._target().tilt,
            guide._target().scale) == (1, 1, BoardTilt.FRONT, 'large')
    add_pair(win)
    guide.refresh()
    settle(guide, app)
    assert guide._target().scale == 'large'
    assert '指定补采' in guide.reason.text() and guide.btn_resume_guidance.isVisible()
    guide.btn_far.click()
    assert guide._target().scale == 'small'
    assert '指定补采：中央正面 · 远距离' in guide.reason.toolTip()
    guide.btn_resume_guidance.click()
    rec = guide._analysis.recommendation
    assert guide._distance_override is None
    assert guide._target().scale == rec.scale
    assert guide._target().zone_col == rec.zone_col
    assert not guide.btn_resume_guidance.isVisible()


def test_distance_entry_disabled_during_save_and_hidden_in_review(stereo_bench, app):
    win = stereo_bench
    guide, frame = prepare(win, app)
    panel = win._calib_panel
    panel._pending_save_tokens.add('pending')
    guide.refresh()
    assert not guide.btn_near.isEnabled() and not guide.btn_far.isEnabled()
    guide.btn_near.click()
    assert guide._distance_override is None
    panel._pending_save_tokens.clear()
    guide.refresh()
    guide.btn_near.click()
    win._stream_view.show_review_images('test-review', {LEFT: frame, RIGHT: frame})
    guide.refresh()
    assert not guide.distance_tools.isVisible()
    assert '回看' in guide.instruction.text()
    win._stream_view.end_review()
    guide.refresh()
    assert guide._distance_override == 'large'
    guide.guide_mode.setCurrentIndex(1)
    assert guide._distance_override is None
    assert not guide.distance_tools.isVisible()
    guide.guide_mode.setCurrentIndex(0)
    settle(guide, app)
    assert guide._target().scale == guide._analysis.recommendation.scale


def test_model_change_clears_explicit_distance(stereo_bench, app):
    guide, _ = prepare(stereo_bench, app)
    guide.btn_near.click()
    guide.panel.set_camera_model(CameraModel.FISHEYE)
    guide.refresh()
    settle(guide, app)
    assert guide._distance_override is None


def test_first_session_preserves_choice_but_replacement_session_clears_it(
        stereo_bench, app, tmp_path):
    guide, _ = prepare(stereo_bench, app)
    guide.btn_near.click()
    assert guide.panel._ensure_capture_session()
    guide.refresh()
    settle(guide, app)
    assert guide._distance_override == 'large'
    guide.panel.set_session(CalibrationSession(tmp_path / 'replacement'))
    guide.refresh()
    settle(guide, app)
    assert guide._distance_override is None


def test_primary_instruction_advances_one_condition_at_a_time(stereo_bench, app, monkeypatch):
    win = stereo_bench
    guide, _ = prepare(win, app)
    _, samples = guide._current_samples()
    pose = {'yaw': 25., 'pitch': 0., 'depth': 1., 'distance_area': .02,
            'distance_reliable': True}
    monkeypatch.setattr(guide, '_pose_measurement', lambda *args: dict(pose))
    for s in samples:
        s.update(center=np.array([.15, .5]), stable=False, sharp=False)
    guide.refresh()
    assert guide._instruction_stage == 'distance' and '靠近' in guide.instruction.text()
    assert guide.schematic.motion == 'approach'
    assert (guide.schematic.canvas._board_point(0, 0, 1)[2]
            < guide.schematic.canvas._board_point(0, 0, 0)[2])
    assert '平移' not in guide.instruction.text()
    pose['distance_area'] = .07
    guide.refresh()
    assert guide._instruction_stage == 'position' and '平移' in guide.instruction.text()
    assert '靠近' not in guide.instruction.text()
    for s in samples:
        s['center'] = np.array([.5, .5])
    guide.refresh()
    assert guide._instruction_stage == 'tilt' and '转正' in guide.instruction.text()
    pose['yaw'] = 0.
    guide.refresh()
    assert guide._instruction_stage == 'quality'
    for s in samples:
        s['sharp'] = True
    guide.refresh()
    assert guide._instruction_stage == 'stability' and '静止' in guide.instruction.text()
    assert guide.schematic.description.text() == guide.instruction.text()
    assert not guide.schematic.timer.isActive()
    assert guide.schematic.progress == 1.
    assert not guide.schematic.replay.isEnabled()
    assert '目标姿态参考' in guide.schematic.caption.text()
    assert not guide.auto_ready()
    for s in samples:
        s['stable'] = True
    guide.refresh()
    assert guide._instruction_stage == 'ready' and '可以采集' in guide.instruction.text()
    assert guide.schematic.description.text() == guide.instruction.text()
    assert not guide.schematic.timer.isActive()
    assert guide.auto_ready()
    pose['distance_area'] = .2
    guide.refresh()
    assert guide._instruction_stage == 'distance' and '远离' in guide.instruction.text()
    assert guide.schematic.motion == 'retreat'
    assert (guide.schematic.canvas._board_point(0, 0, 1)[2]
            > guide.schematic.canvas._board_point(0, 0, 0)[2])
    assert not guide.auto_ready()


def test_conflicting_distance_stops_animation_and_resumes_in_correct_direction(
        stereo_bench, app, monkeypatch):
    guide, _ = prepare(stereo_bench, app)
    guide.btn_near.click()
    areas = {LEFT: .02, RIGHT: .6}
    monkeypatch.setattr(guide, '_pose_measurement', lambda cid, *args:
                        {'yaw': 0., 'pitch': 0., 'depth': .5, 'distance_area': areas[cid],
                         'distance_reliable': True})
    guide.refresh()
    assert '暂停前后移动' in guide.instruction.text()
    assert guide.schematic.motion == 'hold'
    assert not guide.schematic.timer.isActive()
    assert not guide.schematic.replay.isEnabled()
    assert not guide.auto_ready()
    areas[LEFT] = .6
    guide.refresh()
    assert '远离' in guide.instruction.text()
    assert guide.schematic.motion == 'retreat'
    assert guide.schematic.canvas._board_point(0, 0, 1)[2] > guide.schematic.canvas._board_point(0, 0, 0)[2]
    areas.update({LEFT: .02, RIGHT: .02})
    guide.refresh()
    assert '靠近' in guide.instruction.text()
    assert guide.schematic.motion == 'approach'
    assert guide.schematic.canvas._board_point(0, 0, 1)[2] < guide.schematic.canvas._board_point(0, 0, 0)[2]


def test_supplement_does_not_bypass_pose_gate(stereo_bench, app, monkeypatch):
    guide, _ = prepare(stereo_bench, app)
    guide.btn_near.click()
    monkeypatch.setattr(guide, '_pose_measurement', lambda *args: None)
    _, samples = guide._current_samples()
    for s in samples:
        s.update(center=np.array([.5, .5]), distance_area=.16, sharp=True, stable=True)
    guide.refresh()
    assert guide._instruction_stage == 'distance' and '等待距离确认' in guide.instruction.text()
    assert guide.schematic.motion == 'hold'
    assert not guide.schematic.timer.isActive()
    assert not guide.auto_ready()


def test_supplement_keeps_duplicate_and_pairing_gates(stereo_bench, app, monkeypatch):
    guide, _ = prepare(stereo_bench, app)
    guide.btn_near.click()
    monkeypatch.setattr(guide, '_pose_measurement', lambda *args:
                        {'yaw': 0., 'pitch': 0., 'depth': 1., 'distance_area': .16,
                         'distance_reliable': True})
    _, samples = guide._current_samples()
    for s in samples:
        s.update(center=np.array([.5, .5]), sharp=True, stable=True)
    monkeypatch.setattr(guide, '_duplicate_matches', lambda *args: [('rgb', 'old')])
    guide.refresh()
    assert guide._instruction_stage == 'duplicate'
    assert not guide.auto_ready()
    monkeypatch.setattr(guide, '_duplicate_matches', lambda *args: [])
    monkeypatch.setattr(guide.panel._sm, 'get_sync_pair', lambda *args: None)
    guide.refresh()
    assert guide._instruction_stage == 'pairing'
    assert not guide.auto_ready()


def test_far_entry_disabled_when_either_resolution_is_infeasible(stereo_bench, app):
    guide, _ = prepare(stereo_bench, app)
    import time
    guide.panel._sm._latest_frames[RIGHT] = FrameEntry(
        np.zeros((120, 160, 3), np.uint8), time.time())
    guide.refresh()
    assert not guide.btn_far.isEnabled()
    guide.btn_far.click()
    assert guide._distance_override is None


def test_auto_timer_alone_does_not_disable_entry_but_preparation_does(
        stereo_bench, app, monkeypatch):
    guide, _ = prepare(stereo_bench, app)
    guide.panel._auto_timer.start(10000)
    guide.refresh()
    assert guide.btn_near.isEnabled()
    with monkeypatch.context() as patch:
        patch.setattr(guide.panel._white_balance_guard, 'busy', True)
        guide.refresh()
        assert not guide.btn_near.isEnabled()
        assert guide._instruction_stage == 'preparation'
        assert '等待相机参数稳定' in guide.instruction.text()
    guide.panel._auto_timer.stop()


def test_completed_distance_hint_remains_visible_in_compact_mode(stereo_bench, app):
    guide, _ = prepare(stereo_bench, app)
    guide.btn_near.click()
    guide._analysis.recommendation = replace(guide._analysis.recommendation,
                                            missing_distances=('small',))
    guide.set_compact(True)
    guide.refresh()
    assert '本档数量已补足，可返回推荐' in guide.reason.text()
    assert guide.btn_resume_guidance.isVisible()
    assert guide._distance_override == 'large'


def test_distance_buttons_and_instruction_fit_small_windows(stereo_bench, app, tmp_path):
    win = stereo_bench
    guide, frame = prepare(win, app)
    guide.btn_near.click()
    directory = Path(os.environ.get('MULTICAMERA_ACTION_QA_DIR', tmp_path))
    directory.mkdir(parents=True, exist_ok=True)
    for cid in (LEFT, RIGHT):
        win._stream_manager.camera_connected.emit(cid)
    for width, height in ((1000, 680), (1280, 800), (1440, 1024)):
        win.resize(width, height)
        for cid in (LEFT, RIGHT):
            win._stream_view._last_render_ts.pop(cid, None)
            win._stream_view._on_frame(cid, frame, time.time())
        app.processEvents()
        guide.refresh()
        for _ in range(3):
            app.processEvents()
        for widget in (guide.btn_near, guide.btn_far, guide.btn_resume_guidance):
            assert widget.isVisible()
            assert win.rect().contains(widget.mapTo(win, widget.rect().bottomRight()))
        assert win._preview_scroll.horizontalScrollBar().maximum() == 0
        win.grab().save(str(directory / f'capture-actions-{width}.png'))


def test_distance_actions_wrap_with_large_font_in_narrow_sidebar(stereo_bench, app):
    win = stereo_bench
    guide, _ = prepare(win, app)
    guide.btn_near.click()
    win.resize(1000, 680)
    buttons = (guide.btn_near, guide.btn_far, guide.btn_resume_guidance)
    for button in buttons:
        button.setStyleSheet('font-size:20px;')
    for _ in range(4):
        app.processEvents()
    assert guide._distance_wrapped
    index = guide._distance_layout.indexOf(guide.btn_resume_guidance)
    assert guide._distance_layout.getItemPosition(index) == (1, 0, 1, 2)
    viewport = win._guide_scroll.viewport()
    win._guide_scroll.verticalScrollBar().setValue(0)
    app.processEvents()
    assert win._guide_scroll.horizontalScrollBar().maximum() == 0
    for button in buttons:
        assert viewport.rect().contains(button.mapTo(viewport, button.rect().topLeft()))
        assert viewport.rect().contains(button.mapTo(viewport, button.rect().bottomRight()))
    for button in buttons:
        button.setStyleSheet('')
    guide.btn_resume_guidance.click()
    for _ in range(3):
        app.processEvents()
    assert not guide._distance_wrapped
    assert guide.btn_resume_guidance.isHidden()
