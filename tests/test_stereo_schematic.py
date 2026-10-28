"""QPainter motion geometry, rendering and animation lifecycle."""

import os
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from PySide6.QtCore import QRectF
from PySide6.QtGui import QImage
from PySide6.QtWidgets import QScrollArea, QVBoxLayout, QWidget

from multicamera.ui.theme import configure_application_style
from multicamera.ui.widgets.capture_guide import BoardTilt, CaptureTarget, build_default_sequence
from multicamera.ui.widgets.stereo_schematic import StereoSchematic, board_point, project_point
from multicamera.ui.widgets.board_view import (
    board_framing, board_projection, depth_projection, EDGES, BOARD_CORNERS,
    oblique_framing, oblique_projection,
)


@pytest.mark.parametrize('tilt,fixed,moving', [
    (BoardTilt.TILT_LEFT, (1.45, 0), (-1.45, 0)),
    (BoardTilt.TILT_RIGHT, (-1.45, 0), (1.45, 0)),
    (BoardTilt.TILT_UP, (0, 1), (0, -1)),
    (BoardTilt.TILT_DOWN, (0, -1), (0, 1)),
])
def test_rotation_has_fixed_pivot_and_edge_approaches_camera(tilt, fixed, moving):
    target = CaptureTarget(0, 2, tilt)
    start = np.array(board_point(*fixed, 0, target))
    end = np.array(board_point(*fixed, 1, target))
    np.testing.assert_allclose(start, end)
    assert board_point(*moving, 1, target)[2] < board_point(*moving, 0, target)[2]
    a = np.array(board_point(*moving, 0, target))
    b = np.array(board_point(*moving, 1, target))
    assert np.linalg.norm(a - start) == pytest.approx(np.linalg.norm(b - end))


def test_frontal_targets_translate_without_depth_or_shape_change():
    for target in build_default_sequence()[:9]:
        a = np.array(board_point(-1.45, -1, 0, target))
        b = np.array(board_point(-1.45, -1, 1, target))
        np.testing.assert_allclose(b - a, [(target.zone_col - 1) * 1.25,
                                         (target.zone_row - 1) * .8, 0])
    # Every planned endpoint remains in the drawing, even on corner tilt steps.
    for target in build_default_sequence():
        for x, y in [(-1.45, -1), (1.45, -1), (1.45, 1), (-1.45, 1)]:
            for progress in [0, .5, 1]:
                p = project_point(board_point(x, y, progress, target))
                assert 0 < p.x() < 320 and 30 < p.y() < 258


def test_animation_pause_replay_step_review_and_visibility(app, monkeypatch):
    clock = [10.]
    monkeypatch.setattr('multicamera.ui.widgets.stereo_schematic.time.monotonic', lambda: clock[0])
    guide = StereoSchematic()
    guide.set_target(CaptureTarget(1, 1, BoardTilt.TILT_RIGHT))
    assert not guide.timer.isActive()
    guide.show()
    app.processEvents()
    assert guide.timer.isActive()
    guide._elapsed = 2.
    clock[0] += .05
    guide._tick()
    assert 0 < guide.progress < 1
    guide.pause.click()
    assert not guide.timer.isActive() and guide.pause.isChecked()
    value = guide.progress
    guide.set_target(CaptureTarget(1, 1, BoardTilt.TILT_RIGHT))
    assert guide.progress == value and guide.pause.isChecked()
    guide.replay.click()
    assert guide.progress == 0 and guide.timer.isActive()
    guide.set_target(CaptureTarget(1, 1, BoardTilt.TILT_UP))
    assert guide.progress == 0 and '上边缘' in guide.description.text()
    guide.set_target(guide.target, active=False)
    assert not guide.timer.isActive() and not guide.replay.isEnabled()
    guide.set_target(guide.target)
    assert guide.timer.isActive()
    guide.hide()
    assert not guide.timer.isActive()
    guide.show()
    app.processEvents()
    assert guide.timer.isActive()
    guide.set_target(None)
    assert not guide.timer.isActive() and not guide.pause.isEnabled()
    guide.set_target(CaptureTarget(1, 1, BoardTilt.FRONT))
    assert not guide.timer.isActive() and not guide.replay.isEnabled()


@pytest.mark.parametrize('scale', ['small', 'medium', 'large'])
@pytest.mark.parametrize('motion,sign', [('approach', -1), ('retreat', 1)])
def test_explicit_distance_motion_uses_action_not_target_scale(app, scale, motion, sign):
    guide = StereoSchematic()
    target = CaptureTarget(1, 1, BoardTilt.FRONT, scale=scale)
    guide.set_target(target, motion=motion)
    start = np.array(guide.canvas._board_point(0, 0, 0))
    end = np.array(guide.canvas._board_point(0, 0, 1))
    assert sign * (end[2] - start[2]) > 0
    np.testing.assert_array_equal(start[:2], end[:2])
    assert guide.has_motion
    guide.set_target(target, motion='hold')
    assert not guide.has_motion and not guide.replay.isEnabled()
    np.testing.assert_array_equal(guide.canvas._board_point(0, 0, 0),
                                  guide.canvas._board_point(0, 0, 1))


@pytest.mark.parametrize('view', ['board', 'profile', 'spatial'])
def test_qpainter_frames_are_nonblank_and_change(app, tmp_path, view):
    configure_application_style(app)
    guide = StereoSchematic()
    guide.view_selector.setCurrentIndex(guide.view_selector.findData(view))
    guide.resize(820, 400)
    guide.set_target(CaptureTarget(1, 1, BoardTilt.TILT_RIGHT))
    guide.show()
    app.processEvents()
    guide.pause.setChecked(True)
    directory = Path(os.environ.get('MULTICAMERA_UI_QA_DIR', tmp_path))
    directory.mkdir(parents=True, exist_ok=True)
    frames = []
    builds = guide.canvas.cache_builds
    for value in (0., .5, 1.):
        guide.progress = value
        image = guide.canvas.grab().toImage().convertToFormat(QImage.Format_RGBA8888)
        pixels = np.frombuffer(image.bits(), np.uint8).reshape(image.height(), image.bytesPerLine())
        frames.append(pixels.copy())
        assert np.unique(pixels.reshape(-1, 4)[:, :3], axis=0).shape[0] > 30
        guide.grab().save(str(directory / f'stereo-motion-{value}.png'))
    assert guide.canvas.cache_builds == builds
    assert not np.array_equal(frames[0], frames[2])
    if view == 'spatial':
        # The old starting diagram remains unchanged when progress changes.
        w = frames[0].shape[1] // 2
        np.testing.assert_array_equal(frames[0][:, :w], frames[2][:, :w])


def test_board_view_is_unmirrored_and_has_image_axis_directions():
    center = board_projection((0., 0., 4.7))
    assert board_projection((1., 0., 4.7)).x() > center.x()
    assert board_projection((-1., 0., 4.7)).x() < center.x()
    assert board_projection((0., -1., 4.7)).y() < center.y()
    assert board_projection((0., 1., 4.7)).y() > center.y()
    with pytest.raises(ValueError):
        board_projection((0., 0., -1.))


@pytest.mark.parametrize('tilt', list(EDGES))
def test_depth_panel_moving_edge_approaches_camera_and_pivot_does_not_move(tilt):
    target = CaptureTarget(1, 1, tilt)
    fixed, moving, _, _ = EDGES[tilt]
    for xy in fixed:
        a = depth_projection(board_point(*xy, 0, target), tilt)
        b = depth_projection(board_point(*xy, 1, target), tilt)
        assert a == b
    xy = tuple((a + b) / 2 for a, b in zip(*moving))
    a = depth_projection(board_point(*xy, 0, target), tilt)
    b = depth_projection(board_point(*xy, 1, target), tilt)
    if tilt in (BoardTilt.TILT_UP, BoardTilt.TILT_DOWN):
        assert b.x() < a.x()  # Side view: camera on the left.
    else:
        assert b.y() > a.y()  # Top view: camera below the board.


def test_view_switch_and_fixed_moving_edge_caption_preserve_animation_state(app):
    guide = StereoSchematic()
    guide.set_target(CaptureTarget(1, 1, BoardTilt.TILT_LEFT))
    guide.show()
    app.processEvents()
    assert guide.view_mode == 'board'
    assert '固定右边' in guide.caption.text() and '左边靠近镜头' in guide.caption.text()
    guide.pause.setChecked(True)
    guide.progress = .6
    guide.view_selector.setCurrentIndex(guide.view_selector.findData('spatial'))
    assert guide.view_mode == 'spatial'
    assert guide.progress == .6 and guide.pause.isChecked()
    guide.view_selector.setCurrentIndex(0)
    assert guide.progress == .6 and guide.pause.isChecked()
    guide.set_compact(True)
    assert guide.view_options.isHidden()
    guide.expand.setChecked(True)
    assert not guide.view_options.isHidden()
    guide.close()


def test_depth_view_switch_keeps_front_geometry_target_and_pause(app):
    guide = StereoSchematic()
    guide.set_target(CaptureTarget(1, 1, BoardTilt.TILT_DOWN))
    guide.show()
    app.processEvents()
    guide.pause.setChecked(True)
    guide.progress = .65
    target = guide.target
    before = guide.canvas.grab().toImage()
    guide.view_selector.setCurrentIndex(guide.view_selector.findData('profile'))
    after = guide.canvas.grab().toImage()
    assert guide.view_mode == 'profile'
    assert guide.target is target and guide.progress == .65 and guide.pause.isChecked()
    # Only the auxiliary pane changes; the main board image is identical.
    width = int(before.width() * .6)
    assert before.copy(0, 0, width, before.height()) == after.copy(0, 0, width, after.height())
    assert before != after
    guide.close()


@pytest.mark.parametrize('scale', ['small', 'medium', 'large'])
def test_board_framing_keeps_all_planned_motions_inside_view(scale):
    rect = QRectF(10, 30, 260, 140)
    for target in build_default_sequence():
        target = replace(target, scale=scale)
        def point(x, y, t):
            return board_point(x, y, t, target)
        center, focal = board_framing(rect, point)
        for t in np.linspace(0, 1, 21):
            for x, y in BOARD_CORNERS[:-1]:
                assert rect.contains(center + board_projection(point(x, y, t)) * focal)


@pytest.mark.parametrize('scale', ['small', 'medium', 'large'])
def test_oblique_view_fits_board_and_fixed_camera_for_all_targets(scale):
    rect = QRectF(8, 30, 140, 180)
    for target in build_default_sequence():
        target = replace(target, scale=scale)
        def point(x, y, t):
            return board_point(x, y, t, target)
        center, focal = oblique_framing(rect, point)
        for t in np.linspace(0, 1, 21):
            for x, y in BOARD_CORNERS[:-1]:
                assert rect.contains(center + oblique_projection(point(x, y, t)) * focal)
        for x in (-1.3, 1.3):
            for y in (-.42, .42):
                for z in (-.78, .68):
                    camera = tuple(v * .45 for v in (x, y, z))
                    assert rect.contains(center + oblique_projection(camera) * focal)


def test_scroll_visibility_stops_animation_and_restores_it_without_lost_pause(app):
    scroll = QScrollArea()
    body = QWidget()
    layout = QVBoxLayout(body)
    spacer = QWidget()
    spacer.setMinimumHeight(600)
    layout.addWidget(spacer)
    guide = StereoSchematic()
    layout.addWidget(guide)
    scroll.setWidget(body)
    scroll.setWidgetResizable(True)
    scroll.resize(700, 300)
    guide.set_target(CaptureTarget(1, 1, BoardTilt.TILT_RIGHT))
    scroll.show()
    app.processEvents()
    app.processEvents()
    assert guide.canvas.visibleRegion().isEmpty() and not guide.timer.isActive()
    scroll.verticalScrollBar().setValue(scroll.verticalScrollBar().maximum())
    app.processEvents()
    assert not guide.canvas.visibleRegion().isEmpty() and guide.timer.isActive()
    guide.pause.click()
    guide.set_target(CaptureTarget(1, 1, BoardTilt.TILT_LEFT))
    assert guide.pause.isChecked() and not guide.timer.isActive()
    scroll.verticalScrollBar().setValue(0)
    scroll.verticalScrollBar().setValue(scroll.verticalScrollBar().maximum())
    app.processEvents()
    assert guide.pause.isChecked() and not guide.timer.isActive()
    guide.replay.click()
    assert guide.timer.isActive()
    guide.canvas.grab()
    count = guide.canvas.cache_builds
    guide.resize(guide.width() + 30, guide.height())
    guide.canvas.grab()
    assert guide.canvas.cache_builds > count
