"""QPainter motion geometry, rendering and animation lifecycle."""

import os
from pathlib import Path

import numpy as np
import pytest
from PySide6.QtGui import QImage
from PySide6.QtWidgets import QScrollArea, QVBoxLayout, QWidget

from multicamera.ui.theme import configure_application_style
from multicamera.ui.widgets.capture_guide import BoardTilt, CaptureTarget, build_default_sequence
from multicamera.ui.widgets.stereo_schematic import StereoSchematic, board_point, project_point


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


def test_qpainter_frames_are_nonblank_and_change(app, tmp_path):
    configure_application_style(app)
    guide = StereoSchematic()
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
    # The starting diagram remains unchanged when only animation progress changes.
    w = frames[0].shape[1] // 2
    np.testing.assert_array_equal(frames[0][:, :w], frames[2][:, :w])


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
