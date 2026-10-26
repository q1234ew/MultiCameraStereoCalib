"""Schematic stereo housing topology and front-facing optics."""

import numpy as np
from PySide6.QtGui import QColor, QImage, QPainter

from multicamera.ui.widgets.stereo_camera_model import (
    LENS_CENTERS, camera_faces, draw_stereo_camera, visible_faces,
)
from multicamera.ui.widgets.stereo_schematic import VIEW_EYE, project_point


def test_one_housing_two_parallel_forward_barrels_and_valid_faces():
    faces = camera_faces()
    assert camera_faces() is faces
    assert sum(face.part == 'body_front' for face in faces) == 1
    assert sum(face.part == 'body_rear' for face in faces) == 1
    assert sum(face.part == 'lens_rim' for face in faces) == 2
    assert LENS_CENTERS[0][0] == -LENS_CENTERS[1][0]
    assert LENS_CENTERS[0][1:] == LENS_CENTERS[1][1:] == (0., .66)
    for face in faces:
        assert np.isfinite(face.points).all()
        assert np.linalg.norm(face.normal) > 0
        if face.part == 'lens_rim':
            assert face.normal[2] > 0
        for point in face.points:
            pixel = project_point(point)
            assert 0 < pixel.x() < 320 and 32 < pixel.y() < 253


def test_front_view_culls_back_panels_and_keeps_both_lens_faces():
    faces = visible_faces(VIEW_EYE)
    assert not any(face.part == 'body_rear' for face in faces)
    assert sum(face.part == 'body_front' for face in faces) == 1
    assert sum(face.part == 'lens_rim' for face in faces) == 2
    for face in faces:
        assert np.dot(face.normal, np.array(VIEW_EYE) - face.center) > 0
    side_faces = [face for face in faces if face.part == 'body_side']
    assert any(face.normal[1] < 0 for face in side_faces)
    assert not any(face.normal[1] > 0 for face in side_faces)


def test_projection_keeps_vertical_axis_and_top_edge_upright():
    assert VIEW_EYE[1] < 0
    for z in (0., 4.7):
        assert project_point((0, -.4, z)).y() < project_point((0, .4, z)).y()


def test_model_renders_both_optics_at_native_and_high_dpi(app):
    for ratio in (1, 2):
        image = QImage(320 * ratio, 290 * ratio, QImage.Format_RGBA8888)
        image.setDevicePixelRatio(ratio)
        image.fill(QColor('#202023'))
        painter = QPainter(image)
        painter.setRenderHint(QPainter.Antialiasing)
        draw_stereo_camera(painter, project_point, VIEW_EYE)
        painter.end()
        for center in LENS_CENTERS:
            point = project_point(center)
            assert image.pixelColor(round(point.x() * ratio), round(point.y() * ratio)) != QColor('#202023')
