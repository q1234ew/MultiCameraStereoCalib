"""Native-pixel region recovery must not weaken original-image admission."""

import cv2
import numpy as np
import pytest

from multicamera.calibration.aux_capture_quality import assess_aux_frame
from multicamera.calibration.planar import (
    PatternBoardConfig,
    PlanarDetectionResult,
    PlanarPatternDetector,
)


BOARD = PatternBoardConfig.ir_chessboard()


def scene(size=(1280, 720), origin=(480, 220), spacing=36, levels=(25, 225)):
    image = np.full((size[1], size[0]), 130, np.uint8)
    x, y = origin
    for row in range(BOARD.rows + 1):
        for col in range(BOARD.cols + 1):
            image[y + row * spacing:y + (row + 1) * spacing,
                  x + col * spacing:x + (col + 1) * spacing] = levels[(row + col) % 2]
    return image


def invalid(image):
    return PlanarDetectionResult(False, np.empty((0, 3), np.float32),
                                 np.empty((0, 2), np.float32), image.shape[1::-1],
                                 BOARD.pattern_type)


def require_region(monkeypatch, detector, *, forbid_enlargement=False):
    """Reproduce a full-image search miss, retaining actual ROI SB and quality."""
    original = detector._detect_sb
    calls = []

    def detect(gray, scale, roi, inverted):
        calls.append((scale, roi, inverted))
        if roi is None or (forbid_enlargement and scale != 1):
            return invalid(gray)
        return original(gray, scale, roi, inverted)

    monkeypatch.setattr(detector, 'detect', lambda image, **kw: invalid(image))
    monkeypatch.setattr(detector, '_detect_sb', detect)
    return calls


@pytest.mark.parametrize('size,origin,spacing', [
    ((640, 480), (30, 40), 24),
    ((1280, 720), (850, 430), 36),
    ((1920, 1080), (1050, 180), 55),
])
def test_rgb_recovers_full_board_at_new_positions_in_original_coordinates(
    monkeypatch, size, origin, spacing,
):
    image = scene(size, origin, spacing)
    before = image.copy()
    detector = PlanarPatternDetector(BOARD)
    calls = require_region(monkeypatch, detector)
    result = detector.detect_offline(image, modality='rgb')
    assert result.valid and result.num_points == 24
    assert result.image_size == size
    assert result.detection_info['method'] == 'sb_rgb_roi_native'
    assert result.detection_info['coordinate_space'] == 'original_pixels'
    assert assess_aux_frame(image, result, BOARD, modality='rgb').accepted
    expected = (np.array(origin) + np.array([1, 1]) * spacing - .5,
                np.array(origin) + np.array([6, 4]) * spacing - .5)
    np.testing.assert_allclose(result.image_points.min(axis=0), expected[0], atol=.6)
    np.testing.assert_allclose(result.image_points.max(axis=0), expected[1], atol=.6)
    assert all(scale == 1 for scale, roi, _ in calls if roi is not None)
    assert sum(roi is not None for _, roi, _ in calls) <= 3
    np.testing.assert_array_equal(image, before)


def test_ir_automatic_region_uses_native_pixels_before_enlargement(monkeypatch):
    image = scene((256, 192), (50, 50), 18)
    detector = PlanarPatternDetector(BOARD)
    calls = require_region(monkeypatch, detector, forbid_enlargement=True)
    result = detector.detect_offline(image, modality='ir')
    assert result.valid and result.num_points == 24
    assert result.detection_info['method'] == 'sb_roi_native'
    assert assess_aux_frame(image, result, BOARD, modality='ir').accepted
    assert calls[-1][0] == 1 and calls[-1][1] is not None
    assert not any(scale == 2 for scale, _, _ in calls)
    assert result.image_size == (256, 192)


@pytest.mark.parametrize('modality', ['rgb', 'ir'])
def test_failed_regions_do_not_remove_full_image_paths(monkeypatch, modality):
    image = scene((256, 192), (50, 50), 18)
    detector = PlanarPatternDetector(BOARD)
    monkeypatch.setattr(detector, '_rgb_board_rois', lambda gray: [])
    monkeypatch.setattr(detector, '_thermal_roi', lambda gray: None)
    result = detector.detect_offline(image, modality=modality)
    assert result.valid
    assert assess_aux_frame(image, result, BOARD, modality=modality).accepted
    assert result.detection_info['roi'] is None


def test_rgb_regions_are_relocated_after_board_moves_and_disappears(monkeypatch):
    detector = PlanarPatternDetector(BOARD)
    require_region(monkeypatch, detector)
    centres = []
    for origin in ((40, 40), (750, 430)):
        image = scene(origin=origin)
        result = detector.detect_offline(image, modality='rgb')
        assert result.valid
        centres.append(result.image_points.mean(axis=0))
    assert np.linalg.norm(centres[1] - centres[0]) > 500
    blank = np.full_like(image, 130)
    result = detector.detect_offline(blank, modality='rgb')
    assert not result.valid
    assert not assess_aux_frame(blank, result, BOARD, modality='rgb').accepted


@pytest.mark.parametrize('modality', ['rgb', 'ir'])
@pytest.mark.parametrize('failure', ['contrast', 'blur', 'margin', 'incomplete'])
def test_region_results_still_face_all_original_image_quality_gates(
    monkeypatch, modality, failure,
):
    # Supply an otherwise valid geometric observation, as a false positive or
    # weak image can come from any search. The real admission must reject it.
    origin = (0, 0) if failure == 'margin' else (50, 50)
    spacing = 18
    levels = (125, 127) if failure == 'contrast' else (25, 225)
    image = scene((256, 192), origin, spacing, levels)
    if failure == 'blur':
        image = cv2.GaussianBlur(image, (0, 0), spacing * .25)
    detector = PlanarPatternDetector(BOARD)
    grid = np.indices((4, 6))[::-1].transpose(1, 2, 0).reshape(-1, 2)
    points = np.array(origin) + (grid + 1) * spacing - .5
    if failure == 'incomplete':
        points = points[:-1]
    candidate = PlanarDetectionResult(True, detector._object_points.copy(),
                                      points.astype(np.float32), (256, 192), BOARD.pattern_type)
    monkeypatch.setattr(detector, 'detect', lambda image, **kw: invalid(image))
    monkeypatch.setattr(detector, '_rgb_board_rois', lambda gray: [(25, 25, 220, 150)])
    monkeypatch.setattr(detector, '_thermal_roi', lambda gray: (25, 25, 220, 150))
    monkeypatch.setattr(detector, '_detect_sb', lambda gray, scale, roi, inv:
                        candidate if roi is not None else invalid(gray))
    result = detector.detect_offline(image, modality=modality)
    quality = assess_aux_frame(image, result, BOARD, modality=modality)
    assert not quality.accepted
    reason = {'contrast': 'low_board_contrast', 'blur': 'blurred_board_edges',
              'margin': 'board_margin_too_small',
              'incomplete': 'invalid_or_incomplete_grid'}[failure]
    assert reason in quality.reasons
    assert not any(attempt['image_accepted'] for attempt in result.detection_info['attempts'])


def test_rgb_unrelated_rectangles_are_not_admitted(monkeypatch):
    image = np.full((480, 640), 130, np.uint8)
    for y in (70, 180, 290):
        for x in (100, 210, 320):
            cv2.rectangle(image, (x, y), (x + 45, y + 35), 20, -1)
    detector = PlanarPatternDetector(BOARD)
    require_region(monkeypatch, detector)
    result = detector.detect_offline(image, modality='rgb')
    assert not assess_aux_frame(image, result, BOARD, modality='rgb').accepted


def test_region_locator_handles_small_or_empty_scenes():
    detector = PlanarPatternDetector(BOARD)
    for shape in ((20, 20), (192, 256), (720, 1280)):
        assert detector._rgb_board_rois(np.full(shape, 130, np.uint8)) == []
