"""Image-grid numbering is deterministic, reversible, and not physical correspondence."""

from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace
import tracemalloc

import cv2
import numpy as np
import pytest

from multicamera.calibration import grid_order
from multicamera.calibration.planar import (
    PatternBoardConfig, PatternType, PlanarDetectionResult, PlanarPatternDetector,
)


ORDERS = ('identity', 'reverse_rows', 'reverse_cols', 'reverse_both')
BOARD = PatternBoardConfig.ir_chessboard()


def permute(points, rows, cols, order):
    grid = np.asarray(points).reshape(rows, cols, -1)
    if order in ('reverse_rows', 'reverse_both'):
        grid = grid[::-1]
    if order in ('reverse_cols', 'reverse_both'):
        grid = grid[:, ::-1]
    return grid.reshape(len(points), -1).copy()


def detection(board=BOARD, angle=0):
    objects = PlanarPatternDetector._make_object_points(board)
    points = objects[:, :2].astype(float) / board.square_size * 20
    radians = np.radians(angle)
    rotation = np.array([[np.cos(radians), -np.sin(radians)],
                         [np.sin(radians), np.cos(radians)]])
    points = np.round((points - points.mean(axis=0)) @ rotation.T + [128, 96], 10)
    return PlanarDetectionResult(True, objects, points, (256, 192), board.pattern_type,
        dict(method='fixture', source_detector='fixture_detector', custom={'value': [1, 2]}))


def reference(points, board, size):
    allowed = grid_order.allowed_orders(PlanarPatternDetector._make_object_points(board),
                                       board.rows, board.cols)
    variants = [permute(points, board.rows, board.cols, order) for order in allowed]
    return min(variants, key=lambda p: (p[0, 0] / size[0] + p[0, 1] / size[1],
                                        p[0, 1], p[0, 0]))


@pytest.mark.parametrize('angle', [0, 90, 180, 270, 32])
@pytest.mark.parametrize('order', ORDERS)
def test_exact_permutations_share_image_numbering_and_are_reversible(angle, order):
    raw = detection(angle=angle)
    raw.image_points = permute(raw.image_points, BOARD.rows, BOARD.cols, order)
    before = deepcopy(raw)
    result = grid_order.canonicalize_detection(raw, BOARD)
    np.testing.assert_array_equal(result.image_points, reference(raw.image_points, BOARD, raw.image_size))
    np.testing.assert_array_equal(result.object_points, raw.object_points)
    np.testing.assert_array_equal(raw.image_points, before.image_points)
    np.testing.assert_array_equal(raw.object_points, before.object_points)
    assert raw.detection_info == before.detection_info
    assert result is not raw and result.valid
    assert not np.shares_memory(result.image_points, raw.image_points)
    assert not np.shares_memory(result.object_points, raw.object_points)
    info = result.detection_info
    restored = grid_order.reorder_points(result.image_points, BOARD.rows, BOARD.cols,
                                        info['normalization_order'])
    np.testing.assert_array_equal(restored, raw.image_points)
    assert info['scope'] == 'image_grid_only'
    assert info['physical_correspondence_verified'] is False
    assert info['method'] == 'fixture' and info['source_detector'] == 'fixture_detector'
    assert info['custom'] == {'value': [1, 2]}
    assert len(info['raw_fingerprint']) == len(info['normalized_fingerprint']) == 64
    assert (info['raw_fingerprint'] == info['normalized_fingerprint']) == (
        info['normalization_order'] == 'identity')
    again = grid_order.canonicalize_detection(result, BOARD)
    np.testing.assert_array_equal(again.image_points, result.image_points)
    assert again.detection_info == info  # Preserve the original detector lineage on repeated calls.
    result.detection_info['custom']['value'].append(3)
    assert raw.detection_info['custom']['value'] == [1, 2]


def test_canonical_origin_uses_normalized_xy_score_then_y_x_tiebreak():
    raw = detection()
    raw.image_points[:, 0] *= 2
    raw.image_size = (512, 192)
    result = grid_order.canonicalize_detection(raw, BOARD)
    np.testing.assert_array_equal(result.image_points, reference(raw.image_points, BOARD, raw.image_size))
    # Endpoints (20, 140) and (180, 20) tie on x/w+y/h; smaller y wins.
    raw.image_size = (320, 240)
    x, y = np.meshgrid(np.arange(6), np.arange(4))
    raw.image_points = np.column_stack((20 + 32 * x.ravel() + 10 * y.ravel(),
                                        140 - 24 * x.ravel() + 10 * y.ravel())).astype(float)
    result = grid_order.canonicalize_detection(raw, BOARD)
    np.testing.assert_array_equal(result.image_points, reference(raw.image_points, BOARD, raw.image_size))
    np.testing.assert_array_equal(result.image_points[0], [180, 20])


@pytest.mark.parametrize('rows,expected', [(4, ['identity', 'reverse_both']),
                                        (5, ['identity', 'reverse_rows'])])
def test_asymmetric_grids_only_normalize_rigid_orders(rows, expected):
    board = PatternBoardConfig(cols=6, rows=rows, square_size=.08,
                               pattern_type=PatternType.CIRCLES_GRID, asymmetric=True)
    raw = detection(board)
    assert grid_order.allowed_orders(raw.object_points, rows, 6) == expected
    first = grid_order.canonicalize_detection(raw, board)
    for order in expected:
        alternate = replace(raw, image_points=permute(raw.image_points, rows, 6, order))
        result = grid_order.canonicalize_detection(alternate, board)
        np.testing.assert_array_equal(result.image_points, first.image_points)
        assert result.detection_info['normalization_order'] in expected
    illegal = replace(raw, image_points=permute(raw.image_points, rows, 6, 'reverse_cols'))
    different = grid_order.canonicalize_detection(illegal, board)
    assert different.detection_info['normalization_order'] in expected
    assert not np.array_equal(first.image_points, different.image_points)


def test_allowed_orders_use_rigid_fit_without_quadratic_point_distances():
    board = PatternBoardConfig(cols=100, rows=100, square_size=.04)
    objects = PlanarPatternDetector._make_object_points(board)
    before = objects.copy()
    tracemalloc.start()
    try:
        assert grid_order.allowed_orders(objects, 100, 100) == list(ORDERS)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 32 * 1024 * 1024
    np.testing.assert_array_equal(objects, before)
    rotation = cv2.Rodrigues(np.array([.2, -.4, .6]))[0]
    rotated = objects.astype(float) @ rotation + [4., -3., 2.]
    assert grid_order.allowed_orders(rotated, 100, 100) == list(ORDERS)


@pytest.mark.parametrize('change', ['incomplete', 'nonfinite', 'wrong_shape', 'degenerate'])
def test_invalid_object_grids_are_rejected(change):
    objects = detection().object_points.copy()
    if change == 'incomplete':
        objects = objects[:-1]
    elif change == 'nonfinite':
        objects[0, 0] = np.nan
    elif change == 'wrong_shape':
        objects = objects[:, :2]
    else:
        objects[:] = 0
    with pytest.raises(ValueError):
        grid_order.allowed_orders(objects, BOARD.rows, BOARD.cols)


@pytest.mark.parametrize('change', ['invalid', 'incomplete', 'nonfinite', 'wrong_shape',
                                  'duplicate', 'size', 'pattern', 'object_grid'])
def test_invalid_detections_fail_closed_without_moving_raw_points(change):
    raw = detection()
    if change == 'invalid':
        raw.valid = False
    elif change == 'incomplete':
        raw.image_points = raw.image_points[:-1]
    elif change == 'nonfinite':
        raw.image_points[0, 0] = np.nan
    elif change == 'wrong_shape':
        raw.image_points = raw.image_points.reshape(-1)
    elif change == 'duplicate':
        raw.image_points[0] = raw.image_points[1]
    elif change == 'size':
        raw.image_size = (0, 192)
    elif change == 'pattern':
        raw.pattern_type = PatternType.CIRCLES_GRID
    else:
        raw.object_points[0, 0] += .01
    before = deepcopy(raw)
    result = grid_order.canonicalize_detection(raw, BOARD)
    assert not result.valid
    assert result.detection_info['normalization_order'] is None
    assert result.detection_info['scope'] == 'image_grid_only'
    assert result.detection_info['physical_correspondence_verified'] is False
    np.testing.assert_array_equal(raw.image_points, before.image_points)
    np.testing.assert_array_equal(raw.object_points, before.object_points)


@pytest.mark.parametrize('rows,cols,order', [(0, 6, 'identity'), (4, 0, 'identity'),
    (True, 6, 'identity'), (4.5, 6, 'identity'), (4, 6, 'transpose')])
def test_reordering_rejects_invalid_shape_and_unsupported_orders(rows, cols, order):
    with pytest.raises(ValueError):
        grid_order.reorder_points(detection().image_points, rows, cols, order)


def test_square_board_transpose_is_not_silently_treated_as_a_flip():
    board = PatternBoardConfig(cols=4, rows=4, square_size=.08)
    raw = detection(board)
    transposed = replace(raw, image_points=raw.image_points.reshape(4, 4, 2).transpose(1, 0, 2)
                         .reshape(-1, 2).copy())
    assert grid_order.allowed_orders(raw.object_points, 4, 4) == list(ORDERS)
    first = grid_order.canonicalize_detection(raw, board)
    second = grid_order.canonicalize_detection(transposed, board)
    assert not np.array_equal(first.image_points, second.image_points)
    assert second.detection_info['scope'] == 'image_grid_only'


def mock_classic(monkeypatch, points):
    monkeypatch.setattr(cv2, 'findChessboardCorners', lambda *a, **k: (True, points[:, None].copy()))
    monkeypatch.setattr(cv2, 'cornerSubPix', lambda gray, corners, *a, **k: corners)


@pytest.mark.parametrize('order', ORDERS)
def test_classic_detection_preserves_detector_coordinates_and_provenance(monkeypatch, order):
    points = permute(detection().image_points.astype(np.float32), 4, 6, order)
    mock_classic(monkeypatch, points)
    image = np.full((192, 256), 100, np.uint8)
    before = image.copy()
    result = PlanarPatternDetector(BOARD).detect(image, adaptive_subpixel=True)
    np.testing.assert_array_equal(result.image_points, detection().image_points.astype(np.float32))
    assert result.detection_info['method'] == 'classic_native'
    assert result.detection_info['source_detector'] == 'cv2.findChessboardCorners'
    assert result.detection_info['normalization_order'] == order
    np.testing.assert_array_equal(image, before)


@pytest.mark.parametrize('scale,roi,inverted', [(1, None, False), (1, None, True),
    (2, None, False), (2, (12, 16, 210, 150), False), (2, (12, 16, 210, 150), True)])
def test_sb_mapping_and_variants_use_original_pixel_numbering(monkeypatch, scale, roi, inverted):
    points = permute(detection().image_points.astype(np.float32), 4, 6, 'reverse_both')
    offset = np.array(roi[:2] if roi else (0, 0))
    work_points = ((points - offset + .5) * scale - .5).astype(np.float32)
    captured = []

    def find(work, *args, **kwargs):
        captured.append(work.copy())
        return True, work_points[:, None].copy()

    monkeypatch.setattr(cv2, 'findChessboardCornersSB', find)
    image = np.tile(np.arange(256, dtype=np.uint8), (192, 1))
    before = image.copy()
    result = PlanarPatternDetector(BOARD)._detect_sb(image, scale, roi, inverted)
    np.testing.assert_array_equal(result.image_points, detection().image_points.astype(np.float32))
    assert result.detection_info['normalization_order'] == 'reverse_both'
    assert result.detection_info['source_detector'] == 'cv2.findChessboardCornersSB'
    assert result.detection_info['coordinate_space'] == 'original_pixels'
    assert result.detection_info['scale'] == scale and result.detection_info['inverted'] is inverted
    assert result.image_size == (256, 192)
    assert len(captured) == 1
    if scale == 1 and roi is None:
        np.testing.assert_array_equal(captured[0], 255 - image if inverted else image)
    np.testing.assert_array_equal(image, before)


@pytest.mark.parametrize('wanted', ['classic_native', 'sb_native', 'sb_inverse', 'sb_2x',
                                   'sb_roi_2x', 'sb_roi_inverse_2x', 'rejected_all'])
def test_offline_selection_retains_normalization_metadata_on_every_path(monkeypatch, wanted):
    from multicamera.calibration import aux_capture_quality
    points = detection().image_points.astype(np.float32)
    mock_classic(monkeypatch, permute(points, 4, 6, 'reverse_rows'))
    detector = PlanarPatternDetector(BOARD)
    roi = (12, 16, 210, 150)
    monkeypatch.setattr(detector, '_thermal_roi', lambda gray: roi)
    original = detector._detect_sb

    def sb(gray, scale, region, inverted):
        offset = np.array(region[:2] if region else (0, 0))
        work_points = ((permute(points, 4, 6, 'reverse_cols') - offset + .5) * scale - .5)
        monkeypatch.setattr(cv2, 'findChessboardCornersSB',
                            lambda *a, **k: (True, work_points.astype(np.float32)[:, None].copy()))
        return original(gray, scale, region, inverted)

    monkeypatch.setattr(detector, '_detect_sb', sb)
    monkeypatch.setattr(aux_capture_quality, 'assess_aux_frame',
        lambda image, result, *a, **k: SimpleNamespace(accepted=result.detection_info['method'] == wanted))
    image = np.full((192, 256), 100, np.uint8)
    result = detector.detect_offline(image, modality='rgb' if wanted == 'classic_native' else 'ir')
    np.testing.assert_array_equal(result.image_points, points)
    info = result.detection_info
    if wanted == 'rejected_all':
        assert info['method'] == 'sb_native'
        assert info['attempts'][-1]['method'] == 'sb_roi_inverse_2x'
        assert not any(attempt['image_accepted'] for attempt in info['attempts'])
    else:
        assert info['method'] == wanted and info['attempts'][-1]['method'] == wanted
    assert info['normalization_order'] == ('reverse_rows' if wanted == 'classic_native' else 'reverse_cols')
    assert info['scope'] == 'image_grid_only' and info['physical_correspondence_verified'] is False
    assert len(info['raw_fingerprint']) == len(info['normalized_fingerprint']) == 64
    assert info['source_detector'] == ('cv2.findChessboardCorners' if wanted == 'classic_native'
                                       else 'cv2.findChessboardCornersSB')


def test_failed_detection_fallback_has_no_normalization_or_invented_observations(monkeypatch):
    monkeypatch.setattr(cv2, 'findChessboardCorners', lambda *a, **k: (False, None))
    monkeypatch.setattr(cv2, 'findChessboardCornersSB', lambda *a, **k: (False, None))
    image = np.full((192, 256), 100, np.uint8)
    before = image.copy()
    detector = PlanarPatternDetector(BOARD)
    monkeypatch.setattr(detector, '_thermal_roi', lambda gray: None)
    result = detector.detect_offline(image, modality='ir')
    assert not result.valid and result.image_points.shape == (0, 2)
    assert result.detection_info['normalization_order'] is None
    assert result.detection_info['scope'] == 'image_grid_only'
    assert result.detection_info['physical_correspondence_verified'] is False
    assert result.detection_info['method'] == 'sb_native'
    assert len(result.detection_info['attempts']) == 4
    np.testing.assert_array_equal(image, before)


@pytest.mark.parametrize('rows', [4, 5])
def test_circles_detector_uses_asymmetric_allowed_orders(monkeypatch, rows):
    board = PatternBoardConfig(cols=6, rows=rows, square_size=.08,
                               pattern_type=PatternType.CIRCLES_GRID, asymmetric=True)
    raw = detection(board)
    expected = reference(raw.image_points, board, raw.image_size)
    for order in grid_order.allowed_orders(raw.object_points, rows, 6):
        points = permute(raw.image_points, rows, 6, order).astype(np.float32)
        monkeypatch.setattr(cv2, 'findCirclesGrid', lambda *a, **k: (True, points[:, None].copy()))
        result = PlanarPatternDetector(board).detect_offline(np.zeros((192, 256), np.uint8))
        np.testing.assert_array_equal(result.image_points, expected.astype(np.float32))
        assert result.detection_info['source_detector'] == 'cv2.findCirclesGrid'
        assert result.detection_info['scope'] == 'image_grid_only'


def chessboard_image():
    image = np.full((192, 256), 125, np.uint8)
    for row in range(5):
        for col in range(7):
            image[40 + row * 20:60 + row * 20, 40 + col * 20:60 + col * 20] = (
                0 if (row + col) % 2 == 0 else 255)
    return image


@pytest.mark.parametrize('turns', [0, 1, 2, 3])
def test_real_classic_sb_and_inverse_keep_same_grid_orientation(turns):
    image = np.ascontiguousarray(np.rot90(chessboard_image(), turns))
    before = image.copy()
    detector = PlanarPatternDetector(BOARD)
    classic = detector.detect(image, adaptive_subpixel=True)
    inverse = detector.detect(255 - image, adaptive_subpixel=True)
    sb = detector._detect_sb(image, 1, None, False)
    sb_inverse = detector._detect_sb(image, 1, None, True)
    assert all(result.valid for result in (classic, inverse, sb, sb_inverse))
    for result in (inverse, sb, sb_inverse):
        np.testing.assert_allclose(result.image_points, classic.image_points, atol=.25, rtol=0)
        assert result.detection_info['scope'] == 'image_grid_only'
        assert result.detection_info['physical_correspondence_verified'] is False
    np.testing.assert_array_equal(image, before)
