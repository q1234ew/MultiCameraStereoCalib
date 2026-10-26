"""Distinguish board completeness, image coverage and duplicate composition."""

import numpy as np
import pytest

from multicamera.board.charuco_board import CharucoBoard, CharucoBoardConfig
from multicamera.board.detector import DetectionResult
from multicamera.calibration.assessment import DataAssessor, FrameQuality
from multicamera.calibration.intrinsic import IntrinsicCalibrator
from multicamera.ui.widgets.board_motion import BoardTilt, TILT_ACTIONS, tilt_point
from multicamera.ui.widgets.capture_guide import TILT_POSE_HINT, board_corners_f


def detection(board, shift=0.):
    ids = np.arange(board.num_corners, dtype=np.int32)[:, None]
    i = ids.ravel()
    points = np.column_stack((.51 + shift + (i % 14) * .001,
                              .51 + (i // 14) * .001)) * (1280, 720)
    return DetectionResult(points.astype(np.float32)[:, None], ids, [], None, (1280, 720))


def test_complete_board_is_not_complete_image_coverage():
    board = CharucoBoard(CharucoBoardConfig())
    cal = IntrinsicCalibrator(board)
    assert cal.data_revision == 0
    cal.add_detection(detection(board))
    assert cal.data_revision == 1
    assert cal.compute_coverage() == 1.
    assert cal.compute_image_coverage() == pytest.approx(1 / 16)
    cal.add_detection(detection(board, -.3))
    assert cal.compute_image_coverage() == pytest.approx(2 / 16)
    cal.pop_last_frame()
    assert cal.compute_image_coverage() == pytest.approx(1 / 16)
    cal.pop_last_frame()
    assert cal.compute_image_coverage() == 0
    assert cal.data_revision == 4
    cal.reset()
    assert cal.data_revision == 5


def test_duplicate_matches_ids_not_order_and_never_changes_data():
    board = CharucoBoard(CharucoBoardConfig())
    cal = IntrinsicCalibrator(board)
    original = detection(board)
    cal.add_detection(original)
    reordered = DetectionResult(original.charuco_corners[::-1].copy(),
                                original.charuco_ids[::-1].copy(), [], None, (1280, 720))
    assert cal.find_similar_detection(reordered) == 0
    assert cal.find_similar_detection(original, exclude_last=True) is None
    assert cal.find_similar_detection(detection(board, .03)) is None
    reordered.image_size = (640, 480)
    assert cal.find_similar_detection(reordered) is None
    partial = DetectionResult(original.charuco_corners[:6], original.charuco_ids[:6],
                              [], None, (1280, 720))
    assert cal.find_similar_detection(partial) is None
    assert cal.num_frames == 1


def test_concentrated_samples_cannot_receive_high_uniformity_or_coverage_score():
    assessor = DataAssessor()
    qualities = [FrameQuality(i, 100., 128., 126, False, False, False) for i in range(20)]
    concentrated = np.zeros((4, 4))
    concentrated[2, 2] = 100
    poor = assessor.assess_dataset('cam', qualities, 1., concentrated)
    broad = assessor.assess_dataset('cam', qualities, 1., np.ones((4, 4)))
    assert poor.coverage == broad.coverage == 1.
    assert poor.distribution_uniformity == pytest.approx(0.)
    assert broad.distribution_uniformity == pytest.approx(1.)
    assert poor.image_coverage == pytest.approx(1 / 16)
    assert poor.score < 50 < broad.score
    assert any('视场' in text for text in poor.suggestions)


@pytest.mark.parametrize('tilt', list(TILT_ACTIONS))
def test_legacy_diagram_and_animation_have_identical_near_edge(tilt):
    assert TILT_POSE_HINT[tilt] == TILT_ACTIONS[tilt][0]
    quad = board_corners_f(0, 0, 100, 70, tilt)
    left, right = np.linalg.norm(quad[3] - quad[0]), np.linalg.norm(quad[2] - quad[1])
    top, bottom = np.linalg.norm(quad[1] - quad[0]), np.linalg.norm(quad[2] - quad[3])
    if tilt == BoardTilt.TILT_LEFT:
        assert left > right and tilt_point(-1.45, 0, 1, tilt)[2] < 4.7
    elif tilt == BoardTilt.TILT_RIGHT:
        assert right > left and tilt_point(1.45, 0, 1, tilt)[2] < 4.7
    elif tilt == BoardTilt.TILT_UP:
        assert top > bottom and tilt_point(0, -1, 1, tilt)[2] < 4.7
    else:
        assert bottom > top and tilt_point(0, 1, 1, tilt)[2] < 4.7
