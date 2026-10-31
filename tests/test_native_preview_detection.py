"""RGB preview detection must not trade source pixels for an implicit resize."""

import hashlib
import json
import os
from pathlib import Path
import time

import cv2
import numpy as np
import pytest

from multicamera.board.charuco_board import CharucoBoard, CharucoBoardConfig
from multicamera.board.detector import CharucoDetector, DetectionResult
from multicamera.streaming.stream_manager import StreamManager
from multicamera.ui.widgets import stream_view
from multicamera.ui.widgets.stream_view import MultiStreamView


@pytest.fixture
def view(app):
    widget = MultiStreamView(StreamManager(), CharucoBoard(CharucoBoardConfig()))
    yield widget
    widget.close()
    widget.deleteLater()


@pytest.mark.parametrize('shape', [(720, 1280, 3), (2160, 3840, 3), (720, 1280)])
def test_rgb_detector_receives_exact_native_frame_without_resizing(view, monkeypatch, shape):
    frame = np.zeros(shape, np.uint8)
    points = np.array([[[100., 100.]], [[200., 100.]], [[100., 200.]], [[200., 200.]]], np.float32)
    expected = DetectionResult(points, np.arange(4, dtype=np.int32)[:, None], [], None,
                               frame.shape[1::-1])
    received = []

    def detect(image):
        received.append(image)
        return expected

    monkeypatch.setattr(view._detector, 'detect', detect)
    monkeypatch.setattr(stream_view.cv2, 'resize',
                        lambda *args, **kwargs: pytest.fail('RGB preview must not resize source pixels'))
    result = view._detect_for_preview(frame)
    assert received[0] is frame
    assert result is expected
    np.testing.assert_array_equal(result.charuco_corners, points)


def test_native_preview_matches_direct_detector_and_does_not_modify_frame(view):
    board = view._board.generate_image(pixel_size=40, margin=20)
    frame = np.full((720, 1280, 3), 220, np.uint8)
    h, w = board.shape
    frame[100:100 + h, 200:200 + w] = cv2.cvtColor(board, cv2.COLOR_GRAY2BGR)
    before = frame.copy()
    direct = view._detector.detect(frame)
    preview = view._detect_for_preview(frame)
    assert preview.num_corners == direct.num_corners == view._board.num_corners
    assert preview.image_size == (1280, 720)
    np.testing.assert_array_equal(preview.charuco_ids, direct.charuco_ids)
    np.testing.assert_allclose(preview.charuco_corners, direct.charuco_corners, atol=0, rtol=0)
    np.testing.assert_array_equal(frame, before)


def test_rgb_detection_retains_per_camera_two_fps_cap(view, monkeypatch):
    clock = [100.]
    frame = np.zeros((720, 1280, 3), np.uint8)
    calls = []

    def detect(image):
        calls.append(image)
        return DetectionResult(None, None, [], None, image.shape[1::-1])

    monkeypatch.setattr(stream_view.time, 'monotonic', lambda: clock[0])
    monkeypatch.setattr(view._detector, 'detect', detect)
    monkeypatch.setattr(view, '_draw_detection', lambda image, *args, **kwargs: image)
    for step in range(4):
        clock[0] = 100. + .2 * step
        view._on_frame('left', frame, time.time())
        view._on_frame('right', frame, time.time())
    assert view._detect_min_interval == .5
    assert view.UI_FPS_CAP == 15
    assert len(calls) == 4  # Each camera detects at t=0 and t=0.6 only.
    assert all(image is frame for image in calls)


def test_auxiliary_planar_preview_keeps_existing_size_and_coordinate_policy(view, monkeypatch):
    from multicamera.calibration.planar import PlanarPatternDetector, PatternBoardConfig
    view._multimodal_mode = True
    detector = PlanarPatternDetector(PatternBoardConfig.ir_chessboard())
    small = np.full((192, 256, 3), 255, np.uint8)
    for row in range(5):
        for col in range(7):
            if (row + col) % 2 == 0:
                small[40 + row * 20:60 + row * 20, 40 + col * 20:60 + col * 20] = 0
    frame = cv2.resize(small, (1280, 960), interpolation=cv2.INTER_NEAREST)
    received = []

    def detect(image):
        received.append(image.shape[1::-1])
        return detector.detect(image)

    monkeypatch.setattr(view._planar_detector, 'detect', detect)
    result = view._detect_for_preview(frame)
    assert received == [(960, 720)]
    assert result.image_size == (1280, 960) and result.num_corners == 24
    direct = detector.detect(cv2.resize(frame, (960, 720), interpolation=cv2.INTER_AREA))
    np.testing.assert_allclose(result.charuco_corners.reshape(-1, 2), direct.image_points / .75,
                               atol=1e-4, rtol=0)


@pytest.mark.skipif(not os.environ.get('MULTICAMERA_PREVIEW_SESSION'), reason='Field images opt-in')
def test_field_native_preview_matches_raw_detection_and_preserves_originals(view, tmp_path):
    root = Path(os.environ['MULTICAMERA_PREVIEW_SESSION'])
    board = CharucoBoard.from_dict(json.loads((root / 'session.json').read_text())['board'])
    view.update_board(board)
    detector = CharucoDetector(board)
    rows = []
    for key in ('0005', '0006', '0019'):
        row = {'frame': key}
        for side in ('left', 'right'):
            path = root / f'images/rgb_pair_1/{side}_{key}.png'
            before = hashlib.sha256(path.read_bytes()).hexdigest()
            frame = cv2.imread(str(path))
            raw = detector.detect(frame)
            preview = view._detect_for_preview(frame)
            assert preview.valid and preview.num_corners == raw.num_corners
            np.testing.assert_array_equal(preview.charuco_ids, raw.charuco_ids)
            np.testing.assert_allclose(preview.charuco_corners, raw.charuco_corners, atol=0, rtol=0)
            assert hashlib.sha256(path.read_bytes()).hexdigest() == before
            scaled = detector.detect(cv2.resize(frame, (960, 540), interpolation=cv2.INTER_AREA))
            row[side] = {'native': raw.num_corners, 'preview': preview.num_corners,
                         'previous_scaled': scaled.num_corners}
        rows.append(row)
    (tmp_path / 'preview_comparison.json').write_text(json.dumps(rows, indent=2))
