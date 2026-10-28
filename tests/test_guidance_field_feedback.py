"""Regressions for small-board duplicates and physically practical pose scheduling."""

from copy import deepcopy
import os
from pathlib import Path

import cv2
import numpy as np
import pytest

from multicamera.board.charuco_board import CharucoBoard, CharucoBoardConfig
from multicamera.board.detector import CharucoDetector, DetectionResult
from multicamera.calibration import adaptive_guidance as ag
from multicamera.calibration.intrinsic import IntrinsicCalibrator
from multicamera.calibration.models import CameraModel
from test_adaptive_guidance import observation, intrinsics, diverse
from test_multimodal_preview import LEFT, RIGHT
from test_stereo_capture_ui import sample

pytest_plugins = ['test_stereo_capture_ui']


@pytest.fixture
def board():
    return CharucoBoard(CharucoBoardConfig(cols=9, rows=7, square_length=.035, marker_length=.025))


@pytest.mark.parametrize('model', list(CameraModel))
def test_small_board_movement_is_not_duplicate_but_subpixel_noise_is(board, model):
    a = observation(board, model=model, depth=1.4)
    b = deepcopy(a)
    b.frame_id = 'changed'
    b.corners[:, :, 0] += 4
    assert not ag._similar(a, b)
    cal = IntrinsicCalibrator(board, model)
    cal.add_detection(DetectionResult(a.corners, a.ids, [], None, a.image_size), frame_id='old')
    assert cal.find_similar_detection(DetectionResult(b.corners, b.ids, [], None, b.image_size)) is None
    noise = deepcopy(a)
    noise.corners += .15
    assert ag._similar(a, noise)
    assert cal.find_similar_detection(
        DetectionResult(noise.corners, noise.ids, [], None, noise.image_size)) == 0
    tilted = observation(board, model=model, depth=1.4, angles=(25., 0, 0))
    assert not ag._similar(a, tilted)


def test_repeated_unfulfilled_tilt_cannot_dominate_all_next_requests(board):
    data = {camera: [observation(board, i, center=(.004 * i, 0)) for i in range(22)]
            for camera in ('L', 'R')}
    recommendations = [ag.analyze_guidance(board, CameraModel.PINHOLE,
                       {c: seq[:n] for c, seq in data.items()},
                       {c: intrinsics() for c in data}).recommendation for n in range(23)]
    assert len({r.tilt for r in recommendations[6:]}) >= 4
    assert all(len({r.tilt for r in recommendations[i:i + 4]}) > 1 for i in range(19))
    assert all('倾角可估计' in r.reason for r in recommendations)


def test_grouped_height_route_and_local_pose_variation(board):
    data = diverse(board, count=22)
    for sequence in data.values():
        for i, obs in enumerate(sequence):
            obs.corners[:, :, 0] += 3 * i
    recs = [ag.analyze_guidance(board, CameraModel.PINHOLE,
            {c: seq[:n] for c, seq in data.items()},
            {c: intrinsics() for c in data}).recommendation for n in range(23)]
    assert all((r.zone_row, r.zone_col) == (1, 1) for r in recs[:6])
    assert len({r.tilt for r in recs[:6]}) >= 4
    heights = [r.zone_row for r in recs]
    assert [h for i, h in enumerate(heights) if not i or h != heights[i - 1]] == [1, 0, 2]
    assert all(ag._candidate_feasible(board, (r.zone_row, r.zone_col, r.tilt, r.scale),
                                     [(1280, 960)] * 2) for r in recs)


def test_duplicate_requires_same_saved_pair_not_unrelated_eye_matches(stereo_bench):
    win = stereo_bench
    panel, guide = win._calib_panel, win._stereo_capture
    a, _, _ = sample(win)
    b = deepcopy(a)
    b.charuco_corners[:, :, 0] += 40
    for cid in (LEFT, RIGHT):
        panel._intr_cals[cid] = IntrinsicCalibrator(panel._board)
        for source, det in (('a', a), ('b', b)):
            panel._intr_cals[cid].add_detection(det, frame_id=source)
    panel._live_preview_indices['rgb'] = [(0, 0, 'a'), (1, 1, 'b')]
    pair = panel._sm.stereo_pairs['rgb']
    assert guide._duplicate_source(pair, (a, b)) is None
    assert guide._duplicate_source(pair, (a, a)) == 'a'
    panel._guidance_failed_saves.add((id(panel._session), 'rgb', 'a'))
    assert guide._duplicate_source(pair, (a, a)) is None


@pytest.mark.skipif(not os.environ.get('MULTICAMERA_FEEDBACK_IMAGES'), reason='Field data opt-in')
def test_real_feedback_images_agree_on_duplicate_classification():
    source = Path(os.environ['MULTICAMERA_FEEDBACK_IMAGES'])
    board = CharucoBoard(CharucoBoardConfig())
    detector = CharucoDetector(board)
    cals = {role: IntrinsicCalibrator(board) for role in ('left', 'right')}
    history = []
    for path in sorted(source.glob('left_*.png')):
        fid = path.stem.split('_', 1)[1]
        observations = []
        for role in cals:
            image = cv2.imdecode(np.frombuffer((source / f'{role}_{fid}.png').read_bytes(), np.uint8),
                                 cv2.IMREAD_COLOR)
            det = detector.detect(image)
            assert det.valid
            obs = ag.GuidanceObservation(det.charuco_corners, det.charuco_ids, det.image_size, fid)
            expected = next((i for i in range(len(history) - 1, -1, -1)
                             if ag._similar(obs, history[i][role])), None)
            assert cals[role].find_similar_detection(det) == expected
            cals[role].add_detection(det, frame_id=fid)
            observations.append(obs)
        history.append(dict(zip(cals, observations)))
    assert len(history) == 20
