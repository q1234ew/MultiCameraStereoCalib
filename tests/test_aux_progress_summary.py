"""Fast auxiliary guidance must never wait for calibration or pose measurement."""

from dataclasses import replace

import cv2
import numpy as np
import pytest

from multicamera.calibration import aux_guidance as guidance
from multicamera.calibration.models import CameraIntrinsics, CameraModel
from multicamera.calibration.planar import (
    PatternBoardConfig, PatternType, PlanarDetectionResult, PlanarPatternDetector,
)
from multicamera.ui.threads import aux_capture_worker as workers


def samples(count):
    board = PatternBoardConfig.ir_chessboard()
    obj = PlanarPatternDetector._make_object_points(board)
    rows = []
    for i in range(count):
        pair = []
        for size, spacing in (((640, 480), 40), ((256, 192), 16)):
            points = obj[:, :2] / board.square_size * spacing + [30 + 2 * i, 35]
            pair.append(PlanarDetectionResult(
                True, obj.copy(), points.astype(np.float32), size, board.pattern_type))
        rows.append((str(i), *pair))
    intrinsics = tuple(CameraIntrinsics(
        np.array([[size[0] * .8, 0., size[0] / 2],
                  [0., size[0] * .8, size[1] / 2], [0., 0., 1.]]),
        np.zeros(5), size, rms_error=.2,
    ) for size in ((640, 480), (256, 192)))
    return board, rows, intrinsics


@pytest.fixture
def forbid_heavy_analysis(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('Progress summaries must not fit intrinsics or measure poses')

    for name in ('_bootstrap', 'measure_pose', 'measure_composition'):
        monkeypatch.setattr(guidance, name, forbidden)
    monkeypatch.setattr(guidance.AuxIntrinsicCalibrator, 'calibrate', forbidden)
    for name in ('calibrateCamera', 'solvePnP', 'solvePnPGeneric', 'solvePnPRansac'):
        monkeypatch.setattr(cv2, name, forbidden)
    monkeypatch.setattr(cv2.fisheye, 'calibrate', forbidden)


@pytest.mark.parametrize('count', [0, 1, 2])
@pytest.mark.parametrize('warm', [False, True])
def test_first_pass_advances_by_valid_unique_pairs_only(count, warm, forbid_heavy_analysis):
    board, rows, intrinsics = samples(count)
    if rows:
        _, rgb, aux = rows[0]
        rows = [('invalid-first', replace(rgb, valid=False), aux)] + rows + [
            ('repeat', rgb, aux),
            ('invalid-last', rgb, replace(aux, image_points=aux.image_points[:-1])),
        ]
    result = guidance.summarize_aux_progress(
        board, rows, intrinsics if warm else (None, None))
    assert isinstance(result, guidance.AuxGuidanceAnalysis)
    assert result.provisional is True
    assert result.phase == 'initial'
    assert result.target == guidance.fit_target(board, guidance.initial_plan()[count], (256, 192))
    assert result.unique_count == count
    assert result.duplicate_count == int(count > 0)
    assert result.initial_total == 17
    assert result.pose_count == 0 and result.distance_counts == {} and result.poses == []
    assert result.missing == () and result.repair_count == 0
    assert '待更新' in result.summary
    assert result.intrinsics == (intrinsics if warm else (None, None))
    if count:
        assert result.sample_decisions['invalid-first'] == 'invalid_detection'
        assert result.sample_decisions['repeat'] == 'duplicate'
        assert result.sample_decisions['invalid-last'] == 'invalid_detection'


@pytest.mark.parametrize('count', [17, 18, 21, 24])
@pytest.mark.parametrize('warm', [False, True])
def test_finished_first_pass_waits_for_statistics_without_repairs(
        count, warm, forbid_heavy_analysis):
    board, rows, intrinsics = samples(count)
    result = guidance.summarize_aux_progress(
        board, rows, intrinsics if warm else (None, None))
    assert result.unique_count == count
    assert result.phase == 'refining' and result.target is None
    assert result.provisional is True
    assert result.repair_count == 0 and result.missing == ()
    assert result.pose_count == 0 and result.distance_counts == {}


def test_fast_and_full_share_invalid_mixed_size_and_duplicate_decisions(monkeypatch):
    board, rows, _ = samples(2)
    _, rgb, aux = rows[0]
    nan_points = aux.image_points.copy()
    nan_points[0, 0] = np.nan
    incoming = [
        ('invalid-first', replace(rgb, image_size=(0, 480)), aux),
        rows[0],
        ('repeat', rgb, aux),
        ('reverse', replace(rgb, image_points=rgb.image_points[::-1]),
         replace(aux, image_points=aux.image_points[::-1])),
        ('mixed-rgb', replace(rgb, image_size=(1280, 960)), aux),
        ('mixed-ir', rgb, replace(aux, image_size=(512, 384))),
        ('mixed-and-invalid', replace(rgb, image_size=(1280, 960)),
         replace(aux, valid=False)),
        ('bad-count', rgb, replace(aux, image_points=aux.image_points[:-1])),
        ('nan', rgb, replace(aux, image_points=nan_points)),
        ('bad-board', replace(rgb, object_points=rgb.object_points * 2), aux),
        ('wrong-pattern', rgb, replace(aux, pattern_type=PatternType.CIRCLES_GRID)),
        ('bad-size', rgb, replace(aux, image_size=(True, 192))),
        ('bad-shape', rgb, replace(aux, image_points=aux.image_points.ravel())),
        ('collapsed', rgb, replace(aux, image_points=np.zeros_like(aux.image_points))),
        rows[1],
        ('rgb-repeat-only', rgb, rows[1][2]),
    ]
    monkeypatch.setattr(guidance, '_bootstrap', lambda *args: None)
    monkeypatch.setattr(guidance, 'measure_pose', lambda *args: None)
    fast = guidance.summarize_aux_progress(board, incoming)
    full = guidance.analyze_aux_guidance(board, CameraModel.PINHOLE, incoming)
    expected = {
        'invalid-first': 'invalid_detection', '0': 'included', 'repeat': 'duplicate',
        'reverse': 'duplicate', 'mixed-rgb': 'size_mismatch', 'mixed-ir': 'size_mismatch',
        'mixed-and-invalid': 'invalid_detection', 'bad-count': 'invalid_detection',
        'nan': 'invalid_detection', 'bad-board': 'invalid_detection',
        'wrong-pattern': 'invalid_detection', 'bad-size': 'invalid_detection',
        'bad-shape': 'invalid_detection', 'collapsed': 'invalid_detection',
        '1': 'included', 'rgb-repeat-only': 'included',
    }
    assert fast.sample_decisions == full.sample_decisions == expected
    assert fast.unique_count == full.unique_count == 3
    assert fast.duplicate_count == full.duplicate_count == 2
    assert fast.target == full.target and fast.phase == full.phase == 'initial'
    assert fast.provisional is True and full.provisional is False


@pytest.mark.parametrize('size, phase', [((32, 24), 'blocked'), ((256, 192), 'initial')])
def test_fast_progress_keeps_geometric_feasibility(size, phase, forbid_heavy_analysis):
    board, rows, _ = samples(1)
    rows = [(key, rgb, replace(aux, image_size=size)) for key, rgb, aux in rows]
    result = guidance.summarize_aux_progress(board, rows)
    assert result.phase == phase
    assert result.target == guidance.fit_target(board, guidance.initial_plan()[1], size)
    assert result.unique_count == 1


def test_cleanup_copies_without_resizing_or_reordering_sources(forbid_heavy_analysis):
    board, rows, _ = samples(1)
    key, rgb, aux = rows[0]
    rgb = replace(rgb, image_points=rgb.image_points.reshape(-1, 1, 2),
                  object_points=rgb.object_points.reshape(-1, 1, 3))
    aux = replace(aux, image_points=aux.image_points[::-1].copy())
    rows = [(key, rgb, aux), ('repeat', rgb, aux)]
    before = [(d.object_points.copy(), d.image_points.copy(), d.image_size) for d in (rgb, aux)]
    for det in (rgb, aux):
        det.object_points.flags.writeable = False
        det.image_points.flags.writeable = False
    result = guidance.summarize_aux_progress(board, rows)
    assert result.unique_count == result.duplicate_count == 1
    clean, _, _, sizes = guidance._clean_aux_pairs(board, rows)
    assert sizes == ((640, 480), (256, 192))
    for det, copied, (obj, points, size) in zip((rgb, aux), clean[0], before):
        np.testing.assert_array_equal(det.object_points, obj)
        np.testing.assert_array_equal(det.image_points, points)
        assert det.image_size == copied.image_size == size
        np.testing.assert_array_equal(copied.image_points, points.reshape(-1, 2))
        np.testing.assert_array_equal(copied.object_points, obj.reshape(-1, 3))
        assert not np.shares_memory(copied.object_points, det.object_points)
        assert not np.shares_memory(copied.image_points, det.image_points)


def test_progress_rejects_mismatched_intrinsics_without_bootstrap(forbid_heavy_analysis):
    board, rows, intrinsics = samples(1)
    result = guidance.summarize_aux_progress(
        board, rows, (intrinsics[0], replace(intrinsics[1], image_size=(512, 384))))
    assert result.intrinsics[0] is intrinsics[0]
    assert result.intrinsics[1] is None


def test_progress_worker_emits_key_and_provisional_result(app, forbid_heavy_analysis):
    board, rows, intrinsics = samples(2)
    key = ('session', 7)
    worker = workers.AuxGuidanceProgressWorker(key, board, rows, intrinsics)
    received = []
    worker.signals.completed.connect(lambda *args: received.append(args))
    worker.run()
    assert len(received) == 1
    received_key, result, error = received[0]
    assert received_key == key and error == ''
    assert result.provisional is True and result.unique_count == 2
    assert result.target == guidance.initial_plan()[2]


def test_progress_worker_reports_errors_with_same_signal(app, monkeypatch):
    def fail(*args):
        raise ValueError('summary failed')

    monkeypatch.setattr(workers, 'summarize_aux_progress', fail)
    worker = workers.AuxGuidanceProgressWorker('key', PatternBoardConfig.ir_chessboard(), [])
    received = []
    worker.signals.completed.connect(lambda *args: received.append(args))
    worker.run()
    assert received == [('key', None, 'summary failed')]
