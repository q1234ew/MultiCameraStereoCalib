from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock

import cv2
import numpy as np
import pytest

from multicamera.calibration import auxiliary, joint_aux
from multicamera.calibration.aux_capture_quality import assess_aux_frame
from multicamera.calibration.models import CameraIntrinsics
from multicamera.calibration.planar import PlanarPatternDetector, PatternBoardConfig
from multicamera.ui.threads.import_worker import AuxImageImportWorker
from multicamera.ui.widgets.calib_panel import CalibrationPanel
from test_offline_aux_loop import board_image


def setup_geometry(monkeypatch, old_error=2.06, new_error=.14):
    image = board_image()
    detector = PlanarPatternDetector(PatternBoardConfig.ir_chessboard())
    old = detector.detect_offline(image)
    intr = CameraIntrinsics(np.array([[200., 0, 128], [0, 200, 96], [0, 0, 1]]),
                            np.zeros(5), (256, 192))
    monkeypatch.setattr(auxiliary, '_solve_board_pose', lambda *a: (np.zeros(3), np.ones(3)))
    monkeypatch.setattr(auxiliary, '_reprojection_error',
                        lambda det, *a: old_error if det is old else new_error)
    return image, detector, old, intr


def test_native_sb_recovers_geometry_with_original_quality_and_provenance(monkeypatch):
    image, detector, old, intr = setup_geometry(monkeypatch)
    original = image.copy(), old.image_points.copy(), deepcopy(intr.to_dict())
    result = detector.recheck_rgb_geometry(image, old, intr)
    assert result is not old and result.detection_info['method'] == 'sb_geometry_native'
    assert result.image_size == old.image_size and result.num_points == 24
    assert assess_aux_frame(image, result, detector.config, modality='rgb').accepted
    evidence = result.detection_info['geometry_recheck']
    assert evidence['status'] == 'recovered' and evidence['rms_limit_px'] == 1.5
    np.testing.assert_array_equal(evidence['original_points'], original[1])
    np.testing.assert_array_equal(image, original[0])
    np.testing.assert_array_equal(old.image_points, original[1])
    assert intr.to_dict() == original[2]


@pytest.mark.parametrize('error', [0., 1.5])
def test_healthy_or_boundary_geometry_does_not_run_sb(monkeypatch, error):
    image, detector, old, intr = setup_geometry(monkeypatch, old_error=error)
    monkeypatch.setattr(detector, '_detect_sb', Mock(side_effect=AssertionError('No retry')))
    assert detector.recheck_rgb_geometry(image, old, intr) is old


@pytest.mark.parametrize('error', [1.50001, float('inf')])
def test_sb_geometry_above_existing_limit_is_never_selected(monkeypatch, error):
    image, detector, old, intr = setup_geometry(monkeypatch, new_error=error)
    assert detector.recheck_rgb_geometry(image, old, intr) is old
    assert old.detection_info['geometry_recheck']['status'] == 'unresolved'


def test_geometrically_good_but_incomplete_sb_cannot_replace_original(monkeypatch):
    image, detector, old, intr = setup_geometry(monkeypatch)
    bad = deepcopy(old)
    bad.image_points = bad.image_points[:-1]
    monkeypatch.setattr(detector, '_detect_sb', lambda *a: bad)
    assert detector.recheck_rgb_geometry(image, old, intr) is old
    assert old.detection_info['geometry_recheck']['sb_quality_reasons']


def test_missing_or_mismatched_fixed_optics_never_used(monkeypatch):
    image, detector, old, intr = setup_geometry(monkeypatch)
    monkeypatch.setattr(detector, '_detect_sb', Mock(side_effect=AssertionError('No retry')))
    assert detector.recheck_rgb_geometry(image, old, None) is old
    intr.image_size = (640, 480)
    assert detector.recheck_rgb_geometry(image, old, intr) is old
    assert old.detection_info['geometry_recheck']['status'] == 'skipped'


def test_worker_snapshots_intrinsics_and_preserves_history_and_source_hashes(app, tmp_path, monkeypatch):
    image, detector, old, intr = setup_geometry(monkeypatch)
    for side in ('rgb_left', 'aux'):
        assert cv2.imwrite(str(tmp_path / f'{side}_0029.png'), image)
    prior = dict(state='待复核', reasons=['历史稳定性未通过'])
    import json
    (tmp_path / 'quality_0029.json').write_text(json.dumps(prior))
    monkeypatch.setattr(PlanarPatternDetector, 'detect_offline', lambda *a, **kw: deepcopy(old))
    monkeypatch.setattr(auxiliary, '_reprojection_error', lambda det, *a:
                        2.06 if det.detection_info.get('method') == 'classic_native' else .14)
    worker = AuxImageImportWorker(str(tmp_path), detector.config, offline_recheck=True,
                                  rgb_intrinsics=intr)
    intr.camera_matrix[0, 0] = 300
    assert worker.rgb_intrinsics.camera_matrix[0, 0] == 200
    result, errors = [], []
    worker.finished_ok.connect(result.append)
    worker.failed.connect(errors.append)
    worker.run()
    assert result and not errors
    assert worker.offline_reviews['0029']['capture_review'] == prior
    assert not worker.offline_reviews['0029']['included']
    assert worker.offline_reviews['0029']['physical_corner_order'] == 'unconfirmed'
    assert any('[0029]' in text and '重检恢复' in text for text in worker.geometry_messages)
    assert worker.offline_reviews['0029']['sources'][0]['sha256']


def test_changed_intrinsics_discard_completed_import_without_replacing_data():
    logs = []
    intr = CameraIntrinsics(np.eye(3), np.zeros(5), (256, 192))
    worker = SimpleNamespace(read_errors=[], board_config='board', import_context=None,
                             rgb_intrinsics_context={'different': True})
    panel = SimpleNamespace(_aux_import_worker=worker, _log=logs.append,
        _set_aux_import_busy=lambda _: None, _aux_board_config=lambda: 'board',
        _rgb_left_camera_id=lambda: 'rgb', _intrinsics={'rgb': intr})
    CalibrationPanel._on_import_aux_images_ready(panel, [])
    assert any('RGB固定内参' in text for text in logs)


def test_initializer_failure_identifies_frame_sensor_and_threshold(monkeypatch):
    from test_offline_aux_loop import synthetic_pairs, BOARD
    rgb, aux, _, records = synthetic_pairs()
    monkeypatch.setattr(joint_aux, '_correspondence_report',
                        lambda *a: dict(status='ambiguous', unresolved_ids=['0008']))
    monkeypatch.setattr(joint_aux, '_rotation_hypotheses', lambda *a: [])
    target = records[-1][1]
    monkeypatch.setattr(joint_aux, '_reprojection_error',
                        lambda det, *a: 2.06 if det is target else .1)
    with pytest.raises(ValueError, match=r'\[0008\] RGB.*2.060 px > 1.5 px'):
        joint_aux._fit(records, BOARD, rgb, 'fisheye_2')
