"""Selection uses matching source IDs and preserves source observations."""

import json
import threading

import cv2
import numpy as np
import pytest

from multicamera.board.charuco_board import CharucoBoard, CharucoBoardConfig
from multicamera.board.detector import DetectionResult
from multicamera.calibration import selection
from multicamera.calibration.intrinsic import IntrinsicCalibrator
from multicamera.calibration.models import CameraIntrinsics, CameraModel, MultiCameraRig
from multicamera.calibration.stereo import StereoCalibrator
from multicamera.ui import main_window
from test_calibration_workflow import seed_observations, wait_until


@pytest.fixture
def dataset():
    board = CharucoBoard(CharucoBoardConfig())
    left, right = (IntrinsicCalibrator(board, min_frames=5) for _ in range(2))
    stereo = StereoCalibrator(board, min_frames=5)
    image = np.indices((64, 96)).sum(axis=0) % 2 * 200 + 28
    image = image.astype(np.uint8)
    ids = np.arange(board.num_corners, dtype=np.int32)[:, None]
    points = np.column_stack((np.arange(len(ids)) % 14 + 10,
                              np.arange(len(ids)) // 14 + 10)).astype(np.float32)[:, None]
    det = DetectionResult(points, ids, [], None, (96, 64))
    for index in range(10):
        frame_id = f'{index * 2 + 1:04d}'
        assert left.add_detection(det, image, frame_id)
        assert right.add_detection(det, np.zeros_like(image) if index == 0 else image, frame_id)
        assert stereo.add_detection_pair(det, det, frame_id)
    return left, right, stereo


def test_quality_selection_scores_both_sides_of_same_source_pair(dataset):
    left, right, stereo = dataset
    filtered_l, filtered_r, filtered_s, info = selection.select_calibrators(*dataset, 'quality')
    assert '0001' in filtered_l.frame_ids
    assert '0001' not in filtered_r.frame_ids and '0001' not in filtered_s.frame_ids
    assert info['stereo']['excluded_ids'] == ['0001', '0019']
    assert info['left']['used_count'] == info['right']['used_count'] == info['stereo']['used_count'] == 8
    assert all(cal.num_frames == 10 for cal in dataset)
    assert all(image is None for image in filtered_l.stored_images)
    assert np.any(right.stored_images[1])
    assert not np.shares_memory(filtered_s._obj_points[0], stereo._obj_points[1])
    assert filtered_l.frame_ids == left.frame_ids[:8]


def test_all_policy_is_identity_with_complete_provenance(dataset):
    *chosen, info = selection.select_calibrators(*dataset, 'all')
    assert all(selected is source for selected, source in zip(chosen, dataset))
    assert all(not info[side]['excluded_ids'] for side in ('left', 'right', 'stereo'))
    assert info['stereo']['used_ids'] == dataset[2].frame_ids
    with pytest.raises(ValueError):
        selection.select_calibrators(*dataset, 'unrecognized')


def test_corner_policy_and_minimum_never_silently_discard_below_limit(dataset):
    for cal in dataset:
        cal.min_frames = 9
    *chosen, info = selection.select_calibrators(*dataset, 'corners')
    assert all(cal.num_frames == 9 and cal.ready for cal in chosen)
    assert info['stereo']['used_ids'] == dataset[2].frame_ids[:9]


def test_new_intrinsic_provenance_json_and_legacy_roundtrip():
    intr = CameraIntrinsics(np.eye(3), np.zeros(5), (640, 480), per_frame_errors=[.3, .5],
                            frame_ids=['0008', '0011'], per_frame_error_metric='euclidean_rms_px')
    restored = CameraIntrinsics.from_dict(json.loads(json.dumps(intr.to_dict())))
    assert restored.frame_ids == ['0008', '0011']
    assert restored.per_frame_error_metric == 'euclidean_rms_px'
    old = intr.to_dict()
    old.pop('frame_ids')
    old.pop('per_frame_error_metric')
    restored = CameraIntrinsics.from_dict(old)
    assert restored.frame_ids is None and restored.per_frame_error_metric is None


@pytest.mark.parametrize('model', list(CameraModel))
def test_strategy_flags_belong_to_selected_model(app, monkeypatch, tmp_path, model):
    monkeypatch.setattr(main_window, 'sessions_dir', lambda: tmp_path / 'sessions')
    win = main_window.MainWindow()
    panel = win._calib_panel
    try:
        panel.set_camera_model(model)
        panel._combo_calib_strategy.setCurrentIndex(1)
        intrinsic, stereo = panel._calib_strategy_flags()
        if model == CameraModel.FISHEYE:
            assert intrinsic == cv2.fisheye.CALIB_FIX_K3 | cv2.fisheye.CALIB_FIX_K4
            assert stereo == cv2.fisheye.CALIB_FIX_INTRINSIC
        else:
            assert intrinsic & cv2.CALIB_ZERO_TANGENT_DIST
            assert stereo == cv2.CALIB_FIX_INTRINSIC
        panel._combo_calib_strategy.setCurrentIndex(2)
        intrinsic, stereo = panel._calib_strategy_flags()
        assert intrinsic == 0
        assert stereo == (cv2.fisheye.CALIB_USE_INTRINSIC_GUESS if model == CameraModel.FISHEYE
                          else cv2.CALIB_USE_INTRINSIC_GUESS)
    finally:
        win.close()


@pytest.mark.parametrize('strategy', ['standard', 'joint_refine'])
def test_selection_runs_in_worker_and_saved_results_keep_ids(app, monkeypatch, tmp_path, strategy):
    monkeypatch.setattr(main_window, 'sessions_dir', lambda: tmp_path / 'sessions')
    win = main_window.MainWindow()
    panel = win._calib_panel
    seed_observations(win, count=10)
    main_thread = threading.get_ident()
    observed_threads = []
    original = selection.image_scores

    def tracked(cal):
        observed_threads.append(threading.get_ident())
        return original(cal)

    monkeypatch.setattr(selection, 'image_scores', tracked)
    panel._combo_data_filter.setCurrentIndex(panel._combo_data_filter.findData('quality'))
    panel._combo_calib_strategy.setCurrentIndex(panel._combo_calib_strategy.findData(strategy))
    try:
        panel._calibrate_pair('rgb')
        wait_until(app, lambda: not panel.has_active_background_tasks())
        assert observed_threads and all(thread != main_thread for thread in observed_threads)
        pair = win._rig.pairs['rgb']
        assert pair.calibration_info['selection']['stereo']['used_count'] == 8
        assert len(pair.left_intrinsics.frame_ids) == len(pair.left_intrinsics.per_frame_errors) == 8
        assert pair.left_intrinsics.rms_error < .001
        assert pair.right_intrinsics.rms_error < .001
        restored = MultiCameraRig.from_dict(json.loads(json.dumps(win._rig.to_dict())))
        assert restored.pairs['rgb'].calibration_info == pair.calibration_info
        assert restored.pairs['rgb'].left_intrinsics.per_frame_error_metric == 'euclidean_rms_px'
    finally:
        wait_until(app, lambda: not panel.has_active_background_tasks())
        win.close()
