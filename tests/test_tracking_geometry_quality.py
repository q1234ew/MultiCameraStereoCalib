"""Ground-view geometry and visible human evidence without GPU or camera access."""
import json

import numpy as np
import pytest

from scripts.jetson.tracking_pose import person_pose_mask
from scripts.jetson.tracking_service import TrackingRuntime
from scripts.jetson.tracking_site import floor_view


def camera(rotation=None, model='pinhole'):
    k = np.array([[500., 0, 320], [0, 500, 240], [0, 0, 1]])
    transform = np.eye(4)
    transform[:3, :3] = np.diag([1., -1., -1.]) if rotation is None else rotation
    transform[:3, 3] = -transform[:3, :3] @ [0, 0, 2]
    return {'image_size': [640, 480], 'K': k.tolist(), 'K_rect': k.tolist(),
            'D': [0.] * (4 if model == 'fisheye' else 5), 'R': np.eye(3).tolist(),
            'model': model, 'camera_from_ground': transform.tolist(),
            'ground_from_rectified_pixel': np.linalg.inv(k @ transform[:3, [0, 1, 3]]).tolist()}


@pytest.mark.parametrize('scale', [-2., 1., 3.])
def test_downward_camera_floor_coverage_and_transform_direction(scale):
    calibration = camera()
    calibration['ground_from_rectified_pixel'] = (
        np.asarray(calibration['ground_from_rectified_pixel']) * scale).tolist()
    view = floor_view(calibration)
    points = np.asarray(view['polygon_xy_m'])
    assert view['camera_xyz_m'] == pytest.approx([0, 0, 2])
    assert points.min(axis=0) == pytest.approx([-1.28, -.956])
    assert points.max(axis=0) == pytest.approx([1.276, .96])
    json.dumps(view, allow_nan=False)


def test_view_crossing_horizon_is_clipped_and_all_points_are_in_front():
    calibration = camera(np.array([[1., 0, 0], [0, 0, -1], [0, 1, 0]]))
    view = floor_view(calibration)
    points = np.asarray(view['polygon_xy_m'])
    assert len(points) >= 3 and np.isfinite(points).all()
    assert abs(points).max() <= 20 + 1e-8
    transform = np.asarray(calibration['camera_from_ground'])
    for xy in points:
        assert (transform @ [*xy, 0, 1])[2] > 0
    assert points[:, 1].min() > 0


def test_upward_camera_and_missing_calibration_have_no_floor_view():
    assert floor_view(camera(np.eye(3))) is None
    assert floor_view({}) is None


def test_fisheye_coverage_uses_the_calibrated_distortion_model():
    calibration = camera(model='fisheye')
    first = floor_view(calibration)
    assert first and np.isfinite(first['polygon_xy_m']).all()
    calibration['D'] = [.15, 0, 0, 0]
    second = floor_view(calibration)
    assert not np.allclose(first['polygon_xy_m'], second['polygon_xy_m'])


def test_runtime_views_are_available_without_people_or_verified_capture_clock(tmp_path):
    runtime = TrackingRuntime({'devices': [{'id': 'device_1'}, {'id': 'device_2'}]},
                              tmp_path / 'unused.pt')
    runtime.site = {'world_frame': 'floor:test', 'cameras': {'device_1': camera()}}
    snapshot = runtime.snapshot()
    assert snapshot['tracks'] == []
    assert snapshot['cross_camera']['state'] == 'blocked'
    views = snapshot['cross_camera']['camera_views']
    assert len(views) == 1 and views[0]['camera_id'] == 'device_1'
    assert views[0]['world_frame'] == 'floor:test'
    runtime.site = None
    assert runtime.snapshot()['cross_camera']['camera_views'] == []


def pose():
    result = np.zeros((17, 3))
    for index, xy in {0: [100, 30], 5: [80, 65], 6: [120, 65],
                      7: [65, 100], 8: [135, 100], 11: [85, 120], 12: [115, 120],
                      15: [85, 180], 16: [115, 180]}.items():
        result[index] = [*xy, .9]
    return result


@pytest.mark.parametrize('kind', ['empty', 'weak', 'face_only', 'outside', 'nonfinite'])
def test_nonhuman_or_unusable_pose_evidence_is_rejected(kind):
    points = pose()
    if kind == 'empty':
        points[:] = 0
    elif kind == 'weak':
        points[:, 2] = .2
    elif kind == 'face_only':
        points[:] = [100, 40, .9]
        points[5:, 2] = 0
    elif kind == 'outside':
        points[:, 0] += 700
    else:
        points[:] = np.nan
    assert not person_pose_mask([[40, 10, 160, 190]], [points], [640, 480])[0]


@pytest.mark.parametrize('kind', ['full', 'seated', 'cropped', 'occluded'])
def test_visible_person_does_not_require_both_ankles_or_a_complete_body(kind):
    points = pose()
    if kind in ('seated', 'cropped'):
        points[11:, 2] = 0
    elif kind == 'occluded':
        points[[6, 8, 12, 16], 2] = 0
    assert person_pose_mask([[40, 10, 160, 190]], [points], [640, 480])[0]


def test_candidate_filter_preserves_input_order_and_empty_updates():
    valid, weak = pose(), pose()
    weak[:, 2] = .1
    mask = person_pose_mask([[40, 10, 160, 190]] * 3, [weak, valid, weak], [640, 480])
    assert mask.tolist() == [False, True, False]
    assert person_pose_mask(np.empty((0, 4)), np.empty((0, 17, 3)), [640, 480]).size == 0
    with pytest.raises(ValueError, match='align'):
        person_pose_mask([[40, 10, 160, 190]], [], [640, 480])


def test_confident_cropped_limb_is_allowed_only_at_the_image_edge():
    points = np.zeros((17, 3))
    points[7], points[9] = [20, 80, .9], [30, 110, .9]
    assert person_pose_mask([[0, 10, 160, 190]], [points], [640, 480], [.45])[0]
    assert not person_pose_mask([[5, 10, 160, 190]], [points], [640, 480], [.9])[0]
    assert not person_pose_mask([[0, 10, 160, 190]], [points], [640, 480], [.3])[0]
    points[9] = points[7]
    assert not person_pose_mask([[0, 10, 160, 190]], [points], [640, 480], [.9])[0]


def test_geometry_revision_changes_when_calibration_changes_in_same_world_frame(tmp_path):
    import copy

    runtime = TrackingRuntime({'devices': [{'id': 'device_1'}]}, tmp_path / 'unused.pt')
    runtime.site = {'world_frame': 'floor:test', 'cameras': {'device_1': camera()}}
    first = runtime.snapshot()['cross_camera']['geometry_revision']
    assert first and runtime.snapshot()['cross_camera']['geometry_revision'] == first
    runtime.site = copy.deepcopy(runtime.site)
    runtime.site['cameras']['device_1']['D'][0] = .02
    changed = runtime.snapshot()['cross_camera']
    assert changed['world_frame'] == 'floor:test'
    assert changed['geometry_revision'] != first
    runtime.site = None
    assert runtime.snapshot()['cross_camera']['geometry_revision'] is None
