"""Identity and time/geometry contracts, independent of torch or cameras."""
import pytest

from scripts.jetson.person_association import GlobalAssociator
from scripts.jetson.tracking_service import TrackingRuntime


def obs(camera='device_1', local=1, epoch='a', x=1.0, sample=1, stamp=100.0):
    return {'camera_id': camera, 'local_id': local, 'epoch': epoch, 'sample_id': sample,
            'world_xy_m': [x, 1.0], 'world_frame': 'floor:test', 'position_verified': True,
            'capture_time_s': stamp, 'capture_uncertainty_s': 0.01}


def test_requires_independent_frames_before_linking():
    manager = GlobalAssociator()
    manager.update([obs()], 1)
    b = obs('device_2', x=1.1)
    for _ in range(4):
        result = manager.update([b], 1.01)
    assert len(result) == 2 and not any(t['global_id'] for t in result)
    for sample in (2, 3):
        b['sample_id'] = sample
        result = manager.update([b], 1.01 + sample * 0.01)
    assert len(result) == 1 and result[0]['global_id']
    assert len(result[0]['sources']) == 2


@pytest.mark.parametrize('change', [
    {'position_verified': False}, {'world_xy_m': None}, {'capture_time_s': None},
    {'capture_uncertainty_s': None}, {'capture_uncertainty_s': -1},
    {'capture_uncertainty_s': 0.2}, {'capture_time_s': 101},
    {'world_frame': 'other'}, {'world_frame': None}, {'world_xy_m': [float('nan'), 1]},
])
def test_missing_or_incompatible_evidence_never_merges(change):
    manager = GlobalAssociator(evidence_frames=1)
    manager.update([obs()], 1)
    other = obs('device_2')
    other.update(change)
    result = manager.update([other], 1.1)
    assert len(result) == 2
    assert all(t['global_id'] is None for t in result)


def test_no_two_local_people_from_same_camera_in_one_global_track():
    manager = GlobalAssociator(evidence_frames=1)
    manager.update([obs()], 1)
    result = manager.update([obs('device_2', local=1), obs('device_2', local=2)], 1.1)
    assert len(result) == 3  # Equal competitors must both remain unassociated.
    for track in result:
        cameras = [source['camera_id'] for source in track['sources']]
        assert len(cameras) == len(set(cameras))


def test_close_competing_people_are_ambiguous():
    manager = GlobalAssociator(evidence_frames=1)
    manager.update([obs(local=1, x=1), obs(local=2, x=1.2)], 1)
    result = manager.update([obs('device_2', x=1.1)], 1.1)
    assert len(result) == 3 and all(t['global_id'] is None for t in result)


def test_reconnect_epoch_prevents_id_reuse_and_silent_merge():
    manager = GlobalAssociator()
    first = manager.update([obs()], 1)[0]['track_uuid']
    result = manager.update([obs(epoch='reconnected')], 1.1)
    assert len(result) == 2
    assert result[1]['track_uuid'] != first
    assert not any(t['global_id'] for t in result)


def test_expiry_runs_without_new_frames():
    manager = GlobalAssociator(ttl_s=1)
    manager.update([obs()], 1)
    assert manager.snapshot(2.1) == []


def test_link_splits_when_geometry_diverges():
    manager = GlobalAssociator(evidence_frames=1)
    manager.update([obs()], 1)
    assert len(manager.update([obs('device_2')], 1.1)) == 1
    result = manager.update([obs('device_2', x=5, sample=2)], 1.2)
    assert len(result) == 2
    assert not any(t['global_id'] for t in result)


def test_runtime_clears_stale_images_and_tracks_without_fresh_frames(tmp_path):
    config = {'devices': [{'id': 'device_1'}]}
    runtime = TrackingRuntime(config, tmp_path / 'unused.pt')
    runtime.state = 'running'
    camera = runtime.cameras['device_1']
    camera.update(jpeg=b'image', tracks=[obs()], processed_mono=1)
    result = runtime.snapshot()
    assert result['cameras'][0]['state'] == 'unavailable'
    assert result['cameras'][0]['tracks'] == []
    assert runtime.jpeg('device_1') is None
    assert result['cross_camera']['state'] == 'blocked'


def test_empty_source_frame_marks_old_tracks_unobserved():
    manager = GlobalAssociator()
    manager.update([obs()], 1)
    result = manager.update([], 1.1, camera_id='device_1')
    assert len(result) == 1 and not result[0]['observed']


def test_floor_projection_uses_distortion_model_and_confident_ankles():
    import numpy as np

    from scripts.jetson.tracking_site import project_ankles

    k = np.array([[500., 0, 320], [0, 500, 240], [0, 0, 1]])
    transform = np.diag([1., -1., -1., 1.])
    transform[2, 3] = 2
    pixel_from_ground = k @ transform[:3, [0, 1, 3]]
    camera = {'K': k.tolist(), 'D': [0.] * 5, 'R': np.eye(3).tolist(),
              'K_rect': k.tolist(), 'model': 'pinhole', 'image_size': [640, 480],
              'ground_from_rectified_pixel': np.linalg.inv(pixel_from_ground).tolist(),
              'camera_from_ground': transform.tolist()}
    pose = np.zeros((17, 3))
    pose[15] = [315, 240, .9]
    pose[16] = [325, 240, .9]
    assert project_ankles(camera, pose, (640, 480)) == pytest.approx([0, 0])
    assert project_ankles(camera, pose, (1280, 720)) is None
    pose[15, 2] = .1
    assert project_ankles(camera, pose, (640, 480)) is None


def test_invalid_site_file_fails_closed(tmp_path):
    from scripts.jetson.tracking_site import load_site

    site = tmp_path / 'site.json'
    for content in ('[]', '{"floor_verified":false}', '{bad'):
        site.write_text(content)
        data, error = load_site(site, [])
        assert data is None and error


def test_floor_calibration_transform_direction_and_activation_contract(tmp_path):
    import hashlib

    import cv2
    import numpy as np

    from scripts.jetson.tracking_calibrate import build_site

    folder = tmp_path / 'device_1'
    folder.mkdir()
    path = folder / 'calibration_mm.yml'
    fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_WRITE)
    fs.startWriteStruct('camera_intrinsics', cv2.FileNode_MAP)
    fs.startWriteStruct('left', cv2.FileNode_MAP)
    k = np.array([[500., 0, 320], [0, 500, 240], [0, 0, 1]])
    fs.write('K', k)
    fs.write('D', np.zeros(5))
    fs.write('model', 'pinhole')
    fs.endWriteStruct()
    fs.endWriteStruct()
    fs.startWriteStruct('stereo_rectify', cv2.FileNode_MAP)
    fs.write('R1', np.eye(3))
    fs.write('P1', np.column_stack([k, np.zeros(3)]))
    fs.endWriteStruct()
    fs.release()
    devices = [{'id': 'device_1', 'calibration_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                'resolution': {'left': [640, 480]}}]
    tag_from_camera = np.eye(4)
    tag_from_camera[2, 3] = -2
    report = {'quality_passed': True, 'anchor_tag_id': 5, 'cameras': {
        'device_1': {'quality_passed': True, 'T_world_camera': tag_from_camera.tolist(),
                     'statistics': {}}}}
    site = build_site(report, devices, tmp_path)
    import json

    from scripts.jetson.tracking_site import load_site

    site_path = tmp_path / 'site.json'
    site_path.write_text(json.dumps(site))
    validated, error = load_site(site_path, devices)
    assert validated is not None and error is None
    camera = site['cameras']['device_1']
    assert camera['camera_height_m'] == 2
    assert site['world_frame'].startswith('floor:')
    assert not site['measurement_accuracy_verified']
    ground_from_pixel = np.asarray(camera['ground_from_rectified_pixel'])
    xyh = ground_from_pixel @ [320, 240, 1]
    assert (xyh[:2] / xyh[2]).tolist() == pytest.approx([0, 0])
    report['quality_passed'] = False
    with pytest.raises(ValueError, match='fresh'):
        build_site(report, devices, tmp_path)


def test_partial_ground_site_has_explicit_participants_and_full_config_hash(tmp_path):
    import copy
    import hashlib
    import json

    import numpy as np

    from scripts.jetson.tracking_site import load_site

    devices = [{'id': name, 'calibration_sha256': 'bundle-'+name,
                'resolution': {'left': [640, 480]}} for name in ('device_1','device_2','device_3')]
    k = np.array([[500.,0,320],[0,500,240],[0,0,1]])
    transform = np.eye(4)
    transform[2,3] = 2
    projection = k @ transform[:3,[0,1,3]]
    camera = {'K': k.tolist(), 'D': [0]*5, 'R': np.eye(3).tolist(),
              'K_rect': k.tolist(), 'camera_from_ground': transform.tolist(),
              'ground_from_rectified_pixel': np.linalg.inv(projection).tolist(),
              'image_size': [640,480]}
    site = {'schema_version':1, 'floor_verified':True, 'translation_unit':'m',
            'world_frame':'floor:test', 'participating_camera_ids':['device_1','device_3'],
            'device_config_sha256':hashlib.sha256(json.dumps(devices,sort_keys=True).encode()).hexdigest(),
            'cameras':{d['id']:dict(camera,calibration_sha256=d['calibration_sha256'])
                       for d in devices if d['id']!='device_2'}}
    path=tmp_path/'site.json'
    path.write_text(json.dumps(site))
    valid,error=load_site(path,devices)
    assert valid is not None and error is None
    for ids in (['device_1'], ['device_1','device_1'], ['device_1','unknown'],
                ['device_1','device_2']):
        bad=copy.deepcopy(site)
        bad['participating_camera_ids']=ids
        path.write_text(json.dumps(bad))
        assert load_site(path,devices)[0] is None
    path.write_text(json.dumps(site))
    changed=copy.deepcopy(devices)
    changed[0]['calibration_sha256']='replacement'
    assert load_site(path,changed)[0] is None
    del site['participating_camera_ids']
    path.write_text(json.dumps(site))
    assert load_site(path,devices)[0] is None  # Legacy files still require full coverage.


def test_partial_ground_readiness_excludes_third_camera_but_requires_pair_clock():
    import time

    devices=[{'id':name} for name in ('device_1','device_2','device_3')]
    runtime=TrackingRuntime({'devices':devices},'unused.pt')
    runtime.state='running'
    runtime.site={'world_frame':'floor:test','cameras':{'device_1':{},'device_3':{}}}
    runtime.site_error=None
    for name in ('device_1','device_3'):
        runtime.cameras[name].update(processed_mono=time.monotonic(),capture_clock_ok=True)
    result=runtime.snapshot()['cross_camera']
    assert result['state']=='ready'
    assert result['excluded_camera_ids']==['device_2']
    runtime.cameras['device_3']['capture_clock_ok']=False
    result=runtime.snapshot()['cross_camera']
    assert result['geometry_ready'] and result['state']=='blocked'


def test_ground_projections_without_clock_remain_separate_local_positions():
    manager=GlobalAssociator(evidence_frames=1)
    first=obs()
    first['capture_time_s']=None
    second=obs('device_3',x=1.1)
    second['capture_time_s']=None
    manager.update([first],1)
    result=manager.update([second],1.1)
    assert len(result)==2
    assert all(t['world_xy_m'] is not None and t['global_id'] is None for t in result)
    assert all(not t['position_time_verified'] for t in result)


@pytest.mark.parametrize('reverse', [False, True])
def test_source_competition_chooses_clear_closest_independent_of_order(reverse):
    manager = GlobalAssociator()
    manager.update([obs(x=1)], 1)
    for sample in range(1, 4):
        candidates = [obs('device_2', local=1, x=1.50, sample=sample),
                      obs('device_2', local=2, x=1.05, sample=sample)]
        result = manager.update(candidates[::-1] if reverse else candidates, 1 + sample * .01)
    linked = [track for track in result if track['global_id']]
    assert len(linked) == 1
    assert {(s['camera_id'], s['local_id']) for s in linked[0]['sources']} == {
        ('device_1', 1), ('device_2', 2)}


@pytest.mark.parametrize('reverse', [False, True])
def test_source_competition_rejects_close_rivals(reverse):
    manager = GlobalAssociator()
    manager.update([obs()], 1)
    for sample in range(1, 5):
        candidates = [obs('device_2', local=1, x=1.1, sample=sample),
                      obs('device_2', local=2, x=1.2, sample=sample)]
        result = manager.update(candidates[::-1] if reverse else candidates, 1 + sample * .01)
    assert len(result) == 3
    assert all(t['global_id'] is None for t in result)


def test_alternating_three_fps_frames_keep_established_identity():
    manager = GlobalAssociator()
    gid = None
    for sample in range(1, 12):
        stamp = 100 + sample / 3
        first = manager.update([obs(sample=sample, stamp=stamp)], 1 + sample / 3)
        if gid:
            assert len(first) == 1 and first[0]['global_id'] == gid
            assert not first[0]['position_time_verified']
        second = manager.update([obs('device_2', sample=sample, stamp=stamp + .05)],
                                1 + sample / 3 + .05)
        if second[0]['global_id']:
            gid = second[0]['global_id']
            assert len(second) == 1 and second[0]['position_time_verified']
    assert gid is not None
    assert not any(event['kind'] == 'split' for event in manager.events)


def test_wait_for_peer_is_bounded_and_does_not_average_old_positions():
    manager = GlobalAssociator(evidence_frames=1)
    manager.update([obs(x=1)], 1)
    gid = manager.update([obs('device_2', x=1.1, stamp=100.05)], 1.05)[0]['global_id']
    result = manager.update([obs(x=3, stamp=100.4, sample=2)], 1.4)
    assert len(result) == 1 and result[0]['global_id'] == gid
    assert result[0]['world_xy_m'] == pytest.approx([3, 1])
    assert not result[0]['position_time_verified']
    result = manager.update([obs(x=3, stamp=101, sample=3)], 2)
    assert len(result) == 2 and all(t['global_id'] is None for t in result)


@pytest.mark.parametrize('absence', ['empty', 'aged', 'reconnect'])
def test_absent_observations_cannot_supply_new_association_evidence(absence):
    manager = GlobalAssociator(evidence_frames=1)
    manager.update([obs()], 1)
    now = 1.1
    if absence == 'empty':
        manager.update([], 1.05, camera_id='device_1')
    elif absence == 'aged':
        now = 1.9
    else:
        manager.update([obs(epoch='new', x=5)], 1.05)
    result = manager.update([obs('device_2')], now)
    assert not any(t['global_id'] for t in result)


def test_trajectory_records_new_samples_and_expires_by_time_and_count():
    manager = GlobalAssociator()
    for sample in range(400):
        result = manager.update([obs(x=sample / 100, sample=sample)], 1 + sample / 50)
    points = result[0]['trajectory']
    assert len(points) == 256
    assert points[-1]['age_s'] == 0
    assert manager.snapshot(1 + 399 / 50 + .1)[0]['trajectory'][-1]['age_s'] == .1
    assert len(manager.snapshot(1 + 399 / 50 + .2)[0]['trajectory']) == 256
    # At 3 FPS the time bound, rather than the count bound, applies.
    manager = GlobalAssociator()
    for sample in range(40):
        result = manager.update([obs(sample=sample)], 1 + sample / 3)
    assert 24 <= len(result[0]['trajectory']) <= 25
    assert all(0 <= p['age_s'] <= 8 for p in result[0]['trajectory'])


def test_trajectory_breaks_on_absence_and_invalid_geometry():
    manager = GlobalAssociator()
    manager.update([obs(sample=1)], 1)
    assert len(manager.update([obs(x=1.2, sample=2)], 1.1)[0]['trajectory']) == 2
    assert manager.update([], 1.2, camera_id='device_1')[0]['trajectory'] == []
    assert len(manager.update([obs(sample=3)], 1.3)[0]['trajectory']) == 1
    invalid = obs(sample=4)
    invalid['position_verified'] = False
    assert manager.update([invalid], 1.4)[0]['trajectory'] == []
    assert len(manager.update([obs(sample=5)], 1.5)[0]['trajectory']) == 1
    assert manager.snapshot(2.4)[0]['trajectory'] == []
    assert len(manager.update([obs(sample=6)], 2.5)[0]['trajectory']) == 1


def test_trajectory_does_not_bridge_reconnect_membership_or_world_frame():
    manager = GlobalAssociator(evidence_frames=1)
    first = manager.update([obs()], 1)[0]
    manager.update([obs(sample=2)], 1.1)
    joined = manager.update([obs('device_2')], 1.2)[0]
    assert len(joined['trajectory']) == 1
    assert len(joined['sources']) == 2
    result = manager.update([obs(epoch='new', x=4)], 1.3)
    reconnected = next(t for t in result if t['track_uuid'] != first['track_uuid'])
    assert len(reconnected['trajectory']) == 1
    changed = obs(epoch='new', x=4, sample=2)
    changed['world_frame'] = 'floor:new'
    result = manager.update([changed], 1.4)
    changed_track = next(t for t in result if t['world_frame'] == 'floor:new')
    assert len(changed_track['trajectory']) == 1
    manager.snapshot(5)
    assert not manager.trajectories



def test_repeated_source_sample_does_not_extend_the_trajectory():
    manager = GlobalAssociator()
    manager.update([obs()], 1)
    result = manager.update([obs()], 1.1)
    assert len(result[0]['trajectory']) == 1
    assert result[0]['trajectory'][0]['age_s'] == .1


def test_runtime_session_reset_drops_trajectory_history(tmp_path):
    runtime = TrackingRuntime({'devices': [{'id': 'device_1'}]}, tmp_path / 'unused.pt')
    runtime.state = 'running'
    runtime.manager.update([obs()], 1)
    runtime.manager.update([obs(sample=2)], 1.1)
    assert runtime.manager.trajectories
    runtime.configure({'new_track_conf': .4}, runtime.options.revision)
    assert runtime.manager.trajectories == {}
    assert runtime.snapshot()['tracks'] == []
