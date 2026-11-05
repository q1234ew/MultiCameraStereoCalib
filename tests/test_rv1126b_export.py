"""RV1126B candidate fidelity, unit conversion and non-destructive export."""

import hashlib
import json

import cv2
import numpy as np
import pytest

from multicamera.calibration.models import (
    AuxCameraCalibration, CameraIntrinsics, CameraModel, MultiCameraRig,
    StereoCalibResult, StereoPairCalibration,
)
from multicamera.io.rv1126b import export_bundle


@pytest.fixture
def source(tmp_path):
    k = np.array([[460., 0., 640.], [0., 460., 360.], [0., 0., 1.]])
    intr = CameraIntrinsics(k, np.zeros((1, 14)), (1280, 720))
    stereo = StereoCalibResult(np.eye(3), np.array([[-.05], [0.], [0.]]),
                               np.eye(3), np.eye(3))
    pair = StereoPairCalibration('rgb', 'left', 'right', intr, intr, stereo)
    ir_k = np.array([[180., 0., 128.], [0., 180., 96.], [0., 0., 1.]])
    ir = CameraIntrinsics(ir_k, np.array([[-.2, .1, 0., 0.]]), (256, 192),
                          CameraModel.FISHEYE)
    transform = np.eye(4)
    transform[:3, 3] = [.025, .001, .003]
    aux = AuxCameraCalibration('ir', 'ir', ir, transform, calibration_info={
        'status': 'offline_candidate_pending_physical_validation',
        'rgb_binding': {'left_id': 'left', 'right_id': 'right',
                        'fixed_intrinsics': intr.to_dict()},
    })
    rig = MultiCameraRig(pairs={'rgb': pair}, reference_camera='left', aux_cameras={'ir': aux})
    path = tmp_path / 'source.json'
    path.write_text(json.dumps(rig.to_dict()), encoding='utf-8')
    return path


def bundle(source, output):
    return export_bundle(source, output, pair_name='rgb', aux_id='ir',
                         device_id='device_1', endpoint='192.0.2.1')


def read(path):
    return cv2.FileStorage(path.read_text(encoding='utf-8'),
                           cv2.FILE_STORAGE_READ | cv2.FILE_STORAGE_MEMORY)


def test_units_models_and_original_bytes(source, tmp_path):
    original = source.read_bytes()
    output = bundle(source, tmp_path / '标定 候选')
    assert source.read_bytes() == original == (output / 'source_rig.json').read_bytes()
    fs = read(output / 'calibration_mm.yml')
    assert fs.getNode('translation_unit').string() == 'mm'
    intr = fs.getNode('camera_intrinsics')
    assert intr.getNode('left').getNode('D').mat().size == 14
    assert intr.getNode('ir').getNode('model').string() == 'fisheye'
    assert intr.getNode('ir').getNode('D').mat().size == 4
    np.testing.assert_allclose(fs.getNode('extrinsics').getNode('T_RL').getNode('t').mat(),
                               [[-50.], [0.], [0.]])
    np.testing.assert_allclose(fs.getNode('extrinsics').getNode('T_IR_L').getNode('t').mat(),
                               [[25.], [1.], [3.]])
    p1, p2, q = [fs.getNode('stereo_rectify').getNode(k).mat() for k in ['P1', 'P2', 'Q']]
    x = np.array([10., 20., 1000., 1.])
    left_pixel, right_pixel = p1 @ x, p2 @ x
    back = q @ [left_pixel[0] / left_pixel[2], left_pixel[1] / left_pixel[2],
                left_pixel[0] / left_pixel[2] - right_pixel[0] / right_pixel[2], 1.]
    np.testing.assert_allclose(back[:3] / back[3], x[:3])
    fs.release()
    manifest = json.loads((output / 'manifest.json').read_text())
    assert not manifest['activation_allowed'] and not manifest['physical_binding_verified']
    for name, expected in manifest['files'].items():
        assert hashlib.sha256((output / name).read_bytes()).hexdigest() == expected


def test_existing_directory_cannot_be_replaced(source, tmp_path):
    output = bundle(source, tmp_path / 'candidate')
    before = {p.name: p.read_bytes() for p in output.iterdir()}
    with pytest.raises(FileExistsError):
        bundle(source, output)
    assert before == {p.name: p.read_bytes() for p in output.iterdir()}


def test_incomplete_adapter_sources_fail_before_directory_creation(source, tmp_path):
    with pytest.raises(ValueError, match='Incomplete'):
        export_bundle(source, tmp_path / 'candidate', pair_name='rgb', aux_id='ir',
                      device_id='device_1', endpoint='192.0.2.1', adapter_dir=tmp_path)
    assert not (tmp_path / 'candidate').exists()


def test_adapter_sources_are_included_in_hash_manifest(source, tmp_path):
    from pathlib import Path
    adapter = Path(__file__).resolve().parents[1] / 'cpp' / 'rv1126b'
    output = export_bundle(source, tmp_path / 'candidate', pair_name='rgb', aux_id='ir',
                           device_id='device_1', endpoint='192.0.2.1', adapter_dir=adapter)
    manifest = json.loads((output / 'manifest.json').read_text())
    for name in ['calibration_adapter.hpp', 'compatibility_test.cpp', 'CMakeLists.txt']:
        assert (output / name).read_bytes() == (adapter / name).read_bytes()
        assert hashlib.sha256((output / name).read_bytes()).hexdigest() == manifest['files'][name]


@pytest.mark.parametrize('defect', ['binding', 'unitless_legacy', 'nonfinite', 'reference'])
def test_invalid_inputs_never_create_bundle(source, tmp_path, defect):
    data = json.loads(source.read_text())
    if defect == 'binding':
        data['aux_cameras']['ir']['calibration_info']['rgb_binding']['right_id'] = 'wrong'
    elif defect == 'unitless_legacy':
        del data['aux_cameras']['ir']['calibration_info']['rgb_binding']
    elif defect == 'reference':
        data['reference_camera'] = 'right'
    else:
        data['aux_cameras']['ir']['T_rgb_left_aux'][0][3] = float('nan')
    source.write_text(json.dumps(data), encoding='utf-8')
    with pytest.raises(ValueError):
        bundle(source, tmp_path / 'candidate')
    assert not (tmp_path / 'candidate').exists()


def test_write_failure_removes_only_own_incomplete_candidate(source, tmp_path, monkeypatch):
    from pathlib import Path
    original = Path.write_bytes
    def fail(path, value):
        if path.name == 'test_vectors.yml':
            raise OSError('disk full')
        return original(path, value)
    monkeypatch.setattr(Path, 'write_bytes', fail)
    with pytest.raises(OSError, match='disk full'):
        bundle(source, tmp_path / 'candidate')
    assert source.exists() and not (tmp_path / 'candidate').exists()


@pytest.mark.parametrize('model', ['pinhole', 'fisheye'])
def test_projection_vectors_have_multiple_depths_and_correct_model(source, tmp_path, model):
    data = json.loads(source.read_text())
    intr = data['aux_cameras']['ir']['intrinsics']
    intr['model'] = model
    if model == 'pinhole':
        intr['dist_coeffs'] = [[.01, -.005, .001, .001, 0.]]
    source.write_text(json.dumps(data), encoding='utf-8')
    output = bundle(source, tmp_path / 'candidate')
    fs = read(output / 'test_vectors.yml')
    xyz = fs.getNode('ir_xyz_mm').mat()
    rgb = fs.getNode('rgb_xyz_mm').mat()
    assert len(xyz) == 27
    np.testing.assert_array_equal(np.unique(xyz[:, 2]), [500., 1000., 2000.])
    np.testing.assert_allclose(rgb - xyz, np.tile([25., 1., 3.], (27, 1)))
    fs.release()


def test_fisheye_stereo_is_rectified_in_millimetres(source, tmp_path):
    data = json.loads(source.read_text())
    pair = data['pairs']['rgb']
    pair['stereo']['model'] = 'fisheye'
    for key in ['left_intrinsics', 'right_intrinsics']:
        pair[key]['model'] = 'fisheye'
        pair[key]['dist_coeffs'] = [[.01, -.005, 0., 0.]]
        pair[key]['camera_matrix'][0][0] = 800.
        pair[key]['camera_matrix'][1][1] = 800.
    data['aux_cameras']['ir']['calibration_info']['rgb_binding']['fixed_intrinsics'] = (
        pair['left_intrinsics'])
    source.write_text(json.dumps(data), encoding='utf-8')
    output = bundle(source, tmp_path / 'candidate')
    fs = read(output / 'calibration_mm.yml')
    p2 = fs.getNode('stereo_rectify').getNode('P2').mat()
    assert abs(p2[0, 3] / p2[0, 0]) == pytest.approx(50.)
    fs.release()
