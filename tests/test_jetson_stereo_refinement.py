"""Refinement must correct geometry without fitting an arbitrary depth multiplier."""
import json
import hashlib

import cv2
import numpy as np
import pytest

from scripts.jetson.refine_stereo import (
    apply_rotation_override, fit_geometry, fit_rotation_with_plane, load_stereo, write_candidate,
)


def correspondences():
    rng = np.random.default_rng(734)
    rotation = cv2.Rodrigues(np.array([.001, .005, -.002]))[0]
    translation = np.array([-50., .1, .4])
    points = rng.uniform([-.6, -.4, .8], [.6, .4, 4], (500, 3))
    right = (rotation@points.T).T+translation*.001
    left_uv, right_uv = points[:, :2]/points[:, 2:], right[:, :2]/right[:, 2:]
    plane = rng.uniform([-.2, -.15, 1.2], [.2, .15, 1.2], (60, 3))
    plane_right = (rotation@plane.T).T+translation*.001
    initial = cv2.Rodrigues(np.array([.002, -.002, .001]))[0]@rotation
    return initial, rotation, translation, left_uv, right_uv, plane_right, plane[:, :2]/plane[:, 2:]


def test_metric_rotation_recovers_tilt_and_yaw_with_translation_fixed():
    initial, truth, translation, left, right, plane, target = correspondences()
    r, t, report = fit_rotation_with_plane(left, right, plane, target, initial, translation)
    assert np.allclose(t, translation, atol=1e-10, rtol=0)
    assert np.linalg.norm(cv2.Rodrigues(r@truth.T)[0]) < 1e-5
    assert report['translation_fixed'] and report['metric_plane_points'] == 60
    assert np.max(report['candidate_plane_reprojection_px']) < 1e-3
    with pytest.raises(ValueError, match='metric plane'):
        fit_rotation_with_plane(left, right, plane*1000, target, initial, translation)


def test_scene_fit_keeps_metric_baseline_and_rejects_insufficient_points():
    initial, truth, translation, left, right, *_ = correspondences()
    r, t, _ = fit_geometry(left, right, initial, translation)
    assert np.linalg.norm(t) == pytest.approx(np.linalg.norm(translation), abs=1e-10)
    assert np.linalg.norm(cv2.Rodrigues(r@truth.T)[0]) < 1e-4
    with pytest.raises(ValueError):
        fit_geometry(left[:4], right[:4], initial, translation)


def source_calibration(path):
    fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_WRITE)
    k = np.array([[480., 0, 640], [0, 480, 360], [0, 0, 1]])
    fs.startWriteStruct('camera_intrinsics', cv2.FileNode_MAP)
    for role in ('left', 'right'):
        fs.startWriteStruct(role, cv2.FileNode_MAP)
        fs.write('K', k)
        fs.write('D', np.zeros(5))
        fs.endWriteStruct()
    fs.endWriteStruct()
    fs.startWriteStruct('extrinsics', cv2.FileNode_MAP)
    for name, t in [('T_RL', [-50., 0, 0]), ('T_IR_L', [1., 2, 3])]:
        fs.startWriteStruct(name, cv2.FileNode_MAP)
        fs.write('R', np.eye(3))
        fs.write('t', np.array(t).reshape(3, 1))
        fs.endWriteStruct()
    fs.endWriteStruct()
    fs.startWriteStruct('stereo_rectify', cv2.FileNode_MAP)
    p1 = np.column_stack((k, np.zeros(3)))
    p2 = p1.copy()
    p2[0, 3] = -24000
    for name, value in [('R1', np.eye(3)), ('R2', np.eye(3)), ('P1', p1),
                        ('P2', p2), ('Q', np.eye(4))]:
        fs.write(name, value)
    fs.endWriteStruct()
    fs.release()
    return k


def test_candidate_preserves_intrinsics_ir_translation_and_pixel_projection(tmp_path):
    source, destination = tmp_path/'source.yml', tmp_path/'candidate.yml'
    k = source_calibration(source)
    before = source.read_bytes()
    rotation = cv2.Rodrigues(np.array([.001, .002, .001]))[0]
    digest = write_candidate(source, destination, rotation, np.array([-50., 0, 0]))
    assert source.read_bytes() == before
    cameras, r, t = load_stereo(destination)
    assert np.array_equal(cameras['left']['K'], k) and np.allclose(r, rotation)
    assert np.array_equal(t, [-50., 0, 0])
    old, new = (cv2.FileStorage(str(p), cv2.FILE_STORAGE_READ) for p in (source, destination))
    try:
        for name in ('P1', 'P2'):
            a = old.getNode('stereo_rectify').getNode(name).mat()
            b = new.getNode('stereo_rectify').getNode(name).mat()
            assert np.array_equal(a, b)
        assert np.array_equal(new.getNode('extrinsics').getNode('T_IR_L').getNode('t').mat(),
                              [[1.], [2.], [3.]])
    finally:
        old.release()
        new.release()
    manifest = tmp_path/'override.json'
    spec = {'schema_version': 1, 'mode': 'rotation_only_metric_plane',
            'source_calibration_sha256': hashlib.sha256(before).hexdigest(),
            'candidate_calibration_sha256': digest, 'rotation': rotation.tolist(),
            'translation_mm': [-50., 0, 0]}
    manifest.write_text(json.dumps(spec))
    assert apply_rotation_override(source, tmp_path/'rebuilt.yml', manifest) == digest
    spec['status'] = 'withdrawn'
    manifest.write_text(json.dumps(spec))
    with pytest.raises(ValueError, match='withdrawn'):
        apply_rotation_override(source, tmp_path/'withdrawn.yml', manifest)
    assert not (tmp_path/'withdrawn.yml').exists()
    spec.pop('status')

    spec['translation_mm'][0] = -55
    manifest.write_text(json.dumps(spec))
    with pytest.raises(ValueError, match='unchanged translation'):
        apply_rotation_override(source, tmp_path/'unsafe.yml', manifest)
    with pytest.raises(ValueError, match='overwrite'):
        write_candidate(source, source, rotation, np.array([-50., 0, 0]))
