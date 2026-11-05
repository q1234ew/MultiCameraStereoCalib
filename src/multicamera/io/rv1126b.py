"""Isolated, model-aware RV1126B C++ compatibility bundles (never activate devices)."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from pathlib import Path

import cv2
import numpy as np

from ..calibration.models import CameraModel, MultiCameraRig
from .export import _validate_rig


def _project(points, intr):
    points = np.asarray(points, np.float64).reshape(-1, 1, 3)
    if intr.model == CameraModel.FISHEYE:
        return cv2.fisheye.projectPoints(points, np.zeros(3), np.zeros(3),
                                        intr.camera_matrix, intr.dist_coeffs)[0]
    return cv2.projectPoints(points, np.zeros(3), np.zeros(3),
                             intr.camera_matrix, intr.dist_coeffs)[0]


def _vectors(pair, aux):
    intr = aux.intrinsics
    w, h = intr.image_size
    pixels = np.array([[[x, y]] for y in (0., h / 2., h - 1.)
                       for x in (0., w / 2., w - 1.)], np.float64)
    if intr.model == CameraModel.FISHEYE:
        rays = cv2.fisheye.undistortPoints(pixels, intr.camera_matrix, intr.dist_coeffs)
    else:
        rays = cv2.undistortPointsIter(pixels, intr.camera_matrix, intr.dist_coeffs,
                                     None, None, (3, 100, 1e-12))
    rays = np.column_stack((rays.reshape(-1, 2), np.ones(len(rays))))
    ir = np.vstack([rays * depth for depth in (500., 1000., 2000.)])
    transform = aux.T_rgb_left_aux
    rgb = ir @ transform[:3, :3].T + transform[:3, 3] * 1000.
    if np.any(rgb[:, 2] <= 0):
        raise ValueError('Test points behind RGB camera')
    return {'ir_xyz_mm': ir, 'rgb_xyz_mm': rgb,
            'ir_pixels': _project(ir, intr).reshape(-1, 2),
            'rgb_pixels': _project(rgb, pair.left_intrinsics).reshape(-1, 2)}


def export_bundle(source: Path, output: Path, *, pair_name: str, aux_id: str,
                  device_id: str, endpoint: str, adapter_dir: Path | None = None,
                  device_mac: str | None = None) -> Path:
    """Write a new candidate directory; refuse replacement and retain source bytes."""
    source, output = Path(source), Path(output)
    if not re.fullmatch(r'[A-Za-z0-9_-]+', device_id):
        raise ValueError('Device ID must be an ASCII path-safe identifier')
    if device_mac is not None and not re.fullmatch(r'(?:[0-9a-fA-F]{2}:){5}[0-9a-fA-F]{2}',
                                                  device_mac):
        raise ValueError('Invalid device MAC')
    raw = source.read_bytes()
    rig = MultiCameraRig.from_dict(json.loads(raw))
    _validate_rig(rig)
    pair, aux = rig.pairs[pair_name], rig.aux_cameras[aux_id]
    binding = aux.calibration_info.get('rgb_binding', {})
    if binding.get('left_id') != pair.left_id or binding.get('right_id') != pair.right_id:
        raise ValueError('Auxiliary result is not bound to the selected stereo pair')
    if rig.reference_camera != pair.left_id or aux.modality != 'ir':
        raise ValueError('Bundle requires RGB-left reference and IR auxiliary')
    left, right = pair.left_intrinsics, pair.right_intrinsics
    stereo = pair.stereo
    translation = np.asarray(stereo.T, np.float64).reshape(3, 1) * 1000.
    if stereo.model == CameraModel.FISHEYE:
        rect = cv2.fisheye.stereoRectify(
            left.camera_matrix, left.dist_coeffs, right.camera_matrix, right.dist_coeffs,
            left.image_size, stereo.R, translation, flags=cv2.CALIB_ZERO_DISPARITY)
    else:
        rect = cv2.stereoRectify(
            left.camera_matrix, left.dist_coeffs, right.camera_matrix, right.dist_coeffs,
            left.image_size, stereo.R, translation, flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)
    vectors = _vectors(pair, aux)
    if not all(np.isfinite(value).all() for value in vectors.values()):
        raise ValueError('Non-finite projection test vectors')
    adapter_files = []
    if adapter_dir is not None:
        adapter_files = [Path(adapter_dir) / name for name in (
            'calibration_adapter.hpp', 'compatibility_test.cpp', 'CMakeLists.txt', 'README.md',
            'legacy_loader_probe.cpp')]
        if any(not f.is_file() for f in adapter_files):
            raise ValueError('Incomplete C++ adapter source directory')
    output.mkdir(parents=True, exist_ok=False)
    try:
        # ASCII memory serialization also works for Unicode output paths on Windows.
        fs = cv2.FileStorage('calibration.yml', cv2.FILE_STORAGE_WRITE |
                             cv2.FILE_STORAGE_MEMORY | cv2.FILE_STORAGE_FORMAT_YAML)
        for k, v in {'version': 'multicamera_rv1126b_v1', 'coord_frame': 'C_L',
                     'translation_unit': 'mm', 'device_id': device_id,
                     'binding_status': 'physical_device_unverified',
                     'qualification': 'candidate_pending_physical_validation',
                     'requires_model_aware_adapter': 1,
                     'stereo_transform_direction': 'left_to_right',
                     'ir_transform_direction': 'ir_to_left'}.items():
            fs.write(k, v)
        fs.startWriteStruct('camera_intrinsics', cv2.FileNode_MAP)
        for name, camera_id, intr in [('left', pair.left_id, left),
                                      ('right', pair.right_id, right),
                                      ('ir', aux_id, aux.intrinsics)]:
            fs.startWriteStruct(name, cv2.FileNode_MAP)
            fs.write('camera_id', camera_id)
            fs.write('model', intr.model.value)
            fs.write('width', intr.image_size[0])
            fs.write('height', intr.image_size[1])
            fs.write('K', intr.camera_matrix)
            fs.write('D', intr.dist_coeffs.reshape(1, -1))
            fs.endWriteStruct()
        fs.endWriteStruct()
        fs.startWriteStruct('extrinsics', cv2.FileNode_MAP)
        for name, rotation, t in [('T_RL', stereo.R, translation),
                                  ('T_IR_L', aux.T_rgb_left_aux[:3, :3],
                                   aux.T_rgb_left_aux[:3, 3:4] * 1000.)]:
            fs.startWriteStruct(name, cv2.FileNode_MAP)
            fs.write('R', rotation)
            fs.write('t', t)
            fs.endWriteStruct()
        fs.endWriteStruct()
        fs.startWriteStruct('stereo_rectify', cv2.FileNode_MAP)
        for name, value in zip(('R1', 'R2', 'P1', 'P2', 'Q'), rect):
            fs.write(name, value)
        fs.endWriteStruct()
        (output / 'calibration_mm.yml').write_bytes(fs.releaseAndGetString().encode('utf-8'))
        fs = cv2.FileStorage('vectors.yml', cv2.FILE_STORAGE_WRITE |
                             cv2.FILE_STORAGE_MEMORY | cv2.FILE_STORAGE_FORMAT_YAML)
        for name, value in vectors.items():
            fs.write(name, value)
        (output / 'test_vectors.yml').write_bytes(fs.releaseAndGetString().encode('utf-8'))
        (output / 'source_rig.json').write_bytes(raw)
        for f in adapter_files:
            (output / f.name).write_bytes(f.read_bytes())
        manifest = {'schema_version': 1, 'device_id': device_id, 'endpoint_hint': endpoint,
                    'expected_mac': device_mac.lower() if device_mac else None,
                    'pair_name': pair_name, 'left_id': pair.left_id, 'right_id': pair.right_id,
                    'aux_id': aux_id, 'source_sha256': hashlib.sha256(raw).hexdigest(),
                    'source_path': str(source.resolve()), 'translation_unit': 'mm',
                    'activation_allowed': False, 'physical_binding_verified': False,
                    'aux_status': aux.calibration_info.get('status', 'unknown'),
                    'warning': 'Do not use legacy pinhole-only IR projection or fixed Z=1 overlay.',
                    'files': {f.name: hashlib.sha256(f.read_bytes()).hexdigest()
                              for f in sorted(output.iterdir())}}
        (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n',
                                              encoding='utf-8')
    except Exception:
        shutil.rmtree(output)
        raise
    return output
