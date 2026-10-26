"""Direct exporter contracts, model fidelity, and atomic file replacement."""

import copy
import json
from pathlib import Path
from unittest.mock import Mock

import cv2
import numpy as np
import pytest

from multicamera.calibration.models import (
    AuxCameraCalibration, CameraExtrinsics, CameraIntrinsics, CameraModel,
    MultiCameraRig, StereoCalibResult, StereoPairCalibration,
)
from multicamera.io import export


EXPORTERS = [
    (export.export_rig_json, ".json"),
    (export.export_rig_opencv_yaml, ".yml"),
    (export.export_rig_opencv_standard_yaml, ".yml"),
    (export.export_rig_kalibr_yaml, ".yaml"),
    (export.export_rig_yaml, ".yml"),
]


@pytest.mark.parametrize('defect', ['rotation', 'zero_baseline', 'distortion_matrix', 'size', 'homogeneous'])
def test_structurally_invalid_calibration_cannot_replace_previous(tmp_path, rig, defect):
    pair = rig.pairs['rgb']
    if defect == 'rotation':
        pair.stereo.R = np.eye(3) * 2
    elif defect == 'zero_baseline':
        pair.stereo.T = np.zeros((3, 1))
    elif defect == 'distortion_matrix':
        pair.left_intrinsics.dist_coeffs = np.zeros((2, 2))
    elif defect == 'size':
        pair.right_intrinsics.image_size = (320, 240)
    else:
        pair.left_intrinsics.camera_matrix[2, 0] = 1
    path = tmp_path / 'previous.yml'
    path.write_bytes(b'previous')
    with pytest.raises(ValueError):
        export.export_rig_opencv_standard_yaml(rig, path)
    assert path.read_bytes() == b'previous'


@pytest.fixture
def rig():
    left = CameraIntrinsics(
        np.array([[800., 0., 320.], [0., 810., 240.], [0., 0., 1.]]),
        np.array([[.1, -.02, .001, -.001, 0.]]), (640, 480),
    )
    right = copy.deepcopy(left)
    right.camera_matrix[0, 0] = 820.
    rotation = cv2.Rodrigues(np.array([.12, -.07, .04]))[0]
    translation = np.array([[-.1], [.01], [.02]])
    stereo = StereoCalibResult(rotation, translation, np.eye(3), np.eye(3))
    stereo.R1, stereo.R2 = np.eye(3), np.eye(3)
    stereo.P1 = np.column_stack((left.camera_matrix, np.zeros(3)))
    stereo.P2 = np.column_stack((right.camera_matrix, translation))
    stereo.Q = np.eye(4)
    pair = StereoPairCalibration("rgb", "left", "right", left, right, stereo)
    return MultiCameraRig(
        pairs={"rgb": pair}, reference_camera="left", extrinsics={
            "left": CameraExtrinsics(np.eye(3), np.zeros((3, 1))),
            "right": CameraExtrinsics(rotation, translation),
        },
    )


@pytest.mark.parametrize("model", list(CameraModel))
@pytest.mark.parametrize("writer", [export.export_rig_opencv_standard_yaml, export.export_rig_opencv_yaml])
def test_opencv_models_and_matrix_aliases_are_explicit_and_consistent(tmp_path, rig, model, writer):
    pair = rig.pairs["rgb"]
    pair.stereo.model = model
    for intr in (pair.left_intrinsics, pair.right_intrinsics):
        intr.model = model
        if model == CameraModel.FISHEYE:
            intr.dist_coeffs = np.array([[.01, -.02, .03, -.04]])
    path = tmp_path / "stereo.yml"
    assert writer(rig=rig, path=str(path)) == path
    fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
    try:
        assert fs.isOpened()
        distortion = "equidistant" if model == CameraModel.FISHEYE else "radtan"
        for key in ("camera_model", "left_model", "right_model", "left_camera_model", "right_camera_model"):
            assert fs.getNode(key).string() == model.value
        for key in ("distortion_model", "left_distortion_model", "right_distortion_model"):
            assert fs.getNode(key).string() == distortion
        for key, expected in (("M1", pair.left_intrinsics.camera_matrix),
                              ("D1", pair.left_intrinsics.dist_coeffs),
                              ("M2", pair.right_intrinsics.camera_matrix),
                              ("D2", pair.right_intrinsics.dist_coeffs),
                              ("R", pair.stereo.R), ("T", pair.stereo.T),
                              ("R1", pair.stereo.R1), ("R2", pair.stereo.R2),
                              ("P1", pair.stereo.P1), ("P2", pair.stereo.P2), ("Q", pair.stereo.Q)):
            np.testing.assert_allclose(fs.getNode(key).mat(), expected)
        if writer is export.export_rig_opencv_yaml:
            assert fs.getNode("pair_rgb_left_model").string() == model.value
            assert fs.getNode("pair_rgb_right_distortion_model").string() == distortion
            assert path.read_text().count("\npair_rgb_R:") == 1
    finally:
        fs.release()


def test_standard_rejects_models_inconsistent_with_stereo_solver(tmp_path, rig):
    rig.pairs["rgb"].right_intrinsics.model = CameraModel.FISHEYE
    rig.pairs["rgb"].right_intrinsics.dist_coeffs = np.zeros((1, 4))
    path = tmp_path / "mixed.yml"
    with pytest.raises(ValueError, match='inconsistent camera models'):
        export.export_rig_opencv_standard_yaml(rig, path, pair_name="rgb")
    assert not path.exists()


def test_opencv_rational_model_is_not_mislabeled_as_four_parameter_radtan(tmp_path, rig):
    rig.pairs["rgb"].left_intrinsics.dist_coeffs = np.zeros(8)
    path = export.export_rig_opencv_standard_yaml(rig, tmp_path / "rational.yml")
    fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
    try:
        assert fs.getNode("left_distortion_model").string() == "rational_polynomial"
        assert fs.getNode("D1").mat().size == 8
    finally:
        fs.release()


@pytest.mark.parametrize("coefficient", [4, 5, 7, 11, 13])
@pytest.mark.parametrize("value", [1.01e-10, -1.01e-10, .2])
def test_kalibr_rejects_nonzero_unrepresentable_coefficients(tmp_path, rig, coefficient, value):
    dist = np.zeros(14)
    dist[coefficient] = value
    rig.pairs["rgb"].left_intrinsics.dist_coeffs = dist
    target = tmp_path / "chain.yaml"
    target.write_bytes(b"previous calibration")
    with pytest.raises(ValueError, match="left.*Kalibr radtan.*cannot represent"):
        export.export_rig_kalibr_yaml(rig, target)
    with pytest.raises(ValueError, match="truncation tolerance: 1e-10"):
        export.validate_rig_kalibr_chain(rig)
    assert target.read_bytes() == b"previous calibration"
    assert list(tmp_path.iterdir()) == [target]


@pytest.mark.parametrize("value", [0., 1e-10, -1e-10, 1e-12])
def test_kalibr_allows_tiny_discarded_coefficients(tmp_path, rig, value):
    rig.pairs["rgb"].left_intrinsics.dist_coeffs = np.array([.1, .2, .3, .4, value])
    path = export.export_rig_kalibr_yaml(rig, tmp_path / "chain.yaml")
    assert "distortion_coeffs: [0.1, 0.2, 0.3, 0.4]" in path.read_text()
    export.validate_rig_kalibr_chain(rig)


def test_kalibr_fisheye_keeps_all_four_coefficients(tmp_path, rig):
    rig.pairs['rgb'].stereo.model = CameraModel.FISHEYE
    for intr in (rig.pairs["rgb"].left_intrinsics, rig.pairs["rgb"].right_intrinsics):
        intr.model = CameraModel.FISHEYE
        intr.dist_coeffs = np.array([.1, -.2, .3, -.4])
    path = export.export_rig_kalibr_yaml(rig, tmp_path / "fish.yaml")
    assert "camera_model: pinhole" in path.read_text()
    assert "distortion_model: equidistant" in path.read_text()
    assert "distortion_coeffs: [0.1, -0.2, 0.3, -0.4]" in path.read_text()


def add_shared_camera_pair(rig):
    pair = copy.deepcopy(rig.pairs["rgb"])
    pair.pair_name, pair.left_id, pair.right_id = "second", "right", "third"
    pair.left_intrinsics = copy.deepcopy(rig.pairs["rgb"].right_intrinsics)
    rig.pairs["second"] = pair
    return pair.left_intrinsics


@pytest.mark.parametrize("writer,suffix", EXPORTERS)
@pytest.mark.parametrize("field", ["camera_matrix", "dist_coeffs", "image_size", "model"])
def test_conflicting_same_camera_intrinsics_are_rejected(tmp_path, rig, writer, suffix, field):
    intr = add_shared_camera_pair(rig)
    if field == "camera_matrix":
        intr.camera_matrix[0, 0] += 1
    elif field == "dist_coeffs":
        intr.dist_coeffs[0, 0] += .01
    elif field == "image_size":
        intr.image_size = (1280, 960)
    else:
        intr.model = CameraModel.FISHEYE
    target = tmp_path / f"previous{suffix}"
    target.write_bytes(b"previous")
    with pytest.raises(ValueError, match="Conflicting intrinsics.*right"):
        writer(rig, target)
    assert target.read_bytes() == b"previous"


@pytest.mark.parametrize("writer,suffix", EXPORTERS)
def test_same_intrinsics_with_different_shape_and_metrics_are_compatible(tmp_path, rig, writer, suffix):
    intr = add_shared_camera_pair(rig)
    intr.dist_coeffs = intr.dist_coeffs.reshape(-1)
    intr.rms_error = .7
    intr.per_frame_errors = [.4]
    path = tmp_path / f"shared{suffix}"
    assert writer(rig, path) == path
    assert path.stat().st_size > 0


@pytest.mark.parametrize("namespace", ["camera", "pair", "extrinsic", "aux"])
def test_opencv_rejects_sanitized_key_collisions(tmp_path, rig, namespace):
    pair = rig.pairs["rgb"]
    if namespace == "camera":
        pair.left_id, pair.right_id = "a-b", "a_b"
    elif namespace == "pair":
        rig.pairs = {"a-b": pair, "a_b": copy.deepcopy(pair)}
    elif namespace == "extrinsic":
        rig.extrinsics = {"a-b": rig.extrinsics["left"], "a_b": rig.extrinsics["right"]}
    else:
        for camera_id in ("a-b", "a_b"):
            rig.aux_cameras[camera_id] = AuxCameraCalibration(
                camera_id, "ir", pair.left_intrinsics, np.eye(4),
            )
    target = tmp_path / "old.yml"
    target.write_bytes(b"previous")
    expected_namespace = "camera" if namespace == "aux" else namespace
    with pytest.raises(ValueError, match=f"OpenCV {expected_namespace} key collision"):
        export.export_rig_opencv_yaml(rig, target)
    assert target.read_bytes() == b"previous"


@pytest.mark.parametrize("writer,suffix", EXPORTERS)
@pytest.mark.parametrize("field", ["K", "D", "rms", "per_frame", "R", "T", "E", "F", "Q", "extrinsic", "aux"])
def test_nonfinite_calibration_values_cannot_overwrite_previous_file(tmp_path, rig, writer, suffix, field):
    pair = rig.pairs["rgb"]
    if field == "K":
        pair.left_intrinsics.camera_matrix[0, 0] = np.nan
    elif field == "D":
        pair.left_intrinsics.dist_coeffs[0, 0] = np.inf
    elif field == "rms":
        pair.stereo.rms_error = np.nan
    elif field == "per_frame":
        pair.left_intrinsics.per_frame_errors = [np.inf]
    elif field == "extrinsic":
        rig.extrinsics["right"].T[0, 0] = np.inf
    elif field == "aux":
        pose = np.eye(4)
        pose[0, 3] = -np.inf
        rig.aux_cameras["ir"] = AuxCameraCalibration("ir", "ir", pair.left_intrinsics, pose)
    else:
        getattr(pair.stereo, field)[0, 0] = np.nan
    target = tmp_path / f"old{suffix}"
    target.write_bytes(b"previous")
    with pytest.raises(ValueError, match="non-finite"):
        writer(rig, target)
    assert target.read_bytes() == b"previous"
    assert list(tmp_path.iterdir()) == [target]


@pytest.mark.parametrize("writer,suffix", EXPORTERS)
def test_historical_missing_rms_defaults_to_zero_and_exports(tmp_path, rig, writer, suffix):
    data = rig.to_dict()
    for section in ("left_intrinsics", "right_intrinsics", "stereo"):
        data["pairs"]["rgb"][section].pop("rms_error")
    old = MultiCameraRig.from_dict(data)
    assert old.pairs["rgb"].left_intrinsics.rms_error == 0
    assert writer(old, tmp_path / f"legacy{suffix}").is_file()


def test_json_uses_strict_serialization(tmp_path, rig, monkeypatch):
    dumps = Mock(wraps=json.dumps)
    monkeypatch.setattr(export.json, "dumps", dumps)
    path = export.export_rig_json(rig, tmp_path / "rig.json")
    assert dumps.call_args.kwargs["allow_nan"] is False
    assert json.loads(path.read_text())["reference_camera"] == "left"


def test_empty_writer_output_cannot_replace_previous_file(tmp_path, rig, monkeypatch):
    target = tmp_path / "old.json"
    target.write_bytes(b"previous")
    monkeypatch.setattr(export.json, "dumps", lambda *a, **kw: "")
    with pytest.raises(OSError, match="non-empty file"):
        export.export_rig_json(rig, target)
    assert target.read_bytes() == b"previous"
    assert list(tmp_path.iterdir()) == [target]


@pytest.mark.parametrize("field", ["K", "D", "rectification"])
def test_invalid_matrix_shapes_and_coefficient_counts_are_rejected(tmp_path, rig, field):
    pair = rig.pairs["rgb"]
    if field == "K":
        pair.left_intrinsics.camera_matrix = np.eye(4)
    elif field == "D":
        pair.left_intrinsics.dist_coeffs = np.zeros(6)
    else:
        pair.stereo.P2 = None
    target = tmp_path / "old.yml"
    target.write_bytes(b"previous")
    with pytest.raises(ValueError, match="shape|coefficient count"):
        export.export_rig_opencv_yaml(rig, target)
    assert target.read_bytes() == b"previous"


@pytest.mark.parametrize("writer,suffix", [EXPORTERS[0], EXPORTERS[3]])
@pytest.mark.parametrize("existing", [True, False])
def test_partial_text_write_failure_is_atomic(tmp_path, rig, monkeypatch, writer, suffix, existing):
    target = tmp_path / f"calibration{suffix}"
    if existing:
        target.write_bytes(b"previous")
    write_text = Path.write_text

    def fail(path, text, *args, **kwargs):
        assert path != target and path.parent.parent == tmp_path
        assert target.read_bytes() == b"previous" if existing else not target.exists()
        write_text(path, text[:20], *args, **kwargs)
        raise OSError("simulated disk full")

    monkeypatch.setattr(Path, "write_text", fail)
    with pytest.raises(OSError, match="disk full"):
        writer(rig, target)
    assert list(tmp_path.iterdir()) == ([target] if existing else [])
    if existing:
        assert target.read_bytes() == b"previous"


@pytest.mark.parametrize("standard", [True, False])
def test_partial_opencv_write_failure_is_atomic_and_releases_storage(tmp_path, rig, monkeypatch, standard):
    target = tmp_path / "old.yml"
    target.write_bytes(b"previous")
    storages = []
    factory = cv2.FileStorage

    def storage(*args):
        value = factory(*args)
        storages.append(value)
        return value

    def fail(fs, *args):
        fs.write("partial", 123)
        assert target.read_bytes() == b"previous"
        raise OSError("simulated FileStorage failure")

    monkeypatch.setattr(export.cv2, "FileStorage", storage)
    helper = "_write_opencv_standard_stereo_fields" if standard else "_write_opencv_rig"
    monkeypatch.setattr(export, helper, fail)
    writer = export.export_rig_opencv_standard_yaml if standard else export.export_rig_opencv_yaml
    with pytest.raises(OSError, match="FileStorage failure"):
        writer(rig, target)
    assert target.read_bytes() == b"previous"
    assert storages and not storages[0].isOpened()
    assert list(tmp_path.iterdir()) == [target]


@pytest.mark.parametrize("writer,suffix", EXPORTERS)
def test_replace_failure_does_not_destroy_target_or_leave_staging(tmp_path, rig, monkeypatch, writer, suffix):
    target = tmp_path / f"old{suffix}"
    target.write_bytes(b"previous")

    def fail(path, destination):
        assert Path(destination) == target and path != target
        raise PermissionError("replacement denied")

    monkeypatch.setattr(Path, "replace", fail)
    with pytest.raises(PermissionError, match="replacement denied"):
        writer(rig, path=target)
    assert target.read_bytes() == b"previous"
    assert list(tmp_path.iterdir()) == [target]


def test_opencv_open_failure_releases_storage_and_preserves_target(tmp_path, rig, monkeypatch):
    storage = Mock()
    storage.isOpened.return_value = False
    monkeypatch.setattr(export.cv2, "FileStorage", lambda *a: storage)
    target = tmp_path / "old.yml"
    target.write_bytes(b"previous")
    with pytest.raises(OSError, match="Cannot open OpenCV export"):
        export.export_rig_opencv_yaml(rig, target)
    storage.release.assert_called_once()
    assert target.read_bytes() == b"previous"


def test_bundle_finishes_all_formats_before_replacing_previous_members(tmp_path, rig):
    paths = export.export_rig_calibration_bundle(rig, tmp_path)
    previous = {path: path.read_bytes() for path in paths}
    rig.pairs["rgb"].left_intrinsics.dist_coeffs[0, 4] = .2
    with pytest.raises(ValueError, match="Kalibr radtan"):
        export.export_rig_calibration_bundle(rig, tmp_path)
    assert all(path.read_bytes() == data for path, data in previous.items())
    assert set(tmp_path.iterdir()) == set(paths)


def test_noncommuting_stereo_and_aux_chain_keeps_transform_direction(tmp_path, rig):
    aux_to_left = np.eye(4)
    aux_to_left[:3, :3] = cv2.Rodrigues(np.array([-.2, .1, .04]))[0]
    aux_to_left[:3, 3] = [.03, -.01, .02]
    rig.aux_cameras["ir"] = AuxCameraCalibration("ir", "ir", rig.pairs["rgb"].left_intrinsics, aux_to_left)
    right_from_left = rig.extrinsics["right"].transformation_matrix
    expected = np.linalg.inv(aux_to_left) @ np.linalg.inv(right_from_left)
    np.testing.assert_allclose(export._relative_transform(rig, "right", "ir"), expected)
    np.testing.assert_allclose(export._relative_transform(rig, "ir", "right") @ expected, np.eye(4), atol=1e-12)
    path = export.export_rig_kalibr_yaml(rig, tmp_path / "chain.yaml")
    rows = path.read_text().split("cam2:\n")[1].split("T_cn_cnm1:\n")[1].strip().splitlines()
    actual = np.array([json.loads(row.strip()[2:]) for row in rows])
    np.testing.assert_allclose(actual, expected, atol=1e-11)
