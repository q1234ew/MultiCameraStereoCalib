"""Headless candidate framework contracts; no Jetson/network access required."""

import asyncio
import copy
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pytest
from aiohttp.test_utils import TestClient, TestServer

from scripts.jetson.core import Frame, Pairer, Rectifier, depth_points, load_config
from scripts.jetson.service import DeviceWorker, inspect_devices, make_app


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "packaging/jetson/devices.json"


def test_candidate_configuration():
    config = load_config(CONFIG)
    assert len(config["devices"]) == 3
    device = config["devices"][1]
    assert not device["physical_binding_verified"]
    assert device["raw_ports"] == {"left": 8082, "right": 8080, "ir": 8081}
    assert "historical" in device["binding_note"]
    assert device["source_session"] == "calib_20261104_174237"


@pytest.mark.parametrize("field,value", [
    ("translation_unit", "m"), ("calibration_sha256", "bad"),
    ("physical_binding_verified", "true"), ("source_session", "unknown"),
    ("raw_ports", {"left": 8090, "right": 8080, "ir": 8081}),
    ("resolution", {"left": [1280, 720]}),
    ("calibration_path", "relative/calibration.yml"),
])
def test_reject_invalid_binding(tmp_path, field, value):
    config = json.loads(CONFIG.read_text())
    config["devices"][0][field] = value
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError):
        load_config(path)


@pytest.fixture
def candidate(tmp_path, request):
    config = copy.deepcopy(load_config(CONFIG))
    device = config["devices"][0]
    device["resolution"] = {side: [8, 6] for side in ("left", "right", "ir")}
    path = tmp_path / "calibration.yml"
    fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_WRITE)
    for key, value in {"device_id": device["id"], "translation_unit": "mm",
                       "coord_frame": "C_L", "stereo_transform_direction": "left_to_right",
                       "ir_transform_direction": "ir_to_left"}.items():
        fs.write(key, value)
    k = np.array([[10., 0., 4.], [0., 10., 3.], [0., 0., 1.]])
    fs.startWriteStruct("camera_intrinsics", cv2.FileNode_MAP)
    for side in ("left", "right", "ir"):
        fs.startWriteStruct(side, cv2.FileNode_MAP)
        fs.write("width", 8)
        fs.write("height", 6)
        model = getattr(request, "param", "pinhole")
        fs.write("model", model)
        fs.write("K", k)
        fs.write("D", np.zeros((1, 4 if model == "fisheye" else 5)))
        fs.endWriteStruct()
    fs.endWriteStruct()
    fs.startWriteStruct("stereo_rectify", cv2.FileNode_MAP)
    for index in ("1", "2"):
        fs.write("R" + index, np.eye(3))
        fs.write("P" + index, np.column_stack((k, np.zeros(3))))
    fs.endWriteStruct()
    fs.release()
    device["calibration_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return config, device, path


def test_pairing_isolation_skew_age_and_resolution(candidate):
    config, device, _ = candidate
    pairer = Pairer(device, config["max_skew_ms"], config["max_age_ms"])
    image = np.zeros((6, 8, 3), np.uint8)
    def frame(side, timestamp, ident="device_1"):
        return Frame(ident, side, timestamp, image)
    with pytest.raises(ValueError, match="another device"):
        pairer.push(frame("left", 0, "device_2"), 0)
    with pytest.raises(ValueError, match="resolution"):
        pairer.push(Frame("device_1", "left", 0, image[:2]), 0)
    assert pairer.push(frame("left", 0), 0) is None
    assert pairer.push(frame("right", 80_000_000), 80_000_000) is None
    assert pairer.push(frame("left", 90_000_000), 90_000_000) is not None
    assert pairer.push(frame("left", 90_000_000), 90_000_000) is None
    assert pairer.push(frame("left", 100_000_000), 700_000_000) is None
    assert not pairer.pending
    assert pairer.dropped == 3


@pytest.mark.parametrize("candidate", ["pinhole", "fisheye"], indirect=True)
def test_calibration_cache_and_fail_closed(candidate):
    _, device, path = candidate
    rectifier = Rectifier(device, path)
    maps = rectifier.maps["left"]
    image = np.arange(144, dtype=np.uint8).reshape(6, 8, 3)
    for _ in range(2):
        result = rectifier.rectify(Frame("device_1", "left", 0, image))
        np.testing.assert_array_equal(result, cv2.remap(image, *maps, cv2.INTER_LINEAR))
    assert rectifier.maps["left"] is maps
    other = dict(device, id="device_2")
    with pytest.raises(ValueError, match="identity"):
        Rectifier(other, path)
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="hash"):
        Rectifier(device, path)


def test_depth_has_explicit_units_and_no_default_z():
    k = np.eye(3)
    depth = np.array([[1000., 0.], [np.nan, 2000.]])
    points = depth_points(depth, k, unit="mm", expected_size=(2, 2), stride=1)
    np.testing.assert_allclose(points, [[0., 0., 1.], [2., 2., 2.]])
    with pytest.raises(ValueError):
        depth_points(depth, k, unit="unknown", expected_size=(2, 2))
    with pytest.raises(ValueError):
        depth_points(depth, k, unit="mm", expected_size=(3, 2))


def test_replay_worker_depth_binding(candidate, tmp_path):
    config, device, calibration = candidate
    recording = dict(device_id="device_1", calibration_sha256=device["calibration_sha256"],
                     left=np.zeros((6, 8, 3), np.uint8), right=np.zeros((6, 8, 3), np.uint8),
                     left_timestamp_ns=1_000_000, right_timestamp_ns=2_000_000,
                     depth=np.full((6, 8), 1500.), depth_unit="mm",
                     depth_frame="rectified_left_axial_z", depth_timestamp_ns=1_000_000)
    path = tmp_path / "replay.npz"
    np.savez(path, **recording)
    result = DeviceWorker(device, config, calibration).replay(path)
    assert result["point_count"] == 4
    assert result["point_unit"] == "m"
    assert result["pair_skew_ms"] == 1
    assert not result["physical_binding_verified"]
    recording["depth_timestamp_ns"] = 2_000_000
    np.savez(path, **recording)
    with pytest.raises(ValueError, match="Depth"):
        DeviceWorker(device, config, calibration).replay(path)


def test_missing_depth_never_invents_points(candidate, tmp_path):
    config, device, calibration = candidate
    path = tmp_path / "replay.npz"
    recording = dict(device_id="device_1", calibration_sha256=device["calibration_sha256"],
                     left=np.zeros((6, 8, 3), np.uint8), right=np.zeros((6, 8, 3), np.uint8),
                     left_timestamp_ns=0, right_timestamp_ns=1)
    np.savez(path, **recording)
    result = DeviceWorker(device, config, calibration).replay(path)
    assert result["point_count"] is None
    assert result["depth_status"] == "not_supplied"
    recording["device_id"] = "device_2"
    np.savez(path, **recording)
    with pytest.raises(ValueError, match="binding"):
        DeviceWorker(device, config, calibration).replay(path)


def test_one_device_failure_does_not_block_another(candidate, tmp_path):
    config, device, calibration = candidate
    directory = tmp_path / device["id"]
    directory.mkdir()
    (directory / "calibration_mm.yml").write_bytes(calibration.read_bytes())
    workers, states = inspect_devices(config, tmp_path)
    assert list(workers) == ["device_1"]
    assert states["device_1"]["calibration"] == "loaded_candidate"
    assert states["device_2"]["calibration"] == "error"


def test_missing_bundle_isolated_and_status_read_only(tmp_path):
    workers, states = inspect_devices(load_config(CONFIG), tmp_path)
    assert workers == {}
    assert len(states) == 3
    assert all(s["calibration"] == "error" for s in states.values())
    async def check():
        async with TestClient(TestServer(make_app(states))) as client:
            response = await client.get("/health")
            assert response.status == 200
            assert not (await response.json())["ready"]
            assert (await client.get("/ready")).status == 503
            assert (await client.get("/api/devices/device_2")).status == 200
            assert (await client.get("/api/devices/unknown")).status == 404
            assert (await client.post("/health")).status == 405
    asyncio.run(check())
