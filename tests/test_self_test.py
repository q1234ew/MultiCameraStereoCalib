"""Self-test must exercise production calibration, fail closed, and remain headless."""

import json
import subprocess
import sys

import cv2
import pytest

from multicamera import diagnostics, self_test


def test_full_calibration_without_legacy_binding(tmp_path, monkeypatch):
    monkeypatch.delattr(cv2.aruco, "calibrateCameraCharuco", raising=False)
    fish_stereo_calls = []
    original = cv2.fisheye.stereoCalibrate

    def capture(object_points, *args, **kwargs):
        fish_stereo_calls.append([len(points) for points in object_points])
        return original(object_points, *args, **kwargs)

    monkeypatch.setattr(cv2.fisheye, "stereoCalibrate", capture)
    path = tmp_path / "reports" / "self_test.json"
    assert self_test.run_self_test(path) == 0
    report = json.loads(path.read_text())
    assert report["status"] == "passed"
    assert report["stage"] == "complete"
    assert not report["opencv"]["apis"][diagnostics.LEGACY_API]
    assert len(report["opencv"]["native_sha256"]) == 64
    assert report["calibration"]["frames"] == 24
    assert report["calibration"]["detected_corners"] == 126
    assert report["calibration"]["baseline_m"] == pytest.approx(.12, abs=.0001)
    assert len(report["calibration"]["exports"]) == 4
    unicode_io = report['calibration']['unicode_file_io']
    assert unicode_io['pixel_exact']
    assert len(unicode_io['images']) == 2
    assert all('标定 文件' in name for name in unicode_io['exports'])
    health = report['calibration']['health_checks']
    assert health['health']['status'] == 'ok'
    assert health['invalid_lens_rejected'] and health['overzoom_rejected']
    assert health['rational_denominator_preserved']
    assert health['projection_baseline_rejected'] and health['zero_q_rejected']
    assert health['wrong_depth_scale_rejected']
    fish = report["calibration"]["fisheye"]
    assert fish["model"] == "fisheye"
    assert fish["frames"] == fish["compatible_frames"] == 16
    assert len(set(fish["original_common_corner_counts"])) > 1
    assert 6 <= fish["compatible_common_corners"] < 126
    assert "鱼眼兼容" in fish["variable_corner_rejection"]
    assert fish["baseline_m"] == pytest.approx(.12, abs=.0001)
    assert max(fish[key] for key in ("left_rms", "right_rms", "stereo_rms",
                                    "rectified_rms_px")) < .001
    assert fish["rectified_error_axis"] == "y"
    assert fish["rectified_corner_count"] == sum(fish["original_common_corner_counts"])
    assert len(fish["exports"]) == 4
    # The default variable-count attempt must be rejected before calling OpenCV.
    assert fish_stereo_calls == [[fish["compatible_common_corners"]] * 16]


@pytest.mark.parametrize("api,stage", [
    ("calibrate", "fisheye_intrinsics_left"),
    ("stereoCalibrate", "fisheye_stereo_calibration"),
    ("stereoRectify", "fisheye_stereo_calibration"),
])
def test_fisheye_calibration_failure_fails_self_test(tmp_path, monkeypatch, api, stage):
    def fail(*args, **kwargs):
        raise RuntimeError("simulated fisheye calibration failure")

    monkeypatch.setattr(cv2.fisheye, api, fail)
    path = tmp_path / "failed-fisheye.json"
    assert self_test.run_self_test(path) == 1
    report = json.loads(path.read_text())
    assert report["status"] == "failed"
    assert report["stage"] == stage
    assert "simulated fisheye calibration failure" in report["error"]


def test_unicode_image_write_failure_fails_self_test(tmp_path, monkeypatch):
    from multicamera.io import image_io

    def fail(*args):
        raise OSError('simulated Unicode image write failure')

    monkeypatch.setattr(image_io, 'write_png', fail)
    path = tmp_path / 'failed-unicode.json'
    assert self_test.run_self_test(path) == 1
    report = json.loads(path.read_text())
    assert report['status'] == 'failed' and report['stage'] == 'unicode_image_roundtrip'
    assert 'Unicode image write failure' in report['error']


@pytest.mark.parametrize("failure", ["import", "api", "calibration", "conflicting_distribution"])
def test_failure_report(tmp_path, monkeypatch, failure):
    path = tmp_path / "failed.json"
    if failure == "import":
        def fail(**kwargs):
            raise ImportError("simulated cv2 DLL import failure")
        monkeypatch.setattr(diagnostics, "opencv_runtime_info", fail)
    elif failure == "api":
        monkeypatch.delattr(cv2, "calibrateCamera")
    elif failure == "conflicting_distribution":
        original = diagnostics.opencv_runtime_info

        def conflict(**kwargs):
            info = original(**kwargs)
            info["distributions"]["opencv-python"] = cv2.__version__
            return info
        monkeypatch.setattr(diagnostics, "opencv_runtime_info", conflict)
    else:
        def fail(report):
            raise RuntimeError("simulated calibration failure")
        monkeypatch.setattr(self_test, "_exercise_calibration", fail)
    assert self_test.run_self_test(path) == 1
    report = json.loads(path.read_text())
    assert report["status"] == "failed"
    assert report["error"]
    assert report["stage"] != "complete"


def test_external_native_library_rejected_in_frozen_mode(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "_MEIPASS", str(tmp_path / "fake-bundle"), raising=False)
    path = tmp_path / "external.json"
    assert self_test.run_self_test(path) == 1
    assert "outside the bundle" in json.loads(path.read_text())["error"]


def test_report_write_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(self_test, "_exercise_calibration", lambda report: {})
    assert self_test.run_self_test(tmp_path) == 2


def test_windowed_exe_without_stdio(tmp_path, monkeypatch):
    monkeypatch.setattr(self_test, "_exercise_calibration", lambda report: {})
    monkeypatch.setattr(sys, "stdout", None)
    monkeypatch.setattr(sys, "stderr", None)
    assert self_test.run_self_test(tmp_path / "windowed.json") == 0


def test_entrypoint_does_not_import_gui_or_open3d(tmp_path):
    code = """
import importlib.abc
import runpy
import sys
class NoGui(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname.split('.')[0] in {'PySide6', 'open3d'}:
            raise ImportError('Self-test imported GUI or point cloud code: ' + fullname)
sys.meta_path.insert(0, NoGui())
sys.argv = ['multicamera', '--self-test-calibration', '--report', sys.argv[1]]
runpy.run_module('multicamera', run_name='__main__')
"""
    path = tmp_path / "headless.json"
    result = subprocess.run([sys.executable, "-c", code, str(path)],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(path.read_text())
    assert report["status"] == "passed"
    assert report["calibration"]["fisheye"]["compatible_frames"] == 16
