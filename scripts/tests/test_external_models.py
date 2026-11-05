"""Frozen external evaluation must not train on new images or hide rig changes."""
import copy
from pathlib import Path
import sys

import numpy as np
import pytest

pytest.importorskip("mrcal")
sys.path.insert(0, str(Path(__file__).parents[1]))
import test_mrcal_external as external
import compare_mrcal_spline as spline
import validate_mrcal as validation
import test_validate_mrcal as base_tests


@pytest.fixture
def geometry():
    return base_tests.geometry.__wrapped__()


def records_for(inputs):
    objects = external.mrcal.ref_calibration_object(8, 6, .08).reshape(-1, 3)
    records = []
    for frame in range(3):
        ids = np.arange(len(objects))
        xy = [spline.project(inputs, objects, inputs["rt_ref_frame"][frame], c) for c in (0, 1)]
        records.append({"capture_id": str(frame), "ids": ids.tolist(), "left": xy[0], "right": xy[1]})
    return objects, records


def test_external_evaluation_keeps_camera_parameters_and_never_invokes_calibration(geometry, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("External data must not enter camera calibration")

    monkeypatch.setattr(external.mrcal, "optimize", forbidden)
    objects, records = records_for(geometry)
    before = external.model_fingerprint(geometry)
    result = external.evaluate_frozen(geometry, records, objects, (1280, 720), validation.opencv_model(geometry))
    assert external.model_fingerprint(geometry) == before
    assert result["stereo"]["transfer"]["all"]["rms_px"] < 1e-6
    assert result["monocular"]["residuals"]["all"]["rms_px"] < 1e-6


def test_changed_rig_can_raise_transfer_error_while_monocular_fit_stays_good(geometry):
    changed = copy.deepcopy(geometry)
    changed["rt_cam_ref"][0, 3] += .01
    objects, records = records_for(changed)
    result = external.evaluate_frozen(geometry, records, objects, (1280, 720), validation.opencv_model(geometry))
    assert result["stereo"]["transfer"]["all"]["rms_px"] > 1
    assert result["monocular"]["residuals"]["all"]["rms_px"] < 1e-6
    measured = result["monocular"]["relative_pose_diagnostic"]
    assert measured["translation_vector_difference_mm_median"] == pytest.approx(10., abs=1e-4)
