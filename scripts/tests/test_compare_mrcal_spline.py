"""Native spline projection, evaluation and uncertainty validity checks."""
from pathlib import Path
import sys

import numpy as np
import pytest

pytest.importorskip("mrcal")
sys.path.insert(0, str(Path(__file__).parents[1]))
import compare_mrcal_spline as spline
import validate_mrcal as validation
import test_validate_mrcal as base_tests


@pytest.fixture
def geometry():
    return base_tests.geometry.__wrapped__()


def test_regions_use_observed_pixels_and_keep_corner_overlap_explicit():
    xy = np.array([[10., 10.], [50., 50.], [90., 50.], [20., 80.]])
    masks = spline.region_masks(xy, (100, 100))
    np.testing.assert_array_equal(masks["edge"], [True, False, True, False])
    np.testing.assert_array_equal(masks["corners"], [True, False, False, False])
    np.testing.assert_array_equal(masks["center"], ~masks["edge"])


def test_spline_initialization_does_not_replace_measurements_or_mutate_source(geometry):
    import mrcal
    original = geometry["intrinsics"].copy()
    initial, diagnostic = spline.spline_seed(geometry, spline.SPLINES["spline12x8"])
    np.testing.assert_array_equal(geometry["intrinsics"], original)
    np.testing.assert_array_equal(initial["observations_board"], geometry["observations_board"])
    assert initial["do_apply_regularization"]
    assert not initial["do_optimize_intrinsics_core"]
    assert initial["intrinsics"].shape == (2, mrcal.lensmodel_num_params(initial["lensmodel"]))
    assert all(d["approximation"]["rms_px"] < 1 for d in diagnostic)


def test_generic_pose_evaluation_works_on_native_spline_without_kd_conversion(geometry):
    import mrcal
    initial, _ = spline.spline_seed(geometry, spline.SPLINES["spline12x8"])
    rendered = validation.render_observations(initial)
    objects = mrcal.ref_calibration_object(8, 6, .08).reshape(-1, 3)
    records = []
    for frame in range(3):
        pair = rendered["observations_board"][frame * 2:frame * 2 + 2].reshape(2, -1, 3)
        valid = (pair[..., 2] > 0).all(axis=0)
        ids = np.flatnonzero(valid)
        records.append({"capture_id": str(frame), "ids": ids.tolist(),
                        "left": pair[0, valid, :2].tolist(), "right": pair[1, valid, :2].tolist()})
    measured = spline.evaluate(initial, records, objects, (1280, 720), validation.opencv_model(geometry))
    assert measured["transfer"]["all"]["rms_px"] < 1e-6
    assert measured["joint_pose"]["all"]["rms_px"] < 1e-6


def test_nan_uncertainty_cannot_be_reported_as_valid(geometry, monkeypatch):
    monkeypatch.setattr(spline.mrcal, "projection_uncertainty",
                        lambda rays, **kwargs: np.full((len(rays), 2, 2), np.nan))
    with pytest.raises(ValueError, match="Invalid projection covariance"):
        spline.uncertainty_map(geometry)


def test_monte_carlo_keeps_regularization_after_exact_render(geometry, tmp_path, monkeypatch):
    geometry["do_apply_regularization"] = True

    def check_flags(inputs):
        assert inputs["do_apply_regularization"] is True
        raise RuntimeError("stop after flag check")

    monkeypatch.setattr(spline, "solve", check_flags)
    with pytest.raises(RuntimeError, match="stop after flag check"):
        spline.monte_carlo(geometry, tmp_path, "test", samples=30)


def test_zero_inverse_sentinel_is_not_accepted_as_a_valid_ray(monkeypatch):
    monkeypatch.setattr(spline.mrcal, "unproject", lambda *args, **kwargs: np.zeros((1, 3)))
    with pytest.raises(ValueError, match="Invalid inverse ray"):
        spline.checked_unproject(np.array([[640., 360.]]), "unused", np.zeros(4))


def test_nonfinite_roundtrip_cannot_bypass_tolerance_check(monkeypatch):
    monkeypatch.setattr(spline.mrcal, "unproject", lambda *args, **kwargs: np.array([[0., 0., 1.]]))
    monkeypatch.setattr(spline.mrcal, "project", lambda *args, **kwargs: np.full((1, 2), np.nan))
    with pytest.raises(ValueError, match="Invalid inverse roundtrip"):
        spline.checked_unproject(np.array([[640., 360.]]), "unused", np.zeros(4))
