"""Virtual seed validation must never filter actual calibration observations."""
import copy
from pathlib import Path
import sys

import numpy as np
import pytest

mrcal = pytest.importorskip("mrcal")
sys.path.insert(0, str(Path(__file__).parents[1]))
from assess_mrcal_necessity import seed_from_valid_rays  # noqa: E402
from compare_mrcal_session import MODELS  # noqa: E402


def source_model():
    observations = np.zeros((2, 9, 14, 3))
    observations[..., 2] = 1
    observations[0, 0, 0, 2] = -1
    return {"lensmodel": "LENSMODEL_OPENCV5", "imagersizes": np.array([[1280, 720]] * 2),
            "intrinsics": np.array([[460., 460., 640., 360., 0., 0., 0., 0., 0.]] * 2),
            "observations_board": observations, "do_optimize_intrinsics_core": True,
            "do_apply_regularization": False}


@pytest.mark.parametrize("failure", ["nan", "zero", "wrong_roundtrip"])
def test_invalid_virtual_rays_are_explicit_and_real_observations_unchanged(monkeypatch, failure):
    source = source_model()
    original = copy.deepcopy(source)
    unproject = mrcal.unproject

    def perturbed(*args, **kwargs):
        rays = unproject(*args, **kwargs)
        rays[0] = {"nan": [np.nan] * 3, "zero": [0.] * 3,
                   "wrong_roundtrip": [0., 0., 1.]}[failure]
        return rays

    monkeypatch.setattr(mrcal, "unproject", perturbed)
    fitted, diagnostics = seed_from_valid_rays(source, MODELS["spline12x8_fov120"])
    for camera in (0, 1):
        assert diagnostics[camera]["valid_virtual_rays"] == 424
        assert diagnostics[camera]["invalid_virtual_pixels"] == [[0., 0.]]
        assert np.isfinite(fitted["intrinsics"][camera]).all()
    np.testing.assert_array_equal(fitted["observations_board"], original["observations_board"])
    np.testing.assert_array_equal(source["intrinsics"], original["intrinsics"])
    assert source["do_optimize_intrinsics_core"]
    assert not fitted["do_optimize_intrinsics_core"]
    assert fitted["do_apply_regularization"]


def test_insufficient_valid_initialization_is_rejected(monkeypatch):
    monkeypatch.setattr(mrcal, "unproject", lambda q, *a, **kw: np.zeros((len(q), 3)))
    with pytest.raises(ValueError, match="Insufficient valid"):
        seed_from_valid_rays(source_model(), MODELS["spline12x8_fov120"])
