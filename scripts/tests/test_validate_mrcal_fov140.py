"""Check full-border inverse evaluation and configurable edge uncertainty queries."""
from pathlib import Path
import sys

import numpy as np
import pytest

pytest.importorskip("mrcal")
sys.path.insert(0, str(Path(__file__).parents[1]))
import compare_mrcal_spline as spline
import validate_mrcal_fov140 as experiment
import test_validate_mrcal as base_tests


@pytest.fixture
def geometry():
    return base_tests.geometry.__wrapped__()


def test_dense_inverse_grid_includes_full_border_on_known_model(geometry):
    records = []
    for frame in range(3):
        pair = geometry["observations_board"][frame * 2:frame * 2 + 2].reshape(2, -1, 3)
        valid = (pair[..., 2] > 0).all(axis=0)
        records.append({"left": pair[0, valid, :2], "right": pair[1, valid, :2]})
    result = experiment.inverse_checks(geometry, records, geometry)
    for camera in result["cameras"]:
        assert camera["grid_queries"] == 65 * 37
        assert camera["grid_valid"] == 65 * 37
        assert camera["zero_ray_count"] == 0
        assert camera["holdout_inverse_valid"] == camera["holdout_corners"]
        assert camera["common_physical_ray_orientation"]["negative_determinants"] == 0


def test_monte_carlo_uses_requested_edge_locations_and_family_size(geometry, tmp_path):
    geometry["do_apply_regularization"] = True
    result = spline.monte_carlo(geometry, tmp_path, "edge_test", 30,
                                query_fractions=(.025, .5, .975), family_queries=162)
    saved = np.load(tmp_path / "edge_test_monte_carlo.npz")
    np.testing.assert_allclose(saved["pixels"][0], [.025 * 1279, .025 * 719])
    np.testing.assert_allclose(saved["pixels"][-1], [.975 * 1279, .975 * 719])
    assert result["family_queries"] == 162
    assert result["samples"] == 30
