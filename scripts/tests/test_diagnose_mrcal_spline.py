"""Independent covariance algebra and strict inverse validity regression checks."""
from pathlib import Path
import sys

import numpy as np
import pytest

pytest.importorskip("mrcal")
sys.path.insert(0, str(Path(__file__).parents[1]))
import diagnose_mrcal_spline as diagnosis


def test_pivoted_qr_regularized_covariance_matches_small_independent_reference():
    rng = np.random.default_rng(5)
    observed = rng.normal(size=(30, 5))
    observed[:, -1] = 0  # unobserved direction, constrained only by the prior
    jacobian = np.concatenate((observed, .1 * np.eye(5)))
    gradient = rng.normal(size=(7, 2, 5))
    computed = diagnosis.qr_covariance(jacobian, gradient, len(observed))
    hessian = jacobian.T @ jacobian
    response = np.linalg.solve(hessian, gradient.reshape(-1, 5).T)
    response = (observed @ response).T.reshape(7, 2, 30)
    expected = np.einsum("...ik,...jk->...ij", response, response) * .04
    np.testing.assert_allclose(computed, expected, atol=1e-12, rtol=1e-12)


def test_inverse_validity_rejects_zero_and_nonfinite_roundtrip():
    rays = np.array([[0., 0., 1.], [0., 0., 0.], [0., 0., 1.], [0., 0., 1.]])
    pixels = np.zeros((4, 2))
    projected = np.array([[0., 0.], [0., 0.], [np.nan, 0.], [1., 0.]])
    np.testing.assert_array_equal(diagnosis.valid_inverse(rays, projected, pixels),
                                   [True, False, False, False])


def test_probe_restores_library_function_even_when_native_call_fails(monkeypatch):
    original = diagnosis.analysis._propagate_calibration_uncertainty
    monkeypatch.setattr(diagnosis.mrcal, "cameramodel", lambda **kwargs: object())

    def fail(*args, **kwargs):
        raise RuntimeError("native probe failed")

    monkeypatch.setattr(diagnosis.mrcal, "projection_uncertainty", fail)
    with pytest.raises(RuntimeError, match="native probe failed"):
        diagnosis.capture_covariance({}, 0, np.array([[0., 0., 1.]]))
    assert diagnosis.analysis._propagate_calibration_uncertainty is original
