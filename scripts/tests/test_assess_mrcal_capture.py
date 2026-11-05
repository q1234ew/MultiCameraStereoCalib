"""ROI quality descriptors and missing-query accounting must remain conservative."""
from pathlib import Path
import sys

import numpy as np
import pytest

pytest.importorskip("mrcal")
sys.path.insert(0, str(Path(__file__).parents[1]))
import assess_mrcal_capture as audit


def test_board_roi_metrics_ignore_distant_background():
    image = np.full((100, 100), 128, dtype=np.uint8)
    image[25:75:2, 25:75] = 20
    corners = np.array([[20, 20], [80, 20], [80, 80], [20, 80]])
    altered = image.copy()
    altered[:10] = 255
    altered[-10:] = 0
    assert audit.image_metrics(image, corners) == audit.image_metrics(altered, corners)


def test_coverage_counts_distinct_captures_instead_of_corner_multiplicity():
    xy = [[1., 1.], [2., 1.], [2., 2.]]
    records = [{"capture_id": str(i), "left": xy, "right": xy} for i in range(2)]
    result = audit.coverage(records, (800, 600))
    assert result[0]["grid_6x8_counts"][0][0] == 6
    assert result[0]["grid_6x8_distinct_capture_counts"][0][0] == 2
    assert result[0]["empty_cells"] == 47


def test_masked_uncertainty_preserves_invalid_query_as_none(monkeypatch):
    original = audit.mrcal.unproject

    class Model:
        def __init__(self, **kwargs):
            pass

        def intrinsics(self):
            return "LENSMODEL_PINHOLE", np.array([1000., 1000., 640., 360.])

    def one_failed_inverse(*args, **kwargs):
        rays = original(*args, **kwargs)
        rays[0] = 0
        return rays

    monkeypatch.setattr(audit.mrcal, "cameramodel", Model)
    monkeypatch.setattr(audit.mrcal, "unproject", one_failed_inverse)
    monkeypatch.setattr(audit.mrcal, "projection_uncertainty",
                        lambda rays, **kwargs: np.broadcast_to(np.eye(2) * .04, (len(rays), 2, 2)))
    result = audit.masked_uncertainty({"imagersizes": np.array([[1280, 720]] * 2)})
    for camera in result["cameras"]:
        assert camera["valid_inverse_queries"] == 186
        assert len(camera["invalid_pixels"]) == 1
        for level in camera["levels"]:
            assert level["worst_direction_stdev_px"][0] is None
            assert level["regions"]["all"]["valid_covariance_queries"] == 186
