"""Training boundary and file-role invariants for the within-session comparison."""
from pathlib import Path
import sys

import numpy as np
import pytest

pytest.importorskip("mrcal")
sys.path.insert(0, str(Path(__file__).parents[1]))
import compare_mrcal_session as experiment
import compare_mrcal as common
import validate_mrcal as validation
import test_validate_mrcal as base_tests


def test_training_builder_cannot_ingest_poisoned_holdout_records():
    geometry = base_tests.geometry.__wrapped__()
    objects = experiment.mrcal.ref_calibration_object(8, 6, .08).reshape(-1, 3)
    records = []
    for i in range(12):
        obs = geometry["observations_board"][2*i:2*i+2].reshape(2, -1, 3)
        ids = np.flatnonzero((obs[..., 2] > 0).all(axis=0))
        records.append({"capture_id": str(i), "usable": True, "ids": ids.tolist(),
                        "left": obs[0, ids, :2].tolist(), "right": obs[1, ids, :2].tolist()})
    train, holdout, split = common.split_records(records, .25, 2)
    for r in holdout:
        r["left"] = np.full_like(r["left"], np.nan).tolist()
        r["right"] = np.full_like(r["right"], np.nan).tolist()
    data = {"board": {"cols": 9, "rows": 7, "square_length": .08},
            "object_points_m": objects.tolist(), "image_size": [1280, 720], "records": records}
    result = experiment.training_inputs(train, data, validation.opencv_model(geometry))
    assert len(train) == 7 and len(holdout) == 3 and len(split["guard"]) == 2
    assert len(result["rt_ref_frame"]) == 7
    assert result["indices_frame_camintrinsics_camextrinsics"][:, 0].max() == 6
    assert np.isfinite(result["observations_board"]).all()
    # No positive-baseline requirement or physical relabeling is imposed.
    assert result["rt_cam_ref"][0, 3] == pytest.approx(geometry["rt_cam_ref"][0, 3])
    assert result["rt_cam_ref"][0, 3] < 0


def test_requested_model_configurations_are_fixed():
    assert experiment.MODELS == {
        "opencv5": "LENSMODEL_OPENCV5",
        "spline12x8_fov120": "LENSMODEL_SPLINED_STEREOGRAPHIC_order=3_Nx=12_Ny=8_fov_x_deg=120",
        "spline16x10_fov140": "LENSMODEL_SPLINED_STEREOGRAPHIC_order=3_Nx=16_Ny=10_fov_x_deg=140"}
