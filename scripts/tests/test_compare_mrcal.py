"""Numerical protocol tests independent of the application's Qt test fixtures."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

spec = importlib.util.spec_from_file_location("compare_mrcal", Path(__file__).parents[1] / "compare_mrcal.py")
comparison = importlib.util.module_from_spec(spec)
spec.loader.exec_module(comparison)


def synthetic():
    objects = np.zeros((6, 8, 3))
    objects[..., :2] = np.stack(np.meshgrid(np.arange(8), np.arange(6)), axis=-1) * .04
    objects = objects.reshape(-1, 3)
    model = {
        "K": [np.array([[900., 0, 640], [0, 910., 360], [0, 0, 1.]])] * 2,
        "D": [np.array([-.1, .02, .001, -.002, .005])] * 2,
        "R": cv2.Rodrigues(np.array([.005, -.01, .003]))[0],
        "T": np.array([-.12, .001, .002]),
    }
    records = []
    for i in range(12):
        pose = np.array([.22 * np.sin(i), .3 * np.cos(i), .12 * np.sin(i * .3),
                         -.15 + .03 * np.cos(i), -.1 + .02 * np.sin(i), .8 + .06 * i])
        # Partial boards and shuffled detector ordering are represented by canonical IDs.
        ids = np.arange(len(objects))[np.arange(len(objects)) % 7 != i % 7]
        xy = [comparison.project_camera(objects[ids], pose, model, c) for c in range(2)]
        records.append({"capture_id": f"{i:04d}", "ids": ids.tolist(),
                        "left": xy[0].tolist(), "right": xy[1].tolist(), "usable": True})
    return objects, model, records


def test_point_rms_is_not_coordinate_rms():
    result = comparison.error_summary([[3., 4.], [0., 0.]])
    assert result["rms_px"] == pytest.approx(np.sqrt(25 / 2))
    assert result["points"] == 2


def test_holdout_is_whole_contiguous_captures_and_gap_precedes_filtering():
    records = [{"capture_id": str(i), "usable": i != 2} for i in range(36)]
    train, validation, split = comparison.split_records(records)
    assert [r["capture_id"] for r in validation] == [str(i) for i in range(27, 36)]
    assert split["guard"] == ["25", "26"]
    assert "2" in split["train"] and "2" not in [r["capture_id"] for r in train]
    assert not set(split["train"]) & set(split["validation"])


def test_corner_cache_preserves_id_correspondence_and_rejects_changed_source(tmp_path, monkeypatch):
    board = {"cols": 5, "rows": 4, "square_length": .04,
             "marker_length": .03, "dict_type": "DICT_4X4_1000"}
    (tmp_path / "session.json").write_text(json.dumps({"board": board}))
    images = tmp_path / "images" / "rgb"
    images.mkdir(parents=True)
    for side in ("left", "right"):
        cv2.imwrite(str(images / f"{side}_0000.png"), np.zeros((100, 160, 3), np.uint8))
    orders = [np.array([9, 1, 4, 0, 5, 8, 2]), np.array([2, 8, 5, 0, 4, 1, 10])]

    class Detector:
        def __init__(self, board):
            self.side = 0

        def detect(self, image):
            ids = orders[self.side]
            xy = np.c_[ids * 3., ids * 2. + self.side]
            self.side += 1
            return SimpleNamespace(valid=True, num_corners=len(ids), charuco_ids=ids[:, None],
                                   charuco_corners=xy[:, None, :])

    monkeypatch.setattr(comparison, "CharucoDetector", Detector)
    result = comparison.collect_observations(tmp_path, "rgb")
    record = result["records"][0]
    assert record["ids"] == [0, 1, 2, 4, 5, 8]
    np.testing.assert_array_equal(np.array(record["left"])[:, 0], np.array(record["ids"]) * 3)
    np.testing.assert_array_equal(np.array(record["right"])[:, 1], np.array(record["ids"]) * 2 + 1)
    cache = tmp_path / "cache.json"
    comparison.write_json(cache, result)
    assert comparison.collect_observations(tmp_path, "rgb", cache) == result
    cv2.imwrite(str(images / "right_0000.png"), np.ones((100, 160, 3), np.uint8))
    with pytest.raises(ValueError, match="does not match"):
        comparison.collect_observations(tmp_path, "rgb", cache)


def test_transfer_detects_wrong_baseline_and_evaluation_does_not_change_model():
    objects, model, records = synthetic()
    original = comparison.serializable(model)
    correct = comparison.evaluate(records[-3:], objects, (1280, 720), model)
    assert correct["cross_camera_transfer"]["rms_px"] < 1e-6
    assert correct["joint_pose"]["rms_px"] < 1e-6
    assert comparison.serializable(model) == original
    wrong = {**model, "T": model["T"] + np.array([.02, 0., 0.])}
    errors = comparison.evaluate(records[-3:], objects, (1280, 720), wrong)
    assert errors["cross_camera_transfer"]["rms_px"] > 5


def test_opencv5_and_mrcal_recover_synthetic_stereo_with_missing_corners(tmp_path):
    pytest.importorskip("mrcal")
    objects, truth, records = synthetic()
    cv_models = comparison.solve_opencv(records, objects, (1280, 720))
    data = {"board": {"cols": 9, "rows": 7, "square_length": .04}, "image_size": [1280, 720]}
    models = comparison.solve_mrcal(records, objects, data, cv_models["opencv_fixed"], tmp_path)
    for model in models.values():
        np.testing.assert_allclose(model["T"], truth["T"], atol=2e-5)
        np.testing.assert_allclose(model["R"], truth["R"], atol=2e-5)
        errors = comparison.evaluate(records[-2:], objects, (1280, 720), model)
        assert errors["cross_camera_transfer"]["rms_px"] < .001
    import mrcal
    loaded = mrcal.cameramodel(str(tmp_path / "mrcal_joint_1.cameramodel"))
    assert loaded.optimization_inputs() is not None
    assert loaded.intrinsics()[0] == "LENSMODEL_OPENCV5"
