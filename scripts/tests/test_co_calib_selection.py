"""Numerical and data-isolation contracts for the offline selection experiment."""
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

SPEC = importlib.util.spec_from_file_location(
    "co_selection", Path(__file__).parents[1] / "assess_co_calib_selection.py")
co = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(co)


def grid():
    xy = np.stack(np.meshgrid(np.arange(4), np.arange(3)), axis=-1).reshape(-1, 2)
    return np.c_[xy, np.zeros(len(xy))].astype(float)


def test_radial_span_survives_center_crossing_and_scale():
    obj = grid()
    xy = obj[:, :2] * 80 + [280, 220]
    first = co.frame_metrics(xy, obj, (800, 600))
    second = co.frame_metrics(xy * 2, obj, (1600, 1200))
    assert first["span"] == pytest.approx(.24)
    assert first["span"] == pytest.approx(second["span"])
    assert first["iso"] == pytest.approx(1, abs=1e-6)


def test_projective_isotropy_detects_compressed_board():
    obj = grid()
    xy = obj[:, :2] * [100, 10] + [100, 100]
    metrics = co.frame_metrics(xy, obj, (800, 600))
    assert metrics["iso"] == pytest.approx(.1, abs=1e-6)
    assert metrics["homography_rms_px"] < 1e-5


@pytest.mark.parametrize("policy", ["uniform", "corners", "radial", "radial_iso",
                                   "coverage_radial_iso"])
def test_selection_has_fixed_budget_and_does_not_mutate_inputs(policy):
    records = [{"ids": list(range(12))} for _ in range(8)]
    metrics = [[{"span": .1 + i / 20, "iso": .8, "cells": [i, i + 1]}] * 2
               for i in range(8)]
    before = co.json.dumps((records, metrics))
    indices = co.select(records, metrics, policy, 6)
    assert len(set(indices)) == 6
    assert indices == sorted(indices)
    assert min(indices) >= 0 and max(indices) < len(records)
    assert co.json.dumps((records, metrics)) == before


def test_bidirectional_prediction_uses_correct_transform_and_keeps_target_out_of_pose():
    obj = grid() * .04
    board = SimpleNamespace(object_points=obj)
    k = np.array([[650., 0, 400], [0, 650., 300], [0, 0, 1.]])
    intr = SimpleNamespace(camera_matrix=k, dist_coeffs=np.zeros(5))
    rotation = cv2.Rodrigues(np.array([.01, -.03, .02]))[0]
    translation = np.array([-.08, .003, .002])
    rv = np.array([.2, -.15, .05])
    tv = np.array([-.04, -.04, .7])
    left = cv2.projectPoints(obj, rv, tv, k, intr.dist_coeffs)[0].reshape(-1, 2)
    right = cv2.projectPoints(obj, cv2.Rodrigues(rotation @ cv2.Rodrigues(rv)[0])[0],
                              rotation @ tv + translation, k, intr.dist_coeffs)[0].reshape(-1, 2)
    data = {"image_size": (800, 600)}
    record = {"capture_id": "holdout", "ids": list(range(len(obj))),
              "left": left.tolist(), "right": right.tolist()}
    result = SimpleNamespace(model=co.CameraModel.PINHOLE, R=rotation, T=translation)
    actual = co.evaluate(data, [record], intr, intr, result, board)
    assert actual["all"]["rms_px"] < 1e-5
    record["right"] = (right + [3, 0]).tolist()
    changed = co.evaluate(data, [record], intr, intr, result, board)
    assert changed["left_to_right"]["rms_px"] == pytest.approx(3, abs=1e-5)


def test_role_allocation_keeps_pairs_in_each_monocular_set(monkeypatch):
    monkeypatch.setitem(sys.modules, "assess_co_calib_selection", co)
    spec = importlib.util.spec_from_file_location(
        "co_roles", Path(__file__).parents[1] / "assess_co_calib_roles.py")
    roles = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(roles)
    metrics = [[{"span": .4, "iso": .8, "points": 20, "center_fraction": .1,
                 "cells": [i, i + 1]}] * 2 for i in range(8)]
    selected = roles.role_indices(metrics)
    assert selected["stereo"]
    for monocular in selected["mono"]:
        assert len(monocular) == len(set(monocular))
        assert set(selected["stereo"]) <= set(monocular)
    invalid = [[{**side, "iso": .1} for side in pair] for pair in metrics]
    assert roles.role_indices(invalid)["mono"] == [[], []]
