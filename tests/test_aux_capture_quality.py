"""Original-frame quality regressions using rendered boards, not white canvases."""

from dataclasses import replace

import cv2
import numpy as np
import pytest

from multicamera.calibration import aux_capture_quality as quality
from multicamera.calibration.aux_capture_quality import AuxFrameQuality, assess_aux_frame
from multicamera.calibration.models import CameraIntrinsics, CameraModel
from multicamera.calibration.planar import (
    PatternBoardConfig,
    PatternType,
    PlanarDetectionResult,
    PlanarPatternDetector,
)


def chessboard(spacing=20, levels=(30, 220), padding=24):
    board = PatternBoardConfig.ir_chessboard()
    height = (board.rows + 1) * spacing + 2 * padding
    width = (board.cols + 1) * spacing + 2 * padding
    image = np.full((height, width), 125, np.uint8)
    for row in range(board.rows + 1):
        for col in range(board.cols + 1):
            y, x = padding + row * spacing, padding + col * spacing
            image[y:y + spacing, x:x + spacing] = levels[(row + col) % 2]
    xy = np.indices((board.rows, board.cols))[::-1].transpose(1, 2, 0).reshape(-1, 2)
    points = padding + (xy + 1) * spacing - .5
    detection = PlanarDetectionResult(
        True, PlanarPatternDetector._make_object_points(board), points.astype(np.float64),
        (width, height), board.pattern_type,
    )
    return image, detection, board


def intrinsics_for(detection, model=CameraModel.PINHOLE):
    w, h = detection.image_size
    return CameraIntrinsics(
        np.array([[180., 0, w / 2], [0, 180., h / 2], [0, 0, 1.]]),
        np.zeros(4 if model == CameraModel.FISHEYE else 5), detection.image_size, model,
    )


@pytest.mark.parametrize("spacing", [4, 5, 8, 20, 48])
@pytest.mark.parametrize("modality", ["ir", "rgb", "aux"])
def test_sharp_real_chessboards_at_multiple_scales(spacing, modality):
    image, detection, board = chessboard(spacing)
    result = assess_aux_frame(image, detection, board, modality=modality)
    assert isinstance(result, AuxFrameQuality)
    assert result.accepted, (result.reasons, result.metrics)
    assert result.reasons == ()
    assert result.pose is None
    assert result.metrics["min_grid_spacing_px"] == spacing
    assert result.metrics["pattern_agreement"] == 1
    assert result.metrics["edge_width_fraction_p75"] < .4


def test_actual_detector_observation_is_accepted_without_mutation():
    image, _, board = chessboard(32)
    detection = PlanarPatternDetector(board).detect(image)
    assert detection.valid
    before = image.copy(), detection.image_points.copy(), detection.object_points.copy()
    assert assess_aux_frame(image, detection, board).accepted
    for actual, expected in zip((image, detection.image_points, detection.object_points), before):
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("spacing", [4, 8, 20, 40])
def test_scale_relative_blur_is_rejected(spacing):
    image, detection, board = chessboard(spacing)
    blurred = cv2.GaussianBlur(image, (0, 0), spacing * .25)
    result = assess_aux_frame(blurred, detection, board)
    assert not result.accepted
    assert "blurred_board_edges" in result.reasons


def test_mild_blur_remains_usable():
    image, detection, board = chessboard(24)
    image = cv2.GaussianBlur(image, (0, 0), 1.0)
    assert assess_aux_frame(image, detection, board).accepted


@pytest.mark.parametrize("levels", [(124, 127), (0, 3), (251, 254)])
def test_low_local_contrast_is_not_normalized_into_acceptance(levels):
    image, detection, board = chessboard(levels=levels)
    result = assess_aux_frame(image, detection, board)
    assert not result.accepted
    assert "low_board_contrast" in result.reasons


def test_ir_threshold_is_lower_than_rgb_without_brightness_gate():
    image, detection, board = chessboard(levels=(10, 18))
    assert assess_aux_frame(image, detection, board).accepted
    rgb = assess_aux_frame(image, detection, board, modality="rgb")
    assert not rgb.accepted
    assert "low_board_contrast" in rgb.reasons
    for levels in ((0, 20), (235, 255)):
        image, detection, board = chessboard(levels=levels)
        assert assess_aux_frame(image, detection, board).accepted


def test_background_texture_does_not_rescue_flat_or_blurred_board():
    image, detection, board = chessboard()
    background = np.indices(image.shape).sum(axis=0) % 2 * 255
    for board_image in (np.full_like(image, 126), cv2.GaussianBlur(image, (0, 0), 6)):
        combined = background.astype(np.uint8)
        combined[24:-24, 24:-24] = board_image[24:-24, 24:-24]
        assert not assess_aux_frame(combined, detection, board).accepted


def test_random_thermal_texture_is_not_board_edge_evidence():
    image, detection, board = chessboard()
    image = np.random.default_rng(14).integers(0, 256, image.shape, dtype=np.uint8)
    result = assess_aux_frame(image, detection, board)
    assert not result.accepted
    assert "inconsistent_board_signal" in result.reasons


@pytest.mark.parametrize("color_map", [cv2.COLORMAP_JET, cv2.COLORMAP_INFERNO])
def test_ir_false_color_uses_luminance(color_map):
    image, detection, board = chessboard(levels=(45, 185))
    color = cv2.applyColorMap(image, color_map)
    gray = cv2.cvtColor(color.astype(np.float32) / 255, cv2.COLOR_BGR2GRAY)
    colored = assess_aux_frame(color, detection, board)
    monochrome = assess_aux_frame(gray, detection, board)
    assert colored.accepted
    assert colored.accepted == monochrome.accepted
    assert colored.metrics["pattern_contrast"] == pytest.approx(
        monochrome.metrics["pattern_contrast"], abs=1e-6
    )


def test_isoluminant_false_color_does_not_get_channel_contrast_credit():
    image, detection, board = chessboard()
    color = np.empty((*image.shape, 3), np.uint8)
    color[:] = (0, 0, 100)
    color[image > 128] = (0, 51, 0)
    result = assess_aux_frame(color, detection, board)
    assert not result.accepted
    assert "low_board_contrast" in result.reasons


@pytest.mark.parametrize("encoding", ["uint16", "float32", "float255", "bgr", "bgra", "one"])
def test_supported_image_encodings(encoding):
    image, detection, board = chessboard()
    if encoding == "uint16":
        image = image.astype(np.uint16) * 257
    elif encoding == "float32":
        image = image.astype(np.float32) / 255
    elif encoding == "float255":
        image = image.astype(np.float64)
    elif encoding == "bgr":
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif encoding == "bgra":
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGRA)
    else:
        image = image[:, :, None]
    assert assess_aux_frame(image, detection, board).accepted


@pytest.mark.parametrize("image", [
    None, [], np.empty((0, 10), np.uint8), np.zeros((8,), np.uint8),
    np.zeros((8, 8, 2), np.uint8), np.zeros((8, 8, 3, 1), np.uint8),
    np.zeros((8, 8), np.int32), np.zeros((8, 8), bool),
    np.full((8, 8), np.nan), np.full((8, 8), np.inf),
    np.full((8, 8), -1.), np.full((8, 8), 256.),
])
def test_invalid_images_fail_closed(image):
    _, detection, board = chessboard()
    result = assess_aux_frame(image, detection, board)
    assert result.reasons == ("invalid_image",)
    assert result.pose is None


@pytest.mark.parametrize("change", [
    {"valid": False}, {"image_size": (0, 100)}, {"image_size": (188., 148.)},
    {"image_size": (True, 148)}, {"pattern_type": PatternType.CIRCLES_GRID},
    {"object_points": None}, {"image_points": "not points"},
])
def test_invalid_detection_metadata(change):
    image, detection, board = chessboard()
    result = assess_aux_frame(image, replace(detection, **change), board)
    assert not result.accepted
    assert result.pose is None


@pytest.mark.parametrize("field", ["image_points", "object_points"])
@pytest.mark.parametrize("mutation", ["missing", "extra", "nan", "inf", "wrong_shape"])
def test_point_arrays_must_be_complete_finite_and_correctly_shaped(field, mutation):
    image, detection, board = chessboard()
    values = getattr(detection, field).copy()
    if mutation == "missing":
        values = values[:-1]
    elif mutation == "extra":
        values = np.concatenate((values, values[:1]))
    elif mutation == "wrong_shape":
        values = values.ravel()
    else:
        values[0, 0] = float(mutation)
    result = assess_aux_frame(image, replace(detection, **{field: values}), board)
    assert not result.accepted


@pytest.mark.parametrize("mutation", ["scale", "translate", "nonplanar", "reorder"])
def test_object_points_must_match_board_frame_and_order(mutation):
    image, detection, board = chessboard()
    values = detection.object_points.copy()
    if mutation == "scale":
        values *= 1000
    elif mutation == "translate":
        values += .01
    elif mutation == "nonplanar":
        values[0, 2] = .01
    else:
        values = values[::-1]
    assert not assess_aux_frame(image, replace(detection, object_points=values), board).accepted


@pytest.mark.parametrize("mutation", ["collinear", "duplicate", "fold", "outside", "tiny"])
def test_invalid_image_grid_geometry(mutation):
    image, detection, board = chessboard()
    points = detection.image_points.copy()
    if mutation == "collinear":
        points[:, 1] = 50
    elif mutation == "duplicate":
        points[-1] = points[0]
    elif mutation == "fold":
        points[[7, 8]] = points[[8, 7]]
    elif mutation == "outside":
        points[0] = (-1, 20)
    else:
        points = points * .1 + 40
    assert not assess_aux_frame(image, replace(detection, image_points=points), board).accepted


def test_board_margin_uses_outer_squares_not_only_inner_corner_hull():
    image, detection, board = chessboard(padding=2)
    assert not assess_aux_frame(image, detection, board).accepted  # 1.5 px pixel-center margin
    image, detection, board = chessboard(padding=3)
    assert assess_aux_frame(image, detection, board).accepted
    image, detection, board = chessboard(padding=0)
    result = assess_aux_frame(image, detection, board)
    assert result.reasons == ("board_margin_too_small",)


def test_sizes_are_full_resolution_and_intrinsics_are_never_silently_scaled():
    image, detection, board = chessboard()
    wrong = replace(detection, image_size=(image.shape[1] * 2, image.shape[0] * 2))
    assert assess_aux_frame(image, wrong, board).reasons == ("detection_size_mismatch",)
    intr = replace(intrinsics_for(detection), image_size=(640, 480))
    result = assess_aux_frame(image, detection, board, intr)
    assert result.reasons == ("invalid_intrinsics",)
    assert result.pose is None


@pytest.mark.parametrize("model", [CameraModel.PINHOLE, CameraModel.FISHEYE])
def test_model_specific_pose_and_physical_units(model):
    image, detection, board = chessboard(24)
    intr = intrinsics_for(detection, model)
    intr.dist_coeffs[:2] = [-.03, .005]
    rvec, tvec = np.array([.15, -.2, .04]), np.array([-.20, -.12, .85])
    source = detection.image_points.astype(np.float32)
    projected = quality._project(detection.object_points.astype(np.float64), rvec, tvec, intr)
    # Render the same physical board through the selected lens; remapping rays
    # avoids using a pinhole homography for the fisheye regression.
    yy, xx = np.indices(image.shape, dtype=np.float64)
    pixels = np.stack((xx, yy), axis=-1).reshape(-1, 1, 2)
    K = intr.camera_matrix
    if model == CameraModel.FISHEYE:
        rays = cv2.fisheye.undistortPoints(pixels, K, intr.dist_coeffs)
    else:
        rays = cv2.undistortPoints(pixels, K, intr.dist_coeffs)
    rays = np.column_stack((rays.reshape(-1, 2), np.ones(len(pixels))))
    R, _ = cv2.Rodrigues(rvec)
    depth = np.dot(R[:, 2], tvec) / (rays @ R[:, 2])
    obj = (rays * depth[:, None] - tvec) @ R
    scale = 24 / board.square_size
    offset = source[0] - detection.object_points[0, :2] * scale
    maps = (obj[:, :2] * scale + offset).reshape(*image.shape, 2).astype(np.float32)
    rendered = cv2.remap(image, maps[:, :, 0], maps[:, :, 1], cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=125)
    detection = replace(detection, image_points=projected)
    result = assess_aux_frame(rendered, detection, board, intr)
    assert result.accepted, (result.reasons, result.metrics)
    center = R @ detection.object_points.mean(axis=0) + tvec
    assert result.pose["distance_reliable"] is True
    assert result.pose["depth_m"] == pytest.approx(center[2], abs=1e-6)
    assert result.pose["distance_m"] == pytest.approx(np.linalg.norm(center), abs=1e-6)
    assert result.pose["yaw_deg"] == pytest.approx(np.degrees(np.arctan2(R[0, 2], R[2, 2])))
    assert result.pose["reprojection_rms_px"] < 1e-5


@pytest.mark.parametrize("change", [
    {"camera_matrix": np.eye(2)}, {"camera_matrix": np.full((3, 3), np.nan)},
    {"dist_coeffs": np.zeros(3)}, {"dist_coeffs": np.full(5, np.inf)},
    {"model": "unknown"},
])
def test_invalid_intrinsics_do_not_create_pose(change):
    image, detection, board = chessboard()
    result = assess_aux_frame(image, detection, board, replace(intrinsics_for(detection), **change))
    assert result.reasons == ("invalid_intrinsics",)
    assert result.pose is None


@pytest.mark.parametrize("failure", [None, "opencv", "nonfinite", "behind", "wrong"])
def test_failed_or_unreliable_pose_is_not_fabricated(monkeypatch, failure):
    image, detection, board = chessboard()

    def solve(*args):
        if failure == "opencv":
            raise cv2.error("synthetic failure")
        if failure == "nonfinite":
            return np.full(3, np.nan), np.ones(3)
        if failure == "behind":
            return np.zeros(3), np.array([0., 0., -1.])
        if failure == "wrong":
            return np.zeros(3), np.array([0., 0., 1.])
        return None

    monkeypatch.setattr(quality, "_solve_board_pose", solve)
    result = assess_aux_frame(image, detection, board, intrinsics_for(detection))
    assert result.reasons == ("unreliable_pose",)
    assert result.pose is None
    assert result.metrics["pose_status"] == "unreliable"


def test_small_board_does_not_claim_precise_pose():
    image, detection, board = chessboard(4)
    result = assess_aux_frame(image, detection, board, intrinsics_for(detection))
    assert result.pose is None
    assert result.reasons == ("unreliable_pose",)


@pytest.mark.parametrize("asymmetric", [False, True])
def test_circle_grid_support(asymmetric):
    board = PatternBoardConfig(5, 4, .04, PatternType.CIRCLES_GRID, asymmetric)
    obj = PlanarPatternDetector._make_object_points(board)
    points = obj[:, :2] / board.square_size * 20 + 50
    image = np.full((170, 290), 220, np.uint8)
    for point in points:
        cv2.circle(image, tuple(np.rint(point).astype(int)), 4, 30, -1)
    det = PlanarDetectionResult(True, obj, points, image.shape[1::-1], board.pattern_type)
    result = assess_aux_frame(image, det, board)
    assert result.accepted, (result.reasons, result.metrics)


def test_invalid_types_and_mutated_config_are_rejected():
    image, detection, board = chessboard()
    assert not assess_aux_frame(image, None, board).accepted
    assert not assess_aux_frame(image, detection, None).accepted
    assert not assess_aux_frame(image, detection, board, object()).accepted
    assert not assess_aux_frame(image, detection, board, modality="unknown").accepted
    board.cols = 0
    assert not assess_aux_frame(image, detection, board).accepted


def test_opencv_point_shapes_are_supported():
    image, detection, board = chessboard()
    detection = replace(detection, image_points=detection.image_points.reshape(-1, 1, 2),
                        object_points=detection.object_points.reshape(-1, 1, 3))
    assert assess_aux_frame(image, detection, board).accepted


def test_symmetric_order_is_never_claimed_to_be_physically_resolved():
    image, detection, board = chessboard()
    reversed_grid = replace(detection, image_points=detection.image_points[::-1].copy())
    assert assess_aux_frame(image, reversed_grid, board).accepted
    # Quality admission cannot tell which physical corner is the origin.
    assert "ordering_resolved" not in assess_aux_frame(image, detection, board).metrics


@pytest.mark.parametrize("change", [
    {"valid": np.ones(3)}, {"valid": 1}, {"image_size": np.array([188, 148])},
    {"image_points": np.ones((24, 2), dtype=complex)},
    {"object_points": np.ones((24, 3), dtype=bool)},
    {"image_points": [[1], [1, 2]]}, {"pattern_type": np.ones(3)},
])
def test_malformed_detection_fields_fail_closed(change):
    image, detection, board = chessboard()
    result = assess_aux_frame(image, replace(detection, **change), board)
    assert not result.accepted
    assert result.pose is None


def test_large_grid_sampling_stays_within_opencv_remap_limits():
    gray = np.zeros((16, 16), np.float32)
    points = np.full((80000, 65, 2), 3., np.float64)
    assert quality._sample(gray, points).shape == (80000, 65)


def test_pose_with_nonfinite_projection_does_not_leak_fake_metrics(monkeypatch):
    image, detection, board = chessboard()
    monkeypatch.setattr(quality, "_project", lambda *args: np.full((24, 2), np.nan))
    result = assess_aux_frame(image, detection, board, intrinsics_for(detection))
    assert result.pose is None
    assert result.reasons == ("unreliable_pose",)
    assert "pose_rms_px" not in result.metrics


@pytest.mark.parametrize("factor", [.2, .4, .75, 1.5])
def test_resized_original_chessboard_is_evaluated_at_its_actual_pixel_scale(factor):
    image, detection, board = chessboard(40, padding=40)
    image = cv2.resize(image, None, fx=factor, fy=factor, interpolation=cv2.INTER_AREA)
    points = (detection.image_points + .5) * factor - .5
    detection = replace(detection, image_points=points, image_size=image.shape[1::-1])
    result = assess_aux_frame(image, detection, board)
    assert result.accepted, (result.reasons, result.metrics)
    assert result.metrics["min_grid_spacing_px"] == pytest.approx(40 * factor)


@pytest.mark.parametrize("dist", [np.zeros(5), np.zeros((2, 2)), np.full(4, np.nan)])
def test_invalid_fisheye_coefficients_are_rejected(dist):
    image, detection, board = chessboard()
    intr = replace(intrinsics_for(detection, CameraModel.FISHEYE), dist_coeffs=dist)
    result = assess_aux_frame(image, detection, board, intr)
    assert result.pose is None
    assert result.reasons == ("invalid_intrinsics",)


@pytest.mark.parametrize("modality", [None, 1, [], np.ones(3)])
def test_nonstring_modality_is_invalid(modality):
    image, detection, board = chessboard()
    assert assess_aux_frame(image, detection, board, modality=modality).reasons == (
        "invalid_modality",
    )


@pytest.mark.parametrize("field", ["camera_matrix", "dist_coeffs"])
def test_complex_intrinsics_are_not_silently_truncated(field):
    image, detection, board = chessboard()
    intr = intrinsics_for(detection)
    setattr(intr, field, getattr(intr, field).astype(complex) + 1j)
    assert assess_aux_frame(image, detection, board, intr).reasons == ("invalid_intrinsics",)
