"""Deterministic, non-destructive observation selection for background solves."""

from __future__ import annotations

import cv2
import numpy as np

from .models import CameraModel


def top_score_indices(scores, min_frames, keep_ratio=.8):
    keep = min(len(scores), max(min_frames, int(np.ceil(len(scores) * keep_ratio))))
    ranked = sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)
    return sorted(ranked[:keep])


def image_scores(calibrator):
    scores = []
    for image in calibrator.stored_images:
        if image is None:
            scores.append(.6)
            continue
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        brightness = float(gray.mean())
        scores.append(.65 * min(sharpness / 120., 1.) +
                      .35 * max(0., 1. - abs(brightness - 128.) / 128.))
    return scores


def select_calibrators(left, right, stereo, key="all"):
    if key not in {"all", "quality", "corners", "fisheye_common"}:
        raise ValueError(f"Unknown data selection policy: {key}")
    calibrators = [left, right, stereo]
    counts = [left.frame_corner_counts, right.frame_corner_counts, stereo.frame_common_corner_counts]
    if key in {"all", "fisheye_common"}:
        indices = [list(range(cal.num_frames)) for cal in calibrators]
        selected = calibrators
        if key == "fisheye_common":
            if stereo.model != CameraModel.FISHEYE:
                raise ValueError("跨帧共同角点兼容策略仅适用于鱼眼双目标定")
            selected = [left, right, stereo.copy_with_common_ids()]
    else:
        if key == "quality":
            quality_l, quality_r = image_scores(left), image_scores(right)
            quality_by_id = [dict(zip(cal.frame_ids, scores))
                             for cal, scores in ((left, quality_l), (right, quality_r))]
            quality_stereo = [min(side.get(frame_id, .6) for side in quality_by_id)
                              for frame_id in stereo.frame_ids]
            scores = [[.55 * value / max(max(values, default=1), 1) + .45 * quality
                       for value, quality in zip(values, qualities)]
                      for values, qualities in zip(counts, [quality_l, quality_r, quality_stereo])]
        else:
            scores = counts
        indices = [top_score_indices(values, cal.min_frames)
                   for cal, values in zip(calibrators, scores)]
        selected = [left.copy_with_indices(indices[0], keep_images=False),
                    right.copy_with_indices(indices[1], keep_images=False),
                    stereo.copy_with_indices(indices[2])]
    info = {"policy": key, "keep_ratio": 1. if key in {"all", "fisheye_common"} else .8}
    for label, cal, used in zip(("left", "right", "stereo"), calibrators, indices):
        keep = set(used)
        info[label] = {"input_count": cal.num_frames, "used_count": len(used),
                       "used_ids": [cal.frame_ids[i] for i in used],
                       "excluded_ids": [value for i, value in enumerate(cal.frame_ids) if i not in keep]}
    if key == "fisheye_common":
        info["stereo"]["input_corner_counts"] = stereo.frame_common_corner_counts
        info["stereo"]["used_corner_counts"] = selected[2].frame_common_corner_counts
        info["stereo"]["shared_corner_ids"] = selected[2].frame_corner_ids[0].tolist()
    return (*selected, info)
