"""Visible pose evidence for person candidates, independent of GPU dependencies."""
from __future__ import annotations

import numpy as np


def person_pose_mask(boxes, keypoints, image_size, confidence=None):
    """Require several visible landmarks, including two torso landmarks.

    This rejects empty/weak skeletons while allowing seated, cropped and partially
    occluded people. It is a quality gate, not proof that an object is a person.
    """
    boxes = np.asarray(boxes, dtype=float)
    points = np.asarray(keypoints, dtype=float)
    if boxes.size == 0 and points.size == 0:
        return np.zeros(0, dtype=bool)
    if boxes.ndim != 2 or boxes.shape[1] != 4 or points.shape != (len(boxes), 17, 3):
        raise ValueError('Person boxes and COCO17 poses do not align')
    width, height = image_size
    visible = ((points[:, :, 2] >= .4) & (points[:, :, 2] <= 1)
               & np.isfinite(points).all(axis=2)
               & (points[:, :, 0] >= 0) & (points[:, :, 0] < width)
               & (points[:, :, 1] >= 0) & (points[:, :, 1] < height)
               & (points[:, :, 0] >= boxes[:, None, 0])
               & (points[:, :, 0] <= boxes[:, None, 2])
               & (points[:, :, 1] >= boxes[:, None, 1])
               & (points[:, :, 1] <= boxes[:, None, 3]))
    body = ((visible.sum(axis=1) >= 4)
            & (visible[:, [5, 6, 11, 12]].sum(axis=1) >= 2))
    if confidence is not None:
        scores = np.asarray(confidence, dtype=float)
        if scores.shape != (len(boxes),):
            raise ValueError('Person confidence and boxes do not align')
        # A person cropped at the image edge may expose only one connected limb.
        edge = ((boxes[:, 0] <= 1) | (boxes[:, 1] <= 1)
                | (boxes[:, 2] >= width - 1) | (boxes[:, 3] >= height - 1))
        limb = np.zeros(len(boxes), dtype=bool)
        for a, b in [(7, 9), (8, 10), (13, 15), (14, 16)]:
            limb |= (visible[:, a] & visible[:, b]
                     & (np.linalg.norm(points[:, a, :2] - points[:, b, :2], axis=1) > 3))
        body |= edge & limb & np.isfinite(scores) & (scores >= .4)
    return (np.isfinite(boxes).all(axis=1) & (boxes[:, 2] > boxes[:, 0])
            & (boxes[:, 3] > boxes[:, 1]) & body)
