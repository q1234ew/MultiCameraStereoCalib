"""Display verdicts must not confuse valid files with calibration quality."""

import numpy as np
import pytest

from multicamera.calibration.models import (
    CameraIntrinsics, MultiCameraRig, StereoCalibResult, StereoPairCalibration,
)
from multicamera.calibration.quality_summary import summarize_quality


def rig_with_error(error):
    intr = CameraIntrinsics(
        np.array([[900., 0., 640.], [0., 900., 360.], [0., 0., 1.]]),
        np.zeros((1, 5)), (1280, 720), rms_error=.2, per_frame_errors=[.2] * 12,
    )
    stereo = StereoCalibResult(np.eye(3), np.array([[-.1], [0.], [0.]]),
                              np.eye(3), np.eye(3), rms_error=error)
    return MultiCameraRig(pairs={'rgb': StereoPairCalibration('rgb', 'L', 'R', intr, intr, stereo)})


@pytest.mark.parametrize('error,title', [
    (.2, '建议接受'), (.9, '建议补拍'), (2., '不建议直接使用'), (float('nan'), '质量指标无效'),
])
def test_verdict_uses_existing_error_thresholds(error, title):
    assert summarize_quality(rig_with_error(error)).title.startswith(title)


def test_missing_frame_errors_requires_review():
    rig = rig_with_error(.2)
    rig.pairs['rgb'].left_intrinsics.per_frame_errors = None
    result = summarize_quality(rig)
    assert result.title == '建议复核后使用'
    assert any('缺少逐帧误差' in risk for risk in result.risks)


def test_invalid_frame_errors_not_accepted():
    rig = rig_with_error(.2)
    rig.pairs['rgb'].left_intrinsics.per_frame_errors = [float('nan')]
    assert summarize_quality(rig).level == 'poor'
