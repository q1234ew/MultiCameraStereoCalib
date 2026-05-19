"""Background workers for calibration tasks."""

from __future__ import annotations

import traceback

from PySide6.QtCore import QThread, Signal

from ...calibration.auxiliary import AuxIntrinsicCalibrator, CrossModalExtrinsicCalibrator
from ...calibration.models import CameraIntrinsics, StereoPairCalibration


class PairCalibrationWorker(QThread):
    """Run one stereo pair calibration away from the UI thread."""

    log = Signal(str)
    finished_ok = Signal(str, str, str, object, object, object)
    failed = Signal(str, str)

    def __init__(
        self,
        pair_name: str,
        left_id: str,
        right_id: str,
        left_calibrator,
        right_calibrator,
        stereo_calibrator,
        left_intrinsics: CameraIntrinsics | None = None,
        right_intrinsics: CameraIntrinsics | None = None,
        parent=None,
    ):
        super().__init__(parent)
        self.pair_name = pair_name
        self.left_id = left_id
        self.right_id = right_id
        self.left_calibrator = left_calibrator
        self.right_calibrator = right_calibrator
        self.stereo_calibrator = stereo_calibrator
        self.left_intrinsics = left_intrinsics
        self.right_intrinsics = right_intrinsics

    def run(self):
        try:
            left_intr = self.left_intrinsics
            if left_intr is None:
                self.log.emit(f"标定内参: {self.left_id}...")
                left_intr = self.left_calibrator.calibrate()

            right_intr = self.right_intrinsics
            if right_intr is None:
                self.log.emit(f"标定内参: {self.right_id}...")
                right_intr = self.right_calibrator.calibrate()

            self.log.emit(f"标定外参: {self.pair_name}...")
            stereo = self.stereo_calibrator.calibrate(left_intr, right_intr)
            pair_calib = StereoPairCalibration(
                pair_name=self.pair_name,
                left_id=self.left_id,
                right_id=self.right_id,
                left_intrinsics=left_intr,
                right_intrinsics=right_intr,
                stereo=stereo,
            )
            self.finished_ok.emit(
                self.pair_name,
                self.left_id,
                self.right_id,
                left_intr,
                right_intr,
                pair_calib,
            )
        except Exception as exc:
            self.failed.emit(self.pair_name, f"{exc}\n{traceback.format_exc(limit=3)}")


class AuxCalibrationWorker(QThread):
    """Run auxiliary mono intrinsics and AUX-to-RGB_L extrinsics off the UI thread."""

    log = Signal(str)
    finished_ok = Signal(object, object)
    failed = Signal(str)

    def __init__(
        self,
        aux_id: str,
        board_config,
        camera_model,
        detections,
        rgb_left_intrinsics: CameraIntrinsics,
        min_intrinsic_frames: int,
        parent=None,
    ):
        super().__init__(parent)
        self.aux_id = aux_id
        self.board_config = board_config
        self.camera_model = camera_model
        self.detections = list(detections)
        self.rgb_left_intrinsics = rgb_left_intrinsics
        self.min_intrinsic_frames = min_intrinsic_frames

    def run(self):
        try:
            aux_intr_cal = AuxIntrinsicCalibrator(
                self.board_config,
                model=self.camera_model,
                min_frames=self.min_intrinsic_frames,
            )
            cross_cal = CrossModalExtrinsicCalibrator(self.board_config, min_frames=3)

            for _, rgb_det, aux_det in self.detections:
                aux_intr_cal.add_detection(aux_det)
                cross_cal.add_detection_pair(rgb_det, aux_det)

            self.log.emit(f"{self.aux_id}: 辅助单目内参标定...")
            aux_intr = aux_intr_cal.calibrate()
            self.log.emit(f"{self.aux_id}: 内参 RMS={aux_intr.rms_error:.4f}px")
            self.log.emit(f"{self.aux_id}: 跨模态外参标定...")
            extr = cross_cal.calibrate(self.rgb_left_intrinsics, aux_intr)
            self.finished_ok.emit(aux_intr, extr)
        except Exception as exc:
            self.failed.emit(f"{exc}\n{traceback.format_exc(limit=3)}")
