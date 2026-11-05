"""Background workers for calibration tasks."""

from __future__ import annotations

import logging
import traceback
from copy import deepcopy

import cv2
import numpy as np
from PySide6.QtCore import QThread, Signal

from ...calibration.auxiliary import AuxIntrinsicCalibrator, CrossModalExtrinsicCalibrator
from ...calibration.models import CameraIntrinsics, CameraModel, MultiCameraRig, StereoPairCalibration
from ...calibration.quality_summary import summarize_quality
from ...calibration.selection import select_calibrators
from ...calibration.stereo_diagnostics import measure_rectification
from ...calibration.health import check_stereo_health
from ...calibration.validated_selection import calibrate_validated
from ...calibration.stereo import pinhole_stereo_flags

logger = logging.getLogger(__name__)


def _provided_intrinsic_selection(intrinsics, selected):
    ids = list(intrinsics.frame_ids) if intrinsics.frame_ids is not None else None
    return {
        "source": "provided_intrinsics", "used_ids": ids if ids is not None else [],
        "frame_ids_known": ids is not None,
        "used_count": len(ids) if ids is not None else None,
        "input_count": None, "excluded_ids": [], "excluded_ids_known": False,
        "selected_observations": deepcopy(selected),
    }


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
        intrinsic_flags: int = 0,
        stereo_flags: int = 0,
        strategy_name: str = "标准",
        data_filter_key: str = "all",
        parent=None,
        automatic_model: bool = False,
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
        self.intrinsic_flags = intrinsic_flags
        self.stereo_flags = stereo_flags
        self.strategy_name = strategy_name
        self.data_filter_key = data_filter_key
        self.automatic_model = automatic_model

    def run(self):
        try:
            logger.info(
                "Pair calibration start: pair=%s left=%s right=%s strategy=%s "
                "intrinsic_flags=%d stereo_flags=%d frames_left=%s frames_right=%s frames_stereo=%s",
                self.pair_name,
                self.left_id,
                self.right_id,
                self.strategy_name,
                self.intrinsic_flags,
                self.stereo_flags,
                getattr(self.left_calibrator, "num_frames", "?"),
                getattr(self.right_calibrator, "num_frames", "?"),
                getattr(self.stereo_calibrator, "num_frames", "?"),
            )
            self.log.emit(f"标定策略: {self.strategy_name}")
            selection = {}
            left_cal, right_cal, stereo_cal = self.left_calibrator, self.right_calibrator, self.stereo_calibrator
            if left_cal is not None and right_cal is not None:
                left_cal, right_cal, stereo_cal, selection = select_calibrators(
                    left_cal, right_cal, stereo_cal, self.data_filter_key,
                )
                self.log.emit(
                    f"数据筛选[{self.data_filter_key}]: " + ", ".join(
                        f"{side} {selection[side]['used_count']}/{selection[side]['input_count']}"
                        for side in ("left", "right", "stereo")
                    )
                )
            selected_sides = deepcopy(selection)
            automatic = None
            if self.automatic_model and self.left_intrinsics is None and self.right_intrinsics is None:
                automatic = calibrate_validated(left_cal, right_cal, stereo_cal, self.log.emit)
            left_intr = deepcopy(self.left_intrinsics)
            if left_intr is None:
                self.log.emit(f"标定内参: {self.left_id}...")
                left_intr = automatic[0] if automatic else left_cal.calibrate(flags=self.intrinsic_flags)
            else:
                selection["left"] = _provided_intrinsic_selection(left_intr, selected_sides.get("left"))

            right_intr = deepcopy(self.right_intrinsics)
            if right_intr is None:
                self.log.emit(f"标定内参: {self.right_id}...")
                right_intr = automatic[1] if automatic else right_cal.calibrate(flags=self.intrinsic_flags)
            else:
                selection["right"] = _provided_intrinsic_selection(right_intr, selected_sides.get("right"))

            self.log.emit(f"标定外参: {self.pair_name}...")
            before = [(intr.camera_matrix.copy(), intr.dist_coeffs.copy())
                      for intr in (left_intr, right_intr)]
            stereo = automatic[2] if automatic else stereo_cal.calibrate(
                left_intr, right_intr, flags=self.stereo_flags
            )
            fix_flag = (cv2.fisheye.CALIB_FIX_INTRINSIC if stereo.model == CameraModel.FISHEYE
                        else cv2.CALIB_FIX_INTRINSIC)
            if self.stereo_flags and not self.stereo_flags & fix_flag:
                for side, cal, intr, (old_k, old_d) in zip(
                    ("left", "right"), (left_cal, right_cal), (left_intr, right_intr), before,
                ):
                    if (not np.array_equal(intr.camera_matrix, old_k) or
                            not np.array_equal(np.asarray(intr.dist_coeffs).ravel(), old_d.ravel())):
                        if cal is None:
                            raise ValueError("联合精修需要原始观测以复核新内参误差")
                        initial_ids = list(intr.frame_ids) if intr.frame_ids is not None else None
                        cal.recompute_errors(intr)
                        selection[side] = {
                            **selected_sides.get(side, {}), "source": "refined",
                            "initial_frame_ids": initial_ids,
                            "initial_source": ("provided_intrinsics"
                                               if getattr(self, f"{side}_intrinsics") is not None
                                               else "selected_observations"),
                            "used_ids": list(intr.frame_ids), "used_count": len(intr.frame_ids),
                        }
            health = check_stereo_health(left_intr, right_intr, stereo)
            if health['status'] != 'ok':
                raise ValueError('不可用于立体校正：' + '；'.join(health['reasons'])
                                 + '；请使用自动校验择模或补充观测后重试')
            self.log.emit('有效性检查通过：边界反算、径向单调性及校正视野（数值采样）')
            info = {'health': health}
            if selection:
                rectification = measure_rectification(stereo_cal, left_intr, right_intr, stereo)
                rectification['dataset'] = 'selected_stereo_observations'
                info.update({"opencv_version": cv2.__version__, "strategy": self.strategy_name,
                        "intrinsic_flags": int(self.intrinsic_flags),
                        "stereo_flags": int(self.stereo_flags),
                        "camera_model": stereo.model.value,
                        "selection": selection, "board": stereo_cal.board.to_dict(),
                        "rectification": rectification})
                self.log.emit(
                    f"同 ID 校正残差：{rectification['axis']}，{rectification['point_count']} 点，"
                    f"RMS={rectification['rms_px']} px（训练集，非独立验证）"
                )
            if hasattr(self, 'review_info'):
                info['review'] = deepcopy(self.review_info)
                info['review']['image_coverage'] = [cal.compute_image_coverage() if cal else None
                                                    for cal in (left_cal, right_cal)]
            pair_calib = StereoPairCalibration(
                pair_name=self.pair_name,
                left_id=self.left_id,
                right_id=self.right_id,
                left_intrinsics=left_intr,
                right_intrinsics=right_intr,
                stereo=stereo,
                calibration_info=info,
            )
            if automatic:
                info['model_selection'] = automatic[3]
                selected = automatic[3]['selected']
                info['intrinsic_flags'] = (
                    cv2.CALIB_RATIONAL_MODEL | cv2.CALIB_USE_INTRINSIC_GUESS if selected == 'rational'
                    else (cv2.fisheye.CALIB_FIX_K3 | cv2.fisheye.CALIB_FIX_K4
                          if stereo.model == CameraModel.FISHEYE else cv2.CALIB_FIX_K3)
                    if selected == 'constrained' else 0)
                info['stereo_flags'] = ((cv2.fisheye.CALIB_FIX_INTRINSIC | cv2.fisheye.CALIB_FIX_SKEW
                                         if stereo.model == CameraModel.FISHEYE else cv2.CALIB_FIX_INTRINSIC)
                                        | (cv2.CALIB_RATIONAL_MODEL if selected == 'rational' else 0))
            info['requested_stereo_flags'] = int(self.stereo_flags)
            info['requested_intrinsic_flags'] = int(self.intrinsic_flags)
            if stereo.model == CameraModel.PINHOLE:
                info['stereo_flags'] = int(pinhole_stereo_flags(
                    left_intr, right_intr, info.get('stereo_flags', self.stereo_flags)))
            else:
                info['stereo_flags'] = int(info.get('stereo_flags', self.stereo_flags)
                                           or (cv2.fisheye.CALIB_FIX_INTRINSIC
                                               | cv2.fisheye.CALIB_FIX_SKEW))
            quality = summarize_quality(MultiCameraRig(pairs={self.pair_name: pair_calib}))
            if not quality.valid:
                raise ValueError("标定结果无效：" + "；".join(quality.risks))
            self.finished_ok.emit(
                self.pair_name,
                self.left_id,
                self.right_id,
                left_intr,
                right_intr,
                pair_calib,
            )
            logger.info(
                "Pair calibration finished: pair=%s left_rms=%.4f right_rms=%.4f stereo_rms=%.4f",
                self.pair_name,
                left_intr.rms_error,
                right_intr.rms_error,
                stereo.rms_error,
            )
        except Exception as exc:
            logger.exception("Pair calibration failed: pair=%s", self.pair_name)
            self.failed.emit(self.pair_name, f"{exc}\n{traceback.format_exc(limit=3)}")


class AuxCorrespondenceWorker(QThread):
    """Read-only grid symmetry inspection; never modifies admission decisions."""

    finished_ok = Signal(object)
    failed = Signal(str)

    def __init__(self, records, board, rgb_intrinsics, model_key, parent=None):
        super().__init__(parent)
        self.inputs = deepcopy((records, board, rgb_intrinsics, model_key))

    def run(self):
        try:
            from ...calibration.joint_aux import inspect_aux_correspondences
            self.finished_ok.emit(inspect_aux_correspondences(*self.inputs))
        except Exception as exc:
            logger.exception('Auxiliary correspondence inspection failed')
            self.failed.emit(str(exc))


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
        offline_model_key=None,
        offline_reviews=None,
        offline_dataset_reviews=None,
    ):
        super().__init__(parent)
        self.aux_id = aux_id
        self.board_config = board_config
        self.camera_model = camera_model
        self.detections = list(detections)
        self.rgb_left_intrinsics = rgb_left_intrinsics
        self.min_intrinsic_frames = min_intrinsic_frames
        self.offline_model_key = offline_model_key
        self.offline_reviews = deepcopy(offline_reviews or [])
        self.offline_dataset_reviews = deepcopy(offline_dataset_reviews or [])

    def run(self):
        try:
            if self.offline_model_key is not None:
                from ...calibration.joint_aux import calibrate_offline_aux
                from ...calibration.offline_aux import verify_review_sources
                verify_review_sources(self.offline_reviews)
                if {str(row[0]) for row in self.detections} != {
                        review['source_id'] for review in self.offline_reviews}:
                    raise ValueError('求解数据与逐对复核编号不一致')
                aux_intr, extr = calibrate_offline_aux(self.detections, self.board_config,
                    self.rgb_left_intrinsics, self.offline_model_key, self.log.emit)
                verify_review_sources(self.offline_reviews)
                extr.calibration_info['offline_reviews'] = deepcopy(self.offline_reviews)
                extr.calibration_info['offline_dataset_reviews'] = deepcopy(self.offline_dataset_reviews)
                self.finished_ok.emit(aux_intr, extr)
                return
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
