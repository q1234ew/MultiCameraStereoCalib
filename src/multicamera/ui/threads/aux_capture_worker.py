"""Raw planar detection, quality and snapshot analysis outside the UI thread."""

from dataclasses import replace

from PySide6.QtCore import QObject, QRunnable, Signal

from ...calibration.aux_capture_quality import assess_aux_frame
from ...calibration.adaptive_guidance import measure_composition, measure_pose, _valid_intrinsics
from ...calibration.aux_guidance import (
    analyze_aux_guidance, board_adapter, observation, summarize_aux_progress,
)
from ...calibration.health import check_intrinsics
from ...calibration.planar import PlanarPatternDetector


class AuxCaptureSignals(QObject):
    completed = Signal(object, object, str)


class AuxCaptureWorker(QRunnable):
    def __init__(self, key, images, board, intrinsics, modality, *, enhanced=False):
        super().__init__()
        self.signals = AuxCaptureSignals()
        self.key, self.images, self.board = key, images, board
        self.intrinsics, self.modality = intrinsics, modality
        self.enhanced = enhanced

    def run(self):
        try:
            detector = PlanarPatternDetector(self.board)
            detections = [detector.detect_offline(image,
                modality='ir' if i == 1 and self.modality == 'ir' else 'rgb')
                if self.enhanced else detector.detect(image)
                for i, image in enumerate(self.images)]
            quality = []
            adapter = board_adapter(self.board)
            for i, (image, det, intr) in enumerate(zip(self.images, detections, self.intrinsics)):
                # Weak guidance intrinsics must not discard otherwise usable raw images.
                result = assess_aux_frame(image, det, self.board,
                    modality='rgb' if i == 0 or self.modality == 'acoustic_rgb' else 'ir')
                metrics = dict(result.metrics)
                composition = measure_composition(adapter, observation(det)) if result.accepted else None
                if composition:
                    metrics.update(area=composition['area'], center=composition['center'].tolist())
                pose = None
                if (result.accepted and intr is not None
                        and _valid_intrinsics(intr, det.image_size, intr.model)
                        and check_intrinsics(intr)['status'] == 'ok'):
                    pose = measure_pose(adapter, observation(det), intr)
                quality.append(replace(result, metrics=metrics, pose=pose))
            self.signals.completed.emit(self.key, (detections, quality), '')
        except Exception as exc:
            self.signals.completed.emit(self.key, None, str(exc))


class AuxGuidanceProgressWorker(QRunnable):
    def __init__(self, key, board, pairs, intrinsics=(None, None)):
        super().__init__()
        self.signals = AuxCaptureSignals()
        self.key, self.board = key, board
        self.pairs, self.intrinsics = pairs, intrinsics

    def run(self):
        try:
            result = summarize_aux_progress(self.board, self.pairs, self.intrinsics)
            self.signals.completed.emit(self.key, result, '')
        except Exception as exc:
            self.signals.completed.emit(self.key, None, str(exc))


class AuxGuidanceWorker(QRunnable):
    def __init__(self, key, board, model, pairs, intrinsics):
        super().__init__()
        self.signals = AuxCaptureSignals()
        self.key, self.board, self.model = key, board, model
        self.pairs, self.intrinsics = pairs, intrinsics

    def run(self):
        try:
            result = analyze_aux_guidance(self.board, self.model, self.pairs, self.intrinsics)
            self.signals.completed.emit(self.key, result, '')
        except Exception as exc:
            self.signals.completed.emit(self.key, None, str(exc))
