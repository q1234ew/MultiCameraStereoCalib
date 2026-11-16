"""Replay a local person video with one shared detection cache per input size.

Exports MOT rows and frame mapping for later annotated TrackEval evaluation.
Without identity ground truth, ID counts are descriptive, not accuracy scores.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from collections import Counter
from pathlib import Path

import numpy as np

from .tracking_options import TRACKERS, make_tracker
from .tracking_service import WEIGHT_SHA256


class DetectionBatch:
    """Minimal NumPy-backed Ultralytics Boxes contract for cached replay."""

    def __init__(self, boxes):
        self.data = np.asarray(boxes, dtype=np.float32).reshape(-1, 6)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return DetectionBatch(self.data[index])

    @property
    def conf(self):
        return self.data[:, 4]

    @property
    def cls(self):
        return self.data[:, 5]

    @property
    def xyxy(self):
        return self.data[:, :4]

    @property
    def xywh(self):
        result = self.data[:, :4].copy()
        result[:, :2] = (self.data[:, :2] + self.data[:, 2:4]) / 2
        result[:, 2:4] = self.data[:, 2:4] - self.data[:, :2]
        return result


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def detection_cache(video, weights, output, imgsz=640, sample_fps=6, max_frames=120,
                    reuse=False):
    import cv2

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    specification = {'video_sha256': digest(video), 'weight_sha256': digest(weights),
                     'imgsz': imgsz, 'requested_sample_fps': sample_fps,
                     'max_frames': max_frames, 'detection_confidence': .1}
    if specification['weight_sha256'] != WEIGHT_SHA256:
        raise ValueError('Unexpected YOLO26n-pose weight hash')
    meta_path, data_path = output / 'detections.json', output / 'detections.npz'
    if reuse:
        meta = json.loads(meta_path.read_text())
        if meta['specification'] != specification:
            raise ValueError('Replay detection cache does not match requested inputs')
        with np.load(data_path, allow_pickle=False) as data:
            boxes, offsets = data['boxes'], data['offsets']
        return [boxes[offsets[i]:offsets[i+1]] for i in range(len(offsets)-1)], meta
    os.environ.setdefault('YOLO_AUTOINSTALL', 'false')
    import torch
    import ultralytics
    from ultralytics import YOLO

    if not torch.cuda.is_available():
        raise RuntimeError('Replay detection requires CUDA; CPU fallback is not a GPU benchmark')
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    cap = cv2.VideoCapture(str(video))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if not cap.isOpened() or not np.isfinite(fps) or fps <= 0:
        raise ValueError('Cannot read video or source frame rate')
    stride = max(1, round(fps / sample_fps))
    model = YOLO(str(weights), task='pose')
    frames, source_frames, timings = [], [], []
    index = 0
    try:
        while len(frames) < max_frames:
            ok, image = cap.read()
            if not ok:
                break
            index += 1
            if (index - 1) % stride:
                continue
            options = dict(source=image, device=0, imgsz=imgsz, quantize=16,
                           conf=.1, verbose=False, save=False, nms=False)
            with torch.inference_mode():
                if not frames:
                    for _ in range(2):
                        model.predict(**options)
                    torch.cuda.synchronize()
                started = time.perf_counter()
                result = model.predict(**options)[0]
                boxes = result.boxes.data.detach().float().cpu().numpy()[:, :6]
                torch.cuda.synchronize()
                timings.append((time.perf_counter() - started) * 1000)
            if not np.isfinite(boxes).all():
                raise ValueError('Non-finite detection in replay cache')
            frames.append(boxes.copy())
            source_frames.append(index)
    finally:
        cap.release()
    if not frames:
        raise ValueError('Video provided no usable frames')
    sizes = [len(frame) for frame in frames]
    np.savez_compressed(data_path, boxes=np.concatenate(frames),
                        offsets=np.cumsum([0] + sizes, dtype=np.int64))
    parameter = next(model.predictor.model.backend.model.parameters())
    if parameter.device.type != 'cuda' or parameter.dtype != torch.float16:
        raise RuntimeError('Replay did not use CUDA FP16')
    meta = {'schema_version': 1, 'specification': specification,
            'ultralytics': ultralytics.__version__, 'source_fps': fps,
            'source_frame_numbers': source_frames, 'sample_fps': fps / stride,
            'frame_count': len(frames), 'detected_boxes': sum(sizes),
            'warmup_predictions': 2, 'detector_wall_ms': timings,
            'device': str(parameter.device), 'fp16': True,
            'measurement_note': 'Detector timing includes prediction and CPU box transfer; '
                                'excludes source decoding, tracking, JPEG encoding and HTTP.'}
    atomic_json(meta_path, meta)
    return frames, meta


def compare_trackers(frames, meta, output, names=TRACKERS, repeats=3, new_track_conf=.30):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    results = []
    for name in names:
        timings = []
        final_rows = []
        for _ in range(repeats):
            tracker = make_tracker(name, meta['sample_fps'], new_track_conf)
            rows = []
            for index, boxes in enumerate(frames, 1):
                detections = DetectionBatch(boxes)
                started = time.perf_counter()
                tracks = tracker.update(detections, None)
                timings.append((time.perf_counter() - started) * 1000)
                for track in tracks:
                    if not np.isfinite(track).all():
                        raise ValueError('Non-finite track output')
                    x1, y1, x2, y2, ident, confidence = track[:6]
                    rows.append([index, int(ident), float(x1), float(y1), float(x2-x1),
                                 float(y2-y1), float(confidence), -1, -1, -1])
            final_rows = rows
        path = output / (name + '.txt')
        path.write_text(''.join(','.join(str(value) for value in row) + '\n' for row in final_rows))
        histories = Counter(row[1] for row in final_rows)
        results.append({'tracker': name, 'association_median_ms': float(np.median(timings)),
                        'association_p95_ms': float(np.percentile(timings, 95)),
                        'local_ids_created': len(histories),
                        'confirmed_observations': len(final_rows),
                        'median_track_observations': float(np.median(list(histories.values())))
                        if histories else 0, 'mot_file': path.name, 'repeats': repeats})
    report = {'schema_version': 1, 'available': True, 'input_size': meta['specification']['imgsz'],
              'video_sha256': meta['specification']['video_sha256'],
              'weight_sha256': meta['specification']['weight_sha256'],
              'ultralytics': meta['ultralytics'], 'frame_count': len(frames),
              'replay_implementation_sha256': digest(Path(__file__)),
              'sample_fps': meta['sample_fps'], 'results': results,
              'detector_median_ms': float(np.median(meta['detector_wall_ms'])),
              'detector_p95_ms': float(np.percentile(meta['detector_wall_ms'], 95)),
              'identity_accuracy_measured': False, 'IDF1': None, 'HOTA': None,
              'note': 'Shared cached detections; ReID and camera-motion compensation disabled; '
                      'common new-track threshold and 1.5-second lost-track buffer. '
                      'No identity ground truth: ID counts are not ID switches or accuracy. '
                      'Association time excludes the detector and is not end-to-end FPS.',
              'created_at_unix_ms': round(time.time() * 1000)}
    atomic_json(output / 'replay-report.json', report)
    atomic_json(output / 'frame-map.json', {'sampled_to_source': meta['source_frame_numbers']})
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video', type=Path, required=True)
    parser.add_argument('--weights', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--imgsz', type=int, choices=(320, 480, 640, 960), default=640)
    parser.add_argument('--sample-fps', type=float, default=6)
    parser.add_argument('--max-frames', type=int, default=120)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--reuse-cache', action='store_true')
    parser.add_argument('--publish-report', type=Path)
    args = parser.parse_args()
    if not .5 <= args.sample_fps <= 30 or not 10 <= args.max_frames <= 2000:
        parser.error('Use 0.5–30 sample FPS and 10–2000 sampled frames')
    if not 1 <= args.repeats <= 10:
        parser.error('Use 1–10 repeats')
    frames, meta = detection_cache(args.video, args.weights, args.output, args.imgsz,
                                   args.sample_fps, args.max_frames, args.reuse_cache)
    report = compare_trackers(frames, meta, args.output, repeats=args.repeats)
    if args.publish_report:
        atomic_json(args.publish_report, report)
    print(json.dumps(report, allow_nan=False))


if __name__ == '__main__':
    main()
