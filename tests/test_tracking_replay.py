"""Replay cache geometry and exported MOT data do not claim identity accuracy."""
import json

import numpy as np
import pytest

from scripts.jetson.tracking_replay import DetectionBatch, compare_trackers, detection_cache, digest


def test_detection_batch_matches_box_subset_and_empty_contract():
    batch = DetectionBatch([[10, 20, 30, 60, .9, 0], [100, 200, 130, 240, .2, 0]])
    assert batch.xywh.tolist() == [[20, 40, 20, 40], [115, 220, 30, 40]]
    assert len(batch[batch.conf > .5]) == 1
    assert batch[batch.conf < .1].xywh.shape == (0, 4)
    assert batch.xyxy[0].tolist() == [10, 20, 30, 60]


def test_replay_exports_sampled_mot_rows_and_never_invents_id_scores(tmp_path, monkeypatch):
    class Tracker:
        def update(self, detections, image):
            if not len(detections):
                return np.empty((0, 8))
            return np.array([[10, 20, 30, 60, 7, .9, 0, 0]])

    monkeypatch.setattr('scripts.jetson.tracking_replay.make_tracker', lambda *args: Tracker())
    frames = [np.array([[10, 20, 30, 60, .9, 0]]), np.empty((0, 6))]
    meta = {'sample_fps': 3, 'source_frame_numbers': [1, 11], 'ultralytics': 'fixture',
            'detector_wall_ms': [30, 31],
            'specification': {'imgsz': 640, 'video_sha256': 'video', 'weight_sha256': 'weights'}}
    report = compare_trackers(frames, meta, tmp_path, names=['bytetrack'], repeats=2)
    assert report['identity_accuracy_measured'] is False
    assert report['IDF1'] is None and report['HOTA'] is None
    assert report['results'][0]['local_ids_created'] == 1
    row = (tmp_path / 'bytetrack.txt').read_text().strip().split(',')
    assert row[:2] == ['1', '7']
    assert list(map(float, row[2:6])) == [10, 20, 20, 40]
    assert json.loads((tmp_path / 'frame-map.json').read_text())['sampled_to_source'] == [1, 11]


def test_replay_rejects_cache_from_another_model_input(tmp_path, monkeypatch):
    video, weights = tmp_path / 'video.mp4', tmp_path / 'weights.pt'
    video.write_bytes(b'fixture')
    weights.write_bytes(b'weights')
    monkeypatch.setattr('scripts.jetson.tracking_replay.WEIGHT_SHA256', digest(weights))
    output = tmp_path / 'cache'
    output.mkdir()
    (output / 'detections.json').write_text(json.dumps({'specification': {'imgsz': 320}}))
    with pytest.raises(ValueError, match='does not match'):
        detection_cache(video, weights, output, imgsz=640, reuse=True)
