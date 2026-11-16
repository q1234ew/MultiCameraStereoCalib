"""Isolated YOLO26n-pose + per-camera ByteTrack service; localhost HTTP only.

Run in .pose-venv. A shared GPU model consumes bounded latest-frame slots;
no torch imports occur in the existing preview process.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import signal
import statistics
import threading
import time
import urllib.request
import uuid
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .tracking_clock import SourceClock
from .person_association import GlobalAssociator
from .tracking_pose import person_pose_mask
from .tracking_site import floor_view, load_site, project_ankles
from .tracking_options import PIPELINE_KEYS, OptionStore, RevisionConflict, make_tracker

LOG = logging.getLogger(__name__)
WEIGHT_SHA256 = 'eb3bb8268828aeaf515cec23a4bfafd793944a86fe9af94ba7823609c14522a9'


class TrackingRuntime:
    def __init__(self, config, weights, site_path=None, fps=3, model=None, options_path=None,
                 replay_path=None):
        self.devices = config['devices']
        self.weights, self.site_path = Path(weights), site_path
        self.options = OptionStore(options_path, fps)
        self.replay_path = Path(replay_path or "data/tracking/replay/latest.json")
        self.fps = self.options.options["camera_fps"]
        self.generation = 0
        self.engine_ready = False
        self.lock = threading.RLock()
        self.stop = threading.Event()
        self.manager = GlobalAssociator()
        self.cameras = {d['id']: {'id': d['id'], 'state': 'connecting', 'tracks': [],
                                'epoch': str(uuid.uuid4()), 'sequence': 0,
                                'image_size': None, 'last_seen_mono': None,
                                'processed_mono': 0.0, 'frame': None, 'jpeg': None, 'annotated': None,
                                'error': None, 'failures': 0, 'processed_count': 0,
                                'dropped_frames': 0, 'samples': deque(maxlen=90),
                                'timings_ms': deque(maxlen=90),
                                'receive_to_result_ms': None, 'model_speed_ms': None} for d in self.devices}
        self.encode_locks = {device['id']: threading.Lock() for device in self.devices}
        self.model = model
        self.state, self.error = 'loading', None
        self.session_id = str(uuid.uuid4())
        self.site, self.site_error = load_site(site_path, self.devices)
        self.views_site, self.views, self.geometry_revision = None, [], None
        self.started = time.monotonic()
        self.clock_unix_anchor = time.time() - self.started
        self.inference_ms = None
        self.batch_inference_ms = None
        self.batch_size = 0
        self.gpu_info = None
        self.source_threads = []
        self.publisher = None

    def replay_report(self):
        try:
            if self.replay_path.stat().st_size > 1024 * 1024:
                raise ValueError('Replay report is too large')
            data = json.loads(self.replay_path.read_text())
            if data.get('schema_version') != 1 or not isinstance(data.get('results'), list):
                raise ValueError('Invalid replay report')
            return data
        except (OSError, ValueError, AttributeError):
            return {'available': False, 'note': '暂无已完成的回放评测'}

    def configure(self, patch, revision):
        with self.lock:
            if self.state == 'failed':
                raise ValueError('推理服务故障，请先重启服务')
            changed = self.options.update(patch, revision)
            self.fps = self.options.options['camera_fps']
            if changed & PIPELINE_KEYS:
                self.generation += 1
                self.manager = GlobalAssociator()
                self.site, self.site_error = load_site(self.site_path, self.devices)
                self.session_id = str(uuid.uuid4())
                for camera in self.cameras.values():
                    camera.update(epoch=str(uuid.uuid4()), frame=None, jpeg=None, annotated=None, tracks=[],
                                  processed_mono=0.0, processed_sequence=None, last_seen_mono=None,
                                  processed_count=0, dropped_frames=0)
                    camera['samples'].clear()
                    camera['timings_ms'].clear()
            if not self.options.options['enabled']:
                self.state = 'paused'
            else:
                self.state = 'running' if self.engine_ready else 'loading'
            return {'options': dict(self.options.options), 'revision': self.options.revision,
                    'session_reset': bool(changed & PIPELINE_KEYS), 'state': self.state,
                    'session_id': self.session_id}

    def start(self):
        # OpenCV's Python bootstrap can deadlock when several cold threads import
        # its generated submodules concurrently on Python 3.8. Prepare native
        # dependencies once, before starting any camera/inference threads.
        import cv2
        import torch
        from ultralytics import YOLO

        cv2.setNumThreads(1)
        torch.set_num_threads(2)
        self.model_factory = YOLO
        for device in self.devices:
            thread = threading.Thread(target=self.capture_loop, args=(device,), daemon=True)
            thread.start()
            self.source_threads.append(thread)
        thread = threading.Thread(target=self.inference_loop, daemon=True)
        thread.start()
        self.inference_thread = thread

    def capture_loop(self, device):
        import cv2
        import numpy as np

        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        url = f"http://{device['host']}:{device['raw_ports']['left']}/snapshot"
        camera = self.cameras[device['id']]
        source_clock = SourceClock()
        while not self.stop.is_set():
            with self.lock:
                options = dict(self.options.options)
                generation = self.generation
            if not options['enabled']:
                self.stop.wait(.2)
                continue
            started = time.monotonic()
            try:
                with opener.open(url, timeout=1.5) as response:
                    headers_mono = time.monotonic()
                    clock_sample = source_clock.observe(response.headers, started, headers_mono)
                    body = response.read(4 * 1024 * 1024 + 1)
                    if len(body) > 4 * 1024 * 1024 or response.status != 200:
                        raise ValueError('Invalid snapshot response')
                    stamp = response.headers.get('X-Capture-Unix-Ms')
                    uncertainty = response.headers.get('X-Capture-Uncertainty-Ms')
                image = cv2.imdecode(np.frombuffer(body, np.uint8), cv2.IMREAD_COLOR)
                if image is None or list(image.shape[1::-1]) != device['resolution']['left']:
                    raise ValueError('Snapshot resolution differs from camera calibration')
                ended = time.monotonic()
                # Receipt time is exposed for freshness, never relabeled as capture time.
                source_clock_verified = device.get('capture_clock_verified', False)
                capture_s = float(stamp) / 1000 if stamp and source_clock_verified else None
                uncertainty_s = (float(uncertainty) / 1000
                                 if uncertainty and source_clock_verified else None)
                if clock_sample.capture_verified:
                    capture_s = clock_sample.mapped_mono_s + self.clock_unix_anchor
                    uncertainty_s = source_clock.capture_budget(clock_sample)
                if capture_s is not None and (not np.isfinite(capture_s)
                        or abs((ended + self.clock_unix_anchor if clock_sample.capture_verified
                                else time.time()) - capture_s) > 2):
                    raise ValueError('Source capture time is stale or invalid')
                with self.lock:
                    if generation != self.generation or not self.options.options['enabled']:
                        continue
                    camera['source_clock'] = clock_sample.diagnostic()
                    if clock_sample.source_key is not None:
                        previous_source = camera.get('source_epoch')
                        if previous_source != clock_sample.source_key:
                            camera['epoch'] = str(uuid.uuid4())
                            camera['source_epoch'] = clock_sample.source_key
                        elif camera.get('source_sequence') == clock_sample.sequence:
                            continue  # Repeated HTTP snapshot is not a new source frame.
                        camera['source_sequence'] = clock_sample.sequence
                    if (camera['last_seen_mono'] is not None
                            and ended - camera['last_seen_mono'] > 2):
                        camera['epoch'] = str(uuid.uuid4())
                    camera['sequence'] += 1
                    camera['last_seen_mono'] = ended
                    camera['frame'] = (camera['sequence'], image, ended, capture_s,
                                       uncertainty_s, camera['epoch'])
                    camera['capture_clock_ok'] = bool(capture_s is not None and uncertainty_s is not None
                        and np.isfinite(uncertainty_s) and 0 <= uncertainty_s <= 0.1)
                    camera['image_size'] = list(image.shape[1::-1])
                    camera['error'] = None
                    camera['failures'] = 0
            except Exception as exc:
                with self.lock:
                    camera['error'] = str(exc)
                    camera['failures'] += 1
            # Offline cameras back off instead of repeatedly exhausting requests.
            delay = max(1 / options['camera_fps'],
                        min(4, .5 * 2**min(camera['failures'], 3)) if camera['failures'] else 0)
            self.stop.wait(max(0.01, delay - (time.monotonic() - started)))

    def inference_loop(self):
        try:
            os.environ.setdefault('YOLO_AUTOINSTALL', 'false')
            import cv2
            import numpy as np
            import torch

            if not torch.cuda.is_available():
                raise RuntimeError('CUDA unavailable; refusing CPU fallback')
            if hashlib.sha256(self.weights.read_bytes()).hexdigest() != WEIGHT_SHA256:
                raise ValueError('YOLO26n-pose weight hash mismatch')
            model = self.model or self.model_factory(str(self.weights), task='pose')
            trackers, epochs = {}, {}
            with self.lock:
                self.engine_ready = True
                self.state = 'running' if self.options.options['enabled'] else 'paused'
            last_site_check = 0
            while not self.stop.is_set():
                with self.lock:
                    enabled = self.options.options['enabled']
                if not enabled:
                    self.stop.wait(.1)
                    continue
                with self.lock:
                    generation = self.generation
                    options = dict(self.options.options)
                    revision = self.options.revision
                    site = self.site
                    jobs = [(device, self.cameras[device['id']]['frame'])
                            for device in self.devices]
                    jobs = [(device, frame) for device, frame in jobs
                            if options['enabled'] and frame is not None
                            and frame[0] != self.cameras[device['id']].get('processed_sequence')
                            and time.monotonic() - frame[2] <= 1.5]
                progressed = bool(jobs)
                before = time.monotonic()
                results = []
                if jobs:
                    with torch.inference_mode():
                        results = model.predict(source=[frame[1] for _, frame in jobs], device=0,
                                               imgsz=options['imgsz'], quantize=16, conf=0.10,
                                               verbose=False, save=False, nms=False)
                    parameter = next(model.predictor.model.backend.model.parameters())
                    if parameter.device.type != 'cuda' or parameter.dtype != torch.float16:
                        raise RuntimeError('Inference did not remain CUDA FP16')
                    self.batch_inference_ms = round((time.monotonic() - before) * 1000, 2)
                    self.batch_size = len(jobs)
                    if len(results) != len(jobs):
                        raise RuntimeError('Batch results differ from camera inputs')
                for (device, frame), result in zip(jobs, results):
                    camera = self.cameras[device['id']]
                    sequence, image, received, capture_s, uncertainty_s, epoch = frame
                    if epochs.get(device['id']) != epoch:
                        trackers[device['id']] = make_tracker(
                            options['tracker'], options['camera_fps'], options['new_track_conf'])
                        epochs[device['id']] = epoch
                    with self.lock:
                        if generation != self.generation or not self.options.options['enabled']:
                            continue
                    detections = result.boxes.cpu().numpy()
                    keypoints = result.keypoints.data.detach().float().cpu().numpy()
                    keep = person_pose_mask(detections.xyxy, keypoints, image.shape[1::-1],
                                            detections.conf)
                    rejected = int(len(keep) - keep.sum())
                    detections, keypoints = detections[keep], keypoints[keep]
                    rows = trackers[device['id']].update(detections, image)
                    observations = []
                    annotated = image.copy()
                    for row in rows:
                        if not np.isfinite(row).all():
                            continue
                        index = int(row[-1])
                        if not 0 <= index < len(keypoints):
                            continue
                        pose = keypoints[index]
                        if not np.isfinite(pose).all():
                            continue
                        xy = None
                        if site and device['id'] in site['cameras']:
                            xy = project_ankles(site['cameras'][device['id']], pose,
                                                image.shape[1::-1])
                        obs = {'camera_id': device['id'], 'epoch': epoch,
                               'local_id': int(row[4]), 'bbox_xyxy': row[:4].tolist(),
                               'confidence': float(row[5]), 'keypoints': pose.tolist(),
                               'sample_id': str(capture_s) if capture_s is not None else sequence,
                               'world_xy_m': xy,
                               'world_frame': site['world_frame'] if xy is not None else None,
                               'position_verified': xy is not None,
                               'capture_time_s': capture_s,
                               'capture_uncertainty_s': uncertainty_s}
                        observations.append(obs)
                        x1, y1, x2, y2 = map(int, row[:4])
                        if options['draw_boxes']:
                            cv2.rectangle(annotated, (x1, y1), (x2, y2), (80, 220, 180), 2)
                        if options['draw_boxes']:
                            cv2.putText(annotated, f"{device['id']} L-{int(row[4])}",
                                    (x1, max(18, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.65, (80, 220, 180), 2)
                        for a, b in [(5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11),
                                     (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)]:
                            if options['draw_skeleton'] and min(pose[a, 2], pose[b, 2]) > 0.4:
                                cv2.line(annotated, tuple(pose[a, :2].astype(int)),
                                         tuple(pose[b, :2].astype(int)), (80, 220, 180), 2)
                    with self.lock:
                        # A reconnect while inference runs invalidates that old epoch immediately.
                        if camera['epoch'] != epoch or generation != self.generation:
                            continue
                        self.manager.update(observations, time.monotonic(), device['id'])
                        done = time.monotonic()
                        previous = camera.get('processed_sequence')
                        if previous is not None:
                            camera['dropped_frames'] += max(0, sequence - previous - 1)
                        camera['processed_count'] += 1
                        camera['samples'].append(done)
                        camera['timings_ms'].append((done - before) * 1000)
                        camera['receive_to_result_ms'] = round((done - received) * 1000, 2)
                        camera.update(model_speed_ms={k: round(float(v), 2) for k, v in result.speed.items()},
                                      rejected_pose_count=rejected,
                                      render_revision=revision, tracks=observations, processed_sequence=sequence,
                                      processed_mono=time.monotonic(), state='running',
                                      jpeg=None, annotated=annotated, received_mono=received)
                        self.inference_ms = round((time.monotonic() - before) * 1000, 2)
                        self.gpu_info = {'device': str(parameter.device), 'fp16': True,
                                         'allocated_mib': round(torch.cuda.memory_allocated() / 2**20, 1),
                                         'reserved_mib': round(torch.cuda.memory_reserved() / 2**20, 1)}
                if time.monotonic() - last_site_check > 3:
                    site, error = load_site(self.site_path, self.devices)
                    with self.lock:
                        if site != self.site:
                            self.manager = GlobalAssociator()
                        self.site, self.site_error = site, error
                    last_site_check = time.monotonic()
                if not progressed:
                    self.stop.wait(0.02)
        except Exception as exc:
            LOG.exception('Tracking inference stopped')
            with self.lock:
                self.state, self.error = 'failed', str(exc)
                for camera in self.cameras.values():
                    camera.update(jpeg=None, tracks=[], state='failed')

    def snapshot(self):
        now = time.monotonic()
        memory_info = None
        try:
            import psutil

            memory = psutil.virtual_memory()
            memory_info = {
                'system_used_gib': round((memory.total - memory.available) / 2**30, 2),
                'system_total_gib': round(memory.total / 2**30, 2),
                'process_rss_mib': round(psutil.Process().memory_info().rss / 2**20, 1),
            }
        except ImportError:
            pass
        with self.lock:
            cameras = []
            for camera in self.cameras.values():
                age = now - camera['processed_mono'] if camera['processed_mono'] else None
                fresh = bool(self.state == 'running' and age is not None and age <= 2)
                cameras.append({'id': camera['id'], 'epoch': camera['epoch'],
                                'state': 'running' if fresh else 'unavailable',
                                'age_s': round(age, 3) if age is not None else None,
                                'sample_id': camera.get('processed_sequence'),
                                'image_size': camera['image_size'],
                                'tracks': camera['tracks'] if fresh else [],
                                'rejected_pose_count': camera.get('rejected_pose_count', 0) if fresh else 0,
                                'error': camera['error'],
                                'capture_time_verified': bool(fresh and camera.get('capture_clock_ok')),
                                'source_clock': dict(camera.get('source_clock') or {},
                                    fresh=fresh and camera['failures'] == 0),
                                'render_revision': camera.get('render_revision'),
                                'processed_count': camera['processed_count'],
                                'dropped_frames': camera['dropped_frames'],
                                'processed_fps': self.camera_rate(camera, now),
                                'processing_median_ms': round(statistics.median(
                                    camera['timings_ms']), 2) if camera['timings_ms'] else None,
                                'receive_to_result_ms': camera['receive_to_result_ms'] if fresh else None,
                                'model_speed_ms': camera.get('model_speed_ms') if fresh else None})
            tracks = self.manager.snapshot(now) if self.state == 'running' else []
            if self.views_site != self.site:
                self.geometry_revision = (hashlib.sha256(json.dumps(
                    self.site, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
                    if self.site else None)
                self.views = []
                if self.site:
                    for camera_id, calibration in self.site['cameras'].items():
                        view = floor_view(calibration)
                        if view:
                            self.views.append(dict(view, camera_id=camera_id,
                                                   world_frame=self.site['world_frame']))
                self.views_site = self.site
            participants = list(self.site['cameras']) if self.site else []
            clocks_ready = bool(len(participants) >= 2 and all(
                c['capture_time_verified'] for c in cameras if c['id'] in participants))
            return {'schema_version': 1, 'session_id': self.session_id, 'state': self.state,
                    'error': self.error, 'model': 'YOLO26n-pose',
                    'local_tracker': self.options.options['tracker'],
                    'options': dict(self.options.options),
                    'options_revision': self.options.revision,
                    'device': self.gpu_info, 'inference_ms': self.inference_ms,
                    'batch_inference_ms': self.batch_inference_ms, 'batch_size': self.batch_size,
                    'configured_camera_fps': self.fps, 'cameras': cameras,
                    'memory': memory_info,
                    'mqtt': self.publisher.status() if self.publisher else {'enabled': False, 'state': 'disabled'},
                    'tracks': tracks, 'events': list(self.manager.events),
                    'cross_camera': {'state': 'ready' if self.site and clocks_ready else 'blocked',
                        'geometry_ready': self.site is not None,
                        'camera_views': self.views,
                        'geometry_revision': self.geometry_revision,
                        'participating_camera_ids': participants,
                        'excluded_camera_ids': [c['id'] for c in cameras
                                                if c['id'] not in participants],
                        'reason': self.site_error or (None if clocks_ready
                                                    else '源帧采集时间及其误差界限尚未核验'),
                        'world_frame': self.site['world_frame'] if self.site else None},
                    'updated_at_unix_ms': round(time.time() * 1000),
                    'measurement_note': 'Snapshot receipt time is not exposure time; '
                                        'no cross-camera merge without verified geometry and clock.'}

    @staticmethod
    def camera_rate(camera, now):
        samples = [stamp for stamp in camera['samples'] if now - stamp < 10]
        if len(samples) < 2 or samples[-1] == samples[0]:
            return None
        return round((len(samples) - 1) / (samples[-1] - samples[0]), 2)

    def jpeg(self, camera_id):
        """Encode exact inference frames on demand, outside the inference/GPU loop."""
        import cv2

        encode_lock = self.encode_locks.get(camera_id)
        if encode_lock is None or not encode_lock.acquire(timeout=1):
            return None
        try:
            with self.lock:
                camera = self.cameras[camera_id]
                processed = camera['processed_mono']
                if (self.state != 'running' or not processed
                        or time.monotonic() - processed > 2):
                    return None
                if camera['jpeg']:
                    return camera['jpeg']
                image = camera.get('annotated')
                if image is None:
                    return None
                generation, epoch = self.generation, camera['epoch']
                sequence = camera.get('processed_sequence')
                quality = self.options.options['jpeg_quality']
            ok, encoded = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, quality])
            with self.lock:
                if (not ok or self.state != 'running' or generation != self.generation
                        or camera['epoch'] != epoch or time.monotonic() - processed > 2):
                    return None
                body = encoded.tobytes()
                if camera.get('processed_sequence') == sequence:
                    camera['jpeg'] = body
                return body
        finally:
            encode_lock.release()


def make_handler(runtime):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path in ('/state', '/health', '/replay-report'):
                data = runtime.replay_report() if self.path == '/replay-report' else runtime.snapshot()
                body = json.dumps(data, allow_nan=False).encode()
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
            elif self.path.startswith('/frame/'):
                body = runtime.jpeg(self.path[len('/frame/'):])
                if not body:
                    self.send_error(503, 'No fresh tracked frame')
                    return
                self.send_response(200)
                self.send_header('Content-Type', 'image/jpeg')
            else:
                self.send_error(404)
                return
            self.send_header('Cache-Control', 'no-store')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            if self.path != '/control':
                self.send_error(404)
                return
            status = 200
            try:
                length = int(self.headers.get('Content-Length', '0'))
                if not 0 < length <= 4096 or self.headers.get_content_type() != 'application/json':
                    raise ValueError('Expected a bounded JSON control request')
                data = json.loads(self.rfile.read(length))
                if not isinstance(data, dict) or set(data) != {'patch', 'revision'}:
                    raise ValueError('Invalid control request')
                response = runtime.configure(data['patch'], data['revision'])
            except RevisionConflict as exc:
                status, response = 409, {'error': str(exc)}
            except (ValueError, TypeError) as exc:
                status, response = 400, {'error': str(exc)}
            except OSError:
                status, response = 503, {'error': 'Cannot persist tracking settings'}
            body = json.dumps(response, allow_nan=False).encode()
            self.send_response(status)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass
    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('packaging/jetson/devices.json'))
    parser.add_argument('--weights', type=Path, required=True)
    parser.add_argument('--site', type=Path, default=Path('data/tracking/site.json'))
    parser.add_argument('--port', type=int, default=8093)
    parser.add_argument('--camera-fps', type=float, default=3)
    parser.add_argument('--settings', type=Path, default=Path('data/tracking/runtime.json'))
    parser.add_argument('--mqtt', type=Path, default=Path('data/tracking/mqtt.json'))
    args = parser.parse_args()
    if not 0.5 <= args.camera_fps <= 25:
        parser.error('camera-fps must be between 0.5 and 25')
    logging.basicConfig(level=logging.INFO)
    runtime = TrackingRuntime(json.loads(args.config.read_text()), args.weights,
                              args.site, args.camera_fps, options_path=args.settings)
    server = ThreadingHTTPServer(('127.0.0.1', args.port), make_handler(runtime))
    server.daemon_threads = True
    signal.signal(signal.SIGTERM, lambda *_: (runtime.stop.set(),
                  threading.Thread(target=server.shutdown, daemon=True).start()))
    from .tracking_mqtt import MQTTPublisher, load_mqtt_options

    mqtt_error = False
    try:
        mqtt_options = load_mqtt_options(args.mqtt)
    except (OSError, ValueError):
        mqtt_options, mqtt_error = load_mqtt_options(), True
    runtime.publisher = MQTTPublisher(mqtt_options, runtime.snapshot,
                                      [d['id'] for d in runtime.devices])
    if mqtt_error:
        runtime.publisher.state, runtime.publisher.last_error = 'failed', 'MQTT configuration is invalid'
    runtime.start()
    runtime.publisher.start()
    try:
        server.serve_forever()
    finally:
        runtime.stop.set()
        runtime.publisher.close()
        server.server_close()


if __name__ == '__main__':
    main()
