"""MQTT 3.1.1 live telemetry and a strict adapter for the existing fusionmap JSON.

No inbound command subscription. Tracking packets are non-retained QoS 0 with
explicit freshness; availability uses a retained QoS 1 last will.
"""
from __future__ import annotations

import json
import logging
import math
import platform
import re
import threading
import time
import uuid
from pathlib import Path

LEGACY_TOPIC = 'wyzl/iot/service/device/fusionmap/report'


def load_mqtt_options(path=None):
    node = platform.node().split('.')[0] or 'orin'
    defaults = {'enabled': False, 'host': '127.0.0.1', 'port': 1883, 'node_id': node,
                'topic_prefix': 'multicamera/' + node, 'username': None, 'password': None,
                'tls': False, 'ca_file': None, 'interval_s': 1.0, 'advertised_host': None,
                'legacy_enabled': False, 'legacy_mac': None}
    if path and Path(path).exists():
        supplied = json.loads(Path(path).read_text())
        if not isinstance(supplied, dict) or set(supplied) - set(defaults):
            raise ValueError('Invalid MQTT option keys')
        defaults.update(supplied)
    return validate_mqtt_options(defaults)


def validate_mqtt_options(config):
    result = dict(config)
    for key in ('enabled', 'tls', 'legacy_enabled'):
        if type(result.get(key)) is not bool:
            raise ValueError('MQTT booleans must be explicit')
    if (not isinstance(result.get('host'), str) or not result['host'].strip()
            or len(result['host']) > 253 or any(c.isspace() for c in result['host'])):
        raise ValueError('Invalid MQTT broker host')
    if type(result.get('port')) is not int or not 1 <= result['port'] <= 65535:
        raise ValueError('Invalid MQTT broker port')
    for key in ('topic_prefix', 'node_id'):
        value = result.get(key)
        if (not isinstance(value, str) or len(value) > 128
                or not re.fullmatch(r'[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)*', value)):
            raise ValueError('MQTT topic prefix/node contains invalid characters')
    interval = result.get('interval_s')
    if (type(interval) not in (int, float) or not math.isfinite(interval)
            or not .5 <= interval <= 10):
        raise ValueError('Invalid MQTT reporting interval')
    for key in ('username', 'password', 'ca_file', 'advertised_host'):
        if result.get(key) is not None and not isinstance(result[key], str):
            raise ValueError('Invalid MQTT credential/TLS setting')
    if result.get('password') and not result.get('username'):
        raise ValueError('MQTT password requires username')
    mac = result.get('legacy_mac')
    if mac is not None:
        mac = mac.replace(':', '').lower()
        if not re.fullmatch('[0-9a-f]{12}', mac):
            raise ValueError('Legacy MAC must contain exactly 12 hexadecimal digits')
        result['legacy_mac'] = mac
    return result


def topics(config, camera_ids):
    prefix = config['topic_prefix']
    return {'availability': prefix + '/availability', 'status': prefix + '/status',
            'global_tracks': prefix + '/global/tracks',
            'cameras': {ident: prefix + '/cameras/' + ident + '/tracks' for ident in camera_ids},
            'legacy_report': LEGACY_TOPIC}


def finite_xy(value):
    return (isinstance(value, (list, tuple)) and len(value) == 2
            and all(type(v) in (int, float) and math.isfinite(v) for v in value))


def valid_uuid(value):
    try:
        return isinstance(value, str) and str(uuid.UUID(value)) == value.lower()
    except (ValueError, AttributeError):
        return False


def legacy_report(tracks, mac):
    """Do not invent temperature, posture, alarm, sensor identity or coordinates."""
    things, withheld = [], 0
    for track in tracks:
        temperature = track.get('temperature_c')
        if (not isinstance(mac, str) or not re.fullmatch('[0-9a-f]{12}', mac)
                or not valid_uuid(track.get('global_id')) or not track.get('observed')
                or not finite_xy(track.get('world_xy_m'))
                or track.get('position_time_verified') is not True
                or track.get('temperature_verified') is not True
                or track.get('posture_verified') is not True
                or track.get('alarm_verified') is not True
                or type(temperature) not in (int, float) or not math.isfinite(temperature)
                or type(track.get('pos')) is not int or track['pos'] not in (1, 2, 3)
                or type(track.get('alarm')) is not bool):
            withheld += 1
            continue
        things.append({'type': 1, 'track_id': track['global_id'],
                       'location': [int(round(v * 1000)) for v in track['world_xy_m']],
                       'temperature': temperature, 'mac': mac,
                       'pos': track['pos'], 'alarm': track['alarm']})
    return {'uuid': str(uuid.uuid4()), 'code': 0, 'error': '', 'data': {'things': things}}, withheld


def publications(snapshot, config, sequence, now_ms=None):
    now_ms = round(time.time() * 1000) if now_ms is None else now_ms
    mapping = topics(config, [c['id'] for c in snapshot.get('cameras', [])])
    batch = str(uuid.uuid4())

    def envelope(**fields):
        return {'schema_version': 1, 'message_id': str(uuid.uuid4()), 'batch_id': batch,
                'sequence': sequence, 'node_id': config['node_id'],
                'session_id': snapshot.get('session_id'), 'emitted_at_unix_ms': now_ms, **fields}

    # A stopped/failed runtime may contain old diagnostics; never republish its tracks.
    running = snapshot.get('state') == 'running'
    all_tracks = snapshot.get('tracks', []) if running else []
    observed = [track for track in all_tracks if track.get('observed')]
    association = {(source['camera_id'], source['epoch'], source['local_id']): track
                   for track in all_tracks for source in track.get('sources', [])}
    result = []
    for camera in snapshot.get('cameras', []):
        fresh = running and camera.get('state') == 'running'
        age = camera.get('age_s')
        fresh = bool(fresh and type(age) in (int, float) and math.isfinite(age) and 0 <= age <= 2)
        rows = []
        for local in camera.get('tracks', []) if fresh else []:
            row = {key: local.get(key) for key in ('local_id', 'epoch', 'confidence',
                   'bbox_xyxy', 'keypoints', 'capture_time_s', 'capture_uncertainty_s')}
            global_track = association.get((camera['id'], local['epoch'], local['local_id']), {})
            row['track_uuid'] = global_track.get('track_uuid')
            row['global_id'] = global_track.get('global_id')
            position_ok = (local.get('position_verified') is True
                           and camera.get('capture_time_verified') is True
                           and finite_xy(local.get('world_xy_m')))
            row['world_position_m'] = [*local['world_xy_m'], 0.0] if position_ok else None
            row['coordinate_frame'] = local.get('world_frame') if position_ok else None
            row['position_valid'] = position_ok
            row['reference_point'] = 'ankle_midpoint_projected_to_floor'
            row['keypoints_format'] = 'COCO17_x_y_confidence_pixels'
            row['bbox_format'] = 'xyxy_pixels'
            rows.append(row)
        result.append((mapping['cameras'][camera['id']], envelope(
            camera_id=camera['id'], source_state=camera.get('state'),
            sample_id=camera.get('sample_id'), tracks=rows,
            capture_time_verified=camera.get('capture_time_verified', False),
            processed_age_ms=round(age * 1000) if fresh else None,
            valid_for_ms=max(0, round((2-age) * 1000)) if fresh else 0), False))
    global_rows = []
    for track in observed:
        if not track.get('global_id'):
            continue
        row = {key: track.get(key) for key in ('track_uuid', 'global_id', 'sources', 'world_frame')}
        row['position_valid'] = (track.get('position_time_verified') is True
                                 and finite_xy(track.get('world_xy_m')))
        row['world_position_m'] = [*track['world_xy_m'], 0.0] if row['position_valid'] else None
        row['reference_point'] = 'ankle_midpoint_projected_to_floor'
        global_rows.append(row)
    result.append((mapping['global_tracks'], envelope(
        tracks=global_rows, association_state=snapshot.get('cross_camera'),
        pending_local_tracks=sum(not t.get('global_id') for t in observed), valid_for_ms=800), False))
    legacy, withheld = legacy_report(observed, config.get('legacy_mac'))
    result.append((mapping['status'], envelope(
        tracking_state=snapshot.get('state'), error=snapshot.get('error'),
        cameras=[{key: c.get(key) for key in ('id', 'state', 'age_s', 'processed_fps')}
                 for c in snapshot.get('cameras', [])],
        cross_camera=snapshot.get('cross_camera'), memory=snapshot.get('memory'),
        options=snapshot.get('options'),
        legacy={'enabled': config['legacy_enabled'], 'report_uuid': legacy['uuid'],
                'published_objects': len(legacy['data']['things']), 'withheld_objects': withheld,
                'requirements': 'verified position, temperature, posture, alarm and configured MAC'}), True))
    if config['legacy_enabled']:
        result.append((LEGACY_TOPIC, legacy, False))
    return result


class MQTTPublisher:
    def __init__(self, config, provider, camera_ids):
        self.config, self.provider = validate_mqtt_options(config), provider
        self.topic_map = topics(config, camera_ids)
        self.connected = threading.Event()
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        self.client = None
        self.thread = None
        self.state = 'disabled' if not config['enabled'] else 'connecting'
        self.last_error = None
        self.accepted_publications = 0
        self.last_publish_ms = None
        self.client_session = uuid.uuid4().hex

    def availability(self, online):
        # A last-will payload cannot know its future emission time.
        return json.dumps({'schema_version': 1, 'node_id': self.config['node_id'],
                           'client_session_id': self.client_session, 'online': online})

    def start(self):
        try:
            self._start()
        except Exception:
            logging.getLogger(__name__).exception('MQTT client initialization failed')
            with self.lock:
                self.state, self.last_error = 'failed', 'MQTT client could not be initialized'

    def _start(self):
        if not self.config['enabled']:
            return
        import paho.mqtt.client as mqtt

        client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2,
                             client_id='mc-' + self.client_session[:18],
                             clean_session=True, protocol=mqtt.MQTTv311)
        self.client = client
        client.max_queued_messages_set(8)
        client.max_inflight_messages_set(2)
        client.reconnect_delay_set(1, 30)
        if self.config.get('username'):
            client.username_pw_set(self.config['username'], self.config.get('password'))
        if self.config['tls']:
            client.tls_set(ca_certs=self.config.get('ca_file'))
        client.will_set(self.topic_map['availability'], self.availability(False), qos=1, retain=True)

        def on_connect(client, userdata, flags, reason_code, properties):
            if reason_code != 0:
                with self.lock:
                    self.state, self.last_error = 'disconnected', 'MQTT broker rejected connection'
                return
            with self.lock:
                self.state, self.last_error = 'connected', None
            self.connected.set()
            client.publish(self.topic_map['availability'], self.availability(True), qos=1, retain=True)

        def on_disconnect(client, userdata, flags, reason_code, properties):
            self.connected.clear()
            with self.lock:
                self.state = 'stopped' if self.stop_event.is_set() else 'disconnected'

        client.on_connect, client.on_disconnect = on_connect, on_disconnect
        client.connect_async(self.config['host'], self.config['port'], keepalive=15)
        client.loop_start()
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def run(self):
        sequence = 0
        while not self.stop_event.is_set():
            if self.connected.is_set():
                try:
                    # Never hold the MQTT lock while obtaining the tracking snapshot.
                    snapshot = self.provider()
                    sequence += 1
                    for topic, payload, retain in publications(snapshot, self.config, sequence):
                        encoded = json.dumps(payload, ensure_ascii=False, allow_nan=False,
                                             separators=(',', ':'))
                        if len(encoded.encode()) > 1024 * 1024:
                            raise ValueError('MQTT payload exceeds the live telemetry limit')
                        info = self.client.publish(topic, encoded, qos=0, retain=retain)
                        if info.rc == 0:
                            with self.lock:
                                self.accepted_publications += 1
                                self.last_publish_ms = round(time.time() * 1000)
                except Exception:
                    with self.lock:
                        self.last_error = 'MQTT telemetry could not be published'
            self.stop_event.wait(self.config['interval_s'])

    def status(self):
        with self.lock:
            return {'enabled': self.config['enabled'], 'state': self.state,
                    'connected': self.connected.is_set(), 'host': self.config['host'],
                    'port': self.config['port'], 'tls': self.config['tls'],
                    'subscriber_host': self.config.get('advertised_host') or self.config['host'],
                    'authentication': bool(self.config.get('username')),
                    'topics': self.topic_map, 'legacy_enabled': self.config['legacy_enabled'],
                    'accepted_publications': self.accepted_publications,
                    'last_publish_unix_ms': self.last_publish_ms, 'last_error': self.last_error,
                    'live_qos': 0, 'live_retained': False}

    def close(self):
        self.stop_event.set()
        if self.thread:
            self.thread.join(2)
        if self.client:
            if self.connected.is_set():
                try:
                    self.client.publish(self.topic_map['availability'], self.availability(False),
                                        qos=1, retain=True).wait_for_publish(timeout=1)
                except (RuntimeError, ValueError):
                    pass
            self.client.disconnect()
            self.client.loop_stop()
