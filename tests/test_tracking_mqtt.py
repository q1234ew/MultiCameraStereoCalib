"""MQTT contracts: freshness, provenance, privacy and strict legacy serialization."""
import copy
import json

import pytest

from scripts.jetson.tracking_mqtt import (
    LEGACY_TOPIC,
    MQTTPublisher,
    legacy_report,
    load_mqtt_options,
    publications,
    validate_mqtt_options,
)


def config():
    result = load_mqtt_options()
    result.update(node_id='orin', topic_prefix='multicamera/orin', legacy_enabled=True)
    return result


def snapshot():
    return {'state': 'running', 'session_id': 'session', 'control_csrf': 'secret-token',
            'cameras': [{'id': 'device_1', 'state': 'running', 'age_s': .2, 'sample_id': 1,
                         'capture_time_verified': False,
                         'tracks': [{'local_id': 1, 'epoch': 'e', 'confidence': .9,
                                     'bbox_xyxy': [0, 0, 20, 40], 'keypoints': [[0, 0, .9]]*17,
                                     'world_xy_m': [1, 2], 'position_verified': True}]}],
            'tracks': [{'track_uuid': 'local-uuid', 'global_id': None, 'observed': True,
                        'sources': [{'camera_id': 'device_1', 'epoch': 'e', 'local_id': 1}]}],
            'cross_camera': {'state': 'blocked'}}


def test_modern_local_topic_has_no_unverified_world_position_or_false_global_id():
    data = dict((topic, payload) for topic, payload, _ in publications(snapshot(), config(), 1, 1000))
    local = data['multicamera/orin/cameras/device_1/tracks']
    assert local['valid_for_ms'] == 1800
    assert local['tracks'][0]['world_position_m'] is None
    assert not local['tracks'][0]['position_valid']
    assert local['tracks'][0]['global_id'] is None
    assert local['tracks'][0]['keypoints_format'] == 'COCO17_x_y_confidence_pixels'
    assert data['multicamera/orin/global/tracks']['tracks'] == []
    assert data[LEGACY_TOPIC]['data']['things'] == []
    assert 'secret-token' not in json.dumps(data)


@pytest.mark.parametrize('state,age', [('running', 2.1), ('paused', .1), ('failed', .1)])
def test_stale_or_stopped_sources_publish_empty_tracks(state, age):
    snap = snapshot()
    snap['state'] = state
    snap['cameras'][0]['age_s'] = age
    data = dict((topic, payload) for topic, payload, _ in publications(snap, config(), 1))
    assert data['multicamera/orin/cameras/device_1/tracks']['tracks'] == []
    assert data['multicamera/orin/cameras/device_1/tracks']['valid_for_ms'] == 0


def test_legacy_requires_all_verified_fields_and_converts_meters_to_integer_mm():
    person = {'global_id': '11111111-1111-4111-8111-111111111111', 'observed': True, 'world_xy_m': [-.85, .62],
              'position_time_verified': True, 'temperature_c': 32.5, 'temperature_verified': True,
              'posture_verified': True, 'alarm_verified': True, 'pos': 3, 'alarm': False}
    report, count = legacy_report([person], '744dbd2bd088')
    assert count == 0
    assert set(report) == {'uuid', 'code', 'error', 'data'}
    assert report['code'] == 0 and report['error'] == ''
    assert report['data']['things'] == [{'type': 1, 'track_id': '11111111-1111-4111-8111-111111111111',
        'location': [-850, 620], 'temperature': 32.5, 'mac': '744dbd2bd088', 'pos': 3, 'alarm': False}]
    for key in ('temperature_verified', 'posture_verified', 'alarm_verified',
                'position_time_verified'):
        invalid = copy.deepcopy(person)
        invalid[key] = False
        assert legacy_report([invalid], '744dbd2bd088')[0]['data']['things'] == []
    assert legacy_report([person], None)[0]['data']['things'] == []


def test_live_packets_are_not_retained_and_never_contain_credentials():
    options = config()
    options.update(username='publisher', password='super-secret')
    publisher = MQTTPublisher(options, snapshot, ['device_1'])
    assert 'super-secret' not in json.dumps(publisher.status())
    for topic, data, retain in publications(snapshot(), options, 1):
        if topic != options['topic_prefix']+'/status':
            assert not retain
    assert json.loads(publisher.availability(False))['online'] is False
    assert 'emitted_at_unix_ms' not in json.loads(publisher.availability(False))


@pytest.mark.parametrize('patch', [{'topic_prefix': 'bad/#'}, {'port': 0}, {'enabled': 1},
                                  {'legacy_mac': 'unknown'}, {'password': 'x'}])
def test_invalid_mqtt_configuration_is_rejected(patch):
    options = config()
    options.update(patch)
    with pytest.raises(ValueError):
        validate_mqtt_options(options)


@pytest.mark.parametrize('verified', [True, False, None])
def test_pending_peer_does_not_publish_a_verified_global_position(verified):
    snap = snapshot()
    snap['tracks'][0].update(global_id='11111111-1111-4111-8111-111111111111',
                             world_xy_m=[1, 2], position_time_verified=verified)
    data = dict((topic, payload) for topic, payload, _ in publications(snap, config(), 1))
    row = data['multicamera/orin/global/tracks']['tracks'][0]
    assert row['global_id']
    assert row['position_valid'] is (verified is True)
    assert row['world_position_m'] == ([1, 2, 0] if verified is True else None)
