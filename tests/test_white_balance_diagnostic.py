"""The evidence probe must restore AWB and never write to its reference camera."""

from copy import deepcopy
from io import BytesIO
import json

import cv2
import numpy as np
import pytest

from scripts.diagnose_white_balance import experiment, image_metrics


class ProbeClient:
    MAX_SNAPSHOT_BYTES = 1024 * 1024

    def __init__(self, break_manual=False):
        self.http = self
        self.modes = {'http://left/': 1, 'http://right/': 1}
        self.writes = []
        self.break_manual = break_manual

    def config(self, base, payload=None):
        if payload:
            self.writes.append((base, deepcopy(payload)))
            self.modes[base] = payload['white_balance_automatic']
        return {'ok': True, 'controls': {
            'white_balance_automatic': {'value': self.modes[base]},
            'white_balance_temperature': {'value': 4254},
        }}

    def open(self, url, timeout):
        base = url.removesuffix('snapshot')
        if self.break_manual and self.modes[base] == 0:
            return BytesIO(b'broken JPEG')
        image = np.full((64, 64, 3), 120, np.uint8)
        if self.modes[base] == 0:
            image[:, :, 1] = 200
        return BytesIO(cv2.imencode('.jpg', image)[1].tobytes())


def test_probe_defaults_to_read_only_and_preserves_jpegs(tmp_path):
    client = ProbeClient()
    result = experiment('http://left/stream', 'http://right/', tmp_path / 'readonly',
                        client=client, sleep=lambda _: None)
    assert not client.writes
    assert len(result['samples']) == 6
    assert len(list((tmp_path / 'readonly').glob('*.jpg'))) == 6
    assert not result['control_change_authorized']


def test_probe_switches_only_left_and_verifies_restoration(tmp_path):
    client = ProbeClient()
    result = experiment('http://left/', 'http://right/', tmp_path / 'active', change_left=True,
                        client=client, sleep=lambda _: None)
    assert client.writes == [('http://left/', {'white_balance_automatic': 0}),
                             ('http://left/', {'white_balance_automatic': 1})]
    assert result['restoration']['verified']
    assert result['summary']['left']['manual']['max_abs_change'] > .3
    assert result['summary']['right']['manual']['max_abs_change'] == 0
    assert client.modes == {'http://left/': 1, 'http://right/': 1}


def test_probe_restores_even_when_manual_snapshot_fails(tmp_path):
    client = ProbeClient(break_manual=True)
    directory = tmp_path / 'failure'
    with pytest.raises(ValueError, match='Invalid JPEG'):
        experiment('http://left/', 'http://right/', directory, change_left=True,
                   client=client, sleep=lambda _: None)
    assert client.modes['http://left/'] == 1
    report = json.loads((directory / 'report.json').read_text())
    assert report['restoration']['verified'] and 'error' in report


def test_probe_refuses_manual_start_and_existing_output(tmp_path):
    client = ProbeClient()
    client.modes['http://left/'] = 0
    with pytest.raises(ValueError, match='Left AWB is not automatic'):
        experiment('http://left/', 'http://right/', tmp_path / 'manual', change_left=True,
                   client=client, sleep=lambda _: None)
    assert not client.writes
    with pytest.raises(FileExistsError):
        experiment('http://left/', 'http://right/', tmp_path / 'manual', client=client)


def test_metrics_reject_clipped_frame():
    with pytest.raises(ValueError, match='Too few'):
        image_metrics(np.zeros((64, 64, 3), np.uint8))
