"""Offline recommendations must not learn from the frame being evaluated."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import cv2
import pytest

from multicamera.board.charuco_board import CharucoBoard, CharucoBoardConfig
from multicamera.calibration import guidance_replay as replay
from multicamera.calibration.models import CameraModel
from test_adaptive_guidance import diverse


@pytest.fixture
def board():
    return CharucoBoard(CharucoBoardConfig(cols=9, rows=7, square_length=.035, marker_length=.025))


def frame_stream(observations):
    return [(str(i), {'left': left, 'right': right})
            for i, (left, right) in enumerate(zip(observations['L'], observations['R']))]


def test_replay_only_analyzes_observed_prefix(board, monkeypatch):
    original = replay.analyze_guidance
    seen = []

    def tracked(b, model, observations, provided):
        seen.append([[o.frame_id for o in observations[c]] for c in ('left', 'right')])
        assert provided == {}
        return original(b, model, observations, provided)

    monkeypatch.setattr(replay, 'analyze_guidance', tracked)
    report = replay.replay_observations(board, CameraModel.PINHOLE, frame_stream(diverse(board, count=8)))
    assert seen == [[list(map(str, range(n))), list(map(str, range(n)))] for n in range(9)]
    assert report['steps'][0]['before']['intrinsic_cameras'] == []
    for step in report['steps']:
        assert step['motion_stability'] is None
        assert step['synchronization'] is None and step['white_balance'] is None
    json.dumps(report, allow_nan=False)


def test_future_images_cannot_change_past_recommendations(board):
    frames = frame_stream(diverse(board, count=8))
    short = replay.replay_observations(board, CameraModel.PINHOLE, frames[:6])
    full = replay.replay_observations(board, CameraModel.PINHOLE, frames)
    assert full['steps'][:6] == short['steps']
    assert full['steps'][6]['before'] == short['final']


def test_missing_images_do_not_add_evidence(board):
    report = replay.replay_observations(board, CameraModel.PINHOLE,
                                        [('bad', {'left': None, 'right': None})])
    assert report['steps'][0]['unique_sample_change'] == 0
    assert report['steps'][0]['verdict']['composition_match'] is False
    assert report['final']['recommendation']['sample_count'] == 0
    assert len(report['coverage']['left']['missing_zones']) == 9


def test_frame_id_mismatch_is_rejected(board):
    frames = frame_stream(diverse(board, count=1))
    with pytest.raises(ValueError, match='编号'):
        replay.replay_observations(board, CameraModel.PINHOLE, [('wrong', frames[0][1])])


@pytest.mark.parametrize('console_encoding', ['utf-8', 'cp1252', 'ascii'])
def test_cli_preserves_sources_and_does_not_overwrite_output(tmp_path, console_encoding):
    source, output = tmp_path / '中文图片', tmp_path / '回放报告'
    source.mkdir()
    board = CharucoBoard(CharucoBoardConfig())
    image = board.generate_image(pixel_size=40)
    files = [source / f'0001_{role}.png' for role in ('left', 'right')]
    for path in files:
        ok, encoded = cv2.imencode('.png', image)
        assert ok
        path.write_bytes(encoded.tobytes())
    before = [hashlib.sha256(p.read_bytes()).hexdigest() for p in files]
    root = Path(__file__).resolve().parents[1]
    command = [sys.executable, str(root / 'scripts/replay_capture_guidance.py'),
               '--input', str(source), '--output', str(output)]
    env = dict(os.environ, PYTHONIOENCODING=console_encoding, PYTHONUTF8='0')
    result = subprocess.run(command, cwd=root, capture_output=True, encoding='utf-8',
                            env=env, timeout=30)
    assert result.returncode == 0, result.stderr
    assert '回放 0001: L=' in result.stdout
    assert f'报告：{output.resolve() / "index.html"}' in result.stdout
    report = json.loads((output / 'report.json').read_text(encoding='utf-8'))
    assert report['source_hashes_verified']
    assert len(report['steps']) == 1
    assert '__REPLAY_DATA__' not in (output / 'index.html').read_text(encoding='utf-8')
    assert '双目采图引导离线回放' in (output / 'REPORT.md').read_text(encoding='utf-8')
    assert before == [hashlib.sha256(p.read_bytes()).hexdigest() for p in files]
    data = (output / 'report.json').read_bytes()
    repeated = subprocess.run(command, cwd=root, capture_output=True, encoding='utf-8',
                              env=env, timeout=30)
    assert repeated.returncode != 0
    assert 'FileExistsError' in repeated.stderr
    assert 'UnicodeEncodeError' not in repeated.stderr
    assert str(output.resolve()) in repeated.stderr
    assert (output / 'report.json').read_bytes() == data
