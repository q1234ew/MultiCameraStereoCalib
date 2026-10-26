import json
from types import SimpleNamespace

import pytest

from multicamera import build_info


def test_embedded_build_identity(monkeypatch, tmp_path):
    (tmp_path / 'build_identity.json').write_text(json.dumps({'commit': 'abc1234', 'dirty': False}),
                                                encoding='utf-8')
    monkeypatch.setattr(build_info.sys, 'frozen', True, raising=False)
    monkeypatch.setattr(build_info.sys, '_MEIPASS', str(tmp_path), raising=False)
    build_info.build_identity.cache_clear()
    try:
        assert '安装包' in build_info.build_label()
        assert 'abc1234' in build_info.build_label()
    finally:
        build_info.build_identity.cache_clear()


def test_source_reports_uncommitted_changes(monkeypatch):
    monkeypatch.setattr(build_info.sys, 'frozen', False, raising=False)
    monkeypatch.setattr(build_info.subprocess, 'run', lambda args, **kw: SimpleNamespace(
        stdout='abc1234' if 'rev-parse' in args else ' M example.py'))
    build_info.build_identity.cache_clear()
    try:
        assert '含未提交修改' in build_info.build_label()
    finally:
        build_info.build_identity.cache_clear()


def test_old_package_does_not_guess_commit(monkeypatch, tmp_path):
    monkeypatch.setattr(build_info.sys, 'frozen', True, raising=False)
    monkeypatch.setattr(build_info.sys, '_MEIPASS', str(tmp_path), raising=False)
    build_info.build_identity.cache_clear()
    try:
        assert build_info.build_identity()['commit'] == '未知'
    finally:
        build_info.build_identity.cache_clear()


@pytest.mark.parametrize('metadata', [[], {'version': '99.99.99', 'commit': 'abc1234'}])
def test_invalid_embedded_identity_cannot_override_runtime_version(monkeypatch, tmp_path, metadata):
    (tmp_path / 'build_identity.json').write_text(json.dumps(metadata), encoding='utf-8')
    monkeypatch.setattr(build_info.sys, 'frozen', True, raising=False)
    monkeypatch.setattr(build_info.sys, '_MEIPASS', str(tmp_path), raising=False)
    build_info.build_identity.cache_clear()
    try:
        identity = build_info.build_identity()
        assert identity['version'] == build_info.__version__
        assert identity['commit'] == '未知'
    finally:
        build_info.build_identity.cache_clear()
