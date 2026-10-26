"""Read embedded build identity; source checkouts report their actual Git state."""

from functools import lru_cache
import json
from pathlib import Path
import subprocess
import sys

from . import __version__


@lru_cache(maxsize=1)
def build_identity() -> dict:
    if getattr(sys, 'frozen', False):
        try:
            data = json.loads((Path(sys._MEIPASS) / 'build_identity.json').read_text(encoding='utf-8'))
            if not isinstance(data, dict) or data.get('version', __version__) != __version__:
                raise ValueError('Embedded build version does not match runtime version')
            return {**data, 'version': __version__, 'kind': '安装包'}
        except (OSError, ValueError, AttributeError):
            return {'version': __version__, 'commit': '未知', 'kind': '安装包', 'dirty': None}
    root = Path(__file__).resolve().parents[2]
    try:
        def git(*args):
            return subprocess.run(['git', '-C', str(root), *args], check=True,
                                  capture_output=True, text=True, timeout=2).stdout.strip()
        return {'version': __version__, 'commit': git('rev-parse', '--short', 'HEAD'),
                'dirty': bool(git('status', '--porcelain')), 'kind': '源码'}
    except (OSError, subprocess.SubprocessError):
        return {'version': __version__, 'commit': '未知', 'kind': '源码', 'dirty': None}


def build_label() -> str:
    info = build_identity()
    state = ' · 含未提交修改' if info.get('dirty') else ''
    return f"{info['kind']} v{info['version']} · {info['commit']}{state}"
