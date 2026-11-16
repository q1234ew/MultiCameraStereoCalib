"""Bounded immutable RGB-D reuse for explicitly confirmed static scenes."""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import time
from pathlib import Path


class CaptureCache:
    TTL_S = 600
    LIMIT = 6

    def __init__(self, root):
        self.root = Path(root)
        self.index = self.root / 'capture-cache.json'
        try:
            if self.index.stat().st_size > 1024 * 1024:
                raise ValueError('Cache index too large')
            data = json.loads(self.index.read_text())
            if data['schema_version'] != 1 or not isinstance(data['entries'], list):
                raise ValueError('Invalid cache index')
            self.records = [r for r in data['entries'] if isinstance(r, dict)][:self.LIMIT]
        except (OSError, ValueError, TypeError, KeyError):
            self.records = []

    @staticmethod
    def digest(value):
        return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()

    def save(self):
        self.root.mkdir(parents=True, exist_ok=True)
        temporary = self.index.with_suffix('.tmp')
        temporary.write_text(json.dumps({'schema_version': 1, 'entries': self.records}, indent=2))
        temporary.replace(self.index)

    def clear(self):
        # Historical job files and downloadable reports are deliberately preserved.
        self.records = []
        self.save()

    def current(self, record):
        try:
            age = (time.time() * 1000 - record['captured_at_unix_ms']) / 1000
            return (isinstance(record['id'], str)
                    and re.fullmatch('[0-9a-f]{32}', record['id']) is not None
                    and 0 <= age <= self.TTL_S)
        except (TypeError, KeyError):
            return False

    def publish(self, folder, params, context, metrics):
        folder = Path(folder)
        files = {}
        for ident in params['devices']:
            names = ['camera.json'] + [f'{kind}/{i:06d}.png'
                     for kind in ('color', 'depth') for i in range(params['frames'])]
            for name in names:
                relative = f'{ident}/{name}'
                path = folder / 'captures' / relative
                if path.is_symlink():
                    raise ValueError('Cache source cannot be a symbolic link')
                body = path.read_bytes()
                files[relative] = {'size': len(body), 'sha256': hashlib.sha256(body).hexdigest()}
        if sum(f['size'] for f in files.values()) > 128 * 1024 * 1024:
            return
        record = {'id': folder.name, 'devices': sorted(params['devices']),
                  'resolution': params['resolution'], 'quality': params['quality'],
                  'frames': params['frames'], 'context': context,
                  'captured_at_unix_ms': min(m['received_at_unix_ms'] for m in metrics),
                  'metrics': metrics, 'files': files}
        self.records = [record] + [r for r in self.records
                                  if r.get('id') != folder.name and self.current(r)][:self.LIMIT - 1]
        self.save()

    def summaries(self, context_for):
        summaries = []
        for r in self.records:
            try:
                if not self.current(r):
                    continue
                context = context_for(r['devices'])
                summaries.append({k: r[k] for k in ('id', 'devices', 'resolution', 'quality',
                                                   'frames', 'captured_at_unix_ms')} |
                                 {'compatible': context == r['context'],
                                  'age_s': round((time.time() * 1000 - r['captured_at_unix_ms']) / 1000),
                                  'bytes': sum(f['size'] for f in r['files'].values())})
            except (ValueError, TypeError, KeyError, OSError):
                continue
        return summaries

    def restore(self, ident, params, context, destination):
        record = next((r for r in self.records if r.get('id') == ident), None)
        if (not record or not self.current(record) or record['context'] != context
                or record['devices'] != sorted(params['devices'])
                or record['resolution'] != params['resolution']
                or record['quality'] != params['quality'] or record['frames'] < params['frames']):
            raise ValueError('深度缓存已过期或与当前参数不匹配，请重新采集')
        source = self.root / record['id'] / 'captures'
        # Validate all requested files before exposing any reused capture.
        selected = []
        for device in params['devices']:
            for name in ['camera.json'] + [f'{kind}/{i:06d}.png'
                         for kind in ('color', 'depth') for i in range(params['frames'])]:
                relative = f'{device}/{name}'
                path = source / relative
                if path.is_symlink() or source.resolve() not in path.resolve().parents:
                    raise ValueError('深度缓存路径无效，请重新采集')
                body = path.read_bytes()
                if (len(body) != record['files'][relative]['size']
                        or hashlib.sha256(body).hexdigest() != record['files'][relative]['sha256']):
                    raise ValueError('深度缓存文件已变化，请重新采集')
                selected.append((path, Path(destination) / relative))
        for src, dst in selected:
            dst.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.link(src, dst)
            except OSError:
                shutil.copy2(src, dst)
        metrics = record['metrics'][:len(params['devices']) * params['frames']]
        return metrics, {k: record[k] for k in ('id', 'captured_at_unix_ms')}
