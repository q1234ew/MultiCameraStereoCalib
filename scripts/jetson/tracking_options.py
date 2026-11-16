"""Validated, versioned runtime controls without loading GPU dependencies."""
from __future__ import annotations

import json
import math
import uuid
from pathlib import Path

DEFAULT_OPTIONS = {
    'enabled': True, 'camera_fps': 3.0, 'imgsz': 640, 'tracker': 'bytetrack',
    'new_track_conf': 0.30, 'draw_boxes': True, 'draw_skeleton': True, 'jpeg_quality': 78,
}
PIPELINE_KEYS = {'enabled', 'camera_fps', 'imgsz', 'tracker', 'new_track_conf'}
TRACKERS = ('bytetrack', 'ocsort', 'tracktrack')


def validate_options(options):
    if not isinstance(options, dict) or set(options) != set(DEFAULT_OPTIONS):
        raise ValueError('Invalid tracking option keys')
    for key in ('enabled', 'draw_boxes', 'draw_skeleton'):
        if type(options[key]) is not bool:
            raise ValueError(key + ' must be boolean')
    for key, low, high in [('camera_fps', .5, 25), ('new_track_conf', .15, .75)]:
        value = options[key]
        if (type(value) not in (int, float) or not math.isfinite(value)
                or not low <= value <= high):
            raise ValueError(key + ' is outside the supported range')
    if type(options['imgsz']) is not int or options['imgsz'] not in (320, 480, 640, 960):
        raise ValueError('Unsupported pose input size')
    if type(options['jpeg_quality']) is not int or not 50 <= options['jpeg_quality'] <= 90:
        raise ValueError('JPEG quality must be 50–90')
    if options['tracker'] not in TRACKERS:
        raise ValueError('Unsupported tracker')
    return dict(options)


class RevisionConflict(ValueError):
    pass


class OptionStore:
    """Caller serializes reads/updates with the runtime lock."""

    def __init__(self, path=None, fps=3):
        self.path = Path(path) if path else None
        self.options = {**DEFAULT_OPTIONS, 'camera_fps': fps}
        if self.path and self.path.exists():
            self.options = json.loads(self.path.read_text())
        self.options = validate_options(self.options)
        self.revision = uuid.uuid4().hex

    def update(self, patch, expected_revision):
        if expected_revision != self.revision:
            raise RevisionConflict('Settings changed; reload before applying')
        if not isinstance(patch, dict) or not patch or set(patch) - set(DEFAULT_OPTIONS):
            raise ValueError('Invalid tracking setting patch')
        updated = validate_options({**self.options, **patch})
        changed = {key for key in updated if updated[key] != self.options[key]}
        if not changed:
            return set()
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.path.with_name(self.path.name + '.tmp')
            temporary.write_text(json.dumps(updated, indent=2, allow_nan=False) + '\n')
            temporary.replace(self.path)
        self.options = updated
        self.revision = uuid.uuid4().hex
        return changed


def make_tracker(name, fps=3, new_track_conf=.30):
    """Pinned Ultralytics adapters; static cameras, no ReID download or GMC cost."""
    from types import SimpleNamespace

    from ultralytics.trackers.byte_tracker import BYTETracker
    from ultralytics.trackers.oc_sort import OCSORT
    from ultralytics.trackers.track_tracker import TRACKTRACK
    from ultralytics.utils import ROOT, YAML

    factories = {'bytetrack': BYTETracker, 'ocsort': OCSORT, 'tracktrack': TRACKTRACK}
    if name not in factories:
        raise ValueError('Unsupported tracker')
    config = YAML.load(ROOT / 'cfg' / 'trackers' / (name + '.yaml'))
    config.update(track_buffer=max(1, round(fps * 1.5)),
                  new_track_thresh=new_track_conf,
                  track_high_thresh=min(.25, new_track_conf), track_low_thresh=.1,
                  with_reid=False, gmc_method='none')
    # Match the low-confidence rescue policy for the comparison where supported.
    if name == 'ocsort':
        config['use_byte'] = True
    return factories[name](SimpleNamespace(**config))
