"""Conservative online association of anonymous local tracks in a verified floor frame.

Architecture references (no source copied): ByteTrack (FoundationVision),
PoseTrack (ZhenyuX1E), and alexandergg/multicam-3d-tracking.
"""
from __future__ import annotations

import math
import uuid
from collections import deque

import numpy as np


class GlobalAssociator:
    """Require geometry, a bounded capture clock, and repeated unambiguous evidence.

    A local identity is (camera, reconnect epoch, local ID). Members from one
    camera can never be merged. Unknown time or geometry yields local-only IDs.
    TTL uses server monotonic time, including when no camera frames arrive.
    """

    def __init__(self, ttl_s=3.0, evidence_frames=3, distance_m=0.65,
                 max_capture_delta_s=0.25, max_clock_uncertainty_s=0.10):
        self.ttl = ttl_s
        self.evidence_frames = evidence_frames
        self.distance = distance_m
        self.max_delta = max_capture_delta_s
        self.max_uncertainty = max_clock_uncertainty_s
        self.members = {}
        self.pending = {}
        self.trajectories = {}
        self.events = deque(maxlen=60)

    @staticmethod
    def key(obs):
        return (obs['camera_id'], obs['epoch'], int(obs['local_id']))

    def expire(self, now):
        expired = {key for key, value in self.members.items()
                   if now - value['seen_mono'] > self.ttl}
        for key in expired:
            del self.members[key]
        self.pending = {key: value for key, value in self.pending.items()
                        if key in self.members and now - value[2] < 1.0}
        gids = {value['gid'] for value in self.members.values()}
        self.trajectories = {gid: value for gid, value in self.trajectories.items()
                             if gid in gids}

    @staticmethod
    def visible(value, now):
        return value.get('observed', False) and now - value['seen_mono'] <= 0.8

    def time_compatible(self, left, right):
        return (self.eligible(left) and self.eligible(right)
                and abs(left['capture_time_s'] - right['capture_time_s'])
                + left['capture_uncertainty_s'] + right['capture_uncertainty_s']
                <= self.max_delta)

    @staticmethod
    def position_eligible(obs):
        xy = obs.get('world_xy_m')
        return bool(obs.get('position_verified') and obs.get('world_frame') and xy is not None
                    and len(xy) == 2 and all(math.isfinite(v) for v in xy))

    def eligible(self, obs):
        stamp = obs.get('capture_time_s')
        uncertainty = obs.get('capture_uncertainty_s')
        return bool(self.position_eligible(obs)
                    and stamp is not None and math.isfinite(stamp)
                    and uncertainty is not None and math.isfinite(uncertainty)
                    and 0 <= uncertainty <= self.max_uncertainty)

    def pair_cost(self, left, right):
        if (left['camera_id'] == right['camera_id'] or not self.eligible(left)
                or not self.eligible(right)
                or left.get('world_frame') != right.get('world_frame')):
            return None
        delta = abs(left['capture_time_s'] - right['capture_time_s'])
        if (delta + left['capture_uncertainty_s'] + right['capture_uncertainty_s']
                > self.max_delta):
            return None
        distance = math.dist(left['world_xy_m'], right['world_xy_m'])
        return distance if distance <= self.distance else None

    def update(self, observations, now, camera_id=None):
        self.expire(now)
        # Reject duplicate local keys in one source update.
        if len({self.key(obs) for obs in observations}) != len(observations):
            raise ValueError('Duplicate local track keys')
        # Only one source update at a time; otherwise assignment order could
        # silently allow two same-camera detections to claim a global identity.
        if len({obs['camera_id'] for obs in observations}) > 1:
            raise ValueError('Submit one camera per update')
        cameras = {obs['camera_id'] for obs in observations}
        if camera_id is not None:
            if cameras and cameras != {camera_id}:
                raise ValueError('Camera update identity mismatch')
            cameras.add(camera_id)
        for key, value in self.members.items():
            if key[0] in cameras:
                value['observed'] = False
        for obs in observations:
            key = self.key(obs)
            previous = self.members.get(key)
            gid = previous['gid'] if previous else str(uuid.uuid4())
            others = [v for k, v in self.members.items()
                      if k != key and v['gid'] == gid]
            # A newer frame may arrive before its peer. Retain identity for a
            # bounded wait, but split immediately on comparable contrary evidence.
            unsupported = any(
                not self.eligible(obs) or not self.eligible(v['observation'])
                or obs.get('world_frame') != v['observation'].get('world_frame')
                or (self.time_compatible(obs, v['observation'])
                    and self.pair_cost(obs, v['observation']) is None)
                for v in others)
            supported = bool(others) and all(
                self.visible(v, now) and self.pair_cost(obs, v['observation']) is not None
                for v in others)
            support_mono = previous.get('support_mono', now) if previous else now
            if supported:
                support_mono = now
                for value in others:
                    value['support_mono'] = now
            if others and (unsupported or now - support_mono > 0.8):
                gid = str(uuid.uuid4())
                self.events.append({'kind': 'split', 'reason': 'evidence_unavailable',
                                    'camera_id': obs['camera_id'], 'local_id': obs['local_id']})
            self.members[key] = {'gid': gid, 'seen_mono': now, 'observation': dict(obs),
                                 'observed': True, 'support_mono': support_mono}
        # Build the entire cost graph before changing any group. Require a clear
        # best match in both directions, so source ordering cannot decide identity.
        proposals = {}
        incoming = {}
        for obs in observations:
            key = self.key(obs)
            own = self.members[key]
            if not self.eligible(obs):
                self.pending.pop(key, None)
                continue
            own_group = [v for v in self.members.values() if v['gid'] == own['gid']]
            if len(own_group) > 1:
                continue  # Do not union two already-established components.
            candidates = {}
            for value in self.members.values():
                gid = value['gid']
                if gid == own['gid'] or gid in candidates:
                    continue
                members = [v for v in self.members.values() if v['gid'] == gid]
                if not all(self.visible(v, now) for v in members):
                    continue
                group = [v['observation'] for v in members]
                if any(other['camera_id'] == obs['camera_id'] for other in group):
                    continue
                costs = [self.pair_cost(obs, other) for other in group]
                if all(cost is not None for cost in costs):
                    candidates[gid] = max(costs)
                    incoming.setdefault(gid, []).append((max(costs), key))
            ordered = sorted(candidates, key=candidates.get)
            # Similar alternatives are ambiguous, including two nearby people.
            if (not ordered or len(ordered) > 1
                    and candidates[ordered[1]] - candidates[ordered[0]] < 0.20):
                self.pending.pop(key, None)
                continue
            target = ordered[0]
            proposals[key] = target
        for obs in observations:
            key = self.key(obs)
            target = proposals.get(key)
            if target is None:
                self.pending.pop(key, None)
                continue
            rivals = sorted(incoming[target])
            if (rivals[0][1] != key or len(rivals) > 1
                    and rivals[1][0] - rivals[0][0] < 0.20):
                self.pending.pop(key, None)
                continue
            own = self.members[key]
            prior = self.pending.get(key)
            # Require independent source samples, not repeated processing of one frame.
            sample = obs.get('sample_id')
            if prior and prior[0] == target and sample != prior[3]:
                count = prior[1] + 1
            elif prior and prior[0] == target:
                count = prior[1]
            else:
                count = 1
            self.pending[key] = (target, count, now, sample)
            if count >= self.evidence_frames:
                own['gid'] = target
                for value in self.members.values():
                    if value['gid'] == target:
                        value['support_mono'] = now
                self.pending.pop(key, None)
                self.events.append({'kind': 'linked', 'camera_id': obs['camera_id'],
                                    'local_id': obs['local_id'], 'global_id': target,
                                    'evidence_frames': count})
        return self.snapshot(now)

    def snapshot(self, now):
        self.expire(now)
        grouped = {}
        for key, value in self.members.items():
            grouped.setdefault(value['gid'], []).append((key, value))
        result = []
        for gid, values in grouped.items():
            # Aged observations remain inspectable, but are never called visible.
            sources = [{'camera_id': key[0], 'epoch': key[1], 'local_id': key[2],
                        'age_s': round(now - value['seen_mono'], 3)} for key, value in values]
            current = [v for _, v in values if self.visible(v, now)]
            valid = [v['observation'] for v in current if self.eligible(v['observation'])]
            linked = len(values) > 1 and len(valid) == len(values)
            # A single-camera floor projection does not require inter-camera
            # time alignment. Keep it explicitly local; never average unsynced views.
            aligned = bool(valid) and all(self.time_compatible(a, b)
                                         for a in valid for b in valid)
            # Pending peer frames must not turn into averaged unsynchronized positions.
            positions = valid if aligned else []
            latest = max(current, key=lambda v: v['seen_mono']) if current else None
            if latest and (len(values) == 1 or not aligned):
                positions = ([latest['observation']]
                             if self.position_eligible(latest['observation']) else [])
            xy = np.mean([v['world_xy_m'] for v in positions], axis=0).tolist() if positions else None
            frame = positions[0]['world_frame'] if positions else None
            signature = (frame, tuple(sorted(key for key, _ in values)))
            history = self.trajectories.get(gid)
            stamp = latest['seen_mono'] if latest else now
            if (history is None or history['signature'] != signature or xy is None
                    or history['points'] and stamp - history['points'][-1][0] > 0.8):
                history = {'signature': signature, 'points': deque(maxlen=256), 'sample': None}
                self.trajectories[gid] = history
            points = history['points']
            while points and now - points[0][0] > 8:
                points.popleft()
            sample = tuple(sorted((self.key(v['observation']),
                                   v['observation'].get('sample_id', v['seen_mono']))
                                  for v in current))
            if (xy is not None and sample != history['sample']
                    and (not points or stamp > points[-1][0])):
                points.append((stamp, xy))
                history['sample'] = sample
            result.append({'track_uuid': gid, 'global_id': gid if linked else None,
                           'association_state': 'linked' if linked else 'local_only',
                           'sources': sources, 'world_xy_m': xy,
                           'world_frame': frame,
                           'position_time_verified': aligned,
                           'trajectory': [{'age_s': round(now - stamp, 3), 'world_xy_m': point}
                                          for stamp, point in points],
                           'observed': bool(current)})
        return result
