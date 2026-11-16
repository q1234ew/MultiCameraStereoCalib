"""Bound camera monotonic clocks to this process without trusting wall time.

The HTTP send timestamp bounds clock offset, not exposure timing. A mapped
kernel timestamp remains diagnostic until a separately measured source error
budget is supplied for this exact protocol/build/stage.
"""
from __future__ import annotations

import math
import uuid
from collections import deque
from dataclasses import dataclass

PROTOCOL = 'v4l2-monotonic-v1'
BUILD = 'mc-clock-1-42fb525e'


@dataclass(frozen=True)
class ClockSample:
    mapped_mono_s: float | None = None
    uncertainty_s: float | None = None
    capture_verified: bool = False
    mapping_verified: bool = False
    reason: str = 'native_capture_headers_missing'
    source_key: tuple | None = None
    sequence: int | None = None
    stage: str | None = None

    def diagnostic(self):
        return {'protocol': PROTOCOL if self.source_key else None,
                'mapping_verified': self.mapping_verified,
                'capture_verified': self.capture_verified, 'reason': self.reason,
                'mapped_mono_s': self.mapped_mono_s,
                'mapping_uncertainty_ms': (round(self.uncertainty_s * 1000, 3)
                                           if self.uncertainty_s is not None else None),
                'sequence': self.sequence, 'stage': self.stage,
                'source_epoch': list(self.source_key) if self.source_key else None}


class SourceClock:
    def __init__(self, qualification=None, window_s=2.0, drift_ppm=1000):
        self.qualification = qualification or {}
        self.window_s, self.drift = window_s, drift_ppm / 1e6
        self.key = None
        self.bounds = deque(maxlen=128)
        self.last_sequence = self.last_capture = self.last_send = None
        self.independent = 0

    def invalidate(self):
        self.bounds.clear()
        self.last_sequence = self.last_capture = self.last_send = None
        self.independent = 0

    def observe(self, headers, request_mono, headers_mono):
        # HTTPMessage is case-insensitive; plain dictionaries used by tools/tests
        # should obey the same contract.
        h = {k.lower(): v for k, v in headers.items()}
        try:
            if h.get('x-capture-protocol') != PROTOCOL:
                self.invalidate()
                return ClockSample()
            build = h['x-capture-build']
            boot = str(uuid.UUID(h['x-capture-boot-id']))
            stream_ns = int(h['x-capture-stream-ns'])
            capture_ns = int(h['x-capture-monotonic-ns'])
            sequence = int(h['x-capture-sequence'])
            flags = int(h['x-capture-v4l2-flags'])
            stage = h['x-capture-stage']
            send = float(h['x-ustreamer-send-timestamp'])
            if (build != BUILD or h['x-capture-clock'] != 'monotonic'
                    or flags & 0xe000 != 0x2000 or flags & 0x40
                    or stage not in ('driver-soe', 'driver-eof')
                    or (flags & 0x70000) != (0x10000 if stage == 'driver-soe' else 0)
                    or not 0 <= sequence < 2**32 or not 0 < stream_ns <= capture_ns
                    or not all(math.isfinite(v) for v in (send, request_mono, headers_mono))
                    or not 0 <= headers_mono - request_mono <= .2
                    or not 0 <= send - capture_ns / 1e9 <= .5):
                raise ValueError('invalid_native_clock')
            key = (boot, stream_ns, build)
            changed = key != self.key
            if changed:
                self.invalidate()
                self.key = key
            capture = capture_ns / 1e9
            if (self.last_sequence is not None and (sequence < self.last_sequence
                    or capture < self.last_capture or send < self.last_send)):
                raise ValueError('source_counter_regressed')
            if (self.last_sequence is not None and
                    ((sequence == self.last_sequence) != (capture == self.last_capture))):
                raise ValueError('source_counter_timestamp_mismatch')
            self.last_send = send
            if sequence != self.last_sequence:
                self.independent += 1
            self.last_sequence, self.last_capture = sequence, capture
            while self.bounds and headers_mono - self.bounds[0][0] > self.window_s:
                self.bounds.popleft()
            # Send header has six decimal places. Include its quantization.
            self.bounds.append((headers_mono, request_mono - send - 1e-6,
                                headers_mono - send + 1e-6, sequence))
            low = max(lo - (headers_mono - t) * self.drift for t, lo, _, _ in self.bounds)
            high = min(hi + (headers_mono - t) * self.drift for t, _, hi, _ in self.bounds)
            if low > high:
                raise ValueError('clock_offset_discontinuity')
            mapped = capture + (low + high) / 2
            error = (high - low) / 2 + (send - capture) * self.drift
            ready = len({row[3] for row in self.bounds}) >= 3 and error <= .05
            q = self.qualification
            source_error = q.get('source_error_s')
            qualified = (q.get('protocol') == PROTOCOL and q.get('build') == build
                         and q.get('stage') == stage and bool(q.get('evidence'))
                         and isinstance(source_error, (int, float))
                         and not isinstance(source_error, bool) and math.isfinite(source_error)
                         and 0 <= source_error <= .1)
            verified = bool(ready and qualified and error + source_error <= .1)
            return ClockSample(mapped, error, verified, ready,
                               'verified' if verified else ('clock_warming_up' if not ready
                                                           else 'source_error_unqualified'),
                               key, sequence, stage)
        except (KeyError, ValueError, TypeError, OverflowError) as exc:
            self.invalidate()
            return ClockSample(reason=str(exc) if isinstance(exc, ValueError)
                               else 'invalid_native_clock')

    def capture_budget(self, sample):
        if not sample.capture_verified:
            return None
        return sample.uncertainty_s + self.qualification['source_error_s']
