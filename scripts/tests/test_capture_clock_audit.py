import pytest

from scripts.jetson.capture_clock_audit import native_metadata


def headers():
    return {'x-capture-protocol': 'v4l2-monotonic-v1',
            'x-capture-monotonic-ns': '100000000000', 'x-capture-stream-ns': '1000000000',
            'x-capture-sequence': '0', 'x-capture-v4l2-flags': str(0x12000),
            'x-capture-clock': 'monotonic', 'x-capture-send-monotonic-ns': '100030000000',
            'x-capture-boot-id': 'test-boot', 'x-capture-build': 'test-build',
            'x-capture-stage': 'driver-soe'}


def test_native_clock_is_preserved_in_integer_nanoseconds():
    result = native_metadata(headers())
    assert result['stamp_ns'] == 100000000000
    assert result['sequence'] == 0
    assert result['send_lag_ms'] == 30


@pytest.mark.parametrize('key,value', [
    ('x-capture-protocol', 'legacy'), ('x-capture-clock', 'realtime'),
    ('x-capture-v4l2-flags', str(0x12040)), ('x-capture-v4l2-flags', '0'),
    ('x-capture-monotonic-ns', '-1'), ('x-capture-stream-ns', '100000000001'),
    ('x-capture-sequence', '-1'), ('x-capture-sequence', str(2**32)),
    ('x-capture-send-monotonic-ns', '100600000000'), ('x-capture-send-monotonic-ns', '99900000000'),
    ('x-capture-send-monotonic-ns', 'nan'),
])
def test_invalid_capture_evidence_is_rejected(key, value):
    h = headers()
    h[key] = value
    with pytest.raises(ValueError):
        native_metadata(h)


def test_legacy_raw_clock_must_not_change_native_clock_age():
    h = headers()
    h['x-ustreamer-send-timestamp'] = '99.9'
    assert native_metadata(h)['send_lag_ms'] == 30
