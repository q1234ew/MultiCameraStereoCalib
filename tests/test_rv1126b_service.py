"""Packaging checks plus opt-in native HTTP/image regressions for the RV sidecar."""

import hashlib
import http.server
import importlib.util
import json
import os
import re
import socket
import subprocess
import tarfile
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import pytest

from test_rv1126b_export import source  # noqa: F401

from multicamera.io.rv1126b import export_bundle

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('rv_service_package', ROOT / 'scripts/package_rv1126b_service.py')
packager = importlib.util.module_from_spec(spec)
spec.loader.exec_module(packager)


@pytest.fixture
def candidate(source, tmp_path):  # noqa: F811
    return export_bundle(source, tmp_path / 'candidate', pair_name='rgb', aux_id='ir',
                         device_id='device_1', endpoint='192.0.2.1',
                         device_mac='02:00:00:00:00:01')


@pytest.fixture
def elf(tmp_path):
    binary = tmp_path / 'binary'
    data = bytearray(64)
    data[:6] = b'\x7fELF\x02\x01'
    data[18:20] = (183).to_bytes(2, 'little')
    binary.write_bytes(data)
    binary.with_name('THIRD_PARTY_NOTICES.txt').write_text('test notices')
    return binary


def test_service_package_is_isolated_and_hash_complete(candidate, elf, tmp_path):
    before = {p.name: p.read_bytes() for p in candidate.iterdir()}
    out = tmp_path / 'service.20261105.1'
    archive = packager.prepare(candidate, elf, out, 'service.20261105.1', left_port=8082, right_port=8080)
    assert archive.name == 'service.20261105.1.tar.gz'
    config = (out / 'service.yml').read_text()
    assert 'left_port: 8082' in config and 'right_port: 8080' in config
    assert '/calibration-candidates/device_1/0.6.1-compat.20261105.2/' in config
    for line in (out / 'SHA256SUMS.txt').read_text().splitlines():
        digest, name = line.split('  ')
        assert hashlib.sha256((out / name).read_bytes()).hexdigest() == digest
    assert before == {p.name: p.read_bytes() for p in candidate.iterdir()}
    with tarfile.open(archive) as tar:
        assert all(member.isfile() and member.name.startswith(out.name + '/') for member in tar)
    with pytest.raises(FileExistsError):
        packager.prepare(candidate, elf, out, 'service.20261105.1')


def test_dashboard_patch_preserves_original_and_has_rollback(candidate, elf, tmp_path):
    original = tmp_path / 'index.html'
    data = b'<!doctype html>\r\n<html><head></head><body><main>Existing dashboard</main></body></html>'
    original.write_bytes(data)
    out = tmp_path / 'with-dashboard'
    packager.prepare(candidate, elf, out, 'service.1', dashboard_original=original)
    assert original.read_bytes() == (out / 'dashboard-original.html').read_bytes() == data
    patched = (out / 'dashboard.html').read_bytes()
    entry = (ROOT / 'packaging/rv1126b/dashboard-entry.html').read_bytes()
    bootstrap = (ROOT / 'packaging/rv1126b/dashboard-bootstrap.html').read_bytes()
    bootstrap = bootstrap.replace(b'@LEFT_PORT@', b'8080').replace(b'@RIGHT_PORT@', b'8082')
    assert patched.replace(entry + b'\n', b'').replace(bootstrap + b'\n', b'') == data
    assert hashlib.sha256(data).hexdigest() in (out / 'rollback.sh').read_text()
    assert 'cmp -s' in (out / 'rollback.sh').read_text()


@pytest.mark.parametrize('left,right', [(8080, 8082), (8082, 8080)])
def test_dashboard_bootstrap_precedes_legacy_port_initialization(left, right):
    original = ('<html><head><script>const urlPort = window.location.port;'
                'const port = localStorage.getItem("port_stereo") || urlPort;'
                '</script></head><body></body></html>')
    patched = packager.patch_dashboard(original, left, right)
    assert patched.index("localStorage.setItem('port_stereo'") < patched.index('const urlPort')
    assert f'left: {left}, right: {right}' in patched
    assert '[8080, 8082].includes(left)' in patched


def test_dashboard_parser_ignores_body_string_inside_script():
    original = '<html><head><script>const text="</body>";</script></head><body></body></html>'
    patched = packager.patch_dashboard(original, 8080, 8082)
    assert 'const text="</body>";' in patched
    assert patched.index('multicamera-calibration-entry-v1') > patched.index('</head>')


def test_dashboard_routes_status_to_bounded_proxy_and_reports_missing_thermal():
    # Small portable fixture, not a dependency on ignored diagnostic files.
    original = ("<html><head></head><body><script>function pollStatus() {"
                "fetch(url(port, 'state'))} function pollIrThermal() {\n"
                "        if (!j.ok) return;\n        document.getElementById('ir-tmin')"
                "}</script></body></html>")
    patched = packager.patch_dashboard(original, 8080, 8082)
    assert '8090/api/raw-state?port=' in patched
    assert '设备尚未返回温度数据' in patched


@pytest.mark.parametrize('content', ['<body>missing closing', '</body></body>',
                                     '<body>multicamera-calibration-entry-v1</body>'])
def test_dashboard_patch_rejects_ambiguous_or_already_patched(candidate, elf, tmp_path, content):
    original = tmp_path / 'index.html'
    original.write_text(content)
    with pytest.raises(ValueError):
        packager.prepare(candidate, elf, tmp_path / 'out', 'service.1', dashboard_original=original)
    assert not (tmp_path / 'out').exists()


@pytest.mark.parametrize('defect', ['hash', 'mac', 'elf', 'version', 'ports', 'notices'])
def test_service_package_rejects_invalid_input(candidate, elf, tmp_path, defect):
    version, left = 'test.1', 8080
    if defect == 'hash':
        (candidate / 'calibration_mm.yml').write_text('changed')
    elif defect == 'mac':
        manifest = json.loads((candidate / 'manifest.json').read_text())
        manifest['expected_mac'] = 'wrong'
        (candidate / 'manifest.json').write_text(json.dumps(manifest))
    elif defect == 'elf':
        elf.write_bytes(b'Mac binary')
    elif defect == 'version':
        version = '../../bad'
    elif defect == 'ports':
        left = 8090
    else:
        elf.with_name('THIRD_PARTY_NOTICES.txt').unlink()
    with pytest.raises(ValueError):
        packager.prepare(candidate, elf, tmp_path / 'out', version, left_port=left)
    assert not (tmp_path / 'out').exists()


def free_port():
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        return sock.getsockname()[1]


def get(port, path):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(f'http://127.0.0.1:{port}{path}', timeout=15) as response:
            return response.status, response.headers, response.read()
    except urllib.error.HTTPError as error:
        return error.code, error.headers, error.read()


@pytest.fixture
def running_service(candidate, tmp_path):
    binary = os.environ.get('MULTICAMERA_SERVICE_TEST_BINARY')
    if not binary:
        pytest.skip('Native C++ sidecar test requires MULTICAMERA_SERVICE_TEST_BINARY')
    if Path('/sys/class/net').exists():
        mac = next(p.read_text().strip() for p in Path('/sys/class/net').glob('*/address')
                   if p.read_text().strip() != '00:00:00:00:00:00')
    else:
        mac = re.findall(r'ether ([0-9a-f:]{17})', subprocess.check_output(['ifconfig'], text=True))[0]
    images = {}
    calls = []
    for name, shape in [('left', (720, 1280)), ('right', (720, 1280)), ('ir', (192, 256))]:
        frame = np.zeros((*shape, 3), np.uint8)
        frame[::8, :, 1] = 200
        frame[:, ::8, 2] = 200
        images[name] = cv2.imencode('.jpg', frame)[1].tobytes()

    def handler(name):
        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                calls.append((name, self.path))
                body = b'{"ok":true,"result":{"source":{"online":true}}}' if self.path == '/state' else images[name]
                self.send_response(200)
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass
        return Handler

    servers = [http.server.ThreadingHTTPServer(('127.0.0.1', 0), handler(name))
               for name in ['left', 'right', 'ir']]
    for server in servers:
        threading.Thread(target=server.serve_forever, daemon=True).start()
    port = free_port()
    cfg = tmp_path / 'service.yml'
    (tmp_path / 'panel.html').write_bytes((ROOT / 'packaging/rv1126b/panel.html').read_bytes())
    cfg.write_text('%YAML:1.0\n---\n' + f'calibration: "{candidate / "calibration_mm.yml"}"\n'
                   f'device_id: "device_1"\ndevice_mac: "{mac}"\nbind: "127.0.0.1"\n'
                   f'port: {port}\nleft_port: {servers[0].server_port}\n'
                   f'right_port: {servers[1].server_port}\nir_port: {servers[2].server_port}\n')
    log = (tmp_path / 'service.log').open('w')
    process = subprocess.Popen([binary, str(cfg)], stdout=log, stderr=log)
    try:
        for _ in range(100):
            if process.poll() is not None:
                pytest.fail((tmp_path / 'service.log').read_text())
            try:
                if get(port, '/health')[0] == 200:
                    break
            except OSError:
                time.sleep(.05)
        else:
            pytest.fail('Sidecar did not become ready')
        yield port, images, calls, cfg
    finally:
        process.terminate()
        process.wait(timeout=5)
        log.close()
        for server in servers:
            server.shutdown()
            server.server_close()


def test_native_health_parameters_sources_and_rectification(running_service):
    port, _, calls, _ = running_service
    _, _, data = get(port, '/health')
    health = json.loads(data)
    assert health['parameters_loaded'] and not health['physical_validation']
    assert not health['raw_streams_modified']
    _, _, data = get(port, '/api/calibration')
    assert b'translation_unit: mm' in data and b'R2:' in data
    status, _, data = get(port, '/api/sources')
    assert status == 200 and all(s['snapshot_valid'] for s in json.loads(data)['sources'])
    for name in ['left', 'right']:
        status, headers, data = get(port, f'/api/rectified.jpg?camera={name}')
        assert status == 200 and headers['X-Image-Coordinates'] == 'rectified'
        assert cv2.imdecode(np.frombuffer(data, np.uint8), 1).shape == (720, 1280, 3)
    assert calls and all(path == '/snapshot' for _, path in calls)
    status, headers, data = get(port, '/panel')
    assert status == 200 and 'text/html' in headers['Content-Type']
    assert 'RGB+IR 平面验证'.encode() in data


def test_native_raw_state_proxy_is_read_only_and_restricts_ports(running_service):
    port, _, calls, cfg = running_service
    source_port = re.search(r'left_port: (\d+)', cfg.read_text())[1]
    status, headers, body = get(port, '/api/raw-state?port=' + source_port)
    assert status == 200 and json.loads(body)['result']['source']['online']
    assert headers['Access-Control-Allow-Origin'] == '*'
    assert calls == [('left', '/state')]
    assert get(port, '/api/raw-state?port=22')[0] == 400
    assert get(port, '/api/raw-state?port=8090')[0] == 400
    assert calls == [('left', '/state')]


@pytest.mark.parametrize('path', ['/api/project?from=rgb&u=1&v=1',
                                  '/api/project?from=rgb&u=1&v=1&depth_mm=nan',
                                  '/api/project?from=rgb&u=1&v=1&depth_mm=0',
                                  '/api/project?from=ir&u=9999&v=1&depth_mm=1000',
                                  '/api/fusion.jpg', '/api/rectified.jpg?camera=bad'])
def test_native_invalid_requests_fail_without_camera_access(running_service, path):
    port, _, calls, _ = running_service
    assert get(port, path)[0] == 400
    assert not calls


def test_native_projection_and_plane_overlay(running_service):
    port, _, _, _ = running_service
    status, _, data = get(port, '/api/project?from=rgb&u=640&v=360&depth_mm=1000')
    assert status == 200 and json.loads(data)['inside_target']
    status, headers, data = get(port, '/api/fusion.jpg?depth_mm=1000')
    assert status == 200 and headers['X-Hardware-Synchronized'] == 'false'
    assert 'plane' in headers['X-Fusion-Mode']
    assert cv2.imdecode(np.frombuffer(data, np.uint8), 1).shape == (720, 1280, 3)


def test_native_wrong_image_dimensions_are_rejected(running_service):
    port, images, _, _ = running_service
    images['left'] = cv2.imencode('.jpg', np.zeros((12, 12, 3), np.uint8))[1].tobytes()
    assert get(port, '/api/rectified.jpg?camera=left')[0] == 503
    images['left'] = b'not a jpeg'
    assert get(port, '/api/rectified.jpg?camera=left')[0] == 503


@pytest.mark.parametrize('field,value', [('device_id', 'other_device'),
                                         ('device_mac', '00:00:00:00:00:01')])
def test_native_mismatched_binding_refuses_start(running_service, field, value):
    _, _, _, cfg = running_service
    text = cfg.read_text()
    text = re.sub(rf'{field}: "[^"]+"', f'{field}: "{value}"', text)
    bad = cfg.with_name('bad.yml')
    bad.write_text(text)
    result = subprocess.run([os.environ['MULTICAMERA_SERVICE_TEST_BINARY'], str(bad)],
                            capture_output=True, timeout=5)
    assert result.returncode != 0 and b'mismatch' in result.stderr.lower()
