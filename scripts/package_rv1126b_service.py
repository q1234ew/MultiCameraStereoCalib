"""Prepare isolated service payloads; never contact or modify a device."""

import argparse
import hashlib
import io
import json
import re
import shutil
import tarfile
from html.parser import HTMLParser
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def patch_dashboard(text: str, left_port: int, right_port: int) -> str:
    """Insert before initialization, retaining every original character."""
    if 'multicamera-calibration-entry-v1' in text:
        raise ValueError('Dashboard already integrated')
    if 'function pollStatus()' in text:
        old = "fetch(url(port, 'state'))"
        if text.count(old) != 1:
            raise ValueError('Unknown dashboard status reader')
        text = text.replace(old, "fetch('http://' + host + ':8090/api/raw-state?port=' + port)"
                            ".then(function(response) { if (!response.ok) throw new Error('State unavailable'); return response; })")
        text = text.replace('        var sink = j.result || {};',
                            "        if (!j.ok || !j.result || !j.result.source) throw new Error('Invalid state');\n"
                            '        var sink = j.result;')
        text = text.replace('        var src = sink.source || {};',
                            "        var src = sink.source || {};\n"
                            "        document.getElementById('dot-status').classList.toggle('online', src.online !== false);")
        text = text.replace("document.getElementById('status-text').textContent = '未连接';",
                            "document.getElementById('status-text').textContent = '状态读取失败';")
    if 'function pollIrThermal()' in text:
        old = "        if (!j.ok) return;\n        document.getElementById('ir-tmin')"
        if text.count(old) != 1:
            raise ValueError('Unknown dashboard thermal reader')
        text = text.replace(old, """        if (!j.ok) {
          ['ir-tmin', 'ir-tmax', 'ir-tcenter', 'ir-tmaxpos'].forEach(function(id) {
            document.getElementById(id).textContent = '--';
          });
          window.multicameraThermalStatus('设备尚未返回温度数据');
          return;
        }
        window.multicameraThermalStatus('');
        document.getElementById('ir-tmin')""")
    lines = text.splitlines(keepends=True)

    class Locations(HTMLParser):
        def __init__(self):
            super().__init__(convert_charrefs=False)
            self.heads = []
            self.bodies = []

        def absolute_position(self):
            line, column = self.getpos()
            return sum(map(len, lines[:line - 1])) + column

        def handle_starttag(self, tag, attrs):
            if tag == 'head':
                self.heads.append(self.absolute_position() + len(self.get_starttag_text()))

        def handle_endtag(self, tag):
            if tag == 'body':
                self.bodies.append(self.absolute_position())

    parser = Locations()
    parser.feed(text)
    parser.close()
    if len(parser.heads) != 1 or len(parser.bodies) != 1:
        raise ValueError('Dashboard head/body is missing or ambiguous')
    bootstrap = (ROOT / 'packaging/rv1126b/dashboard-bootstrap.html').read_text(encoding='utf-8')
    bootstrap = bootstrap.replace('@LEFT_PORT@', str(left_port)).replace('@RIGHT_PORT@', str(right_port))
    entry = (ROOT / 'packaging/rv1126b/dashboard-entry.html').read_text(encoding='utf-8')
    for offset, snippet in sorted([(parser.heads[0], bootstrap), (parser.bodies[0], entry)], reverse=True):
        text = text[:offset] + snippet + '\n' + text[offset:]
    return text


def prepare(candidate: Path, binary: Path, output: Path, version: str, *,
            left_port: int = 8080, right_port: int = 8082,
            dashboard_original: Path | None = None) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,79}", version):
        raise ValueError("Unsafe version")
    if {left_port, right_port} != {8080, 8082}:
        raise ValueError("Explicit original RGB ports required")
    manifest = json.loads((candidate / 'manifest.json').read_text(encoding='utf-8'))
    device = manifest['device_id']
    mac = str(manifest.get('expected_mac', '')).lower()
    if not re.fullmatch(r'device_[123]', device) or not re.fullmatch(r'(?:[0-9a-f]{2}:){5}[0-9a-f]{2}', mac):
        raise ValueError("Invalid device identity")
    calibration = candidate / 'calibration_mm.yml'
    digest = hashlib.sha256(calibration.read_bytes()).hexdigest()
    if digest != manifest['files']['calibration_mm.yml']:
        raise ValueError("Calibration hash mismatch")
    elf = binary.read_bytes()
    if len(elf) < 64 or elf[:6] != b'\x7fELF\x02\x01' or int.from_bytes(elf[18:20], 'little') != 183:
        raise ValueError("Service must be aarch64 ELF")
    notices = binary.with_name('THIRD_PARTY_NOTICES.txt')
    if not notices.is_file():
        raise ValueError("Missing third-party notices")
    app = f'/opt/fusion-sensor/calibration-services/{device}/{version}'
    remote_calibration = (
        f'/opt/fusion-sensor/calibration-candidates/{device}/'
        '0.6.1-compat.20261105.2/calibration_mm.yml'
    )
    # Deployment version is explicit: this activation requires the previously verified candidate.
    archive = Path(str(output) + '.tar.gz')
    if output.exists() or archive.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    try:
        shutil.copyfile(binary, output / 'calibration_service')
        shutil.copyfile(notices, output / notices.name)
        config = ('%YAML:1.0\n---\n'
                  f'calibration: "{remote_calibration}"\n'
                  f'device_id: "{device}"\ndevice_mac: "{mac}"\n'
                  'bind: "0.0.0.0"\nport: 8090\n'
                  f'left_port: {left_port}\nright_port: {right_port}\nir_port: 8081\n')
        (output / 'service.yml').write_text(config, encoding='utf-8')
        shutil.copyfile(ROOT / 'packaging/rv1126b/panel.html', output / 'panel.html')
        if dashboard_original is not None:
            original = dashboard_original.read_bytes()
            text = original.decode('utf-8')
            patched = patch_dashboard(text, left_port, right_port)
            (output / 'dashboard.html').write_bytes(patched.encode('utf-8'))
            (output / 'dashboard-original.html').write_bytes(original)
        values = {'APP': app, 'CALIBRATION': remote_calibration,
                  'CALIBRATION_SHA': digest, 'MAC': mac,
                  'DASHBOARD_SHA': hashlib.sha256(dashboard_original.read_bytes()).hexdigest()
                  if dashboard_original is not None else 'none'}
        for template, name in [('calibration_service.init', 'service.init'),
                               ('install_service.sh', 'install.sh'),
                               ('rollback_service.sh', 'rollback.sh')]:
            text = (ROOT / 'packaging/rv1126b' / template).read_text(encoding='utf-8')
            for key, value in values.items():
                text = text.replace(f'@{key}@', value)
            (output / name).write_text(text, encoding='utf-8')
        sums = ''.join(f'{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}\n'
                       for path in sorted(output.iterdir()))
        (output / 'SHA256SUMS.txt').write_text(sums, encoding='ascii')
        with archive.open('xb') as stream, tarfile.open(fileobj=stream, mode='w:gz') as tar:
            for path in sorted(output.iterdir()):
                info = tarfile.TarInfo(f'{output.name}/{path.name}')
                data = path.read_bytes()
                info.size = len(data)
                info.mode = 0o755 if path.name in ('calibration_service', 'install.sh', 'service.init', 'rollback.sh') else 0o644
                tar.addfile(info, io.BytesIO(data))
        return archive
    except Exception:
        shutil.rmtree(output)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--version', required=True)
    parser.add_argument('--left-port', type=int, default=8080)
    parser.add_argument('--right-port', type=int, default=8082)
    parser.add_argument('--dashboard-original', type=Path)
    args = parser.parse_args()
    print(prepare(args.candidate, args.binary, args.output, args.version,
                  left_port=args.left_port, right_port=args.right_port,
                  dashboard_original=args.dashboard_original))


if __name__ == '__main__':
    main()
