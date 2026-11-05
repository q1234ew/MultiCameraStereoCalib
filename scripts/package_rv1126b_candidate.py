"""Build per-device, MAC-gated candidate installation archives without activation."""

import argparse
import hashlib
import io
import json
from pathlib import Path
import re
import tarfile


ROOT = Path(__file__).resolve().parents[1]
REQUIRED = {'calibration_mm.yml', 'test_vectors.yml', 'source_rig.json',
            'calibration_adapter.hpp', 'compatibility_test.cpp', 'legacy_loader_probe.cpp',
            'CMakeLists.txt', 'README.md'}


def package(source: Path, output: Path, version: str, runtime_binary: Path | None = None) -> Path:
    source, output = Path(source), Path(output)
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,63}', version) or '..' in version:
        raise ValueError('Invalid candidate version')
    manifest = json.loads((source / 'manifest.json').read_text(encoding='utf-8'))
    device, mac = manifest['device_id'], manifest.get('expected_mac', '')
    if not re.fullmatch(r'[A-Za-z0-9_-]+', device):
        raise ValueError('Invalid device ID')
    if not isinstance(mac, str) or not re.fullmatch(r'(?:[0-9a-f]{2}:){5}[0-9a-f]{2}', mac):
        raise ValueError('An explicit expected device MAC is required')
    if manifest['activation_allowed'] is not False:
        raise ValueError('Only non-activating candidate bundles may be packaged')
    if set(manifest['files']) != REQUIRED:
        raise ValueError('Unexpected or incomplete payload manifest')
    if any(f.is_symlink() or not f.is_file() for f in source.iterdir()):
        raise ValueError('Only regular payload files are permitted')
    if {f.name for f in source.iterdir()} != REQUIRED | {'manifest.json'}:
        raise ValueError('Unlisted payload files')
    payload = {f.name: f.read_bytes() for f in source.iterdir()}
    for name, digest in manifest['files'].items():
        if hashlib.sha256(payload[name]).hexdigest() != digest:
            raise ValueError('Payload hash mismatch: ' + name)
    if runtime_binary is not None:
        binary = Path(runtime_binary).read_bytes()
        if (len(binary) < 64 or binary[:6] != b'\x7fELF\x02\x01' or
                int.from_bytes(binary[18:20], 'little') != 183):
            raise ValueError('Runtime must be a little-endian aarch64 ELF binary')
        payload['compatibility_test'] = binary
        notices = Path(runtime_binary).with_name('THIRD_PARTY_NOTICES.txt').read_bytes()
        payload['THIRD_PARTY_NOTICES.txt'] = notices
        for name in ['compatibility_test', 'THIRD_PARTY_NOTICES.txt']:
            manifest['files'][name] = hashlib.sha256(payload[name]).hexdigest()
        manifest['runtime'] = {'architecture': 'aarch64', 'purpose': 'offline_numeric_test_only'}
        payload['manifest.json'] = (json.dumps(manifest, indent=2) + '\n').encode('utf-8')
    template = (ROOT / 'packaging/rv1126b/install_candidate.sh').read_text(encoding='utf-8')
    installer = template.replace('@DEVICE_ID@', device).replace('@VERSION@', version).replace(
        '@EXPECTED_MAC@', mac).encode('utf-8')
    sums = ''.join(f'{hashlib.sha256(data).hexdigest()}  {name}\n'
                   for name, data in sorted(payload.items())).encode('ascii')
    variant = 'runtime' if runtime_binary is not None else 'sources'
    filename = f'MultiCamera_calibration_{device}_{version}_aarch64_{variant}.tar.gz'
    prefix = filename.removesuffix('.tar.gz')
    output.mkdir(parents=True, exist_ok=True)
    target = output / filename
    # Exclusive creation: never replace a distributed installation archive.
    with target.open('xb') as stream:
        with tarfile.open(fileobj=stream, mode='w:gz') as archive:
            entries = {f'payload/{name}': value for name, value in payload.items()}
            entries.update({'install.sh': installer, 'SHA256SUMS.txt': sums})
            for name, value in sorted(entries.items()):
                info = tarfile.TarInfo(prefix + '/' + name)
                executable = name in {'install.sh', 'payload/compatibility_test'}
                info.size, info.mode = len(value), 0o755 if executable else 0o644
                archive.addfile(info, io.BytesIO(value))
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--version', required=True)
    parser.add_argument('--runtime-binary', type=Path)
    args = parser.parse_args()
    print(package(args.source, args.output, args.version, args.runtime_binary))


if __name__ == '__main__':
    main()
