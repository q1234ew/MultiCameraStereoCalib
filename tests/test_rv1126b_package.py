"""MAC-gated offline device packages must never activate or replace a candidate."""

import hashlib
import json
import os
from pathlib import Path
import runpy
import shutil
import subprocess
import tarfile

import pytest


MODULE = runpy.run_path(str(Path(__file__).resolve().parents[1] /
                           'scripts/package_rv1126b_candidate.py'))
package = MODULE['package']


@pytest.fixture
def candidate(tmp_path):
    source = tmp_path / 'source'
    source.mkdir()
    for name in MODULE['REQUIRED']:
        (source / name).write_bytes(b'test payload\n')
    manifest = {'device_id': 'device_1', 'expected_mac': '02:00:00:00:00:01',
                'activation_allowed': False, 'files': {
                    f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in source.iterdir()}}
    (source / 'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
    return source


def test_archive_has_only_sources_safe_paths_and_exact_checksums(candidate, tmp_path):
    archive = package(candidate, tmp_path / 'out', 'compat_20261105_1')
    with tarfile.open(archive) as tar:
        members = tar.getmembers()
        assert all(m.isfile() and '..' not in Path(m.name).parts for m in members)
        assert all(not m.name.startswith('/') for m in members)
        assert len(members) == 11
        sums = next(m for m in members if m.name.endswith('/SHA256SUMS.txt'))
        for line in tar.extractfile(sums).read().decode().splitlines():
            digest, name = line.split('  ')
            member = next(m for m in members if m.name.endswith('/payload/' + name))
            assert hashlib.sha256(tar.extractfile(member).read()).hexdigest() == digest
    before = archive.read_bytes()
    with pytest.raises(FileExistsError):
        package(candidate, tmp_path / 'out', 'compat_20261105_1')
    assert archive.read_bytes() == before


@pytest.mark.parametrize('defect', ['hash', 'mac', 'activation', 'extra', 'symlink'])
def test_invalid_package_is_refused(candidate, tmp_path, defect):
    manifest_path = candidate / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    if defect == 'hash':
        (candidate / 'calibration_mm.yml').write_bytes(b'changed')
    elif defect == 'extra':
        (candidate / 'unexpected.bin').write_bytes(b'bad')
    elif defect == 'symlink':
        if os.name == 'nt':
            pytest.skip('Creating symlinks requires Windows privileges')
        (candidate / 'unexpected').symlink_to(candidate / 'calibration_mm.yml')
    else:
        manifest['expected_mac' if defect == 'mac' else 'activation_allowed'] = (
            'not a mac' if defect == 'mac' else True)
        manifest_path.write_text(json.dumps(manifest), encoding='utf-8')
    with pytest.raises(ValueError):
        package(candidate, tmp_path / 'out', 'compat_1')
    assert not (tmp_path / 'out').exists()


@pytest.mark.parametrize('version', ['../old', '..', 'x/y', "x'; echo bad"])
def test_version_cannot_escape_paths_or_shell(candidate, tmp_path, version):
    with pytest.raises(ValueError):
        package(candidate, tmp_path / 'out', version)


@pytest.mark.parametrize('header', [b'not ELF', b'\xcf\xfa\xed\xfe' + bytes(64),
                                  b'\x7fELF\x02\x01' + bytes(58)])
def test_runtime_rejects_mac_and_non_aarch64_binaries(candidate, tmp_path, header):
    binary = tmp_path / 'test_binary'
    binary.write_bytes(header)
    with pytest.raises(ValueError, match='aarch64'):
        package(candidate, tmp_path / 'out', 'compat_1', binary)


def test_runtime_binary_and_notices_are_in_manifest_and_executable(candidate, tmp_path):
    header = bytearray(64)
    header[:6] = b'\x7fELF\x02\x01'
    header[18:20] = (183).to_bytes(2, 'little')
    binary = tmp_path / 'test_binary'
    binary.write_bytes(header)
    (tmp_path / 'THIRD_PARTY_NOTICES.txt').write_text('license notices')
    archive = package(candidate, tmp_path / 'out', 'compat_1', binary)
    assert archive.name.endswith('_aarch64_runtime.tar.gz')
    with tarfile.open(archive) as tar:
        members = tar.getmembers()
        member = next(m for m in members if m.name.endswith('/compatibility_test'))
        assert member.mode == 0o755
        manifest = json.loads(tar.extractfile(next(
            m for m in members if m.name.endswith('/manifest.json'))).read())
        assert manifest['files']['compatibility_test'] == hashlib.sha256(header).hexdigest()
        assert 'THIRD_PARTY_NOTICES.txt' in manifest['files']
        assert not manifest['activation_allowed']


@pytest.mark.skipif(os.name == 'nt' or not shutil.which('sha256sum'),
                    reason='POSIX installer with sha256sum')
def test_install_only_matching_mac_refuses_repeat_and_preserves_active_files(candidate, tmp_path):
    archive = package(candidate, tmp_path / 'out', 'compat_1')
    unpacked = tmp_path / 'unpacked'
    with tarfile.open(archive) as tar:
        tar.extractall(unpacked, filter='data')
    directory = next(unpacked.iterdir())
    network = tmp_path / 'net' / 'eth0'
    network.mkdir(parents=True)
    (network / 'address').write_text('02:00:00:00:00:02\n')
    installer = directory / 'install.sh'
    # Replace only the hardware discovery location in this test copy.
    installer.write_text(installer.read_text().replace('/sys/class/net', str(network.parent)))
    active = tmp_path / 'active.yml'
    active.write_bytes(b'active parameters')
    target = tmp_path / 'candidates'
    result = subprocess.run(['sh', str(installer), str(target)], capture_output=True)
    assert result.returncode != 0 and not target.exists()
    (network / 'address').write_text('02:00:00:00:00:01\n')
    result = subprocess.run(['sh', str(installer), str(target)], capture_output=True)
    assert result.returncode == 0, result.stderr
    installed = target / 'device_1' / 'compat_1'
    assert (installed / 'candidate-status.txt').read_text().strip() == 'candidate_only_not_activated'
    before = {f.name: f.read_bytes() for f in installed.iterdir()}
    result = subprocess.run(['sh', str(installer), str(target)], capture_output=True)
    assert result.returncode != 0
    assert before == {f.name: f.read_bytes() for f in installed.iterdir()}
    assert active.read_bytes() == b'active parameters'
