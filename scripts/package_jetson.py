#!/usr/bin/env python3
"""Build a small Jetson installer without models, wheels or runtime archives (never deploys or publishes)."""
from __future__ import annotations

import argparse
import ast
import datetime
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tarfile


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for part in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()


def copy_filtered(source, destination):
    shutil.copytree(source, destination, ignore=shutil.ignore_patterns(
        '.git', '__pycache__', '*.pyc', '.DS_Store', '*.pt', '*.pth', '*.onnx',
        '*.engine', '*.plan', '*.whl', '*.safetensors', '.venv', 'node_modules'), symlinks=False)


def build(source, output, build_id=None):
    tree = ast.parse((source / 'src/multicamera/__init__.py').read_text())
    version = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == '__version__' for t in n.targets))
    when = datetime.datetime.now(datetime.timezone.utc)
    build_id = build_id or when.strftime('%Y%m%dT%H%M%SZ')
    import re
    if not re.fullmatch(r'[A-Za-z0-9_.-]+', build_id):
        raise ValueError('Invalid build ID')
    release = '{}-{}'.format(version, build_id)
    output.mkdir(parents=True, exist_ok=True)
    folder = output / ('MultiCamera_Orin_' + release)
    folder.mkdir(exist_ok=False)
    app = folder / 'app'
    (app / 'scripts').mkdir(parents=True)
    if (source / 'scripts/__init__.py').is_file():
        shutil.copy2(source / 'scripts/__init__.py', app / 'scripts/__init__.py')
    copy_filtered(source / 'scripts/jetson', app / 'scripts/jetson')
    copy_filtered(source / 'vendor/owen_fusion', app / 'vendor/owen_fusion')
    (app / 'src/multicamera').mkdir(parents=True)
    shutil.copy2(source / 'src/multicamera/__init__.py', app / 'src/multicamera/__init__.py')
    installer = source / 'packaging/jetson/installer'
    for name in ['installer.py', 'site.example.json', 'assets.json', 'README.md', 'provision.py', 'provision-assets.json',
                 'site.fresh.example.json', 'site.fresh.jp6.example.json', 'foundation-orin.patch',
                 'foundation-cfg.yaml', 'provision-assets-jp6.json']:
        shutil.copy2(installer / name, folder / name)
    copy_filtered(installer / 'requirements', folder / 'requirements')
    for command, action in [('install', 'install'), ('upgrade', 'upgrade'),
                            ('doctor', 'doctor'), ('rollback', 'rollback')]:
        target = folder / (command + '.sh')
        target.write_text('#!/bin/sh\nset -eu\nbundle_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)\n'
            'export PYTHONDONTWRITEBYTECODE=1\nexec /usr/bin/python3 "$bundle_dir/installer.py" ' + action + ' --bundle "$bundle_dir" "$@"\n')
        target.chmod(0o755)
    try:
        commit = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'],
                                         text=True).strip()
        dirty = bool(subprocess.check_output(['git', '-C', str(source), 'status', '--porcelain']))
    except subprocess.CalledProcessError:
        commit, dirty = None, True
    files = {p.relative_to(folder).as_posix(): digest(p)
             for p in sorted(folder.rglob('*')) if p.is_file()}
    manifest = {'schema_version': 1, 'release_id': release, 'version': version,
                'build_at_utc': when.isoformat(), 'base_commit': commit,
                'workspace_dirty': dirty, 'release_candidate': False,
                'external_resources': True, 'models_embedded': False,
                'profile': 'multi-profile', 'profiles': ['jetpack5-l4t35.4.1-aarch64', 'jetpack6-l4t36.4.3-aarch64'], 'files': files}
    (folder / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    sums = {**files, 'manifest.json': digest(folder / 'manifest.json')}
    (folder / 'SHA256SUMS.txt').write_text(''.join(
        '{}  {}\n'.format(value, name) for name, value in sorted(sums.items())))
    if sum(p.stat().st_size for p in folder.rglob('*') if p.is_file()) > 20 * 1024**2:
        raise ValueError('Code payload exceeds 20 MiB; no archive generated')
    archive = output / (folder.name + '_aarch64.tar.gz')
    if archive.exists():
        raise FileExistsError(str(archive))
    with tarfile.open(archive, 'x:gz', compresslevel=1) as tar:
        tar.add(folder, arcname=folder.name)
    if archive.stat().st_size > 20 * 1024**2:
        raise ValueError('Lightweight installer unexpectedly exceeds 20 MiB; do not distribute')
    checksum = digest(archive)
    archive.with_name(archive.name + '.sha256').write_text(checksum + '  ' + archive.name + '\n')
    return {'folder': str(folder), 'archive': str(archive), 'sha256': checksum, 'bytes': archive.stat().st_size,
            'external_resources': True, 'models_embedded': False, 'release_id': release}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--build-id')
    args = parser.parse_args()
    print(json.dumps(build(args.source.resolve(), args.output.resolve(),
                           build_id=args.build_id), indent=2))


if __name__ == '__main__':
    main()
