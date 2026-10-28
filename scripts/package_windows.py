"""Package a validated onedir app; ZIP and optional Inno Setup run concurrently."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import time
import zipfile


ROOT = Path(__file__).resolve().parents[1]
ZIP_NAME = 'MultiCamera_portable_x64.zip'


def portable_zip(source: Path, output: Path, level: int = 1) -> dict:
    source, output = source.resolve(), output.resolve()
    if source == output or source in output.parents:
        raise ValueError('Packaging output must be outside the source directory')
    files = sorted(path for path in source.rglob('*') if path.is_file())
    if (not files or not (source / 'MultiCamera.exe').is_file()
            or (source / 'MultiCamera.exe').stat().st_size == 0):
        raise ValueError('Expected a validated dist/MultiCamera/MultiCamera.exe')
    if any(path.is_symlink() for path in source.rglob('*')):
        raise ValueError('Portable source must not contain symbolic links')
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    destination = output / ZIP_NAME
    with tempfile.TemporaryDirectory(prefix='.portable-', dir=output) as staging:
        staged = Path(staging) / ZIP_NAME
        with zipfile.ZipFile(staged, 'w', compression=zipfile.ZIP_DEFLATED,
                             compresslevel=level, allowZip64=True) as archive:
            for path in files:
                archive.write(path, path.relative_to(source).as_posix())
        staged.replace(destination)
    return {'phase': 'portable_zip', 'seconds': round(time.perf_counter() - started, 3),
            'compression_level': level, 'file_count': len(files),
            'source_bytes': sum(path.stat().st_size for path in files),
            'output_bytes': destination.stat().st_size, 'file': destination.name}


def installer(source: Path, output: Path, compiler: Path, version: str,
              compression: str = 'normal') -> dict:
    started = time.perf_counter()
    output.mkdir(parents=True, exist_ok=True)
    destination = output / f'MultiCamera_Setup_{version}_x64.exe'
    with tempfile.TemporaryDirectory(prefix='.installer-', dir=output) as staging:
        subprocess.run([
            str(compiler), f'/DMyAppVersion={version}', f'/DMyCompression=lzma2/{compression}',
            f'/DDistDir={source.resolve()}', f'/DOutputRoot={Path(staging).resolve()}',
            str(ROOT / 'packaging/windows/MultiCamera.iss'),
        ], check=True)
        staged = Path(staging) / destination.name
        if not staged.is_file() or staged.stat().st_size == 0:
            raise RuntimeError('Inno Setup did not create a non-empty installer')
        staged.replace(destination)
    return {'phase': 'installer', 'seconds': round(time.perf_counter() - started, 3),
            'compression': f'lzma2/{compression}', 'output_bytes': destination.stat().st_size,
            'file': destination.name}


def package(source: Path, output: Path, *, compiler: Path | None = None,
            version: str | None = None, compression: str = 'normal',
            installer_only: bool = False) -> dict:
    source, output = source.resolve(), output.resolve()
    if source == output or source in output.parents:
        raise ValueError('Packaging output must be outside the source directory')
    if not (source / 'MultiCamera.exe').is_file() or (source / 'MultiCamera.exe').stat().st_size == 0:
        raise ValueError('Expected a non-empty dist/MultiCamera/MultiCamera.exe')
    if any(path.is_symlink() for path in source.rglob('*')):
        raise ValueError('Portable source must not contain symbolic links')
    if installer_only and compiler is None:
        raise ValueError('Installer-only packaging requires an Inno Setup compiler')
    if compiler is not None:
        if not compiler.is_file():
            raise ValueError('Inno Setup compiler was not found')
        if not version or not re.fullmatch(r'(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)', version):
            raise ValueError('Installer requires a numeric MAJOR.MINOR.PATCH version')
    if compression not in ('normal', 'max', 'ultra64'):
        raise ValueError('Unsupported installer compression')
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    report = {'status': 'running', 'parallel': compiler is not None and not installer_only,
              'phases': []}
    failures = []
    with ThreadPoolExecutor(max_workers=2) as pool:
        jobs = []
        if not installer_only:
            jobs.append(('portable_zip', pool.submit(portable_zip, source, output)))
        if compiler is not None:
            jobs.append(('installer', pool.submit(installer, source, output, compiler,
                                                  version, compression)))
        # Always wait for both tasks. A failed installer must not publish a ZIP-only release.
        for phase, future in jobs:
            try:
                report['phases'].append(future.result())
            except Exception as exc:
                failures.append(f'{phase}: {exc}')
    report.update(status='failed' if failures else 'passed', errors=failures,
                  seconds=round(time.perf_counter() - started, 3))
    destination = output / 'packaging_metrics.json'
    with tempfile.TemporaryDirectory(prefix='.metrics-', dir=output) as staging:
        staged = Path(staging) / destination.name
        staged.write_text(json.dumps(report, indent=2), encoding='utf-8')
        staged.replace(destination)
    if failures:
        raise RuntimeError('; '.join(failures))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT / 'dist/MultiCamera')
    parser.add_argument('--output', type=Path, default=ROOT / 'installer_output')
    parser.add_argument('--installer-compiler', type=Path)
    parser.add_argument('--version')
    parser.add_argument('--installer-compression', choices=('normal', 'max', 'ultra64'),
                        default=os.environ.get('MULTICAMERA_INSTALLER_COMPRESSION', 'normal'))
    parser.add_argument('--installer-only', action='store_true')
    args = parser.parse_args()
    report = package(args.source, args.output, compiler=args.installer_compiler,
                     version=args.version, compression=args.installer_compression,
                     installer_only=args.installer_only)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
