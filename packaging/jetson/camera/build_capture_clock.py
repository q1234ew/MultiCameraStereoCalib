"""Reproduce the deployed RK1126B binary before applying the capture-only patch.

Builds in an isolated output directory. Never changes the RK1126B worktree,
existing deploy packages, IR validation outputs, or live devices.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import shutil
import subprocess
import tarfile
from pathlib import Path

COMMIT = '42fb525e73b44221a65f78a41f5f52edee4865dd'
BASE_SHA256 = '24b7063a69b77640865a6e3a6e611ba19d2eb9c840ce8da749e054eedf563803'
FLAGS = ('-D_LARGEFILE_SOURCE -D_LARGEFILE64_SOURCE -D_FILE_OFFSET_BITS=64 '
         '-O2 -g0 -D_FORTIFY_SOURCE=1 -I/deps/include')
CHANGED = {'src/libs/frame.h', 'src/libs/frame.c', 'src/libs/memsink.c',
           'src/ustreamer/device.h', 'src/ustreamer/device.c',
           'src/ustreamer/http/server.c'}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(args, **kwargs):
    return subprocess.run(args, check=True, **kwargs)


def build(rk_root, source, package, image):
    docker = ['docker', 'run', '--rm', '--platform', 'linux/amd64',
              '-v', f'{source}:/work', '-v', f'{rk_root}/rockchip_rv1126b_toolchain:/sdk:ro',
              '-v', f'{rk_root}/FusionSensor/ustreamer-ir/.cross-deps/usr:/deps:ro',
              '-v', f'{package}/tests:/tests:ro', '-w', '/work', image]
    run(docker + ['make', '-C', 'src', 'clean'], stdout=subprocess.DEVNULL)
    run(docker + ['make', '-j4', '-C', 'src', 'ustreamer.bin',
                  'CC=/sdk/bin/aarch64-buildroot-linux-gnu-gcc',
                  'CXX=/sdk/bin/aarch64-buildroot-linux-gnu-g++', f'CFLAGS={FLAGS}',
                  'LDFLAGS=-L/deps/lib -Wl,-rpath-link,/deps/lib', 'WITH_SOUNDIMG=0'])
    return docker


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rk-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True,
                        help='New directory; existing outputs are never overwritten')
    parser.add_argument('--image', default='multicamera-clock-build:20261109')
    args = parser.parse_args()
    rk_root, output = args.rk_root.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    package = Path(__file__).resolve().parent
    archive = subprocess.check_output(['git', '-C', str(rk_root / 'FusionSensor'),
                                       'archive', COMMIT, 'ustreamer-ir'])
    baseline = output / 'baseline'
    baseline.mkdir()
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        # Trusted local Git archive, restricted further for accidental path escape.
        for member in tar.getmembers():
            if member.name.startswith('/') or '..' in Path(member.name).parts:
                raise ValueError('Unsafe archive path')
        tar.extractall(baseline)
    baseline /= 'ustreamer-ir'
    build(rk_root, baseline, package, args.image)
    baseline_sha = digest(baseline / 'src/ustreamer.bin')
    if baseline_sha != BASE_SHA256:
        raise ValueError(f'Unmodified rebuild does not match device baseline: {baseline_sha}')
    source_root = output / 'patched'
    source_root.mkdir()
    source = source_root / 'ustreamer-ir'
    shutil.copytree(baseline, source)
    patch = package / 'ustreamer-clock.patch'
    run(['patch', '-p1', '-d', str(source_root)], input=patch.read_bytes())
    changed = {str(p.relative_to(baseline)) for p in baseline.rglob('*')
               if p.is_file() and digest(p) != digest(source / p.relative_to(baseline))}
    if changed != CHANGED:
        raise ValueError(f'Unexpected source changes: {changed}')
    docker = build(rk_root, source, package, args.image)
    run(docker + ['/sdk/bin/aarch64-buildroot-linux-gnu-gcc', '-D_GNU_SOURCE',
                  '-std=c11', '-O2', '-ffunction-sections', '-fdata-sections',
                  '-I/work/src', '-I/deps/include', '/tests/capture_clock.c', 'src/libs/frame.c',
                  '-L/deps/lib', '-Wl,--gc-sections', '-Wl,-rpath-link,/deps/lib',
                  '-levent', '-levent_pthreads', '-pthread', '-lm', '-o', 'capture-clock-test'])
    manifest = {'schema_version': 1, 'source_project': str(rk_root / 'FusionSensor'),
                'source_commit': COMMIT, 'baseline_sha256': baseline_sha,
                'baseline_package': 'fusion-ustreamer-rk1126b-frame-sync-20261022.tar.gz',
                'patch_sha256': digest(patch), 'compiler_flags': FLAGS,
                'binary_sha256': digest(source / 'src/ustreamer.bin'),
                'test_binary_sha256': digest(source / 'capture-clock-test'),
                'changed_source_files': sorted(changed),
                'infrared_sources_unchanged': True, 'shared_memory_abi_unchanged': True,
                'native_tests_executed': False}
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
