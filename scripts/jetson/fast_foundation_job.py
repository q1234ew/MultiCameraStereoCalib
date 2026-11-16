"""Orin-only, logout-independent TensorRT build and benchmark controller."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import signal
import subprocess
import time
from pathlib import Path


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    temporary.replace(path)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, required=True)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    root, base = args.project.resolve(), args.base.resolve()
    py = str(root / '.fast-stereo-venv/bin/python')
    import numpy as np
    import psutil

    targets = ['1280x736-i8-sim', '1920x1088-i8-sim']
    for target in targets:
        folder = base / target
        export = json.loads((folder / 'export.json').read_text())
        if digest(folder / 'model.onnx') != export['onnx_sha256']:
            raise ValueError('ONNX identity mismatch: ' + target)
    required = [root / 'scripts/jetson/fast_foundation_build_trt.py',
                root / 'scripts/jetson/fast_foundation_smoke.py',
                root / 'scripts/jetson/fast_foundation_trt.py',
                base / 'benchmark.py', base / 'validate.py', base / 'run_guarded.py']
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.verify_only:
        import tensorrt as trt
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError('Orin CUDA is unavailable in this execution environment')
        print(json.dumps({'ready': True, 'all_inputs_on_orin': True, 'targets': targets,
                          'torch': str(torch.__version__), 'tensorrt': trt.__version__}))
        return
    lock = (base / 'job.lock').open('w')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print('An Orin controller is already running', flush=True)
        return
    progress = {'state': 'running', 'phase': 'starting', 'pid': os.getpid(),
                'host_required': False, 'started_unix_s': time.time()}

    def status(phase, **extra):
        progress.update(phase=phase, updated_unix_s=time.time(), **extra)
        write_json(base / 'status.json', progress)

    def valid_build(folder):
        try:
            build = json.loads((folder / 'build.json').read_text())
            return digest(folder / 'model.engine') == build['engine_sha256']
        except (OSError, ValueError, KeyError):
            return False

    def existing_builder(folder):
        for process in psutil.process_iter(['cmdline', 'uids']):
            try:
                command = process.info['cmdline'] or []
                if (len(command) > 1 and Path(command[1]).name == 'fast_foundation_build_trt.py'
                        and str(folder) in command and process.uids().real == os.getuid()
                        and process.status() != psutil.STATUS_ZOMBIE):
                    return process
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        return None

    def wait_existing(process, folder):
        # Replace the SSH-launched deadline monitor, preserving the live build.
        monitor = None
        try:
            parent = process.parent()
            command = parent.cmdline()
            if len(command) > 1 and Path(command[1]).name == 'run_guarded.py':
                monitor = parent
                parent.send_signal(signal.SIGSTOP)
        except psutil.Error:
            pass
        samples = []
        try:
            while process.is_running() and process.status() != psutil.STATUS_ZOMBIE:
                vm = psutil.virtual_memory()
                elapsed = time.time() - process.create_time()
                samples.append({'t':time.time(),'used_gib':vm.used/2**30,
                                'available_gib':vm.available/2**30})
                status('adopted_build_' + folder.name, builder_pid=process.pid,
                       available_gib=round(vm.available/2**30, 2))
                if vm.available < 2*2**30 or elapsed > 3600:
                    process.terminate()
                    try:
                        process.wait(timeout=10)
                    except psutil.TimeoutExpired:
                        process.kill()
                    break
                time.sleep(1)
        except psutil.NoSuchProcess:
            pass
        finally:
            if monitor is not None:
                try:
                    monitor.send_signal(signal.SIGCONT)
                except psutil.Error:
                    pass
            write_json(folder / 'adopted-resource-samples.json', samples)

    def run(command, log):
        with log.open('a') as output:
            subprocess.run(command, cwd=str(root), check=True, stdout=output,
                           stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)

    def build_target(folder):
        if valid_build(folder):
            return
        process = existing_builder(folder)
        if process:
            wait_existing(process, folder)
        if valid_build(folder):
            return
        # A lost SSH process can be recovered using immutable local ONNX files.
        stamp = str(int(time.time()))
        for name in ('model.engine', 'build.json', 'build.log'):
            path = folder / name
            if path.exists():
                path.rename(folder / (name + '.interrupted-' + stamp))
        status('building_' + folder.name)
        run([py, str(base / 'run_guarded.py'), str(folder / 'build.log'), py,
             str(root / 'scripts/jetson/fast_foundation_build_trt.py'),
             '--directory', str(folder), '--verbose'], base / 'controller-build.log')
        if not valid_build(folder):
            raise RuntimeError('Build did not create a verified engine: ' + folder.name)

    def preflight(folder, width, height):
        status('numerical_check_' + folder.name)
        out = base / ('preflight-' + str(width))
        run(['bash', str(root / 'packaging/jetson/run-fast-foundation-stereo.sh'),
             '--backend', 'trt-single', '--width', str(width), '--height', str(height),
             '--engine-dir', str(folder), '--output', str(out)],
            base / ('preflight-' + str(width) + '.log'))
        reference = root / ('deployment/fast-foundation-highres_20261107/%dx%d-i8'
                            % (width, height))
        a = json.loads((reference / 'report.json').read_text())
        b = json.loads((out / 'report.json').read_text())
        if (a['identity']['checkpoint_sha256'] != b['identity']['checkpoint_sha256']
                or a['left_sha256'] != b['left_sha256']
                or a['right_sha256'] != b['right_sha256']):
            raise ValueError('Numerical reference identity mismatch')
        x = np.load(reference / 'disparity_px.npy', allow_pickle=False)
        y = np.load(out / 'disparity_px.npy', allow_pickle=False)
        error = np.abs(x-y)
        result = {'finite': bool(np.isfinite(y).all()), 'mae_px': float(error.mean()),
                  'p95_abs_px': float(np.percentile(error, 95)),
                  'fraction_gt_2px': float((error > 2).mean()),
                  'forward_median_ms': b['forward_median_ms']}
        result['passed'] = (result['finite'] and result['mae_px'] <= .5
                            and result['p95_abs_px'] <= 1 and result['fraction_gt_2px'] <= .01)
        write_json(base / ('preflight-%d-validation.json' % width), result)
        if not result['passed']:
            raise ValueError('Numerical comparison did not pass: ' + folder.name)

    try:
        for target, (width, height) in zip(targets, [(1280,736), (1920,1088)]):
            folder = base / target
            build_target(folder)
            preflight(folder, width, height)
        status('paired_resource_benchmark')
        run([py, str(base / 'benchmark.py')], base / 'benchmark.log')
        status('final_validation')
        run([py, str(base / 'validate.py')], base / 'validation.log')
        measurements = base / 'measurements'
        summary = json.loads((measurements / 'summary.json').read_text())
        validation = json.loads((measurements / 'validation.json').read_text())
        if (len(summary['profiles']) != 4 or not all(p['passed'] for p in summary['profiles'])
                or len(validation['profiles']) != 2
                or not all(p['passed'] for p in validation['profiles'])):
            raise RuntimeError('Benchmark coverage or final validation incomplete')
        lines = ['# Fast-FoundationStereo TensorRT Orin 实测结果', '',
                 '同一20-30-48权重，8次迭代，最大视差192；每档预热2次、测量5次。', '',
                 '| 输入 | 后端 | 热推理中位 ms | 整机已用峰值 GiB |',
                 '| --- | --- | ---: | ---: |']
        for p in summary['profiles']:
            lines.append('| %dx%d | %s | %.2f | %.2f |' % (
                p['width'], p['height'], p['backend'],
                p['inference']['forward_median_ms'], p['peak_system_used_gib']))
        lines += ['', '单路官方样例资源测试，不是现场深度精度或三路并发验收。',
                  '计时不含采集、缩放、上传和写盘；整机/进程/CUDA内存重叠，不可相加。',
                  'TensorRT自有内存不完整计入PyTorch allocator，以整机采样为主。']
        (base / 'RESULTS.md').write_text('\n'.join(lines) + '\n')
        files = [*required, Path(__file__), base / 'measurements/summary.json',
                 base / 'measurements/validation.json', base / 'RESULTS.md']
        for target in targets:
            files += [base / target / n for n in ('build.json','export.json','model.engine')]
        write_json(base / 'job-manifest.json', {'files_sha256':{
            str(p.relative_to(root)):digest(p) for p in files}})
        status('complete', state='complete', finished_unix_s=time.time())
    except Exception as error:
        status('failed', state='failed', error=str(error))
        raise


if __name__ == '__main__':
    main()
