"""Private adapter to the Jetson's existing FoundationStereo CUDA environment."""
from __future__ import annotations

import json
import os
import select
import subprocess
import tempfile
from pathlib import Path

import numpy as np


class FoundationRunner:
    def __init__(self):
        self.process = None
        self.identity = None
        self.log = None

    def _read(self, timeout):
        if not select.select([self.process.stdout], [], [], timeout)[0]:
            self.close()
            raise ValueError('FoundationStereo worker timed out; see model log')
        line = self.process.stdout.readline()
        if not line:
            self.close()
            raise ValueError('FoundationStereo worker exited; see model log')
        return json.loads(line)

    def ensure_ready(self):
        if self.process is None or self.process.poll() is not None:
            root = Path(os.environ.get('FOUNDATION_STEREO_ROOT',
                                       '/home/nvidia/Desktop/Dev/FoundationStereo'))
            checkpoint = root / 'pretrained_models/23-51-11/model_best_bp2.pth'
            if not checkpoint.is_file():
                raise ValueError('FoundationStereo checkpoint is unavailable')
            log_path = Path(os.environ.get('FOUNDATION_STEREO_LOG',
                str(Path.home()/'multicamera-jetson/validation/foundation-worker.log')))
            log_path.parent.mkdir(parents=True, exist_ok=True)
            self.log = log_path.open('a')
            env = dict(os.environ, XFORMERS_DISABLED='1', HF_HUB_OFFLINE='1',
                       OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1')
            self.process = subprocess.Popen(
                [os.environ.get('FOUNDATION_STEREO_PYTHON', '/usr/bin/python3'),
                 str(Path(__file__).with_name('foundation_worker.py')),
                 '--root', str(root), '--checkpoint', str(checkpoint),
                 *(['--expected-sha256', os.environ['FOUNDATION_STEREO_CHECKPOINT_SHA256']]
                   if os.environ.get('FOUNDATION_STEREO_CHECKPOINT_SHA256') else [])],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log,
                text=True, bufsize=1, env=env)
            ready = self._read(180)
            if not ready.get('ready'):
                raise ValueError('FoundationStereo initialization failed')
            self.identity = ready['identity']

    def predict(self, left, right, iterations=8):
        if type(iterations) is not int or iterations not in (8, 16, 32):
            raise ValueError('Unsupported FoundationStereo iteration count')
        self.ensure_ready()
        with tempfile.TemporaryDirectory(prefix='multicamera-foundation-') as directory:
            source, target = Path(directory)/'pair.npz', Path(directory)/'disparity.npy'
            # FoundationStereo expects RGB 0..255, not normalized or BGR.
            np.savez(source, left=left[:, :, ::-1].copy(), right=right[:, :, ::-1].copy())
            self.process.stdin.write(json.dumps({'input': str(source), 'output': str(target), 'iterations': iterations})+'\n')
            self.process.stdin.flush()
            result = self._read(150)
            if not result.get('ok'):
                raise ValueError('FoundationStereo inference failed: '+result.get('error', 'unknown'))
            disparity = np.load(target, allow_pickle=False)
            if disparity.shape != left.shape[:2]:
                raise ValueError('FoundationStereo disparity dimensions mismatch')
            return disparity, result

    def close(self):
        if self.process:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
            for stream in (self.process.stdin, self.process.stdout):
                if stream:
                    stream.close()
            self.process = None
        if self.log:
            self.log.close()
            self.log = None
