"""Linux integration tests with fake network/capture; never access real camera ports."""
from __future__ import annotations

import os
import shutil
import signal
import subprocess
import tempfile
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


@unittest.skipUnless(shutil.which('flock') and shutil.which('setsid'),
                     'Requires Linux camera process-group and lock primitives')
class CameraSupervisorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='camera-supervisor-test-')
        self.root = Path(self.temp.name)
        self.bin = self.root / 'bin'
        self.bin.mkdir()
        for name in ['fusion-supervisor.sh', 'S90fusion-sensor']:
            shutil.copy(ROOT / 'packaging/jetson/camera' / name, self.root / name)
        self.write('bin/ip', '''#!/bin/sh
[ -f "$FUSION_APP_DIR/network" ] || exit 1
case "$*" in
 *link*) echo '2: eth0: <UP,LOWER_UP> state UP' ;;
 *) echo '2: eth0 inet 192.168.1.8/24 scope global eth0' ;;
esac
''')
        self.write('bin/wget', '#!/bin/sh\ntest -f "$FUSION_APP_DIR/health"\n')
        self.write('start_fusion.sh', '''#!/bin/bash
echo "$$" >> "$FUSION_APP_DIR/launches"
sleep 120 &
wait
''')
        (self.root / 'health').touch()
        self.env = {**os.environ, 'PATH': str(self.bin) + ':' + os.environ['PATH'],
                    'FUSION_APP_DIR': str(self.root), 'FUSION_RUN_DIR': str(self.root),
                    'FUSION_CHECK_INTERVAL': '.1', 'FUSION_STARTUP_GRACE': '1',
                    'FUSION_NETWORK_RETRY_INTERVAL': '.1', 'FUSION_MAX_FAILURES': '1'}
        self.log = (self.root / 'log').open('w')
        self.process = None

    def write(self, name, body):
        path = self.root / name
        path.write_text(body)
        path.chmod(0o755)

    def start(self):
        self.process = subprocess.Popen(['bash', str(self.root / 'fusion-supervisor.sh')],
                                        env=self.env, stdout=self.log, stderr=self.log)

    def wait_for(self, predicate, timeout=7):
        until = time.monotonic() + timeout
        while time.monotonic() < until:
            if predicate():
                return
            time.sleep(.05)
        self.fail('Supervisor condition timed out: ' + (self.root / 'log').read_text())

    def launches(self):
        path = self.root / 'launches'
        return path.read_text().splitlines() if path.exists() else []

    def healthy(self):
        path = self.root / 'fusion-sensor.state'
        return path.exists() and path.read_text().strip() == 'healthy'

    def tearDown(self):
        if self.process is not None:
            self.process.terminate()
            self.process.wait(timeout=8)
        self.log.close()
        self.temp.cleanup()

    def test_waits_for_network_and_recovers_killed_capture_group(self):
        self.start()
        time.sleep(.3)
        self.assertEqual(self.launches(), [])
        (self.root / 'network').touch()
        self.wait_for(self.healthy)
        first = int(self.launches()[0])
        os.killpg(first, signal.SIGKILL)
        self.wait_for(lambda: len(self.launches()) >= 2)
        self.wait_for(self.healthy)
        self.assertNotEqual(int(self.launches()[-1]), first)

    def test_partial_endpoint_failure_restarts_live_launcher(self):
        (self.root / 'network').touch()
        self.start()
        self.wait_for(self.healthy)
        (self.root / 'health').unlink()
        self.wait_for(lambda: len(self.launches()) >= 2)
        (self.root / 'health').touch()
        self.wait_for(self.healthy)

    def test_singleton_and_explicit_stop_do_not_respawn(self):
        (self.root / 'network').touch()
        self.start()
        self.wait_for(self.healthy)
        second = subprocess.run(['bash', str(self.root / 'fusion-supervisor.sh')],
                                env=self.env, timeout=3, capture_output=True)
        self.assertEqual(second.returncode, 0)
        self.assertEqual(len(self.launches()), 1)
        child = int(self.launches()[0])
        self.process.terminate()
        self.process.wait(timeout=8)
        time.sleep(.3)
        self.assertFalse((self.root / 'fusion-sensor.pid').exists())
        self.assertEqual(len(self.launches()), 1)
        with self.assertRaises(ProcessLookupError):
            os.killpg(child, 0)

    def test_stop_does_not_kill_unrelated_pid_from_stale_file(self):
        unrelated = subprocess.Popen(['sleep', '30'])
        try:
            (self.root / 'fusion-sensor.pid').write_text(str(unrelated.pid))
            result = subprocess.run(['sh', str(self.root / 'S90fusion-sensor'), 'stop'],
                                    env=self.env, timeout=3, capture_output=True)
            self.assertEqual(result.returncode, 0)
            self.assertIsNone(unrelated.poll())
        finally:
            unrelated.terminate()
            unrelated.wait(timeout=3)


if __name__ == '__main__':
    unittest.main(verbosity=2)
