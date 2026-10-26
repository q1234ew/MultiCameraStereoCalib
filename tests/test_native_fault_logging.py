"""Native stack diagnostics must work without console streams or a Qt app."""

import subprocess
import sys


def run(code, path):
    result = subprocess.run([sys.executable, '-c', code, str(path)], capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stdout + result.stderr


def test_native_stack_log_has_live_descriptor_with_no_stdio(tmp_path):
    run('''
import faulthandler, pathlib, sys
from multicamera import logging_config as log
directory = pathlib.Path(sys.argv[1])
sys.stdout = sys.stderr = None
log._install_fault_handler(directory)
assert faulthandler.is_enabled()
assert log._fault_stream is not None and not log._fault_stream.closed
faulthandler.dump_traceback(file=log._fault_stream, all_threads=True)
''', tmp_path)
    text = (tmp_path / 'multicamera_fault.log').read_text()
    assert 'pid=' in text and 'Current thread' in text


def test_reconfiguration_rotates_before_enabling_and_closes_old_descriptor(tmp_path):
    run('''
import faulthandler, pathlib, sys
from multicamera import logging_config as log
directory = pathlib.Path(sys.argv[1])
log.LOG_MAX_BYTES = 1
log._install_fault_handler(directory)
old = log._fault_stream
log._install_fault_handler(directory)
assert old.closed and faulthandler.is_enabled()
assert (directory / (log.FAULT_LOG_FILE_NAME + '.1')).is_file()
faulthandler.dump_traceback(file=log._fault_stream)
''', tmp_path)
    assert 'Current thread' in (tmp_path / 'multicamera_fault.log').read_text()


def test_fault_log_failure_does_not_stop_application_setup(tmp_path):
    path = tmp_path / 'file-not-directory'
    path.write_text('existing')
    run('''
import pathlib, sys
from multicamera import logging_config as log
log._install_fault_handler(pathlib.Path(sys.argv[1]))
assert log._fault_stream is None
''', path)
    assert path.read_text() == 'existing'
