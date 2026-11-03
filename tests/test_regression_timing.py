"""Timing diagnostics preserve test identities and never substitute for pytest."""

import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from unittest.mock import Mock
import xml.etree.ElementTree as ET

import cv2
import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


timing = load_module('regression_timing', ROOT / 'scripts/report_test_timing.py')


@pytest.mark.parametrize('value', [None, '1', '2'])
def test_thread_limit_is_explicit_and_applied_in_each_test_process(monkeypatch, value):
    config = load_module('test_runtime_config', ROOT / 'tests/conftest.py')
    monkeypatch.delenv('MULTICAMERA_TEST_OPENCV_THREADS', raising=False)
    if value is not None:
        monkeypatch.setenv('MULTICAMERA_TEST_OPENCV_THREADS', value)
    setter = Mock()
    monkeypatch.setattr(cv2, 'setNumThreads', setter)
    monkeypatch.setattr(cv2, 'getBuildInformation', lambda: 'Parallel framework: Concurrency')
    config.pytest_configure(None)
    if value is None:
        setter.assert_not_called()
    else:
        setter.assert_called_once_with(int(value))


@pytest.mark.parametrize('value', ['1', '2'])
def test_gcd_backend_disables_parallel_regions_instead_of_ignoring_positive_limit(monkeypatch, value):
    config = load_module('test_runtime_config', ROOT / 'tests/conftest.py')
    monkeypatch.setenv('MULTICAMERA_TEST_OPENCV_THREADS', value)
    monkeypatch.setattr(cv2, 'getBuildInformation', lambda: '  Parallel framework:    GCD')
    setter = Mock()
    monkeypatch.setattr(cv2, 'setNumThreads', setter)
    config.pytest_configure(None)
    setter.assert_called_once_with(0)


@pytest.mark.parametrize('value', ['', '0', '-1', 'auto', '1.5', '１'])
def test_invalid_thread_limit_fails_instead_of_silently_changing_execution(monkeypatch, value):
    config = load_module('test_runtime_config', ROOT / 'tests/conftest.py')
    monkeypatch.setenv('MULTICAMERA_TEST_OPENCV_THREADS', value)
    setter = Mock()
    monkeypatch.setattr(cv2, 'setNumThreads', setter)
    with pytest.raises(pytest.UsageError, match='positive integer'):
        config.pytest_configure(None)
    setter.assert_not_called()


def junit(tmp_path, order=(0, 1, 2, 3), suite_root=False):
    suite = ET.Element('testsuite', time='4')
    samples = [('ui', 'button', '3', None), ('solver', 'fit', '5', 'failure'),
               ('io', 'missing', '.2', 'skipped'), ('ui', 'broken', '.1', 'error')]
    for index in order:
        module, name, duration, outcome = samples[index]
        case = ET.SubElement(suite, 'testcase', classname=module, name=name, time=duration)
        if outcome:
            ET.SubElement(case, outcome)
    root = suite if suite_root else ET.Element('testsuites')
    if root is not suite:
        root.append(suite)
    path = tmp_path / '耗时 报告.xml'
    ET.ElementTree(root).write(path, encoding='utf-8', xml_declaration=True)
    return path


@pytest.mark.parametrize('suite_root', [False, True])
def test_timing_counts_failures_skips_and_parallel_cpu_time_truthfully(tmp_path, suite_root):
    report = timing.summarize(junit(tmp_path, suite_root=suite_root), limit=2)
    assert report['status'] == 'failed'
    assert report['reported_cases'] == 4 and report['passed'] == 1
    assert report['failures'] == report['errors'] == report['skipped'] == 1
    assert report['junit_elapsed_seconds'] == 4
    assert report['reported_test_seconds'] == 8.3
    assert report['slowest_tests'][0]['test'] == 'solver::fit'
    assert report['slowest_modules'][1]['cases'] == 2
    assert len(report['slowest_tests']) == len(report['slowest_modules']) == 2


def test_collection_digest_is_order_independent_but_keeps_duplicate_cases(tmp_path):
    first = timing.summarize(junit(tmp_path))
    assert timing.summarize(junit(tmp_path, (3, 2, 1, 0)))['collection_sha256'] == first['collection_sha256']
    assert timing.summarize(junit(tmp_path, (0, 1, 2, 3, 0)))['collection_sha256'] != first['collection_sha256']


@pytest.mark.parametrize('value', ['nan', 'inf', '-1', 'bad'])
def test_invalid_durations_are_not_reported_as_successful_zero_time(tmp_path, value):
    path = junit(tmp_path)
    root = ET.parse(path)
    root.find('.//testcase').set('time', value)
    root.write(path, encoding='utf-8')
    with pytest.raises(ValueError):
        timing.summarize(path)


def test_cli_writes_unicode_json_and_appends_summary_without_replacing_it(tmp_path):
    source = junit(tmp_path)
    output, summary = tmp_path / '诊断/metrics.json', tmp_path / 'summary.md'
    summary.write_text('Existing summary\n', encoding='utf-8')
    assert timing.main([str(source), '--output', str(output), '--summary', str(summary)]) == 0
    assert json.loads(output.read_text(encoding='utf-8'))['status'] == 'failed'
    text = summary.read_text(encoding='utf-8')
    assert text.startswith('Existing summary\n') and 'not wall time' in text


def test_empty_report_is_not_a_full_test_pass(tmp_path):
    path = tmp_path / 'empty.xml'
    path.write_text('<testsuites><testsuite/></testsuites>', encoding='utf-8')
    assert timing.summarize(path)['status'] == 'no_tests'


def test_successful_report_keeps_explicit_skips_and_rejects_unknown_xml(tmp_path):
    path = junit(tmp_path, (0, 2))
    report = timing.summarize(path)
    assert report['status'] == 'passed' and report['passed'] == report['skipped'] == 1
    with pytest.raises(ValueError, match='positive'):
        timing.summarize(path, limit=0)
    path.write_text('<html/>', encoding='utf-8')
    with pytest.raises(ValueError, match='JUnit'):
        timing.summarize(path)


def test_ui_data_preparation_detects_each_image_once_and_keeps_independent_records(monkeypatch):
    from multicamera.calibration.planar import PlanarPatternDetector, PatternBoardConfig
    from test_multimodal_preview import board_image
    from test_workbench_ui import bench_pairs

    rgb, ir, calls = board_image(), board_image(), []
    original = PlanarPatternDetector.detect

    def detect(detector, image):
        calls.append(image)
        return original(detector, image)

    monkeypatch.setattr(PlanarPatternDetector, 'detect', detect)
    records = bench_pairs(rgb, ir, PatternBoardConfig.ir_chessboard())
    assert len(calls) == 2 and calls[0] is rgb and calls[1] is ir
    assert [record[0] for record in records] == [str(i) for i in range(8)]
    template = records[0][3]
    assert template.valid
    for record in records:
        assert record[1] is rgb and record[2] is ir
        for side in (3, 4):
            assert record[side].valid
            np.testing.assert_array_equal(record[side].image_points, template.image_points)
    assert not np.shares_memory(records[0][3].image_points, records[1][3].image_points)
    assert not np.shares_memory(records[0][3].image_points, records[0][4].image_points)
    records[0][3].image_points[:] = 0
    assert records[1][3].image_points.any() and records[0][4].image_points.any()


def test_parallel_execution_keeps_every_case_and_propagates_failure(tmp_path):
    suite = tmp_path / 'parallel probe'
    suite.mkdir()
    shutil.copyfile(ROOT / 'tests/conftest.py', suite / 'conftest.py')
    source = '''
import os
import cv2
import pytest
@pytest.mark.parametrize('index', range(3))
def test_case(index, request):
    assert cv2.getNumThreads() == 1
    request.node.user_properties.append(('worker', os.environ['PYTEST_XDIST_WORKER']))
    assert not (__name__ == 'test_b' and index == 1), 'intentional probe failure'
'''
    for name in ('test_a.py', 'test_b.py'):
        (suite / name).write_text(source, encoding='utf-8')
    path = suite / 'results.xml'
    env = dict(os.environ, MULTICAMERA_TEST_OPENCV_THREADS='1', OMP_NUM_THREADS='1',
               OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    env.pop('PYTEST_ADDOPTS', None)
    result = subprocess.run([sys.executable, '-m', 'pytest', '-q', str(suite), '-n', '2',
        '--dist', 'loadfile', '--max-worker-restart=0', f'--junitxml={path}'],
        cwd=suite, env=env, capture_output=True, text=True, encoding='utf-8', timeout=60)
    assert result.returncode == 1, result.stdout + result.stderr
    report = timing.summarize(path)
    assert report['reported_cases'] == 6 and report['passed'] == 5, result.stdout + result.stderr
    assert report['failures'] == 1 and report['errors'] == report['skipped'] == 0
    workers = {node.get('value') for node in ET.parse(path).iter('property')
               if node.get('name') == 'worker'}
    assert workers == {'gw0', 'gw1'}
