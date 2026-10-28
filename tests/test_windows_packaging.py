"""Packaging correctness and concurrency without requiring a Windows compiler."""

import ast
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from threading import Barrier
import zipfile

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('windows_packaging', ROOT / 'scripts/package_windows.py')
packaging = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(packaging)


@pytest.fixture
def source(tmp_path):
    folder = tmp_path / 'input \u6d4b\u8bd5 bundle'
    folder.mkdir()
    (folder / 'MultiCamera.exe').write_bytes(b'fake executable')
    (folder / '_internal').mkdir()
    (folder / '_internal/.hidden').write_bytes(b'hidden runtime resource')
    (folder / '_internal/\u4e2d\u6587 name.txt').write_bytes(b'contents')
    return folder


def test_portable_roundtrip_keeps_all_resources_without_wrapper_folder(source, tmp_path):
    before = {p.relative_to(source).as_posix(): p.read_bytes() for p in source.rglob('*') if p.is_file()}
    output = tmp_path / 'output bundle'
    report = packaging.package(source, output)
    assert report['status'] == 'passed' and not report['parallel']
    assert report['phases'][0]['compression_level'] == 1
    with zipfile.ZipFile(output / packaging.ZIP_NAME) as archive:
        assert archive.testzip() is None
        assert {name: archive.read(name) for name in archive.namelist()} == before
    assert json.loads((output / 'packaging_metrics.json').read_text()) == report
    assert before == {p.relative_to(source).as_posix(): p.read_bytes()
                      for p in source.rglob('*') if p.is_file()}


def test_installer_and_zip_really_run_concurrently(source, tmp_path, monkeypatch):
    compiler = tmp_path / 'ISCC.exe'
    compiler.touch()
    barrier = Barrier(2)
    def phase(name):
        barrier.wait(timeout=10)
        return {'phase': name}
    monkeypatch.setattr(packaging, 'portable_zip', lambda *args: phase('portable_zip'))
    monkeypatch.setattr(packaging, 'installer', lambda *args: phase('installer'))
    result = packaging.package(source, tmp_path / 'output', compiler=compiler, version='0.1.1')
    assert result['parallel'] and result['status'] == 'passed'
    assert {p['phase'] for p in result['phases']} == {'portable_zip', 'installer'}


@pytest.mark.parametrize('failed_phase', ['installer', 'portable_zip'])
def test_any_failed_distribution_fails_the_whole_package(source, tmp_path, monkeypatch, failed_phase):
    compiler = tmp_path / 'ISCC.exe'
    compiler.touch()
    done = []
    def phase(name):
        done.append(name)
        if name == failed_phase:
            raise RuntimeError('injected failure')
        return {'phase': name}
    monkeypatch.setattr(packaging, 'portable_zip', lambda *args: phase('portable_zip'))
    monkeypatch.setattr(packaging, 'installer', lambda *args: phase('installer'))
    output = tmp_path / 'output'
    with pytest.raises(RuntimeError, match=failed_phase):
        packaging.package(source, output, compiler=compiler, version='0.1.1')
    assert set(done) == {'portable_zip', 'installer'}
    report = json.loads((output / 'packaging_metrics.json').read_text())
    assert report['status'] == 'failed' and len(report['errors']) == 1


def test_failed_zip_does_not_replace_previous_archive(source, tmp_path, monkeypatch):
    output = tmp_path / 'output'
    packaging.portable_zip(source, output)
    previous = (output / packaging.ZIP_NAME).read_bytes()
    monkeypatch.setattr(zipfile.ZipFile, 'write', lambda *args, **kwargs:
                        (_ for _ in ()).throw(OSError('disk full')))
    with pytest.raises(OSError):
        packaging.portable_zip(source, output)
    assert (output / packaging.ZIP_NAME).read_bytes() == previous
    assert not list(output.glob('.portable-*'))


@pytest.mark.parametrize('version', ['0.1', 'v0.1.1', '0.1.1-build.1', '01.1.1'])
def test_installer_version_must_remain_numeric(source, tmp_path, version):
    compiler = tmp_path / 'ISCC.exe'
    compiler.touch()
    with pytest.raises(ValueError, match='numeric'):
        packaging.package(source, tmp_path / 'output', compiler=compiler, version=version)


def test_inno_args_preserve_paths_with_spaces(source, tmp_path, monkeypatch):
    output = tmp_path / 'installer output'
    output.mkdir()
    compiler = tmp_path / 'Inno Setup 6/ISCC.exe'
    calls = []
    def compile_stub(command, **kwargs):
        calls.append((command, kwargs))
        staged_output = Path(next(arg.split('=', 1)[1] for arg in command
                                  if arg.startswith('/DOutputRoot=')))
        (staged_output / 'MultiCamera_Setup_0.1.1_x64.exe').write_bytes(b'installer')
    monkeypatch.setattr(packaging.subprocess, 'run', compile_stub)
    result = packaging.installer(source, output, compiler, '0.1.1', 'normal')
    command, kwargs = calls[0]
    assert command[0] == str(compiler)
    assert f'/DDistDir={source.resolve()}' in command
    staged_output = Path(next(arg.split('=', 1)[1] for arg in command
                              if arg.startswith('/DOutputRoot=')))
    assert staged_output.parent == output.resolve()
    assert not staged_output.exists()
    assert '/DMyCompression=lzma2/normal' in command
    assert kwargs == {'check': True}
    assert result['output_bytes'] > 0


def test_success_exit_without_new_installer_cannot_reuse_old_binary(source, tmp_path, monkeypatch):
    output = tmp_path / 'output'
    output.mkdir()
    previous = output / 'MultiCamera_Setup_0.1.1_x64.exe'
    previous.write_bytes(b'previous installer')
    monkeypatch.setattr(packaging.subprocess, 'run', lambda *args, **kwargs: None)
    with pytest.raises(RuntimeError, match='non-empty'):
        packaging.installer(source, output, tmp_path / 'ISCC.exe', '0.1.1')
    assert previous.read_bytes() == b'previous installer'
    assert not list(output.glob('.installer-*'))


def test_invalid_input_and_recursive_output_rejected(source, tmp_path):
    with pytest.raises(ValueError, match='outside'):
        packaging.package(source, source / 'output')
    with pytest.raises(ValueError, match='compiler'):
        packaging.package(source, tmp_path / 'output', installer_only=True)
    (source / 'MultiCamera.exe').write_bytes(b'')
    with pytest.raises(ValueError, match='non-empty'):
        packaging.package(source, tmp_path / 'output')


def test_cli_works_from_another_working_directory(source, tmp_path):
    result = subprocess.run([sys.executable, str(ROOT / 'scripts/package_windows.py'),
        '--source', str(source), '--output', str(tmp_path / 'output')],
        cwd=tmp_path, capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)['status'] == 'passed'


def test_scipy_collection_keeps_runtime_but_excludes_tests():
    tree = ast.parse((ROOT / 'MultiCamera.spec').read_text(encoding='utf-8'))
    filters = [keyword.value for node in ast.walk(tree) if isinstance(node, ast.Call)
               for keyword in node.keywords if keyword.arg == 'filter']
    assert len(filters) == 1
    expression = ast.Expression(filters[0])
    predicate = eval(compile(expression, '<spec filter>', 'eval'))
    assert predicate('scipy.spatial.transform._rotation')
    assert predicate('scipy.optimize._lsq.least_squares')
    assert not predicate('scipy.spatial.tests.test_distance')
    assert not predicate('scipy.optimize.benchmarks.optimize')


def test_workflow_keeps_release_gates_and_fast_packaging_contract():
    workflow = (ROOT / '.github/workflows/build-windows.yml').read_text(encoding='utf-8')
    assert 'python -m pytest -q tests' in workflow
    assert '--self-test-calibration' in workflow and 'test_packaged_calibration.ps1' in workflow
    assert workflow.index('test_packaged_calibration.ps1') < workflow.index('scripts/package_windows.py')
    assert workflow.index('scripts/package_windows.py') < workflow.index('Publish Windows Release')
    assert 'packaging_metrics.json' in workflow
    assert 'Compress-Archive' not in workflow and 'pip install -U' not in workflow
    assert 'cache: "pip"' in workflow and 'cache: "venv"' not in workflow
