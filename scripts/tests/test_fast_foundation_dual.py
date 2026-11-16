"""CPU checks for the pinned conversion gate; GPU equivalence runs on Orin."""
from pathlib import Path
from types import SimpleNamespace
import json
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[2]))
from scripts.jetson.fast_foundation_dual import (
    CHECKPOINT_SHA256, REVISION, build, check_identity, check_metadata, digest, error_metrics,
    import_reference,
)


def metadata():
    return {'checkpoint_sha256': CHECKPOINT_SHA256, 'source_revision': REVISION,
            'image_size': [352, 640], 'valid_iters': 8, 'max_disp': 192,
            'split_equivalence_passed': True, 'cv_group': 8, 'normalize': True}


@pytest.mark.parametrize(('key', 'value'), [
    ('checkpoint_sha256', 'other'), ('image_size', [544, 960]), ('valid_iters', 4),
    ('max_disp', 256), ('split_equivalence_passed', False),
    ('cv_group', 0), ('normalize', 'false'), ('source_revision', 'other'),
])
def test_reject_changed_conversion_contract(key, value):
    changed = metadata()
    changed[key] = value
    with pytest.raises(ValueError):
        check_metadata(changed)


def test_staged_export_does_not_claim_completed_validation():
    staged = metadata()
    staged.pop('split_equivalence_passed')
    staged['validation_strategy'] = 'staged-reference'
    check_metadata(staged)
    staged.pop('validation_strategy')
    with pytest.raises(ValueError, match='validation contract'):
        check_metadata(staged)


def test_export_parent_runs_components_sequentially_without_loading_cuda(tmp_path, monkeypatch):
    from scripts.jetson import fast_foundation_dual as dual

    calls = []
    args = SimpleNamespace(root=tmp_path, checkpoint=tmp_path / 'weights.pth',
                           directory=tmp_path / 'export', component=None, trace_device='cpu')
    monkeypatch.setattr(dual, 'check_identity', lambda *args: None)
    monkeypatch.setattr(dual, 'setup', lambda *args: pytest.fail('Parent must not load Torch'))

    def run(command, check):
        assert args.directory.is_dir()
        assert check
        calls.append(command)

    monkeypatch.setattr(dual.subprocess, 'run', run)
    dual.export(args)
    assert [c[c.index('--component') + 1] for c in calls] == ['feature', 'post']
    assert all(c[c.index('--trace-device') + 1] == 'cpu' for c in calls)
    with pytest.raises(FileExistsError):
        dual.export(args)


def test_serial_feature_runner_preserves_left_storage_across_right_execution():
    from scripts.jetson.fast_foundation_dual import DualEngineRunner, FEATURES

    class Buffer:
        shape = (1, 3, 352, 640)
        device = 'test-device'

        def __init__(self, value=0):
            self.value = value

        def copy_(self, other):
            self.value = other.value

    feature_buffers = {name: Buffer() for name in ('image', *FEATURES[:4], FEATURES[-1])}
    post_buffers = {name: Buffer() for name in (*FEATURES, 'gwc_volume', 'disp')}

    def feature_run():
        for name in (*FEATURES[:4], FEATURES[-1]):
            feature_buffers[name].value = feature_buffers['image'].value * 10

    def kernel(left, right, output, normalize):
        assert left.value != right.value
        output.value = left.value + right.value

    def post_run():
        assert post_buffers[FEATURES[0]].value == post_buffers[FEATURES[-1]].value
        post_buffers['disp'].value = post_buffers['gwc_volume'].value

    runner = DualEngineRunner.__new__(DualEngineRunner)
    runner.serial = True
    runner.feature = SimpleNamespace(buffers=feature_buffers, run=feature_run)
    runner.post = SimpleNamespace(buffers=post_buffers, run=post_run, metadata={'normalize': False})
    runner.gwc = kernel
    assert runner(Buffer(2), Buffer(3)).value == 50
    assert runner(Buffer(4), Buffer(5)).value == 90


@pytest.mark.parametrize('tamper', ['none', 'disparity', 'input', 'iterations'])
def test_cached_full_eager_reference_requires_exact_provenance(tmp_path, tamper):
    source, destination = tmp_path / 'old', tmp_path / 'new'
    source.mkdir()
    destination.mkdir()
    left, right = tmp_path / 'left.png', tmp_path / 'right.png'
    left.write_bytes(b'left image')
    right.write_bytes(b'right image')
    disparity = np.ones((352, 640), dtype=np.float32)
    np.save(source / 'disparity_px.npy', disparity)
    np.save(destination / 'split_torch_disparity.npy', disparity[None, None])
    original = {'passed': True, 'identity': {
        'backend': 'torch', 'checkpoint_sha256': CHECKPOINT_SHA256,
        'source_revision': REVISION, 'input_width': 640, 'input_height': 352,
        'iterations': 4 if tamper == 'iterations' else 8, 'max_disp': 192,
        'eager_correlation': True}, 'left_sha256': digest(left), 'right_sha256': digest(right)}
    (source / 'report.json').write_text(json.dumps(original))
    args = SimpleNamespace(cached_reference=source, directory=destination, left=left, right=right,
                           reference_report_sha256=digest(source / 'report.json'),
                           reference_disparity_sha256=digest(source / 'disparity_px.npy'))
    if tamper == 'disparity':
        np.save(source / 'disparity_px.npy', disparity * 2)
    if tamper == 'input':
        left.write_bytes(b'another pair')
    if tamper == 'none':
        import_reference(args)
        result = json.loads((destination / 'reference.json').read_text())
        assert result['split_comparison']['passed']
        assert result['disparity_sha256'] == digest(destination / 'reference_disparity.npy')
    else:
        with pytest.raises(ValueError, match='reference'):
            import_reference(args)
        assert not (destination / 'reference.json').exists()


def test_build_resumes_verified_components_and_rejects_tampering(tmp_path, monkeypatch):
    from scripts.jetson import fast_foundation_dual as dual

    common = metadata()
    (tmp_path / 'export.json').write_text(json.dumps(common))
    source = Path(dual.__file__).with_name('fast_foundation_gwc.cu')
    library = tmp_path / 'libffs_gwc.so'
    library.write_bytes(b'previous kernel')
    (tmp_path / 'kernel.json').write_text(json.dumps({
        'source_sha256': digest(source), 'library_sha256': digest(library)}))
    for component in ('feature', 'post'):
        folder = tmp_path / component
        folder.mkdir()
        (folder / 'model.onnx').write_bytes(component.encode())
        exported = dict(common, component=component, onnx_sha256=digest(folder / 'model.onnx'))
        (folder / 'export.json').write_text(json.dumps(exported))
        (folder / 'model.engine').write_bytes(b'previous engine')
        (folder / 'build.json').write_text(json.dumps(dict(
            exported, engine_sha256=digest(folder / 'model.engine'))))

    def unexpected_build(*args, **kwargs):
        pytest.fail('Verified existing artifacts must be reused')

    monkeypatch.setattr(dual.subprocess, 'run', unexpected_build)
    build(SimpleNamespace(directory=tmp_path))
    (tmp_path / 'post/model.engine').write_bytes(b'tampered')
    with pytest.raises(ValueError, match='unverified engine'):
        build(SimpleNamespace(directory=tmp_path))


def test_numeric_gate_rejects_localized_large_errors():
    reference = np.zeros((1, 1, 10, 10), dtype=np.float32)
    actual = reference.copy()
    actual.flat[:2] = 3
    result = error_metrics(reference, actual)
    assert result['mae_px'] < .5
    assert result['p95_abs_px'] < 1
    assert result['fraction_gt_2px'] == .02
    assert not result['passed']


@pytest.mark.parametrize('bad', [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize('reference_bad', [False, True])
def test_numeric_gate_rejects_nonfinite_on_either_side(bad, reference_bad):
    reference = np.zeros((2, 2), dtype=np.float32)
    actual = reference.copy()
    (reference if reference_bad else actual)[0, 0] = bad
    assert error_metrics(reference, actual) == {'finite': False, 'passed': False}


def test_numeric_gate_checks_shape_and_promotes_half_subtraction():
    reference = np.full((2, 2), 65000, dtype=np.float16)
    actual = -reference
    assert error_metrics(reference, actual)['max_abs_px'] > 100000
    with pytest.raises(ValueError, match='shape'):
        error_metrics(reference, actual.ravel())
    assert error_metrics(reference, reference)['passed']


def test_pin_is_content_based_and_checks_upstream(tmp_path, monkeypatch):
    checkpoint = tmp_path / '20-30-48.pth'
    checkpoint.write_bytes(b'wrong checkpoint')
    (tmp_path / 'UPSTREAM_REVISION').write_text(REVISION)
    with pytest.raises(ValueError, match='checkpoint'):
        check_identity(tmp_path, checkpoint)
    assert digest(checkpoint) != CHECKPOINT_SHA256
    monkeypatch.setattr('scripts.jetson.fast_foundation_dual.digest',
                        lambda path: CHECKPOINT_SHA256)
    check_identity(tmp_path, checkpoint)
    (tmp_path / 'UPSTREAM_REVISION').write_text('other revision')
    with pytest.raises(ValueError, match='revision'):
        check_identity(tmp_path, checkpoint)
