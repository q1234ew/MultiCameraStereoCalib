"""Release publication reconciles ambiguous writes without replacing any remote content."""

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from urllib.parse import parse_qs, urlsplit

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    'windows_release_publish', ROOT / 'scripts/publish_windows_release.py')
release = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = release
SPEC.loader.exec_module(release)
COMMIT = 'a' * 40


def error(status=502, *, network=False):
    return release.ApiError(f'injected failure (HTTP {status})', status=status, network=network)


class FakeGh:
    def __init__(self):
        self.release = None
        self.assets = []
        self.other_releases = []
        self.tag_commit = COMMIT
        self.faults = {}
        self.hooks = {}
        self.calls = []

    def request(self, method, endpoint, *, payload=None, asset=None):
        parsed = urlsplit(endpoint)
        path = parsed.path.lstrip('/')
        query = parse_qs(parsed.query)
        suffix = path.removeprefix('repos/owner/project/')
        if suffix.startswith('git/ref/tags/'):
            stage = 'tag_commit'
        elif suffix.startswith('releases/tags/'):
            stage = 'find_tag'
        elif suffix == 'releases':
            stage = 'create' if method == 'POST' else 'list_releases'
        elif suffix == 'releases/11/assets':
            stage = 'upload' if method == 'POST' else 'list_assets'
        elif suffix == 'git/tags/' + 'b' * 40:
            stage = 'annotated_tag'
        elif suffix == 'releases/11':
            stage = 'publish' if method == 'PATCH' else 'read_release'
        else:
            raise AssertionError((method, endpoint))
        self.calls.append((stage, method, endpoint, deepcopy(payload)))
        fault = self.faults.get(stage, []).pop(0) if self.faults.get(stage) else None
        if fault and fault[0] == 'before':
            raise fault[1]
        result = self._perform(stage, payload, asset, query)
        if stage in self.hooks:
            self.hooks[stage](self)
        if fault and fault[0] == 'after':
            raise fault[1]
        return deepcopy(result)

    def _perform(self, stage, payload, asset, query):
        if stage == 'tag_commit':
            if self.tag_commit is None:
                raise error(404)
            return dict(object=dict(type='commit', sha=self.tag_commit))
        if stage == 'find_tag':
            if self.release is None or self.release['draft']:
                raise error(404)
            return self.release
        if stage in ('list_releases', 'list_assets'):
            rows = (self.other_releases + ([self.release] if self.release else [])
                    if stage == 'list_releases' else self.assets)
            start = (int(query.get('page', ['1'])[0]) - 1) * 100
            return rows[start:start + 100]
        if stage == 'read_release':
            if self.release is None:
                raise error(404)
            return self.release
        if stage == 'create':
            if self.release is not None:
                raise error(422)
            self.release = dict(payload, id=11)
            return self.release
        if stage == 'upload':
            name = query['name'][0]
            assert name == asset.path.name
            if any(row['name'] == name for row in self.assets):
                raise error(422)
            self.assets.append(dict(name=name, size=asset.size, digest=asset.digest,
                                    state='uploaded'))
            return self.assets[-1]
        if stage == 'publish':
            assert self.release['draft']
            assert payload == {'draft': False, 'make_latest':
                               'false' if self.release['prerelease'] else 'true'}
            self.release['draft'] = False
            self.tag_commit = self.release['target_commitish']
            return self.release
        if stage == 'annotated_tag':
            return dict(object=dict(type='commit', sha=COMMIT))
        raise AssertionError(stage)

    def count(self, stage):
        return sum(call[0] == stage for call in self.calls)

    @property
    def mutations(self):
        return [call for call in self.calls if call[1] != 'GET']


@pytest.fixture
def case(tmp_path):
    paths = [tmp_path / 'MultiCamera_Setup_0.6.0_x64.exe', tmp_path / 'SHA256SUMS.txt']
    for index, path in enumerate(paths):
        path.write_bytes(b'local artifact ' + str(index).encode())
    api, sleeps = FakeGh(), []
    options = dict(repo='owner/project', tag='v0.6.0-build.20261104.1', commit=COMMIT,
                   title='Windows preview', notes='Release notes\n', run_id='38', assets=paths,
                   prerelease=True, attempts=3, api=api, sleep=sleeps.append)
    publisher = release.Publisher(**options)
    return publisher, api, sleeps, options


def draft(publisher, api):
    api.release = dict(id=11, tag_name=publisher.tag, target_commitish=publisher.commit,
        body=publisher.body, name=publisher.title, prerelease=publisher.prerelease, draft=True)


def remote_asset(asset, **changes):
    return dict(name=asset.path.name, size=asset.size, digest=asset.digest,
                state='uploaded', **changes)


def test_new_release_uploads_then_publishes_and_preserves_notes(case):
    publisher, api, sleeps, _ = case
    url = publisher.run()
    assert url == f'https://github.com/owner/project/releases/tag/{publisher.tag}'
    assert api.count('create') == api.count('publish') == 1
    assert api.count('upload') == 2 and not sleeps
    assert api.release['body'].startswith('Release notes\n\n\n<!-- multicamera-publisher:')
    assert '"run_id":"38"' in api.release['body']
    assert COMMIT in api.release['body']
    assert all(call[1] in ('POST', 'PATCH') for call in api.mutations)
    assert not api.release['draft']


@pytest.mark.parametrize('status', [429, 500, 502, 503, 504])
def test_transient_create_failure_retries_after_read_only_absence_check(case, status):
    publisher, api, sleeps, _ = case
    api.faults['create'] = [('before', error(status))]
    publisher.run()
    assert api.count('create') == 2 and sleeps == [2]
    creates = [index for index, call in enumerate(api.calls) if call[0] == 'create']
    assert any(call[0] == 'list_releases' for call in api.calls[creates[0] + 1:creates[1]])


@pytest.mark.parametrize('failure', [error(), error(None, network=True)])
def test_lost_create_response_reconciles_own_draft_without_second_create(case, failure):
    publisher, api, sleeps, _ = case
    api.faults['create'] = [('after', failure)]
    publisher.run()
    assert api.count('create') == 1 and api.count('publish') == 1 and not sleeps


@pytest.mark.parametrize('stage', ['tag_commit', 'find_tag', 'list_releases'])
def test_get_502_never_means_absent_and_exhausts_bounded_read_budget(case, stage):
    publisher, api, sleeps, _ = case
    api.faults[stage] = [('before', error())] * 8
    with pytest.raises(release.PublishError, match='budget exhausted'):
        publisher.run()
    assert api.count(stage) == 3 and sleeps == [2, 4] and not api.mutations


def test_get_502_can_recover_but_does_not_trigger_a_create_on_its_own(case):
    publisher, api, sleeps, _ = case
    draft(publisher, api)
    api.faults['find_tag'] = [('before', error())]
    publisher.run()
    assert api.count('create') == 0 and sleeps == [2]


@pytest.mark.parametrize('stage', ['tag_commit', 'find_tag', 'list_releases', 'create', 'upload',
                                  'publish'])
@pytest.mark.parametrize('status', [401, 403])
def test_auth_failure_is_fatal_without_retry_or_subsequent_mutation(case, stage, status):
    publisher, api, sleeps, _ = case
    api.faults[stage] = [('before', error(status))]
    with pytest.raises(release.ApiError) as failure:
        publisher.run()
    assert failure.value.status == status and api.count(stage) == 1 and not sleeps
    assert api.calls[-1][0] == stage


def test_initial_published_release_is_never_modified_even_when_identity_matches(case):
    publisher, api, _, _ = case
    draft(publisher, api)
    api.release['draft'] = False
    before = deepcopy(api.release)
    with pytest.raises(release.PublishError, match='already published'):
        publisher.run()
    assert api.release == before and not api.mutations


@pytest.mark.parametrize('field,value', [('body', 'foreign notes'), ('target_commitish', 'b' * 40),
    ('tag_name', 'v-other'), ('name', 'another title'), ('prerelease', False)])
def test_foreign_draft_or_identity_changes_cannot_be_reused(case, field, value):
    publisher, api, _, _ = case
    draft(publisher, api)
    if field == 'tag_name':
        # A by-ID response changing the tag must also be rejected before upload.
        api.hooks['read_release'] = lambda fake: fake.release.update(tag_name=value)
    else:
        api.release[field] = value
    with pytest.raises(release.PublishError, match='identity differs'):
        publisher.run()
    assert not api.mutations


@pytest.mark.parametrize('change', ['run_id', 'commit', 'notes', 'assets'])
def test_same_tag_draft_from_another_run_or_manifest_is_rejected(case, change):
    publisher, api, _, options = case
    draft(publisher, api)
    modified = dict(options)
    if change == 'assets':
        modified['assets'][0].write_bytes(b'rebuilt, different binary')
    else:
        modified[change] = {'run_id': '39', 'commit': 'b' * 40, 'notes': 'changed notes'}[change]
    another = release.Publisher(**modified)
    if change == 'commit':
        api.tag_commit = 'b' * 40
    with pytest.raises(release.PublishError, match='identity differs'):
        another.run()
    assert not api.mutations


@pytest.mark.parametrize('stage', ['upload', 'publish'])
@pytest.mark.parametrize('failure', [error(), error(None, network=True)])
def test_lost_write_response_verifies_completed_state_without_repeating_write(case, stage, failure):
    publisher, api, sleeps, _ = case
    api.faults[stage] = [('after', failure)]
    publisher.run()
    assert api.count('upload') == 2 and api.count('publish') == 1 and not sleeps
    assert not api.release['draft']


@pytest.mark.parametrize('status', [409, 422])
def test_nontransient_create_failure_is_not_retried(case, status):
    publisher, api, sleeps, _ = case
    api.faults['create'] = [('before', error(status))]
    with pytest.raises(release.ApiError):
        publisher.run()
    assert api.count('create') == 1 and not sleeps


@pytest.mark.parametrize('stage', ['create', 'upload', 'publish'])
def test_mutation_retry_budget_is_finite_and_never_deletes_partial_state(case, stage):
    publisher, api, sleeps, _ = case
    api.faults[stage] = [('before', error())] * 8
    with pytest.raises(release.PublishError, match='budget exhausted'):
        publisher.run()
    assert api.count(stage) == 3 and sleeps == [2, 4]
    assert all(call[1] != 'DELETE' for call in api.calls)
    if stage != 'create':
        assert api.release['draft'] is True


@pytest.mark.parametrize('field,value', [('digest', None), ('digest', 'sha256:' + '0' * 64),
    ('size', 1), ('state', 'starter')])
def test_existing_asset_requires_exact_digest_size_and_uploaded_state(case, field, value):
    publisher, api, _, _ = case
    draft(publisher, api)
    row = remote_asset(publisher.assets[0])
    row[field] = value
    api.assets = [row]
    before = deepcopy(api.assets)
    with pytest.raises(release.PublishError, match='refusing replacement'):
        publisher.run()
    assert api.assets == before and not api.mutations


def test_existing_identical_assets_can_be_skipped_only_in_own_draft(case):
    publisher, api, _, _ = case
    draft(publisher, api)
    api.assets = [remote_asset(asset) for asset in publisher.assets]
    publisher.run()
    assert api.count('create') == api.count('upload') == 0 and api.count('publish') == 1


def test_lost_upload_with_missing_digest_fails_without_retry_or_publish(case):
    publisher, api, _, _ = case
    api.faults['upload'] = [('after', error())]
    api.hooks['upload'] = lambda fake: fake.assets[-1].pop('digest')
    with pytest.raises(release.PublishError, match='refusing replacement'):
        publisher.run()
    assert api.count('upload') == 1 and api.count('publish') == 0


def test_unexpected_assets_are_preserved_and_stop_publication(case):
    publisher, api, _, _ = case
    draft(publisher, api)
    api.assets = [dict(name='foreign.exe', size=1, digest='sha256:foreign', state='uploaded')]
    before = deepcopy(api.assets)
    with pytest.raises(release.PublishError, match='Unexpected'):
        publisher.run()
    assert api.assets == before and not api.mutations


def test_publish_response_loss_cannot_succeed_when_asset_changed(case):
    publisher, api, _, _ = case
    api.faults['publish'] = [('after', error())]
    api.hooks['publish'] = lambda fake: fake.assets[0].update(digest='sha256:wrong')
    with pytest.raises(release.PublishError, match='refusing replacement'):
        publisher.run()
    assert api.count('publish') == 1


def test_publish_response_loss_checks_local_files_again_before_success(case):
    publisher, api, _, _ = case
    api.faults['publish'] = [('after', error())]
    api.hooks['publish'] = lambda fake: publisher.assets[0].path.write_bytes(b'local change')
    with pytest.raises(release.PublishError, match='Local asset changed'):
        publisher.run()
    assert api.count('publish') == 1


def test_create_response_loss_cannot_adopt_foreign_draft(case):
    publisher, api, _, _ = case
    api.faults['create'] = [('after', error())]
    api.hooks['create'] = lambda fake: fake.release.update(body='another run')
    with pytest.raises(release.PublishError, match='identity differs'):
        publisher.run()
    assert api.count('create') == 1 and api.count('upload') == api.count('publish') == 0


def test_local_changes_during_upload_prevent_publish(case):
    publisher, api, _, _ = case
    api.hooks['upload'] = lambda fake: publisher.assets[0].path.write_bytes(b'changed locally')
    with pytest.raises(release.PublishError, match='Local asset changed'):
        publisher.run()
    assert api.count('publish') == 0 and api.release['draft'] is True


@pytest.mark.parametrize('kind', ['empty', 'directory', 'missing', 'limit', 'duplicate', 'no_assets'])
def test_all_local_assets_are_validated_before_any_remote_call(case, tmp_path, monkeypatch, kind):
    _, api, _, options = case
    options = dict(options)
    path = tmp_path / 'invalid'
    if kind == 'empty':
        path.touch()
    elif kind == 'directory':
        path.mkdir()
    elif kind == 'limit':
        monkeypatch.setattr(release, 'MAX_ASSET_BYTES', 3)
        path.write_bytes(b'abc')
    if kind == 'duplicate':
        other = tmp_path / 'another'
        other.mkdir()
        second = other / options['assets'][0].name
        second.write_bytes(b'different')
        options['assets'] = [options['assets'][0], second]
    else:
        options['assets'] = [] if kind == 'no_assets' else [path]
    with pytest.raises((release.PublishError, OSError)):
        release.Publisher(**options).run()
    assert not api.calls


def test_annotated_tag_is_peeled_and_a_wrong_tag_never_creates_release(case):
    publisher, api, _, _ = case
    original = api._perform
    def annotated(stage, payload, asset, query):
        if stage == 'tag_commit':
            return dict(object=dict(type='tag', sha='b' * 40))
        return original(stage, payload, asset, query)
    api._perform = annotated
    publisher.run()
    assert api.count('annotated_tag') >= 2
    other, bad, _, _ = case
    bad.release = None
    bad._perform = original
    bad.tag_commit = 'c' * 40
    bad.calls.clear()
    with pytest.raises(release.PublishError, match='Remote tag'):
        other.run()
    assert not bad.mutations


def test_draft_inventory_is_paginated_before_create(case):
    publisher, api, _, _ = case
    draft(publisher, api)
    api.other_releases = [dict(tag_name=f'other-{index}') for index in range(100)]
    publisher.run()
    assert api.count('list_releases') == 2 and api.count('create') == 0


@pytest.mark.parametrize('stderr,status,retryable', [
    (b'gh: Bad Gateway (HTTP 502)', 502, True),
    (b'gh: unauthorized (HTTP 401)', 401, False),
    (b'gh: forbidden (HTTP 403)', 403, False),
    (b'gh: Not Found (HTTP 404)', 404, False),
    (b'dial tcp: connection reset by peer', None, True),
    (b'net/http: timeout awaiting response headers', None, True),
    (b'unknown flag: --timeout', None, False),
    (b'unknown flag: --unsupported', None, False),
])
def test_gh_transport_classifies_errors_without_guessing_absence(monkeypatch, stderr, status,
                                                               retryable):
    monkeypatch.setattr(release.subprocess, 'run', lambda *a, **k:
                        subprocess.CompletedProcess(a[0], 1, b'', stderr))
    with pytest.raises(release.ApiError) as failure:
        release.GhApi().request('GET', 'repos/owner/project/releases/tags/v1')
    assert failure.value.status == status and failure.value.retryable is retryable


def test_gh_transport_uses_json_stdin_and_streamed_asset_file(monkeypatch, tmp_path):
    calls = []
    def execute(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0,
            b'HTTP/2.0 201 Created\r\nContent-Type: application/json\r\n\r\n{"id":11}', b'')
    monkeypatch.setattr(release.subprocess, 'run', execute)
    api = release.GhApi(timeout=123)
    assert api.request('POST', 'repos/owner/project/releases', payload={'body': 'notes'}) == {'id': 11}
    assert json.loads(calls[0][1]['input']) == {'body': 'notes'}
    path = tmp_path / 'file with space # 测试.zip'
    path.write_bytes(b'asset')
    api.request('POST', 'https://uploads.github.com/repos/owner/project/releases/11/assets',
                asset=release.Asset.inspect(path))
    command, kwargs = calls[1]
    assert command[command.index('--input') + 1] == str(path)
    assert kwargs['input'] is None and kwargs['timeout'] == 123
    assert '--clobber' not in command and '--hostname' in command


def test_timeout_is_uncertain_network_failure_and_missing_gh_is_fatal(monkeypatch):
    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(args[0], 1)
    monkeypatch.setattr(release.subprocess, 'run', timeout)
    with pytest.raises(release.ApiError) as failure:
        release.GhApi().request('POST', 'endpoint')
    assert failure.value.retryable
    def missing(*args, **kwargs):
        raise FileNotFoundError('gh')
    monkeypatch.setattr(release.subprocess, 'run', missing)
    with pytest.raises(release.PublishError, match='execute gh'):
        release.GhApi().request('GET', 'endpoint')


def test_cli_success_outputs_only_verified_url_and_handles_bom_notes(case, tmp_path, monkeypatch,
                                                                  capsys):
    publisher, api, _, options = case
    notes = tmp_path / 'notes.md'
    notes.write_text('\ufeffRelease notes\r\n', encoding='utf-8')
    monkeypatch.setattr(release, 'GhApi', lambda timeout: api)
    args = ['--repo', 'owner/project', '--tag', publisher.tag, '--commit', COMMIT,
            '--title', publisher.title, '--notes-file', str(notes), '--run-id', '38',
            '--prerelease', '--assets', *map(str, options['assets'])]
    assert release.main(args) == 0
    output = capsys.readouterr()
    assert output.out.strip() == f'https://github.com/owner/project/releases/tag/{publisher.tag}'
    assert not output.err
    assert api.release['body'] == publisher.body


def test_asset_limit_and_full_commit_requirement(case):
    _, api, _, options = case
    assert release.MAX_ASSET_BYTES == 2 * 1024**3
    with pytest.raises(release.PublishError, match='full lowercase'):
        release.Publisher(**dict(options, commit='abcdef0'))
    assert not api.calls


def test_network_create_with_failed_reconciliation_never_blindly_creates_again(case):
    publisher, api, sleeps, _ = case
    api.faults['create'] = [('after', error())]
    api.hooks['create'] = lambda fake: fake.faults.update(
        find_tag=[('before', error())] * 8)
    with pytest.raises(release.PublishError, match='budget exhausted'):
        publisher.run()
    assert api.count('create') == 1 and api.count('upload') == api.count('publish') == 0
    assert api.release['draft'] is True and sleeps == [2, 4]


def test_missing_tag_can_be_created_by_publish_and_is_verified_afterward(case):
    publisher, api, _, _ = case
    api.tag_commit = None
    publisher.run()
    assert api.tag_commit == COMMIT and api.count('create') == api.count('publish') == 1


def test_final_attempt_lost_publish_response_still_reconciles_success(case):
    publisher, api, sleeps, _ = case
    api.faults['publish'] = [('before', error()), ('before', error()), ('after', error())]
    publisher.run()
    assert api.count('publish') == 3 and sleeps == [2, 4]


def test_publish_readback_lag_is_reconciled_read_only_on_next_iteration(case):
    publisher, api, sleeps, _ = case
    api.faults['publish'] = [('after', error())]
    original = api._perform
    state = {'lag': True}
    def lagging(stage, payload, asset, query):
        result = original(stage, payload, asset, query)
        if stage == 'read_release' and api.release and not api.release['draft'] and state['lag']:
            state['lag'] = False
            return dict(result, draft=True)
        return result
    api._perform = lagging
    publisher.run()
    assert api.count('publish') == 1 and sleeps == [2]


@pytest.mark.parametrize('stage', ['upload', 'publish'])
def test_transient_write_before_commit_can_be_retried_without_clobber(case, stage):
    publisher, api, sleeps, _ = case
    api.faults[stage] = [('before', error(429))]
    publisher.run()
    assert api.count(stage) == (3 if stage == 'upload' else 2) and sleeps == [2]


def test_published_during_upload_is_not_edited_again(case):
    publisher, api, _, _ = case
    api.hooks['upload'] = lambda fake: fake.release.update(draft=False)
    with pytest.raises(release.PublishError, match='no longer a draft'):
        publisher.run()
    assert api.count('upload') == 1 and api.count('publish') == 0


def test_cli_failure_returns_nonzero_without_printing_a_release_url(case, tmp_path, monkeypatch,
                                                                 capsys):
    _, api, _, options = case
    api.faults['find_tag'] = [('before', error(403))]
    notes = tmp_path / 'notes.md'
    notes.write_text('notes', encoding='utf-8')
    monkeypatch.setattr(release, 'GhApi', lambda timeout: api)
    args = ['--repo', 'owner/project', '--tag', options['tag'], '--commit', COMMIT,
            '--title', options['title'], '--notes-file', str(notes), '--run-id', '38',
            '--assets', *map(str, options['assets'])]
    assert release.main(args) == 1
    output = capsys.readouterr()
    assert not output.out and 'HTTP 403' in output.err and not api.mutations


def test_cli_through_real_gh_transport_reconciles_fake_http_and_network_response_loss(
        case, tmp_path, monkeypatch, capsys):
    publisher, fake, _, options = case
    fake.faults.update(create=[('after', error(502))], upload=[('after', error(None, network=True))],
                       publish=[('after', error(502))])
    def execute(command, **kwargs):
        assert command[:2] == ['gh', 'api'] and '--clobber' not in command
        method = command[command.index('--method') + 1]
        payload, asset = None, None
        if '--input' in command:
            source = command[command.index('--input') + 1]
            if source == '-':
                payload = json.loads(kwargs['input'])
            else:
                assert 'Content-Type: application/octet-stream' in command
                asset = release.Asset.inspect(source)
        try:
            result = fake.request(method, command[2], payload=payload, asset=asset)
        except release.ApiError as exc:
            if exc.status is None:
                return subprocess.CompletedProcess(command, 1, b'', b'connection reset by peer')
            output = f'HTTP/2.0 {exc.status} Error\r\n\r\n'.encode() + b'{"message":"error"}'
            return subprocess.CompletedProcess(command, 1, output, b'')
        output = b'HTTP/2.0 200 OK\r\n\r\n' + json.dumps(result).encode()
        return subprocess.CompletedProcess(command, 0, output, b'')
    monkeypatch.setattr(release.subprocess, 'run', execute)
    notes = tmp_path / 'notes.md'
    notes.write_text('Release notes\n', encoding='utf-8')
    args = ['--repo', 'owner/project', '--tag', publisher.tag, '--commit', COMMIT,
            '--title', publisher.title, '--notes-file', str(notes), '--run-id', '38',
            '--prerelease', '--assets', *map(str, options['assets'])]
    assert release.main(args) == 0
    assert capsys.readouterr().out.strip().endswith('/releases/tag/' + publisher.tag)
    assert fake.count('create') == fake.count('publish') == 1 and fake.count('upload') == 2


@pytest.mark.parametrize('stage,status', [('create', 201), ('upload', 201), ('publish', 200)])
@pytest.mark.parametrize('returncode', [0, 1])
def test_success_headers_then_network_eof_reconcile_committed_write(
        case, monkeypatch, stage, status, returncode):
    publisher, fake, _, _ = case
    response = subprocess.CompletedProcess(['gh'], returncode,
        f'HTTP/2.0 {status} Success\r\nContent-Type: application/json\r\n\r\n{{"id":'.encode(),
        b'unexpected EOF')
    monkeypatch.setattr(release.subprocess, 'run', lambda *a, **k: response)
    with pytest.raises(release.ApiError) as caught:
        release.GhApi().request('PATCH' if stage == 'publish' else 'POST', 'fake-endpoint')
    assert caught.value.status == status and caught.value.retryable
    fake.faults[stage] = [('after', caught.value)]
    publisher.run()
    assert fake.count('create') == fake.count('publish') == 1 and fake.count('upload') == 2


@pytest.mark.parametrize('status', [401, 403, 404, 409, 422])
def test_explicit_http_errors_do_not_become_retryable_from_network_stderr(monkeypatch, status):
    response = subprocess.CompletedProcess(['gh'], 1,
        f'HTTP/2.0 {status} Error\r\n\r\n'.encode(), b'unexpected EOF')
    monkeypatch.setattr(release.subprocess, 'run', lambda *a, **k: response)
    with pytest.raises(release.ApiError) as caught:
        release.GhApi().request('GET', 'fake-endpoint')
    assert caught.value.status == status and not caught.value.retryable


def test_each_upload_and_publish_is_immediately_preceded_by_owned_release_read(case):
    publisher, fake, _, _ = case
    publisher.run()
    for index, call in enumerate(fake.calls):
        if call[0] in ('upload', 'publish'):
            assert fake.calls[index - 1][0:2] == ('read_release', 'GET')


@pytest.mark.parametrize('change', ['published', 'notes'])
@pytest.mark.parametrize('window', ['hash', 'inventory'])
def test_upload_rechecks_draft_and_identity_after_hash_or_inventory(
        case, monkeypatch, change, window):
    publisher, fake, _, _ = case
    def mutate():
        if change == 'published':
            fake.release['draft'] = False
        else:
            fake.release['body'] = 'foreign edited notes'
    if window == 'hash':
        original = release.Asset.verify_local
        def hash_then_mutate(asset):
            original(asset)
            mutate()
        monkeypatch.setattr(release.Asset, 'verify_local', hash_then_mutate)
    else:
        fake.hooks['list_assets'] = lambda api: mutate()
    with pytest.raises(release.PublishError):
        publisher.run()
    assert fake.count('upload') == fake.count('publish') == 0
    assert fake.calls[-1][0] == 'read_release'


@pytest.mark.parametrize('change', ['published', 'notes'])
@pytest.mark.parametrize('window', ['hash', 'inventory', 'tag'])
def test_publish_rechecks_draft_and_identity_after_all_lengthy_verification(
        case, monkeypatch, change, window):
    publisher, fake, _, _ = case
    state = {'complete_inventories': 0}
    def mutate():
        if change == 'published':
            fake.release['draft'] = False
        else:
            fake.release['body'] = 'foreign edited notes'
    if window == 'hash':
        original = release.Asset.verify_local
        def hash_then_mutate(asset):
            original(asset)
            if len(fake.assets) == len(publisher.assets):
                mutate()
        monkeypatch.setattr(release.Asset, 'verify_local', hash_then_mutate)
    elif window == 'tag':
        fake.hooks['tag_commit'] = lambda api: mutate() if api.release else None
    else:
        def after_inventory(api):
            if len(api.assets) == len(publisher.assets):
                state['complete_inventories'] += 1
                if state['complete_inventories'] == 2:
                    mutate()
        fake.hooks['list_assets'] = after_inventory
    with pytest.raises(release.PublishError):
        publisher.run()
    assert fake.count('upload') == len(publisher.assets) and fake.count('publish') == 0
    assert fake.calls[-1][0] == 'read_release'


def test_committed_create_with_one_invisible_read_is_found_before_next_post(case):
    publisher, fake, sleeps, _ = case
    fake.faults['create'] = [('after', error(502))]
    original = fake._perform
    state = {'lag': True}
    def invisible_once(stage, payload, asset, query):
        if stage == 'list_releases' and fake.release is not None and state['lag']:
            state['lag'] = False
            return []
        return original(stage, payload, asset, query)
    fake._perform = invisible_once
    publisher.run()
    assert fake.count('create') == 1 and sleeps == [2]


@pytest.mark.parametrize('status', [409, 422])
def test_committed_create_with_longer_read_lag_reconciles_conflict_without_another_retry(case,
                                                                                    status):
    publisher, fake, sleeps, _ = case
    fake.faults['create'] = [('after', error(502)), ('before', error(status))]
    original = fake._perform
    state = {'hidden_reads': 2}
    def invisible_twice(stage, payload, asset, query):
        if stage == 'list_releases' and fake.release is not None and state['hidden_reads']:
            state['hidden_reads'] -= 1
            return []
        return original(stage, payload, asset, query)
    fake._perform = invisible_twice
    publisher.run()
    assert fake.count('create') == 2 and sleeps == [2]


@pytest.mark.parametrize('status', [409, 422])
@pytest.mark.parametrize('foreign', ['notes', 'commit', 'published'])
def test_conflicting_create_only_adopts_matching_owned_draft(case, status, foreign):
    publisher, fake, sleeps, _ = case
    original = fake._perform
    def conflict(stage, payload, asset, query):
        if stage == 'create':
            draft(publisher, fake)
            if foreign == 'notes':
                fake.release['body'] = 'foreign edited notes'
            elif foreign == 'commit':
                fake.release['target_commitish'] = 'b' * 40
            else:
                fake.release['draft'] = False
            raise error(status)
        return original(stage, payload, asset, query)
    fake._perform = conflict
    with pytest.raises(release.PublishError):
        publisher.run()
    assert fake.count('create') == 1 and not sleeps
    assert fake.count('upload') == fake.count('publish') == 0


def test_cli_help_explicitly_excludes_concurrent_manual_api_edits(capsys):
    with pytest.raises(SystemExit) as exited:
        release.main(['--help'])
    assert exited.value.code == 0
    output = capsys.readouterr().out
    assert 'compare-and-swap' in output and 'unsupported' in output
