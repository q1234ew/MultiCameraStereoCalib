"""Publish a verified Windows Release without replacing existing releases or assets.

No atomic compare-and-swap is available for these writes. Concurrent manual/API
release edits are unsupported; serialize all publishers and edits externally.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import time
from urllib.parse import quote, urlencode


MAX_ASSET_BYTES = 2 * 1024**3
API_VERSION = '2022-11-28'
NETWORK_ERRORS = re.compile(
    r'timed out|(?:i/o|tls handshake|request|read|write|connection) timeout|'
    r'timeout (?:awaiting|waiting)|client\.timeout|connection reset|connection refused|'
    r'unexpected eof|\beof\b|'
    r'temporary failure|no such host|network is unreachable|context deadline|'
    r'dial tcp|tls handshake|broken pipe|could not resolve|failed to connect|http2:', re.I)


class PublishError(RuntimeError):
    pass


class ApiError(PublishError):
    def __init__(self, message, *, status=None, network=False):
        super().__init__(message)
        self.status = status
        self.retryable = status == 429 or (status is not None and 500 <= status <= 599)
        self.retryable |= network and (status is None or 200 <= status < 300)


@dataclass(frozen=True)
class Asset:
    path: Path
    size: int
    digest: str

    @classmethod
    def inspect(cls, path):
        path = Path(path).resolve(strict=True)
        size = path.stat().st_size
        if not path.is_file() or not 0 < size < MAX_ASSET_BYTES:
            raise PublishError(f'Asset must be a nonempty file smaller than 2 GiB: {path.name}')
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(chunk)
        if path.stat().st_size != size:
            raise PublishError(f'Asset changed during hashing: {path.name}')
        return cls(path, size, 'sha256:' + digest.hexdigest())

    def verify_local(self):
        if Asset.inspect(self.path) != self:
            raise PublishError(f'Local asset changed: {self.path.name}')

    def verify_remote(self, remote):
        if (remote.get('name') != self.path.name or type(remote.get('size')) is not int
                or remote.get('size') != self.size
                or remote.get('digest') != self.digest or remote.get('state') != 'uploaded'):
            raise PublishError(f'Remote asset name/size/SHA256/state differs or is missing: '
                               f'{self.path.name}; refusing replacement')


class GhApi:
    """One gh invocation per request; retry decisions belong to the publisher."""

    def __init__(self, timeout=900):
        self.timeout = timeout

    def request(self, method, endpoint, *, payload=None, asset=None):
        command = ['gh', 'api', endpoint, '--hostname', 'github.com',
                   '--method', method, '--include',
                   '--header', 'Accept: application/vnd.github+json',
                   '--header', f'X-GitHub-Api-Version: {API_VERSION}']
        body = None
        if payload is not None:
            command += ['--input', '-']
            body = json.dumps(payload, ensure_ascii=False).encode('utf-8')
        if asset is not None:
            command += ['--input', str(asset.path), '--header',
                        'Content-Type: application/octet-stream']
        try:
            result = subprocess.run(command, input=body, capture_output=True,
                                    timeout=self.timeout, check=False)
        except subprocess.TimeoutExpired as exc:
            raise ApiError(f'gh {method} timed out; remote outcome unknown', network=True) from exc
        except OSError as exc:
            raise PublishError('Could not execute gh CLI') from exc
        output = result.stdout.replace(b'\r\n', b'\n')
        header, separator, response = output.partition(b'\n\n')
        match = re.match(rb'HTTP/\S+\s+(\d{3})(?:\s|$)', header)
        status = int(match[1]) if match else None
        if status is None:
            match = re.search(rb'\(HTTP (\d{3})\)', result.stderr)
            status = int(match[1]) if match else None
        network = bool(NETWORK_ERRORS.search(result.stderr.decode('utf-8', errors='replace')))
        if result.returncode or status is None or not 200 <= status < 300 or network:
            raise ApiError(f'gh {method} failed (HTTP {status or "unknown"})',
                           status=status, network=network)
        if not separator:
            raise PublishError('gh returned no HTTP response body')
        try:
            return json.loads(response)
        except (ValueError, UnicodeError) as exc:
            raise PublishError('gh returned invalid JSON; refusing an uncertain response') from exc


class Publisher:
    def __init__(self, *, repo, tag, commit, title, notes, run_id, assets, prerelease=False,
                 attempts=4, api=None, sleep=time.sleep):
        if not re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+', repo):
            raise PublishError('repo must be OWNER/REPO')
        if not re.fullmatch(r'[0-9a-f]{40}', commit):
            raise PublishError('commit must be the full lowercase 40-character commit SHA')
        if not tag or any(ord(char) < 32 for char in tag):
            raise PublishError('tag must be nonempty and contain no control characters')
        if not re.fullmatch(r'[1-9][0-9]*', str(run_id)):
            raise PublishError('run-id must be a positive GitHub Actions run ID')
        if not 1 <= attempts <= 6:
            raise PublishError('attempts must be between 1 and 6')
        self.assets = [Asset.inspect(path) for path in assets]
        names = [asset.path.name for asset in self.assets]
        if not names or len(names) != len(set(names)):
            raise PublishError('Provide at least one asset with unique filenames')
        self.repo, self.tag, self.commit = repo, tag, commit
        self.title, self.prerelease = title, prerelease
        self.attempts, self.api, self.sleep = attempts, api or GhApi(), sleep
        self.base = f'repos/{repo}'
        manifest = dict(repo=repo, run_id=str(run_id), commit=commit, tag=tag,
                        notes_sha256=hashlib.sha256(notes.encode('utf-8')).hexdigest(),
                        assets=sorted((asset.path.name, asset.size, asset.digest)
                                      for asset in self.assets))
        marker = json.dumps(manifest, sort_keys=True, ensure_ascii=True, separators=(',', ':'))
        self.body = notes + '\n\n<!-- multicamera-publisher:' + marker + ' -->\n'

    def _pause(self, attempt, error):
        if attempt >= self.attempts:
            raise PublishError(f'Retry budget exhausted: {error}') from error
        print(f'Transient failure; retry {attempt + 1}/{self.attempts}: {error}', file=sys.stderr)
        self.sleep(min(2**attempt, 30))

    def _get(self, endpoint, *, missing_ok=False):
        for attempt in range(1, self.attempts + 1):
            try:
                return self.api.request('GET', endpoint)
            except ApiError as exc:
                if missing_ok and exc.status == 404:
                    return None
                if not exc.retryable:
                    raise
                self._pause(attempt, exc)

    def _list(self, endpoint):
        rows = []
        for page in range(1, 101):
            batch = self._get(f'{endpoint}?per_page=100&page={page}')
            if not isinstance(batch, list):
                raise PublishError('Expected an API list response')
            rows.extend(batch)
            if len(batch) < 100:
                return rows
        raise PublishError('API pagination limit reached; refusing an incomplete inventory')

    def _find(self):
        release = self._get(f'{self.base}/releases/tags/{quote(self.tag, safe="")}',
                            missing_ok=True)
        if release is not None:
            return release
        # The tag endpoint does not reliably expose drafts; inventory them explicitly.
        matches = [row for row in self._list(f'{self.base}/releases')
                   if row.get('tag_name') == self.tag]
        if len(matches) > 1:
            raise PublishError('Multiple releases have the requested tag')
        return matches[0] if matches else None

    def _owned(self, release):
        if (not isinstance(release, dict) or type(release.get('id')) is not int
                or release['id'] <= 0 or release.get('tag_name') != self.tag
                or release.get('target_commitish') != self.commit
                or release.get('body') != self.body or release.get('name') != self.title
                or release.get('prerelease') is not self.prerelease
                or type(release.get('draft')) is not bool):
            raise PublishError('Release identity differs: expected this run, full commit, tag, '
                               'notes, title, prerelease flag, and asset manifest')
        return release

    def _read_owned(self, release_id):
        release = self._owned(self._get(f'{self.base}/releases/{release_id}'))
        if release['id'] != release_id:
            raise PublishError('Release ID changed')
        return release

    def _tag_commit(self, *, required=False):
        reference = self._get(f'{self.base}/git/ref/tags/{quote(self.tag, safe="")}',
                              missing_ok=not required)
        if reference is None:
            return
        obj = reference.get('object', {})
        for _ in range(16):
            if obj.get('type') == 'commit' and obj.get('sha') == self.commit:
                return
            if obj.get('type') != 'tag' or not re.fullmatch(r'[0-9a-f]{40}', obj.get('sha', '')):
                break
            obj = self._get(f'{self.base}/git/tags/{obj["sha"]}').get('object', {})
        raise PublishError('Remote tag does not resolve to the intended full commit')

    def _inventory(self, release_id, *, complete=False):
        expected = {asset.path.name: asset for asset in self.assets}
        found = {}
        for remote in self._list(f'{self.base}/releases/{release_id}/assets'):
            name = remote.get('name')
            if name not in expected or name in found:
                raise PublishError('Unexpected or duplicate remote asset; refusing any changes')
            expected[name].verify_remote(remote)
            found[name] = remote
        if complete and found.keys() != expected.keys():
            raise PublishError('Expected Release assets are missing')
        return found

    def _draft(self):
        payload = dict(tag_name=self.tag, target_commitish=self.commit, name=self.title,
                       body=self.body, draft=True, prerelease=self.prerelease)
        for attempt in range(1, self.attempts + 1):
            # Repeat discovery immediately before every POST, including after read lag.
            existing = self._find()
            if existing is not None:
                if existing.get('draft') is not True:
                    raise PublishError('An already published Release exists; use a new tag')
                return self._owned(existing)
            failure = None
            try:
                self.api.request('POST', f'{self.base}/releases', payload=payload)
            except ApiError as exc:
                if exc.status in (409, 422):
                    existing = self._find()
                    if existing is not None:
                        existing = self._owned(existing)
                        if not existing['draft']:
                            raise PublishError('Release was published before our publish step')
                        return existing
                    raise
                if not exc.retryable:
                    raise
                failure = exc
            # Reconcile even the final failed attempt: the server may have committed it.
            release = self._find()
            if release is not None:
                release = self._owned(release)
                if not release['draft']:
                    raise PublishError('Release was published before our publish step')
                return release
            if failure is None:
                raise PublishError('Successful create was not visible; refusing another create')
            self._pause(attempt, failure)
        raise PublishError('Could not obtain a verified draft')

    def _upload(self, release_id, asset):
        for attempt in range(1, self.attempts + 1):
            if asset.path.name in self._inventory(release_id):
                return
            asset.verify_local()
            endpoint = (f'https://uploads.github.com/{self.base}/releases/{release_id}/assets?'
                        + urlencode({'name': asset.path.name}))
            failure = None
            # Last remote operation before the write; hashing and inventory precede it.
            if not self._read_owned(release_id)['draft']:
                raise PublishError('Release is no longer a draft; refusing asset uploads')
            try:
                self.api.request('POST', endpoint, asset=asset)
            except ApiError as exc:
                if not exc.retryable:
                    raise
                failure = exc
            if asset.path.name in self._inventory(release_id):
                return
            if failure is None:
                raise PublishError('Successful upload was not visible; refusing another upload')
            self._pause(attempt, failure)

    def _published_url(self, release_id):
        self._inventory(release_id, complete=True)
        for asset in self.assets:
            asset.verify_local()
        self._tag_commit(required=True)
        if self._read_owned(release_id)['draft']:
            raise PublishError('Release is not published after verification')
        return f'https://github.com/{self.repo}/releases/tag/{quote(self.tag, safe="")}'

    def run(self):
        self._tag_commit()
        release_id = self._draft()['id']
        for asset in self.assets:
            self._upload(release_id, asset)
        for attempt in range(1, self.attempts + 1):
            if not self._read_owned(release_id)['draft']:
                if attempt == 1:
                    raise PublishError('Release was already published before our publish step')
                return self._published_url(release_id)
            self._inventory(release_id, complete=True)
            for asset in self.assets:
                asset.verify_local()
            self._tag_commit()
            failure = None
            # This narrows the race window, but is not an atomic conditional write.
            if not self._read_owned(release_id)['draft']:
                raise PublishError('Release is no longer a draft; refusing publication')
            try:
                self.api.request('PATCH', f'{self.base}/releases/{release_id}',
                                 payload={'draft': False,
                                          'make_latest': 'false' if self.prerelease else 'true'})
            except ApiError as exc:
                if not exc.retryable:
                    raise
                failure = exc
            release = self._read_owned(release_id)
            if release['draft'] is False:
                return self._published_url(release_id)
            if failure is None:
                raise PublishError('Successful publish still reports draft; refusing another edit')
            self._pause(attempt, failure)
        raise PublishError('Could not verify publication')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('repo', 'tag', 'commit', 'title', 'notes-file', 'run-id'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--assets', nargs='+', required=True, type=Path)
    parser.add_argument('--prerelease', action='store_true')
    parser.add_argument('--attempts', type=int, default=4)
    parser.add_argument('--request-timeout', type=int, default=900)
    args = parser.parse_args(argv)
    try:
        if not 1 <= args.request_timeout <= 3600:
            raise PublishError('request-timeout must be between 1 and 3600 seconds')
        notes = Path(args.notes_file).read_text(encoding='utf-8-sig')
        publisher = Publisher(repo=args.repo, tag=args.tag, commit=args.commit, title=args.title,
                              notes=notes, run_id=args.run_id, assets=args.assets,
                              prerelease=args.prerelease, attempts=args.attempts,
                              api=GhApi(timeout=args.request_timeout))
        print(publisher.run())
        return 0
    except (PublishError, OSError, ValueError) as exc:
        print(f'Release publication failed: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
