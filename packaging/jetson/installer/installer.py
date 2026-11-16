#!/usr/bin/env python3
"""Lightweight Orin deployment controller; models/environments are external resources. Python 3.8+, standard library only."""
import argparse
import contextlib
import datetime
import fcntl
import hashlib
import ipaddress
import json
import os
from pathlib import Path
import platform
import pwd
import re
import secrets
import shutil
import socket
import subprocess
import sys
import time
import urllib.request

ENVIRONMENTS = {'dashboard': '.venv', 'pose': '.pose-venv',
                'fusion': '.fusion-venv', 'foundation': '.foundation-venv'}
STATE_FILES = ['tracking/runtime.json', 'tracking/site.json', 'tracking/mqtt.json',
               'fusion/active_calibration.json']


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value, mode=0o600):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w') as stream:
        os.chmod(temporary, mode)
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')
    temporary.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def safe_relative(name):
    p = Path(name)
    if not isinstance(name, str) or p.is_absolute() or not p.parts or '..' in p.parts:
        raise ValueError('Unsafe manifest path: ' + str(name))
    return p


def verify_bundle(bundle):
    manifest = read_json(bundle / 'manifest.json')
    if manifest.get('schema_version') != 1 or not re.fullmatch(
            r'[A-Za-z0-9_.-]+', manifest.get('release_id', '')):
        raise ValueError('Invalid bundle manifest')
    for name, expected in manifest['files'].items():
        path = bundle / safe_relative(name)
        if path.is_symlink() or not path.is_file() or sha256(path) != expected:
            raise ValueError('Bundle integrity failed: ' + name)
    actual = {p.relative_to(bundle).as_posix() for p in bundle.rglob('*') if p.is_file()}
    expected = set(manifest['files']) | {'manifest.json', 'SHA256SUMS.txt'}
    if actual != expected or any(p.is_symlink() for p in bundle.rglob('*')):
        raise ValueError('Bundle contains unlisted files or symlinks')
    return manifest


def load_site(path):
    site = read_json(path)
    if site.get('schema_version') != 1:
        raise ValueError('Unsupported site schema')
    if not re.fullmatch(r'[a-z_][a-z0-9_-]*', site.get('user', '')):
        raise ValueError('Invalid service user')
    if not re.fullmatch(r'multicamera-[a-z0-9-]+', site.get('service_prefix', '')):
        raise ValueError('Invalid service prefix')
    for key in ['prefix', 'devices_file', 'calibration_dir']:
        p = Path(site[key])
        if not p.is_absolute() or '..' in p.parts or any(
                c.isspace() or c in '%"\\' for c in str(p)):
            raise ValueError('Expected safe absolute path: ' + key)
    if Path(site['prefix']) in [Path('/'), Path('/usr'), Path('/etc'), Path('/home')]:
        raise ValueError('Unsafe installation prefix')
    if site.get('environment_mode') not in ['reuse', 'provision']:
        raise ValueError('Unknown environment mode')
    if site['environment_mode'] == 'provision':
        cfg = site.get('provision', {})
        site['provision'] = cfg
        if set(cfg) - {'python310', 'build_jobs', 'asset_sources', 'pip_cache_dir', 'pip_index_url'}:
            raise ValueError('Unknown provisioning option')
        if type(cfg.get('build_jobs', 2)) is not int or not 1 <= cfg.get('build_jobs', 2) <= 4:
            raise ValueError('build_jobs must be 1–4')
        python = cfg.get('python310', 'build')
        if python != 'build' and (not isinstance(python, str) or not Path(python).is_absolute()):
            raise ValueError('python310 must be build or an absolute interpreter path')
        if cfg.get('pip_cache_dir') and not Path(cfg['pip_cache_dir']).is_absolute():
            raise ValueError('pip_cache_dir must be absolute')
        if cfg.get('pip_index_url') and not cfg['pip_index_url'].startswith('https://'):
            raise ValueError('pip_index_url must use HTTPS')
        sources = cfg.get('asset_sources', {})
        if not isinstance(sources, dict) or any(not isinstance(v, str) for v in sources.values()):
            raise ValueError('Invalid asset_sources')
        site['environments'] = {role: str(Path(site['prefix'])/'shared/pending'/role/'bin/python')
                                for role in ENVIRONMENTS}
        site['models'] = {'foundation_root': str(Path(site['prefix'])/'shared/pending/foundation'),
                          'pose_weights': str(Path(site['prefix'])/'shared/pending/pose.pt')}
    if set(site.get('environments', {})) != set(ENVIRONMENTS):
        raise ValueError('Provide all four environment interpreters')
    if set(site.get('models', {})) != {'foundation_root', 'pose_weights'}:
        raise ValueError('Provide FoundationStereo source root and pose weights')
    for value in list(site['environments'].values()) + list(site['models'].values()):
        if not Path(value).is_absolute() or any(c.isspace() or c in '%\"\\' for c in value):
            raise ValueError('Resource paths must be safe absolute paths')
    ipaddress.ip_address(site['listen'])
    for key in ['preview_port', 'tracking_port']:
        if type(site[key]) is not int or not 1024 <= site[key] <= 65535:
            raise ValueError('Invalid port: ' + key)
    if site['preview_port'] == site['tracking_port']:
        raise ValueError('Preview and tracking ports must differ')
    mqtt = site['mqtt']
    if mqtt.get('mode') not in ['disabled', 'external', 'managed']:
        raise ValueError('Invalid MQTT mode')
    if not re.fullmatch(r'[A-Za-z0-9_.-]+', mqtt.get('node_id', '')):
        raise ValueError('Invalid MQTT node')
    if type(mqtt.get('port')) is not int or not 1024 <= mqtt['port'] <= 65535:
        raise ValueError('Invalid MQTT port')
    ipaddress.ip_address(mqtt['listen'])
    return site


def run(args, cwd=None, env=None, user=None, timeout=900):
    command = [str(a) for a in args]
    if user and os.geteuid() == 0:
        command = ['runuser', '-u', user, '--'] + command
    subprocess.run(command, check=True, cwd=cwd, env=env, timeout=timeout)


def owned(path, site):
    if os.geteuid() == 0:
        identity = pwd.getpwnam(site['user'])
        for folder, directories, files in os.walk(path):
            os.chown(folder, identity.pw_uid, identity.pw_gid)
            for name in directories + files:
                child = os.path.join(folder, name)
                os.chown(child, identity.pw_uid, identity.pw_gid, follow_symlinks=False)


@contextlib.contextmanager
def deployment_lock(prefix):
    prefix.mkdir(parents=True, exist_ok=True)
    with (prefix / '.deployment.lock').open('a') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError('Another deployment is running')
        yield


def probe_environment(python, role, model_root=None):
    code = {
        'dashboard': 'import aiohttp,cv2,numpy; from scripts.jetson.preview import make_preview_app',
        'fusion': 'import open3d,cv2,plotly,numpy,scipy; from scripts.jetson.fusion_workflow import run_fusion',
        'pose': ('import torch,torchvision,ultralytics,lap,paho.mqtt.client; '
                 'assert torch.cuda.is_available(); '
                 'assert torchvision.ops.nms(torch.tensor([[0.,0.,1.,1.]],device="cuda"),'
                 'torch.ones(1,device="cuda"),.5).numel()==1'),
        'foundation': ('import os,sys; sys.path.insert(0,os.environ["FOUNDATION_STEREO_ROOT"]); '
                       'import torch; from core.foundation_stereo import FoundationStereo; '
                       'assert torch.cuda.is_available()'),
    }[role]
    return [str(python), '-c', code]


def health_report(site, release, live=False, gpu=False):
    checks = []
    def check(name, action, required=True):
        try:
            result = action()
            checks.append({'name': name, 'passed': True, 'required': required,
                           'detail': result if result is not None else 'ok'})
        except Exception as exc:
            checks.append({'name': name, 'passed': False, 'required': required,
                           'detail': str(exc)[:500]})
    env = dict(os.environ, HF_HUB_OFFLINE='1', PYTHONDONTWRITEBYTECODE='1',
               FOUNDATION_STEREO_ROOT=str(release / 'models/foundation'),
               OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', YOLO_AUTOINSTALL='false',
               YOLO_CONFIG_DIR=str(Path(site['prefix']) / 'shared/data/yolo-settings'))
    if site.get('managed_resources'):
        env.update(PYTHONNOUSERSITE='1', PYTHONPATH='')
    for role, folder in ENVIRONMENTS.items():
        if role in ['pose', 'foundation'] and not gpu:
            continue
        def probe(role=role, folder=folder):
            run(probe_environment(release / folder / 'bin/python', role),
                cwd=release, env=env, user=site['user'], timeout=120)
        check('environment_' + role, probe)
    def geometry_config():
        run([release / '.venv/bin/python', '-c',
             'from pathlib import Path; from scripts.jetson.core import load_config; '
             'load_config(Path("packaging/jetson/devices.json"))'], cwd=release, env=env,
            user=site['user'], timeout=15)
    check('devices_configuration', geometry_config)
    if live:
        host = site['listen'] if site['listen'] != '0.0.0.0' else '127.0.0.1'
        base = 'http://{}:{}'.format(host, site['preview_port'])
        def fetch(path):
            with urllib.request.urlopen(base + path, timeout=8) as response:
                return response.read()
        check('dashboard_http', lambda: bool(fetch('/')))
        def tracking():
            state = json.loads(fetch('/api/tracking/state'))
            if state.get('state') != 'running':
                raise ValueError('Tracking is not running')
            return {'state': state['state'], 'model': state.get('model')}
        check('tracking_http', tracking)
        def cameras():
            status = json.loads(fetch('/api/status'))
            devices = status.get('devices', [])
            online = [d['id'] for d in devices if all(
                any(c.get('role') == r and c.get('online') for c in d.get('channels', []))
                for r in ['left', 'right'])]
            if len(online) != len(devices) or not devices:
                raise ValueError('Some cameras are offline')
            return online
        check('camera_inputs', cameras, required=False)
        def association():
            state = json.loads(fetch('/api/tracking/state')).get('cross_camera', {})
            return {k: state.get(k) for k in ['state', 'geometry_ready', 'reason']}
        check('cross_camera_readiness', association, required=False)
        if site['mqtt']['mode'] != 'disabled':
            def mqtt():
                state = json.loads(fetch('/api/tracking/state')).get('mqtt', {})
                if not state.get('connected'):
                    raise ValueError('MQTT publisher is not connected')
                return 'connected'
            check('mqtt_connected', mqtt)
    if (release / 'assets.json').is_file():
        def model_identity():
            identity = verify_external_assets(site, release / 'assets.json')
            if not identity['passed']:
                raise ValueError('; '.join(identity['problems']))
            return identity['assets']
        check('external_model_identity', model_identity)
    return {'schema_version': 1, 'at_utc': datetime.datetime.now(
        datetime.timezone.utc).isoformat(), 'release': release.name,
        'passed': all(c['passed'] for c in checks if c['required']),
        'checks': checks, 'gpu_probed': gpu, 'live_probed': live,
        'measurement_accuracy_verified': False,
        'note': 'Software health is separate from physical calibration and identity accuracy.'}


def preflight(site, bundle, manifest):
    problems = []
    if platform.system() != 'Linux' or platform.machine() != 'aarch64':
        problems.append('Requires Linux aarch64 on a validated JetPack profile')
    l4t = Path('/etc/nv_tegra_release')
    requested = site.get('deployment_profile', 'jetpack5-l4t35.4.1-aarch64')
    raw = l4t.read_text() if l4t.exists() else ''
    actual = ('jetpack5-l4t35.4.1-aarch64' if re.search(r'R35 .*REVISION: 4\.1', raw) else
              'jetpack6-l4t36.4.3-aarch64' if re.search(r'R36 .*REVISION: 4\.3', raw) else None)
    if actual is None or requested not in ['auto', actual]:
        problems.append('L4T profile mismatch: configured=' + requested + ', actual=' + str(actual))
    try:
        identity = pwd.getpwnam(site['user'])
        if identity.pw_uid == 0:
            problems.append('Service user must be a non-root account')
        if os.geteuid() not in [0, identity.pw_uid]:
            problems.append('Install as configured service user or root')
    except KeyError:
        problems.append('Configured service user does not exist')
    if not Path(site['devices_file']).is_file() or not Path(site['calibration_dir']).is_dir():
        problems.append('Provide devices_file and calibration_dir for this physical site')
    if not manifest.get('external_resources'):
        problems.append('Expected lightweight bundle with external model/environment resources')
    fresh = site.get('environment_mode') == 'provision'
    if fresh:
        assets_report = {'assets': {}}
    else:
        assets_report = verify_external_assets(site, bundle / 'assets.json')
        problems.extend(assets_report['problems'])
        for role in ENVIRONMENTS:
            source = Path(site['environments'][role])
            if not source.is_file():
                problems.append('Missing environment interpreter: ' + role)
            if (source == Path(site['prefix']) or Path(site['prefix']) in source.parents) and not (
                    site.get('managed_resources') and Path(site['prefix'])/'shared/provisioned' in source.parents):
                problems.append('External interpreter must not point inside installation prefix')
    prefix = Path(site['prefix'])
    if prefix.is_symlink():
        problems.append('Installation prefix must not be a symlink')
    if prefix.exists() and any(prefix.iterdir()) and not (prefix / '.managed-installation.json').is_file():
        problems.append('Refuse nonempty directory without managed installation marker')
    marker = prefix / '.managed-installation.json'
    if marker.is_file() and read_json(marker).get('user') != site['user']:
        problems.append('Service user differs from installation owner')
    size = sum((bundle / p).stat().st_size for p in manifest['files'])
    target = Path(site['prefix'])
    while not target.exists():
        target = target.parent
    required = max((12 * 1024**3 if fresh else 256 * 1024**2), size * 4)
    if shutil.disk_usage(target).free < required:
        problems.append('Insufficient disk space; need at least {} GiB free'.format(
            round(required / 1024**3, 1)))
    if site['mqtt']['mode'] == 'managed' and not fresh:
        for executable in ['mosquitto', 'mosquitto_passwd']:
            if not shutil.which(executable):
                problems.append('Install OS prerequisite: ' + executable)
    if site['mqtt']['mode'] == 'external' and not Path(
            site['mqtt'].get('external_config') or '/nonexistent').is_file():
        problems.append('External MQTT requires protected external_config JSON')
    if fresh:
        if actual is None:
            return {'passed': False, 'problems': problems}
        info = load_provision(bundle).plan(site, bundle)
        return dict(info, passed=not problems, problems=problems)
    return {'passed': not problems, 'problems': problems,
            'runtime_profile': actual,
            'required_free_bytes': required, 'assets': assets_report['assets'],
            'models_copied_bytes': 0, 'environment_mode': 'reuse'}


def verify_external_assets(site, catalog_path):
    catalog = read_json(catalog_path)['assets']
    paths = {'foundation': Path(site['models']['foundation_root']) /
             catalog['foundation']['relative_checkpoint'],
             'pose': Path(site['models']['pose_weights'])}
    problems, assets = [], {}
    for role, path in paths.items():
        expected = catalog[role]
        if not path.is_file():
            problems.append('Missing model resource: ' + role)
            continue
        actual_size = path.stat().st_size
        if actual_size != expected['bytes']:
            problems.append('Model size differs: ' + role)
            continue
        actual_hash = sha256(path)
        if actual_hash != expected['sha256']:
            problems.append('Model hash differs: ' + role)
        assets[role] = {'bytes': actual_size, 'sha256': actual_hash,
                        'expected_sha256': expected['sha256'], 'external': True}
    root = Path(site['models']['foundation_root'])
    for relative in ['core/foundation_stereo.py', 'Utils.py', 'dinov2',
                     'pretrained_models/23-51-11/cfg.yaml']:
        if not (root / relative).exists():
            problems.append('Missing FoundationStereo source/config: ' + relative)
    return {'passed': not problems, 'problems': problems, 'assets': assets}


def link_resources(site, release):
    # The application code is versioned; heavy assets remain at the operator's paths.
    for role, folder in ENVIRONMENTS.items():
        source = Path(site['environments'][role])
        if source.parent.name == 'bin' and (source.parent.parent / 'pyvenv.cfg').is_file():
            (release / folder).symlink_to(source.parent.parent, target_is_directory=True)
        else:
            (release / folder / 'bin').mkdir(parents=True)
            (release / folder / 'bin/python').symlink_to(source)
    (release / 'models/pose').mkdir(parents=True)
    (release / 'models/foundation').symlink_to(site['models']['foundation_root'],
                                             target_is_directory=True)
    (release / 'models/pose/yolo26n-pose.pt').symlink_to(site['models']['pose_weights'])


def configure_mqtt(site, prefix):
    destination = prefix / 'shared/data/tracking/mqtt.json'
    if destination.exists():
        if site['mqtt']['mode'] == 'managed' and not (prefix / 'shared/mqtt/broker.conf').is_file():
            raise ValueError('Imported MQTT credentials belong to an external Broker; choose external mode')
        return  # Upgrades never rotate credentials or replace existing runtime options.
    mode = site['mqtt']['mode']
    if mode == 'external':
        write_json(destination, read_json(site['mqtt']['external_config']))
    elif mode == 'disabled':
        write_json(destination, {'enabled': False})
    else:
        mqtt = site['mqtt']
        directory = prefix / 'shared/mqtt'
        directory.mkdir(parents=True, exist_ok=True)
        passwords = {'mc-publisher': secrets.token_urlsafe(24),
                     'mc-viewer': secrets.token_urlsafe(24)}
        password_file = directory / 'passwords'
        password_file.write_text(''.join(k + ':' + v + '\n' for k, v in passwords.items()))
        password_file.chmod(0o600)
        # -U hashes an existing protected file without exposing passwords in argv.
        run(['mosquitto_passwd', '-U', password_file], timeout=15)
        topic = 'multicamera/' + mqtt['node_id']
        (directory / 'acl').write_text('user mc-publisher\ntopic write ' + topic +
            '/#\ntopic write wyzl/iot/service/device/fusionmap/report\n'
            'user mc-viewer\ntopic read ' + topic +
            '/#\ntopic read wyzl/iot/service/device/fusionmap/report\n')
        (directory / 'broker.conf').write_text(
            'listener {} {}\nallow_anonymous false\npassword_file {}\nacl_file {}\n'
            'persistence false\nlog_dest stderr\n'.format(
                mqtt['port'], mqtt['listen'], password_file, directory / 'acl'))
        write_json(destination, {'enabled': True, 'host': mqtt['listen'], 'port': mqtt['port'],
            'node_id': mqtt['node_id'], 'topic_prefix': topic,
            'username': 'mc-publisher', 'password': passwords['mc-publisher']})
        write_json(directory / 'viewer.json', {'host': mqtt['listen'], 'port': mqtt['port'],
            'username': 'mc-viewer', 'password': passwords['mc-viewer'], 'topic': topic + '/#'})


def prepare_release(site, bundle, manifest):
    prefix = Path(site['prefix'])
    release = prefix / 'releases' / manifest['release_id']
    if release.exists():
        installed = release / '.installation.json'
        if installed.exists() and read_json(installed).get('manifest_sha256') == sha256(
                bundle / 'manifest.json'):
            if read_json(release / '.site.json') != site:
                raise ValueError('Existing release uses a different site config; build a new release ID')
            report = health_report(site, release, gpu=True)
            if not report['passed']:
                raise ValueError('Existing staged environment no longer passes health')
            return release  # Exact repeat does not rebuild or rotate credentials.
        raise ValueError('Release directory exists but is incomplete/different; use a new bundle ID')
    release.mkdir(parents=True)
    write_json(prefix / '.managed-installation.json', {'schema_version': 1, 'user': site['user']})
    shutil.copytree(bundle / 'app', release, dirs_exist_ok=True)
    shutil.copy2(bundle / 'assets.json', release / 'assets.json')
    try:
        link_resources(site, release)
        shared = prefix / 'shared'
        (shared / 'data').mkdir(parents=True, exist_ok=True)
        imports = site.get('import_state_from')
        if imports:
            for relative in STATE_FILES:
                source, destination = Path(imports) / 'data' / relative, shared / 'data' / relative
                if source.is_file() and not destination.exists():
                    write_json(destination, read_json(source))
        config = release / 'packaging/jetson/devices.json'
        config.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(site['devices_file'], config)
        # Per-release calibration snapshot: retained for reproducibility and rollback.
        shutil.copytree(site['calibration_dir'], release / 'bundles')
        (release / 'data').symlink_to(shared / 'data', target_is_directory=True)
        (release / 'validation').mkdir()
        configure_mqtt(site, prefix)
        write_json(release / '.site.json', site)
        owned(prefix, site)
        report = health_report(site, release, gpu=True)
        report['assets'] = verify_external_assets(site, release / 'assets.json')
        report['passed'] &= report['assets']['passed']
        write_json(release / 'validation/install-health.json', report)
        if not report['passed']:
            raise ValueError('Environment smoke failed; see validation/install-health.json')
        write_json(release / '.installation.json', {'release_id': manifest['release_id'],
            'manifest_sha256': sha256(bundle / 'manifest.json'), 'phase': 'staged'})
        owned(release, site)
        return release
    except Exception:
        # Retain failed evidence; never touch the active release or delete user's state.
        write_json(release / 'installation-failed.json', {'phase': 'staging_failed'})
        raise


def unit_text(site, role):
    prefix = Path(site['prefix'])
    release = prefix / 'current'
    common = ('[Unit]\nDescription=MultiCamera managed {role}\n'
        'After=network-online.target\nWants=network-online.target\n\n[Service]\n'
        'Type=simple\nUser={user}\nWorkingDirectory={release}\n'
        'Environment=HOME={home}\n'
        'Environment=PYTHONDONTWRITEBYTECODE=1\nEnvironment=OPENBLAS_NUM_THREADS=1\n'
        'Environment=OMP_NUM_THREADS=1\nEnvironment=HF_HUB_OFFLINE=1\n'
        'Environment=YOLO_AUTOINSTALL=false\n'
        'Environment=FOUNDATION_STEREO_CHECKPOINT_SHA256=60e79bde9c6a00acea551625ff814fe06e5a6806e2c0c9829baee248de87c5f1\n'
        'Environment="FOUNDATION_STEREO_ROOT={release}/models/foundation"\n'
        'Environment="FOUNDATION_STEREO_PYTHON={release}/.foundation-venv/bin/python"\n'
        'Environment="FOUNDATION_STEREO_LOG={prefix}/shared/data/foundation-worker.log"\n'
        'Environment="MULTICAMERA_TRACKING_URL=http://127.0.0.1:{tracking_port}"\n'
        'Environment="YOLO_CONFIG_DIR={prefix}/shared/data/yolo-settings"\n').format(
            role=role, user=site['user'], release=release, prefix=prefix,
            home=pwd.getpwnam(site['user']).pw_dir, tracking_port=site['tracking_port'])
    if site.get('managed_resources'):
        common += 'Environment=PYTHONNOUSERSITE=1\nEnvironment=PYTHONPATH=\n'
    if role == 'preview':
        command = '"{r}/.venv/bin/python" -m scripts.jetson.preview --listen {host} --port {p} --bundle-root "{r}/bundles"'.format(
            r=release, host=site['listen'], p=site['preview_port'])
        memory = '10G'
    elif role == 'tracking':
        command = '"{r}/.pose-venv/bin/python" -m scripts.jetson.tracking_service --weights "{r}/models/pose/yolo26n-pose.pt" --port {p}'.format(r=release, p=site['tracking_port'])
        memory = '5G'
    else:
        command = '"{}" -c "{}"'.format(shutil.which('mosquitto'), prefix / 'shared/mqtt/broker.conf')
        memory = '256M'
    return common + 'ExecStart=' + command + '\nRestart=on-failure\nRestartSec=5\nTimeoutStopSec=35\nKillMode=control-group\nNoNewPrivileges=true\nUMask=0077\nMemoryAccounting=true\nMemoryLimit=' + memory + '\nMemoryMax=' + memory + '\n\n[Install]\nWantedBy=multi-user.target\n'


def switch_current(prefix, release):
    link = prefix / 'current.next'
    if link.exists() or link.is_symlink():
        link.unlink()
    link.symlink_to(release, target_is_directory=True)
    link.replace(prefix / 'current')


def service_names(site):
    roles = ['preview', 'tracking'] + (['mqtt'] if site['mqtt']['mode'] == 'managed' else [])
    return [site['service_prefix'] + '-' + role + '.service' for role in roles]


def activate(site, release):
    if os.geteuid() != 0:
        raise ValueError('--activate requires sudo; staged release is retained')
    prefix = Path(site['prefix'])
    old = (prefix / 'current').resolve() if (prefix / 'current').is_symlink() else None
    if old and old == release:
        report = health_report(site, release, live=True)
        if report['passed']:
            return report
    # Refuse a live port collision from another installation. Existing managed services
    # may own ports when this prefix is upgrading.
    if old is None:
        ports = [site['preview_port'], site['tracking_port']]
        if site['mqtt']['mode'] == 'managed':
            ports.append(site['mqtt']['port'])
        for port in ports:
            with socket.socket() as connection:
                if connection.connect_ex(('127.0.0.1', port)) == 0:
                    raise ValueError('Port {} already in use; existing installation was not stopped'.format(port))
    names = service_names(site)
    if old is None and any((Path(folder) / name).exists()
            for folder in ['/etc/systemd/system', '/lib/systemd/system'] for name in names):
        raise ValueError('Service name already exists; choose a unique service_prefix')
    backup = prefix / 'activation-backups' / datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')
    backup.mkdir(parents=True)
    previous = {}
    for name in names:
        path = Path('/etc/systemd/system') / name
        previous[name] = {'active': subprocess.call(['systemctl', 'is-active', '--quiet', name]) == 0,
                          'enabled': subprocess.call(['systemctl', 'is-enabled', '--quiet', name],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) == 0,
                          'existed': path.exists()}
        if path.exists():
            shutil.copy2(path, backup / name)
    write_json(backup / 'before.json', previous)
    try:
        for name in names:
            role = name[len(site['service_prefix']) + 1:-8]
            (Path('/etc/systemd/system') / name).write_text(unit_text(site, role))
        verify_dir = backup / 'verify'
        verify_dir.mkdir()
        for name in names:
            (verify_dir / name).write_text((Path('/etc/systemd/system') / name).read_text().replace(
                str(prefix / 'current'), str(release)))
        run(['systemd-analyze', 'verify'] + [verify_dir / n for n in names], timeout=30)
        run(['systemctl', 'stop'] + names, timeout=100)
        switch_current(prefix, release)
        run(['systemctl', 'daemon-reload'], timeout=30)
        # MQTT first, tracking second, public UI last.
        for name in reversed(names):
            run(['systemctl', 'enable', '--now', name], timeout=60)
        report = None
        for _ in range(30):
            report = health_report(site, release, live=True)
            if report['passed']:
                break
            time.sleep(2)
        write_json(release / 'validation/activation-health.json', report)
        if not report['passed']:
            raise ValueError('Activation health failed; restoring previous deployment')
        if old and old != release:
            write_json(prefix / 'previous.json', {'release': str(old)})
        write_json(prefix / 'active.json', {'release': str(release), 'report': report})
        return report
    except Exception:
        subprocess.call(['systemctl', 'stop'] + names)
        if old:
            switch_current(prefix, old)
        elif (prefix / 'current').is_symlink():
            (prefix / 'current').unlink()
        for name, before in previous.items():
            path = Path('/etc/systemd/system') / name
            if before['existed']:
                shutil.copy2(backup / name, path)
            elif path.exists():
                path.unlink()
            if not before['enabled']:
                subprocess.call(['systemctl', 'disable', name], stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL)
        subprocess.call(['systemctl', 'daemon-reload'])
        for name, before in previous.items():
            if before['active']:
                subprocess.call(['systemctl', 'start', name])
        raise


def load_provision(bundle):
    import importlib.util
    spec = importlib.util.spec_from_file_location('multicamera_provision', bundle / 'provision.py')
    module = importlib.util.module_from_spec(spec)
    previous = sys.dont_write_bytecode
    try:
        sys.dont_write_bytecode = True
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['plan', 'install', 'upgrade', 'doctor', 'rollback'])
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--bundle', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--system-deps', action='store_true', help='Provision mode: install OS build dependencies')
    parser.add_argument('--activate', action='store_true', help='Switch services; requires sudo')
    parser.add_argument('--gpu', action='store_true', help='Doctor probes CUDA/native ops')
    parser.add_argument('--live', action='store_true', help='Doctor probes running HTTP/MQTT')
    args = parser.parse_args()
    args.bundle = args.bundle.resolve()
    args.config = args.config.resolve()
    site = load_site(args.config)
    prefix = Path(site['prefix'])
    if site['environment_mode'] == 'provision' and args.action in ['doctor', 'rollback']:
        effective = load_provision(args.bundle).Provisioner(site, args.bundle, run, owned)
        site = read_json(effective.root / 'effective-site.json')
    if args.action == 'doctor':
        current = prefix / 'current'
        if not current.is_symlink():
            raise ValueError('No active managed release; use install staging report instead')
        report = health_report(site, current.resolve(), live=args.live, gpu=args.gpu)
        print(json.dumps(report, indent=2))
        return 0 if report['passed'] else 1
    if args.action == 'rollback':
        previous = Path(read_json(prefix / 'previous.json')['release'])
        if not previous.is_dir() or previous.parent != prefix / 'releases':
            raise ValueError('Invalid or missing previous managed release')
        with deployment_lock(prefix):
            report = activate(read_json(previous / '.site.json'), previous)
        print(json.dumps(report, indent=2))
        return 0
    manifest = verify_bundle(args.bundle)
    report = preflight(site, args.bundle, manifest)
    if args.action == 'plan' or not report['passed']:
        print(json.dumps(report, indent=2))
        return 0 if report['passed'] else 1
    if args.activate and os.geteuid() != 0:
        raise ValueError('--activate requires sudo')
    with deployment_lock(prefix):
        if site['environment_mode'] == 'provision':
            write_json(prefix / '.managed-installation.json', {'schema_version': 1, 'user': site['user']})
            fresh = load_provision(args.bundle).Provisioner(site, args.bundle, run, owned)
            site = fresh.apply(system_deps=args.system_deps)
            checked = preflight(site, args.bundle, manifest)
            if not checked['passed']:
                raise ValueError('Provisioned resource preflight failed: ' + '; '.join(checked['problems']))
        release = prepare_release(site, args.bundle, manifest)
        if args.activate:
            report = activate(site, release)
            print(json.dumps(report, indent=2))
            print('Dashboard: http://{}:{}'.format(site['listen'], site['preview_port']))
        else:
            print('Staged and environment-verified; services were not changed: ' + str(release))
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (ValueError, OSError, subprocess.SubprocessError) as exc:
        print('Deployment failed: ' + str(exc), file=sys.stderr)
        sys.exit(1)
