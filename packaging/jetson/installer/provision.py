"""Thin, on-demand JetPack 5 provisioning. Python 3.8+, no global pip installs."""
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import tarfile
import time
import urllib.error
import urllib.parse
import urllib.request

ROLES = {'dashboard': 'dashboard', 'fusion': 'fusion', 'pose': 'gpu', 'foundation': 'gpu'}
APT_PACKAGES = [
    'build-essential', 'python3-dev', 'python3-venv', 'libopenblas-dev', 'libjpeg-dev',
    'libpng-dev', 'zlib1g-dev', 'libssl-dev', 'libbz2-dev', 'libreadline-dev',
    'libsqlite3-dev', 'libffi-dev', 'liblzma-dev', 'libgomp1', 'libgl1', 'libglib2.0-0',
    'patch', 'pkg-config', 'cuda-cudart-11-4', 'cuda-libraries-11-4',
    'cuda-nvcc-11-4', 'cuda-cudart-dev-11-4', 'libcudnn8',
]


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            result.update(part)
    return result.hexdigest()


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.chmod(0o600)
    temp.replace(path)


def verified(path, resource):
    path = Path(path)
    return path.is_file() and (not resource.get('bytes') or path.stat().st_size == resource['bytes']) \
        and digest(path) == resource['sha256']


def fetch_asset(ident, resource, cache, source=None, opener=urllib.request.urlopen):
    """Bounded streaming download, resumable partials, final SHA gate; no unverified use."""
    if not re.fullmatch(r'[a-z0-9-]+', ident):
        raise ValueError('Invalid asset identifier')
    if not re.fullmatch(r'[0-9a-f]{64}', resource.get('sha256', '')):
        raise ValueError('Missing pinned SHA256: ' + ident)
    if source and Path(source).is_absolute():
        # Explicit resource library, not an old Python environment. Never duplicate a model.
        if not verified(source, resource):
            raise ValueError('Local resource size/hash mismatch: ' + ident)
        return Path(source)
    cache.mkdir(parents=True, exist_ok=True)
    filename = resource.get('filename', ident)
    if Path(filename).name != filename:
        raise ValueError('Unsafe asset filename')
    destination = cache / resource['sha256'] / filename
    destination.parent.mkdir(parents=True, exist_ok=True)
    if verified(destination, resource):
        return destination
    urls = [source] if source else resource.get('urls', [])
    if not urls:
        raise ValueError('No direct download source for ' + ident + '; provide an HTTPS mirror or local asset path')
    partial = destination.with_name(destination.name + '.part')
    maximum = resource.get('bytes') or resource['max_bytes']
    last = None
    for url in urls:
        if urllib.parse.urlparse(url).scheme != 'https':
            raise ValueError('Resource URLs must use HTTPS')
        try:
            offset = partial.stat().st_size if partial.exists() else 0
            if offset >= maximum:
                partial.unlink()
                offset = 0
            request = urllib.request.Request(url, headers={'Range': 'bytes={}-'.format(offset)} if offset else {})
            with opener(request, timeout=45) as response:
                resumed = offset and response.status == 206
                if response.status not in [200, 206]:
                    raise ValueError('Unexpected asset HTTP status')
                if response.status == 206:
                    expected_prefix = 'bytes {}-'.format(offset)
                    if not response.headers.get('Content-Range', '').startswith(expected_prefix):
                        raise ValueError('Invalid resume Content-Range')
                count = offset if resumed else 0
                with partial.open('ab' if resumed else 'wb') as output:
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        count += len(chunk)
                        if count > maximum:
                            raise ValueError('Asset exceeds size limit: ' + ident)
                        output.write(chunk)
            if not verified(partial, resource):
                # Do not reuse complete but corrupt bytes on subsequent resume attempts.
                partial.unlink()
                raise ValueError('Downloaded asset size/hash mismatch: ' + ident)
            partial.chmod(0o644)
            partial.replace(destination)
            return destination
        except (OSError, ValueError, urllib.error.URLError) as exc:
            last = exc
    raise ValueError('Resource download failed for {}: {}'.format(ident, last))


def extract_source(archive, destination, max_bytes=1024**3):
    """Extract one upstream source root; reject links, traversal and expansion bombs."""
    if destination.exists():
        raise FileExistsError(str(destination))
    with tarfile.open(archive, 'r:*') as tar:
        members = tar.getmembers()
        total, roots = 0, set()
        for member in members:
            path = PurePosixPath(member.name)
            if path.is_absolute() or '..' in path.parts or not path.parts:
                raise ValueError('Unsafe archive path')
            if not (member.isdir() or member.isfile()):
                raise ValueError('Archive links/devices are forbidden')
            roots.add(path.parts[0])
            total += member.size
        if total > max_bytes or len(roots) != 1 or len(members) > 30000:
            raise ValueError('Unexpected source archive structure/size')
        destination.mkdir(parents=True)
        for member in members:
            relative = PurePosixPath(member.name).parts[1:]
            if not relative:
                continue
            output = destination.joinpath(*relative)
            if member.isdir():
                output.mkdir(parents=True, exist_ok=True)
            else:
                output.parent.mkdir(parents=True, exist_ok=True)
                with tar.extractfile(member) as stream, output.open('wb') as target:
                    shutil.copyfileobj(stream, target, length=1024 * 1024)
                output.chmod(0o755 if member.mode & 0o111 else 0o644)


def selected_profile(site):
    name = site.get('deployment_profile', 'jetpack5-l4t35.4.1-aarch64')
    if name == 'auto':
        raw = Path('/etc/nv_tegra_release').read_text()
        if re.search(r'R35 .*REVISION: 4\.1', raw):
            name = 'jetpack5-l4t35.4.1-aarch64'
        elif re.search(r'R36 .*REVISION: 4\.3', raw):
            name = 'jetpack6-l4t36.4.3-aarch64'
        else:
            raise ValueError('No validated profile for this L4T release')
    mapping = {'jetpack5-l4t35.4.1-aarch64': 'provision-assets.json',
               'jetpack6-l4t36.4.3-aarch64': 'provision-assets-jp6.json'}
    if name not in mapping:
        raise ValueError('Unknown deployment_profile')
    return mapping[name]


def plan(site, bundle):
    catalog = json.loads((bundle / selected_profile(site)).read_text())
    cfg = site['provision']
    assets = catalog['assets']
    required = ['pip', 'torch', 'foundation-source', 'foundation', 'pose']
    if cfg.get('python310', catalog['default_python310']) == 'build':
        required.append('python-source')
    if cfg.get('asset_sources', {}).get('torchvision-wheel') or catalog['prebuilt_torchvision']:
        required.append('torchvision-wheel')
    else:
        required.append('torchvision-source')
    known = sum(assets[n].get('bytes', 0) for n in required)
    return {'environment_mode': 'provision', 'profile': catalog['profile'],
            'known_asset_bytes': known, 'required_asset_ids': required,
            'main_package_contains_models': False, 'minimum_free_bytes': 12 * 1024**3,
            'pip_and_apt_download_bytes': 'resolved separately; not included in known_asset_bytes',
            'system_dependencies': [p for p in APT_PACKAGES if not (p.startswith('cuda-') or p.startswith('libcudnn'))] + catalog['cuda_apt_packages'] + (['mosquitto', 'mosquitto-clients']
                if site['mqtt']['mode'] == 'managed' else []),
            'python310': cfg.get('python310', catalog['default_python310']),
            'google_drive_fallback': bool(assets['foundation'].get('google_drive_folder')),
            'source_policy': 'HTTPS with fixed digest or explicitly supplied local resources',
            'does_not_flash_or_upgrade_jetpack': True}


class Provisioner:
    def __init__(self, site, bundle, command, own):
        self.site, self.bundle, self.command, self.own = site, bundle, command, own
        self.prefix = Path(site['prefix'])
        self.shared = self.prefix / 'shared'
        self.config = site['provision']
        self.profile = json.loads((bundle / selected_profile(site)).read_text())
        self.catalog = self.profile['assets']
        if set(self.config.get('asset_sources', {})) - set(self.catalog):
            raise ValueError('Unknown asset_sources identifier')
        fingerprint = json.dumps({'config': self.config, 'catalog': self.profile,
            'requirements': {p.name: digest(p) for p in sorted((bundle / 'requirements').glob('*'))},
            'patch': digest(bundle / 'foundation-orin.patch'), 'controller': digest(bundle / 'provision.py')}, sort_keys=True)
        self.ident = hashlib.sha256(fingerprint.encode()).hexdigest()[:16]
        self.root = self.shared / 'provisioned' / self.ident
        self.cache = self.shared / 'resource-cache'
        self.state_path = self.root / 'state.json'
        self.env = dict(os.environ, PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1',
            PIP_DISABLE_PIP_VERSION_CHECK='1', PIP_CACHE_DIR=self.config.get('pip_cache_dir', str(self.shared / 'pip-cache')),
            OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', HF_HUB_OFFLINE='1',
            PYTHONPATH='', CUDA_HOME='/usr/local/cuda', MAX_JOBS='1')
        if self.config.get('pip_index_url'):
            self.env['PIP_INDEX_URL'] = self.config['pip_index_url']

    def phase(self, name, **details):
        atomic_json(self.state_path, {'phase': name, 'updated_at_unix_s': time.time(),
            'profile_fingerprint': self.ident, **details})
        print('Provision: ' + name, flush=True)

    def call(self, args, cwd=None, timeout=1800, env=None):
        self.own(self.prefix, self.site)
        self.command(args, cwd=cwd, timeout=timeout, env=env or self.env, user=self.site['user'])

    def asset(self, ident):
        return fetch_asset(ident, self.catalog[ident], self.cache,
                           self.config.get('asset_sources', {}).get(ident))

    def pip(self, python, *arguments, constraints=True):
        args = [python, '-m', 'pip', 'install', '--disable-pip-version-check']
        if constraints:
            args += ['-c', self.bundle / ('requirements/gpu-constraints' + self.profile['requirements_suffix'] + '.txt')]
        self.call(args + list(arguments))

    def ensure_system(self, install):
        if install:
            if os.geteuid() != 0:
                raise ValueError('--system-deps requires sudo')
            packages = [p for p in APT_PACKAGES if not (p.startswith('cuda-') or p.startswith('libcudnn'))] + self.profile['cuda_apt_packages'] + (['mosquitto', 'mosquitto-clients']
                if self.site['mqtt']['mode'] == 'managed' else [])
            broker_existed = shutil.which('mosquitto') is not None
            self.command(['apt-get', 'update'], timeout=1800)
            simulation = subprocess.run(['apt-get', '--simulate', 'install', '--no-install-recommends'] + packages,
                check=True, capture_output=True, text=True, timeout=180)
            for line in simulation.stdout.splitlines():
                if line.startswith(('Inst ', 'Remv ')) and any(n in line for n in
                        ['nvidia-l4t-', 'nvidia-driver-']):
                    raise ValueError('APT would alter the driver/L4T profile; refusing: ' + line)
                if line.startswith(('Inst ', 'Remv ')) and any(
                        int(m) != self.profile['cuda_major'] for m in re.findall(
                            r'cuda-(?:[a-z-]+-)?(\d+)-', line)):
                    raise ValueError('APT would install wrong CUDA major version: ' + line)
            self.command(['apt-get', 'install', '-y', '--no-install-recommends'] + packages,
                         timeout=3600, env=dict(os.environ, DEBIAN_FRONTEND='noninteractive'))
            if self.site['mqtt']['mode'] == 'managed' and not broker_existed:
                self.command(['systemctl', 'stop', 'mosquitto'], timeout=60)
                self.command(['systemctl', 'disable', 'mosquitto'], timeout=60)
        missing = [n for n in ['make', 'gcc', 'patch', 'pkg-config'] if not shutil.which(n)]
        if missing:
            raise ValueError('Missing build tools; rerun with --system-deps: ' + ','.join(missing))
        if not Path('/usr/local/cuda/lib64').is_dir():
            raise ValueError('Install matching JetPack 5 CUDA runtime first; no generic CUDA upgrade')

    def ensure_python(self):
        configured = self.config.get('python310', self.profile['default_python310'])
        if configured != 'build':
            self.call([configured, '-c', 'import sys,venv,ssl; assert sys.version_info[:2]==(3,10)'])
            return Path(configured)
        root = self.root / 'python310'
        python = root / 'bin/python3.10'
        if not (root / '.complete').exists():
            source = self.root / 'build-python'
            if not source.exists():
                extract_source(self.asset('python-source'), source)
            self.call(['./configure', '--prefix=' + str(root), '--with-ensurepip=install'],
                      cwd=source, timeout=300)
            self.call(['make', '-j' + str(self.config.get('build_jobs', 2))], cwd=source, timeout=7200)
            self.call(['make', 'install'], cwd=source, timeout=1800)
            self.call([python, '-c', 'import ssl,sqlite3,bz2,lzma,ctypes,venv,ensurepip'])
            (root / '.complete').touch()
        return python

    def seed_pip(self, python, root):
        marker = root / '.pip-seeded'
        if marker.exists():
            return
        wheel = self.asset('pip')
        # pip is a pure-Python pinned wheel; seed it without touching global packages
        # or depending on Debian's optional ensurepip module.
        code = ('import sys,sysconfig,zipfile,pathlib; '
                'z=zipfile.ZipFile(sys.argv[1]); '
                'assert all(not pathlib.PurePosixPath(n).is_absolute() and '
                '".." not in pathlib.PurePosixPath(n).parts for n in z.namelist()); '
                'z.extractall(sysconfig.get_path("purelib"))')
        self.call([python, '-c', code, wheel])
        self.pip(python, 'setuptools==75.3.2', 'wheel==0.45.1', constraints=False)
        marker.touch()

    def ensure_venv(self, role, base):
        root = self.root / 'environments' / role
        root.parent.mkdir(parents=True, exist_ok=True)
        if not (root / 'pyvenv.cfg').exists():
            self.call([base, '-m', 'venv', '--without-pip', root], timeout=120)
        python = root / 'bin/python'
        self.seed_pip(python, root)
        self.call([python, '-c', 'import site; assert not site.ENABLE_USER_SITE'])
        return python

    def ensure_environments(self, python310):
        gpu = self.ensure_venv('gpu', '/usr/bin/python3')
        dashboard = self.ensure_venv('dashboard', python310)
        fusion = self.ensure_venv('fusion', python310)
        for role, python in [('dashboard', dashboard), ('fusion', fusion)]:
            marker = self.root / ('deps-' + role + '.complete')
            if not marker.exists():
                self.pip(python, '-r', self.bundle / ('requirements/' + role + '.txt'), constraints=False)
                marker.touch()
        marker = self.root / 'deps-gpu.complete'
        if not marker.exists():
            self.pip(gpu, 'pip==25.0.1', 'setuptools==75.3.2', 'wheel==0.45.1', constraints=False)
            self.pip(gpu, '--no-deps', self.asset('torch'))
            self.pip(gpu, '-r', self.bundle / ('requirements/gpu-base' + self.profile['requirements_suffix'] + '.txt'))
            override = self.config.get('asset_sources', {}).get('torchvision-wheel')
            if override or self.profile['prebuilt_torchvision']:
                wheel = self.asset('torchvision-wheel')
            else:
                if not Path('/usr/local/cuda/bin/nvcc').is_file():
                    raise ValueError('CUDA nvcc is needed to build torchvision; install JetPack dev tools '
                                     'or supply a verified torchvision-wheel asset')
                source = self.root / 'build-torchvision'
                if not source.exists():
                    extract_source(self.asset('torchvision-source'), source)
                wheels = self.root / 'built-wheels'
                wheels.mkdir(exist_ok=True)
                env = dict(self.env, BUILD_VERSION='0.16.2+mc.nv23.6',
                    PYTORCH_VERSION='2.1.0a0+41361538.nv23.6', FORCE_CUDA='1',
                    TORCH_CUDA_ARCH_LIST='8.7', TORCHVISION_USE_FFMPEG='0',
                    TORCHVISION_USE_VIDEO_CODEC='0')
                self.call([gpu, '-m', 'pip', 'wheel', '--no-deps', '--no-build-isolation',
                           '--wheel-dir', wheels, source], env=env, timeout=10800)
                built = list(wheels.glob('torchvision-*.whl'))
                if len(built) != 1:
                    raise ValueError('Expected one locally built torchvision wheel')
                wheel = built[0]
                atomic_json(wheels / 'build.json', {'file': wheel.name, 'sha256': digest(wheel),
                    'source_sha256': self.catalog['torchvision-source']['sha256']})
            self.pip(gpu, '--no-deps', wheel)
            self.pip(gpu, '-r', self.bundle / ('requirements/gpu' + self.profile['requirements_suffix'] + '.txt'))
            self.call([gpu, '-c', 'import torch,torchvision; assert torch.cuda.is_available(); '
                'assert torchvision.ops.nms(torch.tensor([[0.,0.,1.,1.]],device="cuda"),'
                'torch.ones(1,device="cuda"),.5).numel()==1'])
            marker.touch()
        return {'dashboard': str(dashboard), 'fusion': str(fusion),
                'pose': str(gpu), 'foundation': str(gpu)}

    def foundation_checkpoint(self, gpu):
        asset = self.catalog['foundation']
        if self.config.get('asset_sources', {}).get('foundation') or asset.get('urls'):
            return self.asset('foundation')
        # Google's official folder has no stable direct file URL. Resolve only the
        # selected checkpoint; never download the complete multi-model folder.
        self.pip(gpu, 'gdown==5.2.0')
        folder = asset['google_drive_folder']
        output = self.cache / (asset['sha256'][:16] + '-model_best_bp2.pth')
        if verified(output, asset):
            return output
        code = ('import gdown,os,resource; resource.setrlimit(resource.RLIMIT_FSIZE,(' +
                str(asset['bytes']) + ',' + str(asset['bytes']) + ')); files=gdown.download_folder(url=' + repr(folder) +
                ',quiet=True,skip_download=True); matches=[f for f in files '
                'if f.path.replace(os.sep,"/").endswith("23-51-11/model_best_bp2.pth") or f.path=="model_best_bp2.pth"]; '
                'assert len(matches)==1,"Cannot uniquely resolve official checkpoint; provide a mirror"; '
                'gdown.download(id=matches[0].id,output=' + repr(str(output) + '.part') +
                ',quiet=False,resume=True)')
        self.call([gpu, '-c', code], timeout=21600)
        part = Path(str(output) + '.part')
        if not verified(part, asset):
            raise ValueError('Official checkpoint size/hash mismatch; unverified bytes not installed')
        part.replace(output)
        return output

    def apply(self, system_deps=False):
        self.root.mkdir(parents=True, exist_ok=True)
        self.cache.mkdir(parents=True, exist_ok=True)
        try:
            self.phase('system')
            self.ensure_system(system_deps)
            self.phase('python')
            python310 = self.ensure_python()
            self.phase('environments')
            environments = self.ensure_environments(python310)
            self.phase('model_source')
            model = self.root / 'models/foundation'
            if not (model / '.patched').exists():
                if model.exists():
                    shutil.rmtree(model)  # only this managed, incomplete source tree; no weights yet
                extract_source(self.asset('foundation-source'), model)
                self.call(['patch', '--batch', '--fuzz=0', '-p1', '-i',
                           self.bundle / 'foundation-orin.patch'], cwd=model, timeout=30)
                cfg = model / 'pretrained_models/23-51-11'
                cfg.mkdir(parents=True, exist_ok=True)
                shutil.copy2(self.bundle / 'foundation-cfg.yaml', cfg / 'cfg.yaml')
                (model / '.patched').touch()
            self.phase('weights')
            checkpoint = self.foundation_checkpoint(environments['foundation'])
            link = model / 'pretrained_models/23-51-11/model_best_bp2.pth'
            if not link.exists():
                link.symlink_to(checkpoint)
            pose = self.asset('pose')
            result = json.loads(json.dumps(self.site))
            result.update(environment_mode='reuse', managed_resources=True, deployment_profile=self.profile['profile'],
                environments=environments, models={'foundation_root': str(model), 'pose_weights': str(pose)})
            result['provision_origin'] = {'profile_fingerprint': self.ident,
                'root': str(self.root), 'isolation': 'no system/user site-packages',
                'gpu_environment_shared_by': ['pose', 'foundation']}
            self.own(self.prefix, self.site)
            self.phase('resources_ready', effective_config=str(self.root / 'effective-site.json'))
            atomic_json(self.root / 'effective-site.json', result)
            return result
        except Exception as exc:
            self.phase('failed', error=str(exc)[:1000])
            raise
