"""Deployment regressions: integrity, isolation, state preservation and rollback."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


installer = module("mc_installer", ROOT / "packaging/jetson/installer/installer.py")
builder = module("mc_builder", ROOT / "scripts/package_jetson.py")


def site(tmp_path):
    config = json.loads((ROOT / "packaging/jetson/installer/site.example.json").read_text())
    config.update(prefix=str(tmp_path / "managed"), user="nvidia")
    return config


def test_light_bundle_is_hashed_and_never_embeds_models_or_environments(tmp_path):
    result = builder.build(ROOT, tmp_path, build_id="unit-test")
    bundle = Path(result["folder"])
    manifest = installer.verify_bundle(bundle)
    assert manifest["models_embedded"] is False
    assert manifest["external_resources"] is True
    assert not (bundle / "runtime").exists()
    assert Path(result["archive"]).stat().st_size < 1024**2
    assert not (bundle / "app/packaging/jetson/devices.json").exists()
    assert not (bundle / "app/data").exists()
    assert (bundle / "install.sh").stat().st_mode & 0o111
    assert (bundle / "app/scripts/jetson/web/human-scene.js").exists()
    assert "exec /usr/bin/python3" in (bundle / "install.sh").read_text()
    with pytest.raises(FileExistsError):
        builder.build(ROOT, tmp_path, build_id="unit-test")
    path = bundle / "app/scripts/jetson/preview.py"
    path.write_text(path.read_text() + "\n# modified\n")
    with pytest.raises(ValueError, match="integrity"):
        installer.verify_bundle(bundle)


@pytest.mark.parametrize("path", ["../secret", "/etc/passwd", "ok/../../secret", ""])
def test_manifest_rejects_path_escape(path):
    with pytest.raises(ValueError):
        installer.safe_relative(path)


@pytest.mark.parametrize(
    "patch",
    [
        {"prefix": "/"},
        {"prefix": "/tmp/../usr"},
        {"prefix": "/tmp/space path"},
        {"user": "root\nExecStart=evil"},
        {"service_prefix": "../service"},
        {"listen": "example.com"},
        {"tracking_port": 8092},
    ],
)
def test_site_rejects_ambiguous_or_injectable_configuration(tmp_path, patch):
    config = site(tmp_path)
    config.update(patch)
    path = tmp_path / "site.json"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError):
        installer.load_site(path)


def test_credentials_are_not_rotated_on_repeat_install(tmp_path, monkeypatch):
    config = site(tmp_path)
    config["mqtt"]["mode"] = "external"
    credentials = tmp_path / "external.json"
    credentials.write_text(json.dumps({"enabled": True, "password": "original"}))
    config["mqtt"]["external_config"] = str(credentials)
    prefix = Path(config["prefix"])
    installer.configure_mqtt(config, prefix)
    target = prefix / "shared/data/tracking/mqtt.json"
    assert target.stat().st_mode & 0o777 == 0o600
    credentials.write_text(json.dumps({"enabled": True, "password": "changed"}))
    installer.configure_mqtt(config, prefix)
    assert installer.read_json(target)["password"] == "original"


def test_managed_broker_hashing_does_not_expose_password_in_process_args(tmp_path, monkeypatch):
    config = site(tmp_path)
    commands = []
    monkeypatch.setattr(installer, "run", lambda args, **kwargs: commands.append(args))
    installer.configure_mqtt(config, Path(config["prefix"]))
    password = installer.read_json(Path(config["prefix"]) / "shared/data/tracking/mqtt.json")[
        "password"
    ]
    assert commands[0][:2] == ["mosquitto_passwd", "-U"]
    assert all(password not in str(arg) for command in commands for arg in command)
    broker = Path(config["prefix"]) / "shared/mqtt/broker.conf"
    assert "allow_anonymous false" in broker.read_text()


def test_service_uses_configured_ports_and_isolated_foundation_python(tmp_path, monkeypatch):
    config = site(tmp_path)
    config["tracking_port"] = 18093
    monkeypatch.setattr(installer.pwd, "getpwnam", lambda _: SimpleNamespace(pw_dir="/home/nvidia"))
    preview = installer.unit_text(config, "preview")
    assert "MULTICAMERA_TRACKING_URL=http://127.0.0.1:18093" in preview
    assert ".foundation-venv/bin/python" in preview
    assert "User=nvidia" in preview and "MemoryLimit=10G" in preview
    assert "--port 18093" in installer.unit_text(config, "tracking")


def test_atomic_switch_keeps_old_release_until_commit(tmp_path):
    old, new = tmp_path / "old", tmp_path / "new"
    old.mkdir()
    new.mkdir()
    installer.switch_current(tmp_path, old)
    assert (tmp_path / "current").resolve() == old
    installer.switch_current(tmp_path, new)
    assert (tmp_path / "current").resolve() == new
    assert old.is_dir()


def test_activation_failure_restores_old_link_and_old_units(tmp_path, monkeypatch):
    config = site(tmp_path)
    prefix = Path(config["prefix"])
    (prefix / "releases").mkdir(parents=True)
    old, new = prefix / "releases/old", prefix / "releases/new"
    old.mkdir()
    new.mkdir()
    installer.switch_current(prefix, old)
    units = tmp_path / "etc-systemd"
    units.mkdir()
    names = installer.service_names(config)
    for name in names:
        (units / name).write_text("old unit " + name)
    real_path = installer.Path
    monkeypatch.setattr(
        installer, "Path", lambda p: units if p == "/etc/systemd/system" else real_path(p)
    )
    monkeypatch.setattr(installer.os, "geteuid", lambda: 0)
    monkeypatch.setattr(installer.pwd, "getpwnam", lambda _: SimpleNamespace(pw_dir="/home/nvidia"))
    monkeypatch.setattr(installer.shutil, "which", lambda _: "/usr/sbin/mosquitto")
    monkeypatch.setattr(installer.subprocess, "call", lambda *args, **kwargs: 0)
    monkeypatch.setattr(installer.time, "sleep", lambda _: None)
    monkeypatch.setattr(installer, "run", lambda *args, **kwargs: None)
    monkeypatch.setattr(installer, "health_report", lambda *args, **kwargs: {"passed": False})
    with pytest.raises(ValueError, match="Activation health failed"):
        installer.activate(config, new)
    assert (prefix / "current").resolve() == old
    validations = list((prefix / "activation-backups").glob("*/verify/*.service"))
    assert len(validations) == len(names)
    assert all(str(new) in p.read_text() for p in validations)
    for name in names:
        assert (units / name).read_text() == "old unit " + name


def test_installed_release_refuses_changed_site_settings(tmp_path):
    config = site(tmp_path)
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "manifest.json").write_text("{}")
    release = Path(config["prefix"]) / "releases/example"
    release.mkdir(parents=True)
    installer.write_json(
        release / ".installation.json",
        {"manifest_sha256": installer.sha256(bundle / "manifest.json")},
    )
    other = dict(config, tracking_port=18093)
    installer.write_json(release / ".site.json", other)
    with pytest.raises(ValueError, match="different site config"):
        installer.prepare_release(config, bundle, {"release_id": "example"})


def test_model_resource_size_checked_before_hash_and_assets_are_not_copied(tmp_path, monkeypatch):
    config = site(tmp_path)
    model = tmp_path / "foundation"
    (model / "pretrained_models/23-51-11").mkdir(parents=True)
    checkpoint = model / "pretrained_models/23-51-11/model_best_bp2.pth"
    checkpoint.write_bytes(b"wrong-size")
    pose = tmp_path / "pose.pt"
    pose.write_bytes(b"pose")
    config["models"] = {"foundation_root": str(model), "pose_weights": str(pose)}
    catalog = tmp_path / "assets.json"
    catalog.write_text(
        json.dumps(
            {
                "assets": {
                    "foundation": {
                        "relative_checkpoint": "pretrained_models/23-51-11/model_best_bp2.pth",
                        "bytes": 99,
                        "sha256": "unused",
                    },
                    "pose": {"bytes": 4, "sha256": installer.sha256(pose)},
                }
            }
        )
    )
    real_hash = installer.sha256

    def check_hash(path):
        assert path != checkpoint  # Wrong size never triggers expensive model hashing.
        return real_hash(path)

    monkeypatch.setattr(installer, "sha256", check_hash)
    report = installer.verify_external_assets(config, catalog)
    assert not report["passed"]
    assert "Model size differs: foundation" in report["problems"]
    assert report["assets"]["pose"]["external"] is True


def test_resources_are_linked_and_originals_untouched(tmp_path):
    config = site(tmp_path)
    source = tmp_path / "source-env"
    (source / "bin").mkdir(parents=True)
    (source / "pyvenv.cfg").write_text("version = 3.10")
    (source / "bin/python").write_text("fixture")
    config["environments"] = {role: str(source / "bin/python") for role in installer.ENVIRONMENTS}
    config["models"] = {
        "foundation_root": str(tmp_path / "foundation"),
        "pose_weights": str(tmp_path / "pose.pt"),
    }
    release = tmp_path / "release"
    release.mkdir()
    installer.link_resources(config, release)
    assert (release / ".venv").is_symlink()
    assert (release / "models/foundation").is_symlink()
    assert (release / "models/pose/yolo26n-pose.pt").is_symlink()
    assert (source / "bin/python").read_text() == "fixture"


def test_first_activation_refuses_port_conflict_without_stopping_anything(tmp_path, monkeypatch):
    config = site(tmp_path)
    prefix = Path(config["prefix"])
    release = prefix / "releases/new"
    release.mkdir(parents=True)
    monkeypatch.setattr(installer.os, "geteuid", lambda: 0)
    sock = SimpleNamespace(connect_ex=lambda _: 0)

    class Socket:
        def __enter__(self):
            return sock

        def __exit__(self, *args):
            pass

    monkeypatch.setattr(installer.socket, "socket", lambda: Socket())
    commands = []
    monkeypatch.setattr(installer, "run", lambda *args, **kwargs: commands.append(args))
    with pytest.raises(ValueError, match="already in use"):
        installer.activate(config, release)
    assert not commands
    assert not (prefix / "current").exists()


def test_package_filter_excludes_model_and_wheel_payload_before_copy(tmp_path):
    source, output = tmp_path / "source", tmp_path / "out"
    source.mkdir()
    (source / "app.py").write_text('print("small")')
    for filename in ["model.pth", "pose.pt", "features.onnx", "torch.whl", "large.engine"]:
        (source / filename).write_bytes(b"not-for-main-package")
    builder.copy_filtered(source, output)
    assert sorted(p.name for p in output.iterdir()) == ["app.py"]


def test_imported_broker_credentials_cannot_silently_turn_into_new_managed_broker(tmp_path):
    config = site(tmp_path)
    prefix = Path(config["prefix"])
    installer.write_json(
        prefix / "shared/data/tracking/mqtt.json",
        {"enabled": True, "username": "old", "password": "keep"},
    )
    with pytest.raises(ValueError, match="external Broker"):
        installer.configure_mqtt(config, prefix)
    assert installer.read_json(prefix / "shared/data/tracking/mqtt.json")["password"] == "keep"


def test_preflight_refuses_unmanaged_nonempty_prefix_without_writes(tmp_path, monkeypatch):
    config = site(tmp_path)
    prefix = Path(config["prefix"])
    prefix.mkdir()
    original = prefix / "unrelated.txt"
    original.write_text("leave alone")
    monkeypatch.setattr(
        installer,
        "verify_external_assets",
        lambda *args: {"passed": True, "problems": [], "assets": {}},
    )
    report = installer.preflight(config, tmp_path, {"external_resources": True, "files": {}})
    assert "Refuse nonempty directory without managed installation marker" in report["problems"]
    assert list(prefix.iterdir()) == [original]
    assert original.read_text() == "leave alone"


def test_cli_normalizes_relative_bundle_before_working_directory_changes(tmp_path, monkeypatch):
    import sys

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys, "argv", ["installer", "plan", "--bundle", "bundle", "--config", "site.json"]
    )
    monkeypatch.setattr(
        installer,
        "load_site",
        lambda p: {"prefix": str(tmp_path / "prefix"), "environment_mode": "reuse"},
    )
    seen = []
    monkeypatch.setattr(installer, "verify_bundle", lambda p: seen.append(p) or {})
    monkeypatch.setattr(
        installer, "preflight", lambda *a: {"passed": False, "problems": ["fixture"]}
    )
    assert installer.main() == 1
    assert seen == [tmp_path / "bundle"]
