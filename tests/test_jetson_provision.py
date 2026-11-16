"""Fresh-device provisioning must not silently inherit a working developer environment."""

import importlib.util
import io
import json
from pathlib import Path
import tarfile

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "provision", ROOT / "packaging/jetson/installer/provision.py"
)
provision = importlib.util.module_from_spec(spec)
spec.loader.exec_module(provision)
installer_spec = importlib.util.spec_from_file_location(
    "installer", ROOT / "packaging/jetson/installer/installer.py"
)
installer = importlib.util.module_from_spec(installer_spec)
installer_spec.loader.exec_module(installer)


class Response(io.BytesIO):
    def __init__(self, data, status=200, headers=None):
        super().__init__(data)
        self.status, self.headers = status, headers or {}


def resource(tmp_path, payload=b"test-payload", filename="file.whl"):
    p = tmp_path / "source"
    p.write_bytes(payload)
    return {
        "filename": filename,
        "bytes": len(payload),
        "sha256": provision.digest(p),
        "urls": ["https://example.test/model"],
    }


def test_fetch_keeps_wheel_basename_and_cache_skips_network(tmp_path):
    asset = resource(tmp_path)
    count = []

    def opener(request, **kwargs):
        count.append(request)
        return Response(b"test-payload")

    p = provision.fetch_asset("model", asset, tmp_path / "cache", opener=opener)
    assert p.name == "file.whl"
    assert provision.verified(p, asset)
    assert provision.fetch_asset("model", asset, tmp_path / "cache", opener=opener) == p
    assert len(count) == 1


def test_partial_resume_requires_correct_content_range(tmp_path):
    asset = resource(tmp_path, b"abcdef")
    cache = tmp_path / "cache"
    part = cache / asset["sha256"] / (asset["filename"] + ".part")
    part.parent.mkdir(parents=True)
    part.write_bytes(b"abc")

    def opener(request, **kwargs):
        assert request.headers["Range"] == "bytes=3-"
        return Response(b"def", 206, {"Content-Range": "bytes 3-5/6"})

    p = provision.fetch_asset("model", asset, cache, opener=opener)
    assert p.read_bytes() == b"abcdef"
    assert not part.exists()


def test_resume_restarts_cleanly_if_server_ignores_range(tmp_path):
    asset = resource(tmp_path, b"abcdef")
    part = tmp_path / "cache" / asset["sha256"] / (asset["filename"] + ".part")
    part.parent.mkdir(parents=True)
    part.write_bytes(b"abc")
    p = provision.fetch_asset(
        "model", asset, tmp_path / "cache", opener=lambda *a, **k: Response(b"abcdef")
    )
    assert p.read_bytes() == b"abcdef"


@pytest.mark.parametrize("body", [b"corrupt-data", b"way-too-many-bytes-for-the-limit"])
def test_bad_or_oversized_download_is_never_committed(tmp_path, body):
    asset = resource(tmp_path)
    with pytest.raises(ValueError, match="download failed"):
        provision.fetch_asset(
            "model", asset, tmp_path / "cache", opener=lambda *a, **k: Response(body)
        )
    assert not (tmp_path / "cache" / asset["sha256"] / asset["filename"]).exists()


def test_explicit_local_model_library_is_verified_without_copying(tmp_path):
    asset = resource(tmp_path)
    source = tmp_path / "source"
    p = provision.fetch_asset("model", asset, tmp_path / "cache", str(source))
    assert p == source
    assert not (tmp_path / "cache").exists()
    source.write_bytes(b"changed")
    with pytest.raises(ValueError, match="Local resource"):
        provision.fetch_asset("model", asset, tmp_path / "cache", str(source))


def test_insecure_transport_and_missing_identity_are_rejected(tmp_path):
    asset = resource(tmp_path)
    with pytest.raises(ValueError, match="HTTPS"):
        provision.fetch_asset("model", asset, tmp_path / "cache", "http://example.test/model")
    asset["sha256"] = ""
    with pytest.raises(ValueError, match="pinned SHA"):
        provision.fetch_asset("model", asset, tmp_path / "cache")


@pytest.mark.parametrize(
    "name,kind",
    [
        ("root/../../escape", tarfile.REGTYPE),
        ("/absolute", tarfile.REGTYPE),
        ("root/link", tarfile.SYMTYPE),
    ],
)
def test_archive_escape_and_links_fail_before_extraction(tmp_path, name, kind):
    archive = tmp_path / "source.tar"
    with tarfile.open(archive, "w") as t:
        m = tarfile.TarInfo(name)
        m.type = kind
        m.size = 0
        m.linkname = "/etc/passwd"
        t.addfile(m, io.BytesIO())
    destination = tmp_path / "out"
    with pytest.raises(ValueError):
        provision.extract_source(archive, destination)
    assert not destination.exists()


def test_valid_source_extracts_single_root_and_enforces_expansion_limit(tmp_path):
    archive = tmp_path / "source.tar"
    with tarfile.open(archive, "w") as t:
        m = tarfile.TarInfo("source/main.py")
        m.size = 3
        t.addfile(m, io.BytesIO(b"abc"))
    with pytest.raises(ValueError, match="size"):
        provision.extract_source(archive, tmp_path / "too-large", max_bytes=2)
    provision.extract_source(archive, tmp_path / "ok")
    assert (tmp_path / "ok/main.py").read_bytes() == b"abc"


def test_fresh_plan_requires_no_existing_environments_and_has_no_downloads(tmp_path, monkeypatch):
    p = ROOT / "packaging/jetson/installer"
    site = json.loads((p / "site.fresh.example.json").read_text())
    f = tmp_path / "site.json"
    f.write_text(json.dumps(site))
    loaded = installer.load_site(f)
    loaded["deployment_profile"] = "jetpack5-l4t35.4.1-aarch64"
    assert loaded["environment_mode"] == "provision"
    monkeypatch.setattr(
        provision.urllib.request, "urlopen", lambda *a, **k: pytest.fail("plan downloaded data")
    )
    info = provision.plan(loaded, p)
    assert info["known_asset_bytes"] > 3_000_000_000
    assert not info["main_package_contains_models"]
    assert "torchvision-source" in info["required_asset_ids"]
    assert info["python310"] == "build"
    assert not (tmp_path / "shared").exists()


def test_commands_use_isolated_venv_and_pinned_nvidia_constraint(tmp_path, monkeypatch):
    p = ROOT / "packaging/jetson/installer"
    site = json.loads((p / "site.fresh.example.json").read_text())
    site["prefix"] = str(tmp_path / "managed")
    site["deployment_profile"] = "jetpack5-l4t35.4.1-aarch64"
    calls = []
    boot = provision.Provisioner(site, p, lambda *a, **k: calls.append((a, k)), lambda *a: None)
    monkeypatch.setattr(boot, "seed_pip", lambda *a: None)
    python = boot.ensure_venv("gpu", "/usr/bin/python3")
    assert any("-m" in a[0] and "venv" in a[0] for a, k in calls)
    assert all("--system-site-packages" not in a[0] for a, k in calls)
    assert boot.env["PYTHONNOUSERSITE"] == "1" and boot.env["PYTHONPATH"] == ""
    boot.pip(python, "numpy==1.24.4")
    assert "-c" in calls[-1][0][0]
    assert "gpu-constraints.txt" in str(calls[-1][0][0])
    assert site["user"] == calls[-1][1]["user"]


def test_failure_records_phase_and_preserves_other_release(tmp_path, monkeypatch):
    p = ROOT / "packaging/jetson/installer"
    site = json.loads((p / "site.fresh.example.json").read_text())
    site["prefix"] = str(tmp_path / "managed")
    site["deployment_profile"] = "jetpack5-l4t35.4.1-aarch64"
    prefix = Path(site["prefix"])
    prefix.mkdir()
    old = prefix / "old"
    old.mkdir()
    (prefix / "current").symlink_to(old)
    boot = provision.Provisioner(site, p, lambda *a, **k: None, lambda *a: None)
    monkeypatch.setattr(
        boot, "ensure_system", lambda *_: (_ for _ in ()).throw(ValueError("missing dependency"))
    )
    with pytest.raises(ValueError, match="missing dependency"):
        boot.apply()
    assert json.loads(boot.state_path.read_text())["phase"] == "failed"
    assert (prefix / "current").resolve() == old


@pytest.mark.parametrize(
    "release,expected",
    [
        ("# R35 (release), REVISION: 4.1", "provision-assets.json"),
        ("# R36 (release), REVISION: 4.3", "provision-assets-jp6.json"),
    ],
)
def test_auto_profile_keeps_cuda_stacks_separate(tmp_path, monkeypatch, release, expected):
    file = tmp_path / "l4t"
    file.write_text(release)
    real_path = provision.Path
    monkeypatch.setattr(
        provision, "Path", lambda p: file if p == "/etc/nv_tegra_release" else real_path(p)
    )
    assert provision.selected_profile({"deployment_profile": "auto"}) == expected


def test_unknown_l4t_never_falls_back_to_jetpack5(tmp_path, monkeypatch):
    file = tmp_path / "l4t"
    file.write_text("# R37 (release), REVISION: 1.0")
    real_path = provision.Path
    monkeypatch.setattr(
        provision, "Path", lambda p: file if p == "/etc/nv_tegra_release" else real_path(p)
    )
    with pytest.raises(ValueError, match="No validated profile"):
        provision.selected_profile({"deployment_profile": "auto"})


def test_jetpack6_uses_system_python_and_prebuilt_cuda126_wheels():
    bundle = ROOT / "packaging/jetson/installer"
    site = json.loads((bundle / "site.fresh.jp6.example.json").read_text())
    info = provision.plan(site, bundle)
    assert info["python310"] == "/usr/bin/python3"
    assert "python-source" not in info["required_asset_ids"]
    assert "torchvision-wheel" in info["required_asset_ids"]
    assert "torchvision-source" not in info["required_asset_ids"]
    assert "cuda-cudart-12-6" in info["system_dependencies"]
    assert not any("11-4" in n for n in info["system_dependencies"])


def test_lazy_profile_import_does_not_mutate_verified_bundle(tmp_path, monkeypatch):
    (tmp_path / "provision.py").write_text("VALUE=42\n")
    import sys

    monkeypatch.setattr(sys, "dont_write_bytecode", False)
    assert installer.load_provision(tmp_path).VALUE == 42
    assert not (tmp_path / "__pycache__").exists()
    assert sys.dont_write_bytecode is False


@pytest.mark.parametrize(
    "simulation,reason",
    [
        ("Inst nvidia-l4t-core (36.5 arm64)", "driver/L4T"),
        ("Inst cuda-cudart-13-0 (13.0 arm64)", "wrong CUDA"),
    ],
)
def test_system_dependency_step_refuses_driver_or_cuda_profile_changes(
    tmp_path, monkeypatch, simulation, reason
):
    from types import SimpleNamespace

    bundle = ROOT / "packaging/jetson/installer"
    site = json.loads((bundle / "site.fresh.jp6.example.json").read_text())
    site["prefix"] = str(tmp_path / "managed")
    calls = []
    boot = provision.Provisioner(
        site, bundle, lambda args, **kw: calls.append(args), lambda *a: None
    )
    monkeypatch.setattr(provision.os, "geteuid", lambda: 0)
    monkeypatch.setattr(
        provision.subprocess, "run", lambda *a, **kw: SimpleNamespace(stdout=simulation)
    )
    with pytest.raises(ValueError, match=reason):
        boot.ensure_system(True)
    assert not any("-y" in command for command in calls)
