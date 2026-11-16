"""Configurable model worker/loopback endpoint retain their original defaults."""

from pathlib import Path
from unittest.mock import MagicMock

from scripts.jetson.foundation_runner import FoundationRunner
from scripts.jetson.preview import make_preview_app


def test_foundation_worker_uses_managed_python_and_log(tmp_path, monkeypatch):
    root = tmp_path / "model"
    checkpoint = root / "pretrained_models/23-51-11/model_best_bp2.pth"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"fixture")
    monkeypatch.setenv("FOUNDATION_STEREO_ROOT", str(root))
    monkeypatch.setenv("FOUNDATION_STEREO_PYTHON", "/managed/.foundation-venv/bin/python")
    monkeypatch.setenv("FOUNDATION_STEREO_LOG", str(tmp_path / "logs/worker.log"))
    process = MagicMock()
    popen = MagicMock(return_value=process)
    monkeypatch.setattr("scripts.jetson.foundation_runner.subprocess.Popen", popen)
    runner = FoundationRunner()
    monkeypatch.setattr(
        runner, "_read", lambda timeout: {"ready": True, "identity": {"model": "test"}}
    )
    runner.ensure_ready()
    assert popen.call_args.args[0][0] == "/managed/.foundation-venv/bin/python"
    assert Path(runner.log.name) == tmp_path / "logs/worker.log"
    runner.close()


def test_preview_uses_configurable_internal_tracking_port(tmp_path, monkeypatch):
    monkeypatch.setenv("MULTICAMERA_TRACKING_URL", "http://127.0.0.1:18093")
    captured = {}

    async def handler(request):
        return None

    def proxy(*args, **kwargs):
        captured.update(kwargs)
        result = MagicMock()
        result.state = result.control = result.replay = result.frame = handler
        result.preview.frame = result.preview.stream = handler
        return result

    monkeypatch.setattr("scripts.jetson.preview.TrackingProxy", proxy)
    # Existing preview fixture has validated per-device calibration; this minimal app
    # only exercises constructor wiring and does not start network/model work.
    monkeypatch.chdir(tmp_path)
    app = make_preview_app({"devices": []}, tmp_path)
    assert captured["base_url"] == "http://127.0.0.1:18093"
    assert app is not None


def test_checkpoint_identity_is_checked_before_model_import(tmp_path, monkeypatch):
    import sys
    import hashlib
    from scripts.jetson import foundation_worker

    file = tmp_path / "checkpoint.pth"
    file.write_bytes(b"checkpoint fixture")
    expected = hashlib.sha256(file.read_bytes()).hexdigest()
    assert foundation_worker.checkpoint_digest(file, expected) == expected
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "worker",
            "--root",
            str(tmp_path),
            "--checkpoint",
            str(file),
            "--expected-sha256",
            "0" * 64,
        ],
    )
    import pytest

    with pytest.raises(ValueError, match="refusing deserialization"):
        foundation_worker.main()


def test_full_checkpoint_initialization_cannot_fetch_redundant_backbone(monkeypatch):
    import sys
    from types import SimpleNamespace
    from scripts.jetson.foundation_worker import checkpoint_backbone_initialization

    calls = []

    def create(name, **kwargs):
        if name == "edgenext_small" and kwargs.get("pretrained"):
            raise AssertionError("Unexpected backbone download")
        calls.append((name, kwargs))
        return "initialized"

    fake = SimpleNamespace(create_model=create)
    monkeypatch.setitem(sys.modules, "timm", fake)
    with checkpoint_backbone_initialization():
        assert fake.create_model("edgenext_small", pretrained=True) == "initialized"
    assert fake.create_model is create
    assert calls[0][1]["pretrained"] is False
    import pytest

    with pytest.raises(RuntimeError):
        with checkpoint_backbone_initialization():
            raise RuntimeError("constructor failed")
    assert fake.create_model is create
