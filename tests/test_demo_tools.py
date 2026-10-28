"""Tests for acceptance tooling boundaries, not duplicate CV accuracy tests."""

import importlib
import json
import sys
from pathlib import Path

import httpx
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
demo = importlib.import_module("demo_scenarios")
freeze = importlib.import_module("freeze_demo")
preflight = importlib.import_module("demo_preflight")


@pytest.mark.parametrize(
    "url",
    [
        "https://example.com",
        "http://127.0.0.1",
        "http://user:pass@localhost:8005",
        "http://localhost:8005/other",
        "http://localhost:8005?key=secret",
        "http://localhost:8005/#part",
        "https://localhost:8005",
    ],
)
def test_external_or_ambiguous_demo_target_is_rejected(url):
    with pytest.raises(ValueError):
        demo.local_url(url)


def test_acceptance_record_cannot_be_overwritten(tmp_path):
    target = tmp_path / "acceptance.json"
    demo.write_new(target, {"passed": True})
    with pytest.raises(FileExistsError):
        demo.write_new(target, {"passed": False})
    assert json.loads(target.read_text())["passed"]


@pytest.mark.parametrize("name", ["../escape", "C:/outside", "stable/name", "-invalid"])
def test_freeze_target_cannot_escape_release_directory(tmp_path, name):
    with pytest.raises(ValueError):
        freeze.release_path(tmp_path, name)


def test_freeze_preserves_existing_release(tmp_path):
    path = tmp_path / "releases/stable-demo"
    path.mkdir(parents=True)
    with pytest.raises(ValueError, match="overwrite"):
        freeze.release_path(tmp_path, "stable-demo")


def test_pin_verification_rejects_modified_file_and_path_escape(tmp_path):
    source = tmp_path / "dependency.py"
    source.write_text("original")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"sources": {"dependency.py": preflight.sha(source)}}))
    assert preflight.check_pins(tmp_path, Path("manifest.json"), "sources") == 1
    source.write_text("changed")
    with pytest.raises(ValueError, match="changed"):
        preflight.check_pins(tmp_path, Path("manifest.json"), "sources")
    manifest.write_text(json.dumps({"sources": {"../outside.py": "0" * 64}}))
    with pytest.raises(ValueError, match="escapes"):
        preflight.check_pins(tmp_path, Path("manifest.json"), "sources")


def test_confirmation_wait_does_not_hang_after_failure():
    with httpx.Client(
        base_url="http://localhost:8005",
        transport=httpx.MockTransport(lambda r: httpx.Response(200, json={"status": "failed"}))
    ) as c:
        with pytest.raises(ValueError, match="before confirmation"):
            demo.DemoClient(c).wait("x", paused=True)


def test_resume_requires_an_actual_restart(tmp_path, monkeypatch):
    demo.write_new(tmp_path / "prepared.json", {"server_pid_before_restart": 123})
    monkeypatch.setattr(demo, "server_pid", lambda api: 123)
    with pytest.raises(ValueError, match="Actually restart"):
        demo.resume(None, tmp_path)


def test_process_lookup_rejects_untrusted_execution_marker():
    with pytest.raises(ValueError, match="Invalid execution"):
        demo.worker_pid("'; Get-ChildItem; '")


def test_followup_check_rejects_unexpected_vision_calls(monkeypatch):
    with httpx.Client() as c:
        api = demo.DemoClient(c)
        monkeypatch.setattr(api, "submit", lambda *a, **k: "run")
        monkeypatch.setattr(api, "wait", lambda *a, **k: {"status": "completed"})
        monkeypatch.setattr(api, "get", lambda *a: [{"tool": "unexpected"}])
        with pytest.raises(ValueError, match="unexpectedly ran vision"):
            api.followup({}, {"report_id": "saved-report"}, "explain", [4])
