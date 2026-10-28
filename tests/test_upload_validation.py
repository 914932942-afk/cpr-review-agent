"""CPU input preflight contracts; real decoding is separately checked with the ML interpreter."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from cpr_agent.api import create_app
from cpr_agent.config import ROOT, Settings
from cpr_agent.upload_validation import INPUT_ERRORS
from fastapi.testclient import TestClient


def reply(value):
    return SimpleNamespace(
        returncode=0,
        stdout=json.dumps(
            {
                "policy_sha256": hashlib.sha256((ROOT / "scripts/vision_worker.py").read_bytes()).hexdigest(),
                **value,
            }
        ).encode(),
    )


@pytest.fixture
def client(tmp_path):
    with TestClient(
        create_app(
            Settings(
                data_dir=tmp_path, vision_mode="real", ml_python=Path(sys.executable), upload_preflight=True
            )
        )
    ) as c:
        yield c


def upload(c):
    return c.post(
        "/api/assets", files={"file": ("clip.mp4", b"\0\0\0\x18ftypmp42" + b"\0" * 24, "video/mp4")}
    )


def test_valid_probe_persists_facts_without_creating_a_run(client, monkeypatch):
    def run(command, **kwargs):
        assert command[1].endswith("probe_upload.py")
        assert not any(k.startswith(("DEEPSEEK_", "CPR_")) for k in kwargs["env"])
        assert kwargs["timeout"] == 35
        return reply(
            {"status": "passed", "metadata": {"decoded_frames": 100, "fps": 25, "dimensions_hw": [720, 1280]}}
        )

    monkeypatch.setattr("cpr_agent.upload_validation.subprocess.run", run)
    result = upload(client)
    assert result.status_code == 201
    assert result.json()["inspection"]["duration_seconds"] == 4
    assert client.get("/api/workspaces").json() == []
    assert client.get("/api/capabilities").json()["upload_preflight"] is True


@pytest.mark.parametrize("code", list(INPUT_ERRORS))
def test_invalid_input_is_precise_and_leaves_no_asset_or_task(client, monkeypatch, code):
    monkeypatch.setattr(
        "cpr_agent.upload_validation.subprocess.run",
        lambda *a, **k: reply({"status": "rejected", "code": code}),
    )
    response = upload(client)
    assert response.status_code == 422 and response.json()["error"]["error_code"] == code
    store = client.app.state.runtime.store
    assert store.list_objects("asset") == [] and store.active_runs() == []
    assert not list(client.app.state.runtime.settings.data_dir.joinpath("media").iterdir())


@pytest.mark.parametrize("kind", ["timeout", "invalid_output", "invalid_metadata", "changed_policy"])
def test_unavailable_probe_never_accepts_the_file(client, monkeypatch, kind):
    def run(*args, **kwargs):
        if kind == "timeout":
            raise subprocess.TimeoutExpired("probe", 35)
        if kind == "invalid_output":
            return SimpleNamespace(returncode=0, stdout=b"not json")
        if kind == "changed_policy":
            return SimpleNamespace(returncode=0, stdout=b'{"policy_sha256":"changed"}')
        return reply(
            {"status": "passed", "metadata": {"decoded_frames": 3, "fps": 25, "dimensions_hw": [720, 1280]}}
        )

    monkeypatch.setattr("cpr_agent.upload_validation.subprocess.run", run)
    assert upload(client).status_code == 503
    assert client.app.state.runtime.store.list_objects("asset") == []
