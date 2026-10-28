"""Reference provenance and graph boundaries. Synthetic bytes are isolated unit fixtures, not CPR evidence."""

from __future__ import annotations

import io
import json
import tarfile
from dataclasses import replace
from pathlib import Path
from uuid import uuid4

import pytest
from cpr_agent.api import create_app
from cpr_agent.config import Settings
from cpr_agent.references import ReferenceCatalogue, ReferenceError, sha
from cpr_agent.vision import RealVisionTool
from fastapi.testclient import TestClient
from test_real_video import make_prediction, submit, uploaded_session, wait


@pytest.fixture
def local_catalog(tmp_path):
    root = tmp_path / "project"
    assets = root / "data/public_cpr/assets"
    assets.mkdir(parents=True)
    records = root / "data/public_cpr/rgb-acceptance-v1/records"
    entries, lines = [], []
    for label, take in [(0, "r00"), (4, "r00"), (4, "r01")]:
        folder = records / f"SupC{label:02d}--{take}--ch0"
        folder.mkdir(parents=True)
        clip = folder / "video.mp4"
        clip.write_bytes(b"unit-reference-container-" + str(label).encode() + take.encode())
        entry = dict(
            id=f"cpr-c{label:02d}-{take}-ch0",
            record_id=f"SupC{label:02d}/{take}/ch0",
            label_id=label,
            dataset_label="Correct" if label == 0 else "Bending Arms",
            view="ch0",
            media_rel=clip.relative_to(root).as_posix(),
            media_sha256=sha(clip),
            manifest_rel=(folder / "manifest.json").relative_to(root).as_posix(),
            annotation_member="./ann/trainlist.txt",
            annotation_line=f"SupC14_Dataset/SupC{label:02d}/{take}/ch0 8 {label}",
            frames=8,
            fps=25,
        )
        manifest = dict(
            dataset_revision="b8667b8fd6a70ac63cc09fc6b763a9459b7c160a",
            original_labels=[label],
            record_id=entry["record_id"],
            video_sha256=entry["media_sha256"],
            annotation_frame_dir=entry["annotation_line"].split()[0],
            label_names=[entry["dataset_label"]],
            view="ch0",
            video_validation={"decoded_frames": 8, "fps": 25},
        )
        (folder / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        entry["manifest_sha256"] = sha(folder / "manifest.json")
        entries.append(entry)
        lines.append(entry["annotation_line"])
    with tarfile.open(assets / "ann.tar.gz", "w:gz") as tar:
        data = "\n".join(lines).encode()
        member = tarfile.TarInfo("./ann/trainlist.txt")
        member.size = len(data)
        tar.addfile(member, io.BytesIO(data))
    path = root / "data/reference-v1/catalog.json"
    path.parent.mkdir()
    value = dict(
        schema="cpr-reference-catalog-v1",
        dataset="CPR-Coach",
        dataset_url="https://huggingface.co/datasets/ShunliWang/CPR-Coach",
        repository_url="https://github.com/Shunli-Wang/CPR-Coach",
        dataset_revision="b8667b8fd6a70ac63cc09fc6b763a9459b7c160a",
        annotation_rel="data/public_cpr/assets/ann.tar.gz",
        annotation_sha256=sha(assets / "ann.tar.gz"),
        license_note="Unit fixture, not a dataset license.",
        entries=entries,
    )
    path.write_text(json.dumps(value), encoding="utf-8")
    return root, path, value


def catalogue(fixture):
    return ReferenceCatalogue(fixture[1], root=fixture[0])


def test_reference_search_is_label_bound_and_excludes_identical_media(local_catalog):
    cat = catalogue(local_catalog)
    request = cat.search([4], exclude_sha=local_catalog[2]["entries"][1]["media_sha256"])
    result = cat.verify(request)
    assert result["status"] == "available"
    assert [item["label_id"] for item in result["items"]] == [0, 4]
    assert result["items"][1]["id"] == "cpr-c04-r01-ch0"
    assert all("media_rel" not in entry for entry in result["items"])
    assert "not professional certification" in result["notice"]


def test_general_reference_does_not_claim_missing_category(local_catalog):
    result = catalogue(local_catalog).verify(catalogue(local_catalog).search([5]))
    assert result["status"] == "available" and [i["label_id"] for i in result["items"]] == [0]
    assert result["unavailable_label_ids"] == [5]


@pytest.mark.parametrize("change", ["path", "label", "annotation", "url", "manifest", "video"])
def test_reference_changes_cannot_be_presented_as_verified(local_catalog, change):
    root, path, value = local_catalog
    cat = catalogue(local_catalog)
    request = cat.search([4])
    if change == "video":
        (root / value["entries"][1]["media_rel"]).write_bytes(b"changed")
    elif change == "manifest":
        (root / value["entries"][1]["manifest_rel"]).write_text("{}", encoding="utf-8")
    else:
        if change == "path":
            value["entries"][1]["media_rel"] = "../../secret.mp4"
        if change == "label":
            value["entries"][1]["label_id"] = 0
        if change == "annotation":
            value["entries"][1]["annotation_line"] = "invented 8 4"
        if change == "url":
            value["dataset_url"] = "javascript:alert(1)"
        path.write_text(json.dumps(value), encoding="utf-8")
    result = cat.verify(request)
    assert all(i["id"] != "cpr-c04-r00-ch0" for i in result["items"])
    if change not in ["video", "manifest"]:
        assert result["status"] == "unavailable"


def test_catalogue_disabled_and_path_escape(local_catalog):
    cat = ReferenceCatalogue(None, root=local_catalog[0])
    assert cat.search([4])["status"] == "unavailable"
    with pytest.raises(ReferenceError):
        catalogue(local_catalog)._path("../../secret.mp4", media=True)


@pytest.fixture
def reference_client(local_catalog, tmp_path, monkeypatch):
    import cpr_agent.references as refs

    original = refs.ReferenceCatalogue
    monkeypatch.setattr(refs, "ReferenceCatalogue", lambda path: original(path, root=local_catalog[0]))
    monkeypatch.setattr(RealVisionTool, "execute", make_prediction)
    settings = Settings(
        data_dir=tmp_path / "runtime",
        vision_mode="real",
        ml_python=Path(__import__("sys").executable),
        context_mode="bound-v1",
        reference_catalog=local_catalog[1],
    )
    with TestClient(create_app(settings)) as client:
        yield client, settings


def ref_turn(client, sid, report_id=None, key=None):
    body = {
        "goal": {"intent": "show_example", "target_label_ids": [4], "analysis_mode": "existing_only"},
        "context": {"source_report_id": report_id},
    }
    response = client.post(
        f"/api/sessions/{sid}/turns", json=body, headers={"Idempotency-Key": key or str(uuid4())}
    )
    assert response.status_code == 202, response.text
    return response.json()["run_id"]


def test_reference_uses_existing_source_without_cv_or_relabelling(reference_client):
    client, _ = reference_client
    sid, _, _ = uploaded_session(client)
    analysed = wait(client, submit(client, sid, targets=[4]))
    key = str(uuid4())
    rid = ref_turn(client, sid, analysed["report_id"], key)
    assert ref_turn(client, sid, analysed["report_id"], key) == rid
    run = wait(client, rid)
    assert run["status"] == "completed"
    assert run["response"]["prediction"] is None and run["response"]["classification_claims"] == []
    assert client.get(f"/api/runs/{rid}/tools").json() == []
    result = client.get(f"/api/runs/{rid}/references").json()
    assert result["status"] == "available"
    assert (
        client.get(f"/api/runs/{analysed['id']}").json()["response"]["prediction"]
        == analysed["response"]["prediction"]
    )
    identity = result["items"][0]["id"]
    assert (
        client.get(
            f"/api/runs/{rid}/reference-clips/{identity}/media", headers={"Range": "bytes=0-9"}
        ).status_code
        == 206
    )
    assert client.get(f"/api/runs/{rid}/reference-clips/cpr-c05-r00-ch0/media").status_code == 410
    events = client.get(f"/api/runs/{rid}/events").json()["events"]
    assert len([e for e in events if e["kind"] == "reference.lookup"]) == 1


def test_missing_current_media_hides_reference_not_a_500(reference_client):
    client, settings = reference_client
    sid, asset, _ = uploaded_session(client)
    analysed = wait(client, submit(client, sid, targets=[4]))
    rid = ref_turn(client, sid, analysed["report_id"])
    wait(client, rid)
    (settings.data_dir / asset["storage_rel"]).unlink()
    response = client.get(f"/api/runs/{rid}/references")
    assert response.status_code == 200 and response.json()["status"] == "unavailable"


@pytest.mark.parametrize(
    "field,value", [("session_id", "other-session"), ("clip_id", "other-clip"), ("requested_label_ids", [5])]
)
def test_lookup_context_cannot_be_rebound(reference_client, field, value):
    client, _ = reference_client
    sid, _, _ = uploaded_session(client)
    rid = ref_turn(client, sid)
    wait(client, rid)
    store = client.app.state.runtime.store
    obj = store.get("reference_lookup", "reference:" + rid)
    store.put("reference_lookup", {**obj, field: value})
    assert client.get(f"/api/runs/{rid}/references").json()["status"] == "unavailable"


def test_catalogue_off_returns_no_invented_references(reference_client):
    client, settings = reference_client
    # A second app uses the same read-only catalogue policy but its own persistence.
    with TestClient(
        create_app(replace(settings, data_dir=settings.data_dir.parent / "disabled", reference_catalog=None))
    ) as disabled:
        sid, _, _ = uploaded_session(disabled)
        rid = ref_turn(disabled, sid)
        run = wait(disabled, rid)
        assert not disabled.get(f"/api/runs/{rid}").json().get("reference_requested")
        assert disabled.get(f"/api/runs/{rid}/references").json()["status"] == "not_requested"
        assert run["response"]["prediction"] is None


@pytest.mark.parametrize("bindings", [None, []])
def test_damaged_saved_binding_is_unavailable_not_500(reference_client, bindings):
    client, _ = reference_client
    sid, _, _ = uploaded_session(client)
    rid = ref_turn(client, sid)
    wait(client, rid)
    store = client.app.state.runtime.store
    obj = store.get("reference_lookup", "reference:" + rid)
    store.put("reference_lookup", {**obj, "entry_bindings": bindings})
    response = client.get(f"/api/runs/{rid}/references")
    assert response.status_code == 200 and response.json()["status"] == "unavailable"


def test_cancelled_reference_is_not_available(reference_client, monkeypatch):
    import time

    import cpr_agent.references as refs

    original = refs.prepare_lookup

    def pause_after_lookup(*args):
        result = original(*args)
        time.sleep(0.3)
        return result

    monkeypatch.setattr(refs, "prepare_lookup", pause_after_lookup)
    client, _ = reference_client
    sid, _, _ = uploaded_session(client)
    rid = ref_turn(client, sid)
    until = time.monotonic() + 5
    while time.monotonic() < until:
        if refs.persisted_lookup(client.app.state.runtime.store, rid) is not None:
            break
        time.sleep(0.01)
    response = client.post(f"/api/runs/{rid}/cancel", headers={"Idempotency-Key": str(uuid4())})
    assert response.status_code == 202
    run = wait(client, rid)
    assert run["status"] == "cancelled" and run["response"] is None
    assert client.get(f"/api/runs/{rid}/references").json()["status"] == "unavailable"


def test_changed_catalogue_invalidates_saved_receipt(local_catalog):
    cat = catalogue(local_catalog)
    request = cat.search([4])
    value = local_catalog[2]
    value["license_note"] = "A changed catalog version"
    local_catalog[1].write_text(json.dumps(value), encoding="utf-8")
    assert cat.verify(request)["status"] == "unavailable"
