"""Read-only navigation to verified presentation records; never runs CV or an LLM."""

import html
import json

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

from .config import ROOT
from .review_context import ContextBinder
from .store import DomainError
from .vision import file_hash

PROJECT = ROOT.parents[1]
MANIFEST = PROJECT / "data/mainline-delivery-v1/demo-cases.json"


def catalogue(settings):
    if not MANIFEST.is_file():
        raise DomainError(503, "DEMO_NOT_READY", "Presentation records have not been prepared")
    value = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if value["pipeline_sha256"] != file_hash(settings.branch_manifest):
        raise DomainError(409, "DEMO_VERSION_CHANGED", "Revalidate the presentation records for this version")
    return value


def router(runtime, settings):
    routes = APIRouter()

    @routes.get("/demo", response_class=HTMLResponse)
    def guide():
        path = PROJECT / "docs/当前主线-演示与交付.html"
        if not path.is_file():
            return HTMLResponse(
                '<h1>Presentation guide is being prepared</h1><a href="/">Open review workspace</a>'
            )
        return HTMLResponse(path.read_text(encoding="utf-8"), headers={"Cache-Control": "no-store"})

    @routes.get("/demo/open/{case_id}", response_class=HTMLResponse)
    def open_case(case_id: str):
        cases = catalogue(settings)["cases"]
        case = next((c for c in cases if c["id"] == case_id), None)
        if case is None:
            raise DomainError(404, "DEMO_CASE_UNKNOWN", "Choose a listed presentation case")
        store = runtime().store
        if store.active_runs():
            raise DomainError(
                409, "DEMO_REVIEW_BUSY", "Wait for the active request before switching presentation records"
            )
        session = store.get("session", case["workspace"]["session_id"])
        if session["clip_id"] != case["workspace"]["clip_id"]:
            raise DomainError(409, "DEMO_SOURCE_CHANGED", "Presentation source changed")
        clip = store.get("clip", session["clip_id"])
        if clip["asset_id"] != case["workspace"]["asset_id"]:
            raise DomainError(409, "DEMO_SOURCE_CHANGED", "Presentation media changed")
        rid = case.get("run_id")
        if rid:
            run = store.get_run(rid)
            if run["session_id"] != session["id"] or run["status"] not in {
                "completed",
                "partially_completed",
            }:
                raise DomainError(409, "DEMO_SOURCE_CHANGED", "Presentation record is not ready")
            binder = ContextBinder(store, settings)
            report, _, _ = binder.report(session, run["report_id"])
            preview = binder.preview(session["id"], report["id"])
            if preview.source_status != "available":
                raise DomainError(410, "DEMO_SOURCE_UNAVAILABLE", "Presentation evidence is unavailable")
        else:
            if ContextBinder(store, settings).preview(session["id"]).source is not None:
                raise DomainError(
                    409,
                    "DEMO_ALREADY_REVIEWED",
                    "This practice has already been analysed; choose a new one for the first-review scenario",
                )
        workspace = {k: case["workspace"][k] for k in ["session_id", "clip_id", "asset_id", "name"]}
        payload = json.dumps({"workspace": workspace, "run_id": rid}, ensure_ascii=False).replace(
            "<", "\\u003c"
        )
        title = html.escape(case["title"])
        return HTMLResponse(
            '<!doctype html><meta charset="utf-8"><title>'
            + title
            + "</title><p>Opening "
            + title
            + "…</p><script>const c="
            + payload
            + ';localStorage.setItem("cpr.workspace",JSON.stringify(c.workspace));if(c.run_id)localStorage.setItem("cpr.runId",JSON.stringify(c.run_id));else localStorage.removeItem("cpr.runId");location.replace("/");</script>',
            headers={"Cache-Control": "no-store"},
        )

    return routes
