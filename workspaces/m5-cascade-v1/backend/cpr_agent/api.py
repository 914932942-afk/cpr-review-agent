from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from contextlib import asynccontextmanager
from logging.handlers import RotatingFileHandler
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException, Query, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from .config import ROOT, Settings
from .context_contracts import ContextResumeRequest, ContextSelection, ContextTurnCreate
from .evidence import EvidenceError, build_review_evidence
from .graph import NORMAL_ROUTES
from .grounded_review import verified_grounded_review
from .models import (
    CANDIDATE_LABELS,
    NODE_NAMES,
    TERMINAL,
    ClipCreate,
    ReviewResponse,
    SessionCreate,
)
from .ports import MODULES, TOOLS
from .practice_continuity import ContinuityUpdate, PracticeContinuity
from .report_view import render_saved_review
from .review_context import ContextBinder
from .review_notebook import NotebookUpdate, ReviewNotebook
from .review_records import ReviewRecords
from .runtime import Runtime
from .store import DomainError, dumps, uid
from .upload_validation import UploadInspector
from .vision import LABEL_NAMES, VisionError, validate_receipt


class JSONFormatter(logging.Formatter):
    def format(self, record):
        value = {
            "time": time.time(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for field in ("run_id", "trace_id"):
            if hasattr(record, field):
                value[field] = getattr(record, field)
        if record.exc_info:
            value["exception"] = self.formatException(record.exc_info)
        return dumps(value)


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    upload_inspector = UploadInspector(settings)

    @asynccontextmanager
    async def lifespan(app):
        settings.data_dir.mkdir(parents=True, exist_ok=True)
        logs = settings.data_dir / "logs"
        logs.mkdir(exist_ok=True)
        handler = RotatingFileHandler(
            logs / "runtime.jsonl", maxBytes=2_000_000, backupCount=3, encoding="utf-8"
        )
        handler.setFormatter(JSONFormatter())
        logger = logging.getLogger("cpr")
        logger.setLevel(logging.INFO)
        logger.addHandler(handler)
        runtime = Runtime(settings)
        app.state.runtime = runtime
        try:
            yield
        finally:
            runtime.close()
            logger.removeHandler(handler)
            handler.close()

    app = FastAPI(
        title="CPR Review Agent",
        version="0.2.0",
        lifespan=lifespan,
        description="Local experimental review. Optional DeepSeek goal parsing is explicitly configured.",
    )

    def runtime() -> Runtime:
        return app.state.runtime

    def bundle_id():
        suffix = (
            ":branches:" + hashlib.sha256(settings.branch_manifest.read_bytes()).hexdigest()
            if settings.vision_pipeline == "branches"
            else ""
        )
        return (
            ("single-view-v15:" + runtime().bundle_sha256 + suffix)
            if runtime().bundle_sha256
            else settings.bundle_id
        )

    def envelope(request, status, code, message, fields=None):
        return JSONResponse(
            status_code=status,
            content={
                "error": {
                    "trace_id": request.state.request_id,
                    "error_code": code,
                    "message": message,
                    "retryable": status in {429, 503},
                    "field_errors": fields or [],
                }
            },
        )

    @app.middleware("http")
    async def request_trace(request: Request, call_next):
        request.state.request_id = uid()
        response = await call_next(request)
        response.headers["X-Request-ID"] = request.state.request_id
        response.headers["X-Content-Type-Options"] = "nosniff"
        return response

    @app.exception_handler(DomainError)
    async def domain_error(request, exc):
        return envelope(request, exc.status, exc.code, exc.message)

    @app.exception_handler(RequestValidationError)
    async def validation_error(request, exc):
        fields = [{"loc": list(e["loc"]), "message": e["msg"]} for e in exc.errors()]
        return envelope(request, 422, "INVALID_INPUT", "Check the request fields", fields)

    @app.exception_handler(HTTPException)
    async def http_error(request, exc):
        return envelope(request, exc.status_code, "HTTP_ERROR", str(exc.detail))

    @app.get("/api/health")
    def health():
        alive = runtime().coordinator.is_alive()
        return {
            "status": "ok" if alive else "unhealthy",
            "mode": "experimental" if settings.vision_mode == "real" else "scaffold",
            "coordinator_alive": alive,
            "llm": settings.llm_mode,
            "llm_details": runtime().gateway.status(),
            "vision": "v15-experimental" if settings.vision_mode == "real" else "placeholder",
            "persistence": "sqlite",
            "executor": "local-thread+ml-process" if settings.vision_mode == "real" else "local-thread",
            "bundle_sha256": runtime().bundle_sha256,
            "vision_pipeline": settings.vision_pipeline,
            "context_binding": settings.context_mode,
            "grounded_explanations": settings.context_mode == "bound-v1",
            "graph_version": settings.graph_version,
        }

    @app.get("/api/capabilities")
    def capabilities():
        module_status = {}
        if settings.vision_mode == "real" and settings.vision_pipeline == "branches":
            from .branch_vision import validate_pipeline

            validate_pipeline(settings)
            profile = json.loads(settings.branch_manifest.read_text(encoding="utf-8"))
            if profile.get("worker_entrypoint") == "scripts/m1_branch_worker.py":
                module_status["M1"] = "Validated stability improvement · active"
            elif profile.get("worker_entrypoint") in {
                "scripts/m1_fast_branch_worker.py",
                "scripts/module_slots_worker.py",
            }:
                module_status["M1"] = "Validated M1 · optimized implementation"
        slot_status = {}
        if settings.vision_mode == "real" and settings.vision_pipeline == "branches":
            from .modules.registry import build_registry

            registry = build_registry()
            for module_id, implementation in profile.get("module_slots", {}).items():
                spec = registry.spec(module_id, implementation)
                slot_status[module_id] = {
                    "implementation_id": implementation,
                    "kind": spec.kind,
                    "contract_version": spec.contract_version,
                    "algorithm_status": "baseline_only" if spec.kind == "baseline" else "member_selected",
                }
        return {
            "mode": "experimental" if settings.vision_mode == "real" else "scaffold",
            "modules": MODULES,
            "module_status": module_status,
            "module_slots": slot_status,
            "context_binding": settings.context_mode,
            "grounded_explanations": settings.context_mode == "bound-v1",
            "tools": (
                ["pose.extract", "skeleton.predict", "rgb.predict", "fusion.predict"]
                if settings.vision_pipeline == "branches"
                else ["video.analyze"]
            )
            if settings.vision_mode == "real"
            else list(TOOLS),
            "reference_tools": ["reference.search"] if settings.reference_catalog is not None else [],
            "practice_continuity": settings.context_mode == "bound-v1" and settings.vision_mode == "real",
            "upload_preflight": settings.upload_preflight and settings.vision_mode == "real",
            "review_workspace_tools": settings.context_mode == "bound-v1" and settings.vision_mode == "real",
            "vision_pipeline": settings.vision_pipeline,
            "candidate_labels": list(CANDIDATE_LABELS),
            "validated_cpr_labels": [],
            "experimental_labels": list(CANDIDATE_LABELS) if settings.vision_mode == "real" else [],
            "label_names": LABEL_NAMES,
            "llm": runtime().gateway.status(),
            "nodes": NODE_NAMES,
            "normal_routes": NORMAL_ROUTES,
            "notice": "v15 single-view clip-level engineering profile; no clinical validation."
            if (settings.vision_mode == "real")
            else "Candidate IDs exercise routing only. They are not validated recognition capabilities.",
        }

    @app.post("/api/assets", status_code=201)
    async def upload_asset(file: UploadFile):
        extension = Path(file.filename or "").suffix.lower()
        if extension not in {".mp4", ".mov", ".webm"}:
            raise DomainError(415, "MEDIA_TYPE", "Use MP4, MOV or WebM")
        asset_id = uid()
        media_dir = settings.data_dir / "media"
        media_dir.mkdir(exist_ok=True)
        target = media_dir / (asset_id + extension)
        part = target.with_suffix(extension + ".part")
        total, checksum, head = 0, hashlib.sha256(), b""
        try:
            with part.open("wb") as out:
                while chunk := await file.read(1024 * 1024):
                    total += len(chunk)
                    if total > settings.max_upload_bytes:
                        raise DomainError(413, "MEDIA_TOO_LARGE", "Upload limit is 50 MiB")
                    if not head:
                        head = chunk[:16]
                    checksum.update(chunk)
                    out.write(chunk)
            valid_magic = head.startswith(b"\x1aE\xdf\xa3") if extension == ".webm" else head[4:8] == b"ftyp"
            if not valid_magic:
                raise DomainError(
                    415, "INVALID_CONTAINER", "Container header does not match the selected format"
                )
            inspection = None
            if settings.upload_preflight and settings.vision_mode == "real":
                inspection = await asyncio.to_thread(upload_inspector.inspect, part)
            part.replace(target)
            asset = {
                "id": asset_id,
                "name": (file.filename or "video")[:180],
                "status": "stored" if settings.vision_mode == "real" else "ready_stub",
                "storage_rel": "media/" + target.name,
                "source": "upload",
                "sha256": checksum.hexdigest(),
                "size_bytes": total,
                "validation_level": "decoded_before_session" if inspection else "storage_and_magic_only",
                **({"inspection": inspection} if inspection else {}),
            }
            return runtime().store.put("asset", asset)
        finally:
            await file.close()
            part.unlink(missing_ok=True)

    @app.get("/api/assets/{asset_id}")
    def get_asset(asset_id: str):
        return runtime().store.get("asset", asset_id)

    @app.get("/api/assets/{asset_id}/media")
    def asset_media(asset_id: str):
        asset = runtime().store.get("asset", asset_id)
        if asset["source"] == "fixture":
            path = ROOT / "fixtures" / "workflow-fixture.mp4"
        else:
            path = (settings.data_dir / asset["storage_rel"]).resolve()
            if not path.is_relative_to(settings.data_dir.resolve()):
                raise DomainError(403, "STORAGE_BOUNDARY", "Invalid storage reference")
        if not path.is_file():
            raise DomainError(410, "MEDIA_UNAVAILABLE", "Media is no longer available")
        return FileResponse(path)

    @app.post("/api/clips", status_code=201)
    def create_clip(body: ClipCreate):
        asset = runtime().store.get("asset", body.asset_id)
        if asset["status"] not in {"ready_stub", "stored"}:
            raise DomainError(409, "ASSET_NOT_READY", "Asset is not ready")
        if settings.vision_mode == "real" and (body.start_seconds != 0 or body.end_seconds is not None):
            if asset["source"] != "fixture":
                raise DomainError(
                    422, "WHOLE_CLIP_ONLY", "Initial real-video profile analyses the complete uploaded clip"
                )
        return runtime().store.put(
            "clip",
            {
                "id": uid(),
                **body.model_dump(),
                "validation_level": ("decode_pending" if settings.vision_mode == "real" else "scaffold_only"),
            },
        )

    @app.post("/api/sessions", status_code=201)
    def create_session(body: SessionCreate):
        if body.clip_id:
            runtime().store.get("clip", body.clip_id)
        return runtime().store.put(
            "session",
            {
                "id": uid(),
                "clip_id": body.clip_id,
                "bundle_id": bundle_id(),
                "created_at": time.time(),
            },
        )

    @app.get("/api/sessions")
    def sessions():
        return runtime().store.list_objects("session")

    @app.get("/api/workspaces")
    def workspaces(limit: int = Query(default=50, ge=1, le=50)):
        return runtime().store.workspaces(bundle_id(), limit)

    @app.get("/api/sessions/{session_id}/history")
    def session_history(
        session_id: str,
        limit: int = Query(default=20, ge=1, le=50),
        before_turn_id: str | None = Query(default=None, max_length=120),
    ):
        return runtime().store.session_history(session_id, limit, before_turn_id)

    @app.get("/api/sessions/{session_id}")
    def get_session(session_id: str):
        return runtime().store.get("session", session_id)

    @app.get("/api/sessions/{session_id}/context")
    def session_context(
        session_id: str, source_report_id: str | None = Query(default=None, min_length=1, max_length=120)
    ):
        if settings.context_mode != "bound-v1":
            raise DomainError(409, "CONTEXT_NOT_ENABLED", "Bound review context is not enabled")
        return (
            ContextBinder(runtime().store, settings)
            .preview(session_id, source_report_id)
            .model_dump(mode="json")
        )

    def continuity():
        if settings.context_mode != "bound-v1":
            raise DomainError(409, "CONTEXT_NOT_ENABLED", "Bound review context is not enabled")
        return PracticeContinuity(runtime().store, settings)

    @app.get("/api/sessions/{session_id}/continuity")
    def practice_continuity(session_id: str):
        return continuity().view(session_id)

    @app.get("/api/sessions/{session_id}/notebook")
    def review_notebook(session_id: str):
        continuity()  # Product review workspace requires bound context.
        return ReviewNotebook(runtime().store).view(session_id)

    @app.post("/api/sessions/{session_id}/notebook")
    def save_review_notebook(
        session_id: str, body: NotebookUpdate, idempotency_key: str = Header(min_length=8, max_length=120)
    ):
        continuity()
        return ReviewNotebook(runtime().store).update(session_id, body, idempotency_key)

    @app.post("/api/sessions/{session_id}/continuity")
    def apply_practice_focus(
        session_id: str, body: ContinuityUpdate, idempotency_key: str = Header(min_length=8, max_length=120)
    ):
        return continuity().update(session_id, body, idempotency_key, bundle_id())

    @app.get("/api/sessions/{session_id}/continuity/recaps/{report_id}")
    def previous_practice_recap(session_id: str, report_id: str):
        service = continuity()
        target = runtime().store.get("session", session_id)
        source = service.source(target, report_id)
        return {**service.metadata(source), "recap": source["recap"], "belongs_to_previous_video": True}

    @app.post("/api/sessions/{session_id}/clip")
    def bind_clip(session_id: str, body: SessionCreate):
        if not body.clip_id:
            raise DomainError(422, "CLIP_REQUIRED", "clip_id is required")
        return runtime().store.bind_clip(session_id, body.clip_id)

    @app.post("/api/demo/session", status_code=201)
    def demo_session():
        fixture = ROOT / "fixtures" / "workflow-fixture.mp4"
        if not fixture.is_file():
            raise DomainError(503, "FIXTURE_MISSING", "The packaged workflow fixture is missing")
        asset = runtime().store.put(
            "asset",
            {
                "id": "workflow-fixture",
                "name": "Generated workflow fixture",
                "source": "fixture",
                "status": "ready_stub",
                "validation_level": "synthetic_fixture",
                "sha256": hashlib.sha256(fixture.read_bytes()).hexdigest(),
                "size_bytes": fixture.stat().st_size,
            },
        )
        clip = create_clip(ClipCreate(asset_id=asset["id"], end_seconds=2))
        session = create_session(SessionCreate(clip_id=clip["id"]))
        return {"session": session, "clip": clip, "asset": asset}

    @app.post("/api/sessions/{session_id}/turns", status_code=202)
    def create_turn(
        session_id: str, body: ContextTurnCreate, idempotency_key: str = Header(min_length=8, max_length=120)
    ):
        payload = body.model_dump(mode="json", exclude={"context"})
        factory = None
        if settings.context_mode == "bound-v1":
            selection = body.context or ContextSelection()
            payload["context_selection"] = selection.model_dump(mode="json")
            binder = ContextBinder(runtime().store, settings)

            def factory(run):
                return binder.snapshot(run, selection).model_dump(mode="json")
        elif body.context is not None:
            raise DomainError(409, "CONTEXT_NOT_ENABLED", "Bound review context is not enabled")
        result = runtime().store.create_turn(
            session_id,
            payload,
            idempotency_key,
            bundle_id(),
            settings.graph_version,
            "experimental" if settings.vision_mode == "real" else "scaffold",
            context_snapshot_factory=factory,
        )
        runtime().notify()
        return result

    @app.get("/api/turns/{turn_id}")
    def get_turn(turn_id: str):
        return runtime().store.get("turn", turn_id)

    @app.get("/api/runs/{run_id}")
    def get_run(run_id: str):
        from .references import persisted_lookup

        value = runtime().store.get_run(run_id)
        if persisted_lookup(runtime().store, run_id) is not None:
            value["reference_requested"] = True
        return value

    @app.get("/api/runs/{run_id}/references")
    def reference_lookup(run_id: str):
        from .references import verify_lookup

        return verify_lookup(runtime().store, settings, run_id)

    @app.get("/api/runs/{run_id}/reference-clips/{reference_id}/media")
    def reference_media(run_id: str, reference_id: str):
        from .references import ReferenceCatalogue, persisted_lookup, verify_lookup

        verified = verify_lookup(runtime().store, settings, run_id, entry_id=reference_id)
        if reference_id not in {entry["id"] for entry in verified.get("items", [])}:
            raise DomainError(410, "REFERENCE_UNAVAILABLE", "The requested reference is unavailable")
        receipt = persisted_lookup(runtime().store, run_id)
        path = ReferenceCatalogue(settings.reference_catalog).media(receipt, reference_id)
        return FileResponse(path, media_type="video/mp4")

    @app.get("/api/runs/{run_id}/tools")
    def get_tools(run_id: str):
        return runtime().store.tools_for_run(run_id)

    @app.get("/api/runs/{run_id}/events")
    async def run_events(run_id: str, request: Request, after_seq: int = Query(default=0, ge=0)):
        runtime().store.get_run(run_id)
        if "text/event-stream" not in request.headers.get("accept", ""):
            return {"events": runtime().store.events(run_id, after_seq)}

        async def stream():
            cursor = after_seq
            while not await request.is_disconnected():
                for event in runtime().store.events(run_id, cursor):
                    cursor = event["seq"]
                    yield f"id: {cursor}\nevent: progress\ndata: {json.dumps(event)}\n\n"
                if runtime().store.get_run(run_id)["status"] in TERMINAL:
                    yield "event: finished\ndata: {}\n\n"
                    return
                yield ": heartbeat\n\n"
                await asyncio.sleep(0.5)

        return StreamingResponse(
            stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"}
        )

    @app.post("/api/runs/{run_id}/resume", status_code=202)
    def resume(
        run_id: str, body: ContextResumeRequest, idempotency_key: str = Header(min_length=8, max_length=120)
    ):
        payload = body.model_dump(mode="json", exclude={"context_confirmation"})
        if settings.context_mode == "bound-v1":
            run = runtime().store.get_run(run_id)
            wait = run.get("wait") or {}
            confirmation = body.context_confirmation
            if not confirmation or confirmation.snapshot_id != wait.get("context_snapshot_id"):
                # Keep exact retries valid after the wait has already been consumed.
                turn = runtime().store.get("turn", run["turn_id"])
                snapshot_id = (turn.get("context_snapshot") or {}).get("snapshot_id")
                if not confirmation or confirmation.snapshot_id != snapshot_id:
                    raise DomainError(
                        409, "CONTEXT_CONFIRMATION_CHANGED", "Confirm the video and report shown in this form"
                    )
            if (
                wait.get("scope_confirmation_required")
                and body.goal.intent != "summarise"
                and not body.goal.target_label_ids
            ):
                raise DomainError(
                    422, "CONTEXT_SCOPE_CONFIRMATION_REQUIRED", "Select a category or choose a full summary"
                )
            payload["context_confirmation"] = confirmation.model_dump(mode="json")
        elif body.context_confirmation is not None:
            raise DomainError(409, "CONTEXT_NOT_ENABLED", "Bound review context is not enabled")
        result = runtime().store.submit_resume(run_id, payload, idempotency_key)
        runtime().notify()
        return result

    @app.post("/api/runs/{run_id}/cancel", status_code=202)
    def cancel(run_id: str):
        result = runtime().store.cancel(run_id)
        runtime().notify()
        return {
            "run_id": result["id"],
            "cancel_requested": result["cancel_requested"],
            "status": result["status"],
        }

    @app.get("/api/preferences")
    def preferences():
        return runtime().store.list_objects("preference")

    @app.get("/api/reports/{report_id}")
    def report(report_id: str):
        return runtime().store.get("report", report_id)

    @app.get("/api/reports/{report_id}/evidence")
    def report_evidence(report_id: str):
        """Revalidate recorded evidence on reopening; never runs inference or changes the report."""
        store = runtime().store
        snapshot = store.get("report", report_id)
        payload = snapshot.get("response") or {}
        if payload.get("mode") != "experimental" or not payload.get("prediction"):
            return {"status": "not_applicable", "review": None, "error_code": None}
        try:
            response = ReviewResponse.model_validate(payload)
            prediction = validate_receipt(response.prediction.model_dump(), store, settings)
            execution = store.get_tool(prediction.execution_id)
            report_run = store.get_run(snapshot["run_id"])
            source_run = store.get_run(execution["run_id"])
            if (
                execution.get("status") != "succeeded"
                or report_run.get("report_id") != snapshot["id"]
                or report_run.get("response") != payload
                or report_run["session_id"] != snapshot["session_id"]
                or execution.get("output") != prediction.model_dump()
                or source_run["clip_id"] != prediction.clip_id
                or report_run["clip_id"] != prediction.clip_id
            ):
                raise EvidenceError("RESULT_SOURCE_MISMATCH")
            review = build_review_evidence(
                prediction,
                response.classification_claims,
                settings.data_dir,
                execution["run_id"],
            )
            return {"status": "available", "review": review.model_dump(), "error_code": None}
        except (ValueError, VisionError, DomainError) as exc:
            code = getattr(exc, "code", "RESULT_RECORD_INVALID")
            return {"status": "unavailable", "review": None, "error_code": code}
        except OSError:
            return {"status": "unavailable", "review": None, "error_code": "RESULT_SOURCE_UNAVAILABLE"}

    @app.get("/api/reports/{report_id}/comparison-options")
    def comparison_options(report_id: str):
        return ReviewRecords(runtime().store, settings).options(report_id)

    @app.get("/api/review-comparison")
    def review_comparison(
        left_report_id: str = Query(min_length=1, max_length=120),
        right_report_id: str = Query(min_length=1, max_length=120),
    ):
        return ReviewRecords(runtime().store, settings).compare(left_report_id, right_report_id)

    @app.get("/api/reports/{report_id}/explanation")
    def report_explanation(report_id: str):
        snapshot = runtime().store.get("report", report_id)
        evidence = report_evidence(report_id)
        run = runtime().store.get_run(snapshot["run_id"])
        if run.get("grounded_review") != snapshot.get("grounded_review"):
            return {"status": "unavailable", "review": None, "error_code": "EXPLANATION_SOURCE_MISMATCH"}
        return verified_grounded_review(snapshot, evidence)

    @app.get("/api/reports/{report_id}/export", response_class=HTMLResponse)
    def export_report(report_id: str):
        snapshot = runtime().store.get("report", report_id)
        evidence = report_evidence(report_id)
        prediction = (snapshot.get("response") or {}).get("prediction") or {}
        video_name = None
        if prediction.get("asset_id"):
            try:
                video_name = runtime().store.get("asset", prediction["asset_id"])["name"]
            except DomainError:
                pass
        return HTMLResponse(
            render_saved_review(snapshot, evidence, video_name, report_explanation(report_id))
        )

    from .mainline_demo import router as mainline_demo_router

    app.include_router(mainline_demo_router(runtime, settings))

    dist = ROOT / "frontend" / "dist"
    if dist.is_dir():
        app.mount("/", StaticFiles(directory=dist, html=True), name="frontend")
    else:

        @app.get("/", response_class=HTMLResponse)
        def not_built():
            return '<h1>API ready</h1><p>Build frontend: npm --prefix frontend run build</p><a href="/docs">API docs</a>'

    return app


app = create_app()
