from __future__ import annotations

import logging
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from filelock import FileLock
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from .config import Settings
from .graph import ReviewGraph
from .llm import create_model_gateway
from .models import ToolRequest
from .ports import ScaffoldPolicy, StubAssessment, StubVisionTool
from .store import Store
from .vision import RealVisionTool, VideoAssessment, VideoPolicy, VisionError, file_hash

log = logging.getLogger("cpr.runtime")


class Runtime:
    def __init__(self, settings: Settings):
        self.settings = settings
        settings.data_dir.mkdir(parents=True, exist_ok=True)
        self.process_lock = FileLock(settings.data_dir / "runtime.lock")
        self.process_lock.acquire(timeout=0)
        self.store = Store(settings.data_dir / "business.sqlite3", settings.owner_id)
        self.checkpoint_conn = sqlite3.connect(
            settings.data_dir / "checkpoints.sqlite3", check_same_thread=False
        )
        self.checkpoint_conn.execute("PRAGMA journal_mode=WAL")
        self.checkpointer = SqliteSaver(self.checkpoint_conn)
        self.stop_event = threading.Event()
        self.gateway = create_model_gateway(settings)
        if settings.vision_mode == "real":
            self.bundle_sha256 = file_hash(settings.vision_manifest)
            assessment, policy = VideoAssessment(self.store, settings), VideoPolicy(settings)
            if settings.vision_pipeline == "branches":
                from .branch_vision import BranchVisionTool, validate_pipeline

                validate_pipeline(settings)
                self.worker = BranchVisionTool(settings, self.store, self.stop_event)
                from .module_runtime import SlotAssessment, SlotPolicy, selection

                if selection(settings) is not None:
                    assessment, policy = (
                        SlotAssessment(self.store, settings),
                        SlotPolicy(settings, self.store),
                    )
            else:
                self.worker = RealVisionTool(settings, self.store, self.stop_event)
        else:
            self.bundle_sha256 = None
            assessment, policy = StubAssessment(), ScaffoldPolicy()
            self.worker = StubVisionTool(settings)
        graph_type = ReviewGraph
        if settings.context_mode == "bound-v1":
            from .context_graph import ContextReviewGraph

            graph_type = ContextReviewGraph
        self.graph = graph_type(
            self.store,
            settings,
            self.checkpointer,
            self.gateway,
            assessment,
            policy,
        ).compiled
        self.wake = threading.Event()
        self.tool_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="vision-tool")
        self.inflight: set[str] = set()
        self.inflight_lock = threading.Lock()
        self.coordinator = threading.Thread(target=self._loop, daemon=True, name="graph-coordinator")
        self.coordinator.start()
        log.info("runtime_started")

    def notify(self):
        self.wake.set()

    def close(self):
        self.stop_event.set()
        self.wake.set()
        self.coordinator.join(timeout=10)
        self.tool_pool.shutdown(wait=True, cancel_futures=True)
        if self.coordinator.is_alive():
            raise RuntimeError("Coordinator did not stop; refusing to close its live database")
        self.checkpoint_conn.close()
        self.store.close()
        self.process_lock.release()
        log.info("runtime_stopped")

    def _loop(self):
        while not self.stop_event.is_set():
            try:
                for run in self.store.active_runs():
                    if self.stop_event.is_set():
                        break
                    self._drive(run)
            except Exception:
                log.exception("coordinator_tick_failed")
            self.wake.wait(0.08)
            self.wake.clear()

    def _drive(self, run):
        config = {"configurable": {"thread_id": f"review:{run['id']}"}, "recursion_limit": 100}
        if run["graph_version"] != self.settings.graph_version:
            self.store.commit_run(run["id"], "failed", None, "GRAPH_VERSION_MISMATCH")
            return
        try:
            snapshot = self.graph.get_state(config)
            command = None
            if snapshot.interrupts:
                pending = snapshot.interrupts[0]
                self.store.save_interrupt_id(run["id"], pending.id)
                run = self.store.get_run(run["id"])
                wait = run["wait"] or {}
                expired_user = (
                    wait.get("kind") == "user" and wait.get("expires_at", float("inf")) < time.time()
                )
                timed_out = wait.get("kind") == "tool" and (run["deadline"] or float("inf")) < time.time()
                if run["cancel_requested"] or expired_user or timed_out:
                    command = Command(resume={pending.id: {"kind": "control"}})
                elif wait.get("kind") == "user":
                    if not run["resume_payload"]:
                        return
                    command = Command(resume={pending.id: run["resume_payload"]})
                elif wait.get("kind") == "tool":
                    execution = self.store.get_tool(wait["execution_id"])
                    if execution["status"] in {"succeeded", "failed", "cancelled"}:
                        command = Command(
                            resume={pending.id: {"kind": "tool_complete", "execution_id": execution["id"]}}
                        )
                    else:
                        self._dispatch(execution)
                        return
                else:
                    raise RuntimeError("Checkpoint wait does not match business wait")
            elif snapshot.values:
                if not snapshot.next:
                    # N12 should have committed. Do not silently leave a run active forever.
                    self.store.commit_run(run["id"], "failed", None, "MISSING_TERMINAL_COMMIT")
                    return
                command = None  # Continue an interrupted process between completed nodes.
            else:
                command = {"run_id": run["id"], "node_seq": 0, "receipts": {}, "failed_tools": []}
            log.info("graph_advance", extra={"run_id": run["id"], "trace_id": run["trace_id"]})
            self.graph.invoke(command, config)
            latest = self.graph.get_state(config)
            if latest.interrupts:
                self.store.save_interrupt_id(run["id"], latest.interrupts[0].id)
        except Exception:
            log.exception("graph_execution_failed", extra={"run_id": run["id"], "trace_id": run["trace_id"]})
            self.store.commit_run(run["id"], "failed", None, "ORCHESTRATOR_ERROR")

    def _dispatch(self, execution):
        with self.inflight_lock:
            if execution["id"] in self.inflight:
                return
            self.inflight.add(execution["id"])
        self.tool_pool.submit(self._execute, execution)

    def _execute(self, execution):
        started = time.perf_counter()
        request = ToolRequest.model_validate(execution["request"])
        try:
            if self.stop_event.is_set():
                return
            current = self.store.update_tool(execution["id"], status="running")
            if current["status"] == "cancelled":
                return
            result = self.worker.execute(request)
            saved = self.store.update_tool(
                execution["id"],
                status="succeeded",
                output=result.model_dump(mode="json"),
                duration_ms=round((time.perf_counter() - started) * 1000, 2),
            )
            self.store.event(
                request.run_id,
                "tool.finished",
                {
                    "execution_id": execution["id"],
                    "tool_id": request.tool_id,
                    "is_stub": result.is_stub,
                    "status": saved["status"],
                },
                dedupe="tool:" + execution["id"],
            )
        except Exception as exc:
            if isinstance(exc, VisionError) and exc.code == "VISION_CANCELLED" and self.stop_event.is_set():
                # A graceful server stop preserves the checkpoint and lets a new runtime retry
                # into a fresh attempt directory. User cancellation still wins in update_tool.
                self.store.update_tool(execution["id"], status="queued", error=None)
                return
            log.exception("tool_execution_failed", extra={"run_id": request.run_id})
            self.store.update_tool(
                execution["id"],
                status="failed",
                error={
                    "code": exc.code
                    if isinstance(exc, VisionError)
                    else ("VISION_TOOL_ERROR" if self.settings.vision_mode == "real" else "STUB_TOOL_ERROR"),
                    "type": type(exc).__name__,
                },
                duration_ms=round((time.perf_counter() - started) * 1000, 2),
            )
        finally:
            with self.inflight_lock:
                self.inflight.discard(execution["id"])
            self.notify()
