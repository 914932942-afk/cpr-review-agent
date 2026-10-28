"""Local adapter. SQL facts and graph checkpoints deliberately use separate files."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from uuid import uuid4

from .models import TERMINAL


def uid() -> str:
    return str(uuid4())


def dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest(value: Any) -> str:
    return hashlib.sha256(dumps(value).encode()).hexdigest()


class DomainError(Exception):
    def __init__(self, status: int, code: str, message: str):
        self.status, self.code, self.message = status, code, message
        super().__init__(message)


class Store:
    def __init__(self, path: Path, owner: str):
        self.owner = owner
        self.lock = threading.RLock()
        self.conn = sqlite3.connect(path, check_same_thread=False, isolation_level=None, timeout=10)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript("""
            PRAGMA journal_mode=WAL;
            PRAGMA foreign_keys=ON;
            CREATE TABLE IF NOT EXISTS objects(
              kind TEXT NOT NULL,id TEXT PRIMARY KEY,owner TEXT NOT NULL,body TEXT NOT NULL,created REAL);
            CREATE INDEX IF NOT EXISTS objects_kind ON objects(kind,owner);
            CREATE TABLE IF NOT EXISTS runs(
              id TEXT PRIMARY KEY,session_id TEXT NOT NULL,status TEXT NOT NULL,body TEXT NOT NULL);
            CREATE INDEX IF NOT EXISTS run_session ON runs(session_id,status);
            CREATE TABLE IF NOT EXISTS tools(
              id TEXT PRIMARY KEY,run_id TEXT NOT NULL,status TEXT NOT NULL,body TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS events(
              run_id TEXT NOT NULL,seq INTEGER NOT NULL,kind TEXT NOT NULL,node_id TEXT,
              body TEXT NOT NULL,created REAL NOT NULL,dedupe TEXT,
              PRIMARY KEY(run_id,seq),UNIQUE(run_id,dedupe));
            CREATE TABLE IF NOT EXISTS requests(
              scope TEXT NOT NULL,key TEXT NOT NULL,hash TEXT NOT NULL,response TEXT NOT NULL,
              PRIMARY KEY(scope,key));
        """)

    @contextmanager
    def tx(self):
        with self.lock:
            self.conn.execute("BEGIN IMMEDIATE")
            try:
                yield self.conn
                self.conn.commit()
            except BaseException:
                self.conn.rollback()
                raise

    def close(self):
        self.conn.close()

    def _object(self, db, kind, object_id):
        row = db.execute(
            "SELECT body FROM objects WHERE kind=? AND id=? AND owner=?", (kind, object_id, self.owner)
        ).fetchone()
        if not row:
            raise DomainError(404, "NOT_FOUND", f"{kind} not found")
        return json.loads(row[0])

    def _put(self, db, kind, obj):
        db.execute(
            "INSERT INTO objects VALUES(?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET body=excluded.body",
            (kind, obj["id"], self.owner, dumps(obj), time.time()),
        )

    def put(self, kind, obj):
        with self.tx() as db:
            self._put(db, kind, obj)
        return obj

    def get(self, kind, object_id):
        with self.lock:
            return self._object(self.conn, kind, object_id)

    def list_objects(self, kind):
        with self.lock:
            return [
                json.loads(r[0])
                for r in self.conn.execute(
                    "SELECT body FROM objects WHERE kind=? AND owner=? ORDER BY created DESC",
                    (kind, self.owner),
                )
            ]

    def workspaces(self, bundle, limit=50):
        """Owner-scoped recent sessions; return display fields, never media storage paths."""
        with self.lock:
            items = []
            for session in self.list_objects("session"):
                clip_id = session.get("clip_id")
                asset_id, name = None, "No video attached"
                if clip_id:
                    try:
                        clip = self._object(self.conn, "clip", clip_id)
                        asset = self._object(self.conn, "asset", clip["asset_id"])
                        asset_id, name = asset["id"], asset["name"]
                    except DomainError:
                        name = "Video record unavailable"
                rows = self.conn.execute(
                    "SELECT body FROM runs WHERE session_id=?", (session["id"],)
                ).fetchall()
                runs = [json.loads(r[0]) for r in rows]
                runs = [r for r in runs if r["owner_id"] == self.owner]
                latest = max(runs, key=lambda r: (r["created_at"], r["id"]), default=None)
                items.append(
                    {
                        "session_id": session["id"],
                        "clip_id": clip_id,
                        "asset_id": asset_id,
                        "name": name,
                        "created_at": session["created_at"],
                        "last_active_at": max([session["created_at"]] + [r["updated_at"] for r in runs]),
                        "bundle_compatible": session["bundle_id"] == bundle,
                        "latest_run_id": latest["id"] if latest else None,
                        "latest_status": latest["status"] if latest else None,
                    }
                )
            items.sort(key=lambda s: (s["last_active_at"], s["session_id"]), reverse=True)
            return items[:limit]

    def session_history(self, session_id, limit=20, before_turn_id=None):
        """Stable cursor pagination, including unfinished and failed turns across restarts."""
        with self.lock:
            self._object(self.conn, "session", session_id)
            params = [self.owner, session_id]
            cursor_sql = ""
            if before_turn_id:
                turn = self._object(self.conn, "turn", before_turn_id)
                if turn["session_id"] != session_id:
                    raise DomainError(404, "NOT_FOUND", "Turn not found in this session")
                created = self.conn.execute(
                    "SELECT created FROM objects WHERE id=?", (before_turn_id,)
                ).fetchone()[0]
                cursor_sql = " AND (created<? OR (created=? AND id<?))"
                params.extend([created, created, before_turn_id])
            params.append(limit + 1)
            rows = self.conn.execute(
                "SELECT body,created FROM objects WHERE kind='turn' AND owner=? "
                "AND json_extract(body,'$.session_id')=?"
                + cursor_sql
                + " ORDER BY created DESC,id DESC LIMIT ?",
                params,
            ).fetchall()
            items = []
            for row in rows[:limit]:
                turn = json.loads(row["body"])
                run = self._run(self.conn, turn["run_id"])
                response = run.get("response") or {}
                goal = turn.get("resolved_goal") or turn.get("goal") or {}
                items.append(
                    {
                        "turn_id": turn["id"],
                        "run_id": run["id"],
                        "created_at": row["created"],
                        "message": turn.get("message", ""),
                        "intent": goal.get("intent"),
                        "target_label_ids": goal.get("target_label_ids", []),
                        "status": run["status"],
                        "error_code": run.get("error_code"),
                        "report_id": run.get("report_id"),
                        "title": response.get("title"),
                        "summary": response.get("message"),
                    }
                )
            recap = None
            row = self.conn.execute(
                "SELECT body FROM objects WHERE kind='report' AND owner=? "
                "AND json_extract(body,'$.session_id')=? "
                "AND json_extract(body,'$.grounded_review.intent')='summarise' "
                "ORDER BY created DESC,id DESC LIMIT 1",
                (self.owner, session_id),
            ).fetchone()
            if row:
                report = json.loads(row[0])
                saved_run = self._run(self.conn, report["run_id"])
                if saved_run["status"] == "completed" and saved_run.get("report_id") == report["id"]:
                    recap = {
                        "run_id": report["run_id"],
                        "report_id": report["id"],
                        "scope_label_ids": report["grounded_review"]["scope_label_ids"],
                        "created_at": report["created_at"],
                    }
            return {
                "items": items,
                "latest_recap": recap,
                "has_more": len(rows) > limit,
                "next_before_turn_id": items[-1]["turn_id"] if len(rows) > limit else None,
            }

    def _run(self, db, run_id):
        row = db.execute("SELECT body FROM runs WHERE id=?", (run_id,)).fetchone()
        if not row:
            raise DomainError(404, "NOT_FOUND", "Run not found")
        obj = json.loads(row[0])
        if obj["owner_id"] != self.owner:
            raise DomainError(404, "NOT_FOUND", "Run not found")
        return obj

    def _write_run(self, db, obj):
        obj["updated_at"] = time.time()
        db.execute(
            "INSERT INTO runs VALUES(?,?,?,?) ON CONFLICT(id) DO UPDATE SET status=excluded.status,body=excluded.body",
            (obj["id"], obj["session_id"], obj["status"], dumps(obj)),
        )

    def get_run(self, run_id):
        with self.lock:
            return self._run(self.conn, run_id)

    def update_run(self, run_id, **updates):
        with self.tx() as db:
            obj = self._run(db, run_id)
            if obj["status"] not in TERMINAL:
                obj.update(updates)
                self._write_run(db, obj)
            return obj

    def active_runs(self):
        with self.lock:
            return [
                json.loads(r[0])
                for r in self.conn.execute("SELECT body FROM runs")
                if json.loads(r[0])["status"] not in TERMINAL
            ]

    def _active_for_session(self, db, session_id):
        return any(
            json.loads(r[0])["status"] not in TERMINAL
            for r in db.execute("SELECT body FROM runs WHERE session_id=?", (session_id,))
        )

    def bind_clip(self, session_id, clip_id):
        with self.tx() as db:
            session = self._object(db, "session", session_id)
            self._object(db, "clip", clip_id)
            if session["clip_id"] or self._active_for_session(db, session_id):
                raise DomainError(409, "SESSION_BOUND_OR_BUSY", "Create a new session to change its video")
            session["clip_id"] = clip_id
            self._put(db, "session", session)
            return session

    def _cached(self, db, scope, key, body):
        row = db.execute(
            "SELECT hash,response FROM requests WHERE scope=? AND key=?", (scope, key)
        ).fetchone()
        if row:
            if row[0] != digest(body):
                raise DomainError(
                    409, "IDEMPOTENCY_CONFLICT", "The key was already used with different input"
                )
            return json.loads(row[1])

    def _remember(self, db, scope, key, body, response):
        db.execute("INSERT INTO requests VALUES(?,?,?,?)", (scope, key, digest(body), dumps(response)))

    def create_turn(
        self, session_id, body, key, bundle, graph_version, mode="scaffold", context_snapshot_factory=None
    ):
        with self.tx() as db:
            scope = f"{self.owner}:turn:{session_id}"
            cached = self._cached(db, scope, key, body)
            if cached:
                return cached
            session = self._object(db, "session", session_id)
            if session["bundle_id"] != bundle:
                raise DomainError(
                    409, "SESSION_BUNDLE_CHANGED", "Create a session for the current model bundle"
                )
            if self._active_for_session(db, session_id):
                raise DomainError(409, "SESSION_BUSY", "Finish or cancel the current run first")
            turn_id, run_id, now = uid(), uid(), time.time()
            turn = {"id": turn_id, "session_id": session_id, "run_id": run_id, **body}
            run = dict(
                id=run_id,
                session_id=session_id,
                turn_id=turn_id,
                owner_id=self.owner,
                clip_id=session["clip_id"],
                bundle_id=session["bundle_id"],
                graph_version=graph_version,
                status="queued",
                revision=1,
                cancel_requested=False,
                created_at=now,
                updated_at=now,
                trace_id=uid(),
                wait=None,
                resume_payload=None,
                response=None,
                report_id=None,
                deadline=None,
                closing_deadline=None,
                mode=mode,
                error_code=None,
            )
            if context_snapshot_factory:
                turn["context_snapshot"] = context_snapshot_factory(run)
            self._put(db, "turn", turn)
            self._write_run(db, run)
            response = {"turn_id": turn_id, "run_id": run_id, "status": "queued"}
            self._remember(db, scope, key, body, response)
            return response

    def set_wait(self, run_id, wait):
        with self.tx() as db:
            run = self._run(db, run_id)
            if run["wait"] and run["wait"]["wait_id"] == wait["wait_id"]:
                return run
            if wait["kind"] == "user":
                wait["remaining_budget"] = max(0, (run["deadline"] or time.time()) - time.time())
            run.update(
                wait=wait,
                status="waiting_user" if wait["kind"] == "user" else "waiting_tool",
                revision=run["revision"] + 1,
            )
            self._write_run(db, run)
            return run

    def save_interrupt_id(self, run_id, interrupt_id):
        with self.tx() as db:
            run = self._run(db, run_id)
            if run["wait"]:
                run["wait"]["interrupt_id"] = interrupt_id
                self._write_run(db, run)

    def submit_resume(self, run_id, body, key):
        with self.tx() as db:
            scope = f"{self.owner}:resume:{run_id}"
            cached = self._cached(db, scope, key, body)
            if cached:
                return cached
            run = self._run(db, run_id)
            wait = run["wait"] or {}
            if (
                run["status"] != "waiting_user"
                or run["resume_payload"]
                or run["cancel_requested"]
                or body["revision"] != run["revision"]
                or body["wait_id"] != wait.get("wait_id")
                or body["interrupt_id"] != wait.get("interrupt_id")
            ):
                raise DomainError(
                    409, "STALE_WAIT", "This form is stale, already submitted, or not a user wait"
                )
            run.update(resume_payload=body, revision=run["revision"] + 1)
            self._write_run(db, run)
            response = {"run_id": run_id, "status": "resume_accepted"}
            self._remember(db, scope, key, body, response)
            return response

    def clear_wait(self, run_id):
        with self.tx() as db:
            run = self._run(db, run_id)
            wait = run["wait"] or {}
            if wait.get("kind") == "user":
                run["deadline"] = time.time() + wait["remaining_budget"]
            run.update(wait=None, resume_payload=None, status="running")
            self._write_run(db, run)

    def cancel(self, run_id):
        return self.update_run(run_id, cancel_requested=True)

    def _event(self, db, run_id, kind, payload, node_id=None, dedupe=None):
        seq = db.execute("SELECT COALESCE(MAX(seq),0)+1 FROM events WHERE run_id=?", (run_id,)).fetchone()[0]
        db.execute(
            "INSERT OR IGNORE INTO events VALUES(?,?,?,?,?,?,?)",
            (run_id, seq, kind, node_id, dumps(payload), time.time(), dedupe),
        )

    def event(self, run_id, kind, payload, node_id=None, dedupe=None):
        with self.tx() as db:
            self._event(db, run_id, kind, payload, node_id, dedupe)

    def events(self, run_id, after=0):
        self.get_run(run_id)
        with self.lock:
            return [
                dict(
                    seq=r["seq"],
                    kind=r["kind"],
                    node_id=r["node_id"],
                    payload=json.loads(r["body"]),
                    created_at=r["created"],
                )
                for r in self.conn.execute(
                    "SELECT * FROM events WHERE run_id=? AND seq>? ORDER BY seq", (run_id, after)
                )
            ]

    def get_tool(self, execution_id):
        with self.lock:
            row = self.conn.execute("SELECT body FROM tools WHERE id=?", (execution_id,)).fetchone()
            if not row:
                raise DomainError(404, "TOOL_NOT_FOUND", "Tool execution not found")
            obj = json.loads(row[0])
            self._run(self.conn, obj["run_id"])
            return obj

    def tools_for_run(self, run_id):
        self.get_run(run_id)
        with self.lock:
            return [
                json.loads(r[0])
                for r in self.conn.execute("SELECT body FROM tools WHERE run_id=?", (run_id,))
            ]

    def create_tool(self, request):
        with self.tx() as db:
            row = db.execute("SELECT body FROM tools WHERE id=?", (request["execution_id"],)).fetchone()
            if row:
                existing = json.loads(row[0])
                if existing["request"]["request_hash"] != request["request_hash"]:
                    raise DomainError(409, "TOOL_HASH_CONFLICT", "Execution input changed")
                return existing
            obj = dict(
                id=request["execution_id"],
                run_id=request["run_id"],
                status="queued",
                request=request,
                output=None,
                error=None,
                duration_ms=None,
            )
            db.execute(
                "INSERT INTO tools VALUES(?,?,?,?)", (obj["id"], obj["run_id"], obj["status"], dumps(obj))
            )
            return obj

    def update_tool(self, execution_id, **updates):
        with self.tx() as db:
            row = db.execute("SELECT body FROM tools WHERE id=?", (execution_id,)).fetchone()
            obj = json.loads(row[0])
            obj.update(updates)
            run = self._run(db, obj["run_id"])
            if run["status"] in TERMINAL or run["cancel_requested"]:
                obj.update(status="cancelled", output=None)
            db.execute(
                "UPDATE tools SET status=?,body=? WHERE id=?", (obj["status"], dumps(obj), execution_id)
            )
            return obj

    def commit_run(
        self,
        run_id,
        status,
        response,
        error_code=None,
        memory_commands=None,
        context_receipt=None,
        grounded_review=None,
    ):
        with self.tx() as db:
            run = self._run(db, run_id)
            if run["status"] in TERMINAL:
                return run
            if run["cancel_requested"]:
                status, response, error_code = "cancelled", None, "USER_CANCELLED"
            if run["closing_deadline"] and time.time() > run["closing_deadline"]:
                if status != "cancelled":
                    status, response, error_code = "failed", None, "CLOSING_TIMEOUT"
            if status in {"completed", "partially_completed"} and response:
                receipts = []
                for command in memory_commands or []:
                    if command["operation"] == "remember_focus":
                        pref = {
                            "id": "focus",
                            "label_ids": command["label_ids"],
                            "source_turn_id": run["turn_id"],
                        }
                        self._put(db, "preference", pref)
                        receipts.append(
                            {"operation": "remember_focus", "saved": True, "label_ids": pref["label_ids"]}
                        )
                    elif command["operation"] == "forget_preference":
                        db.execute("DELETE FROM objects WHERE kind='preference' AND owner=?", (self.owner,))
                        receipts.append({"operation": "forget_preference", "saved": True})
                response["memory_receipts"] = receipts
            if context_receipt is not None:
                context_receipt = dict(context_receipt)
                if not response or status not in {"completed", "partially_completed"}:
                    context_receipt.update(
                        result_execution_id=None, result_report_id=None, reused_result=False
                    )
                run["review_context"] = context_receipt
            if grounded_review is not None and response and status in {"completed", "partially_completed"}:
                run["grounded_review"] = grounded_review
            if response:
                report = dict(
                    id=uid(),
                    run_id=run_id,
                    session_id=run["session_id"],
                    response=response,
                    created_at=time.time(),
                    is_scaffold=response.get("mode") == "scaffold",
                )
                if context_receipt is not None:
                    context_receipt["result_report_id"] = report["id"]
                    report["review_context"] = context_receipt
                if run.get("grounded_review"):
                    report["grounded_review"] = run["grounded_review"]
                self._put(db, "report", report)
                run["report_id"] = report["id"]
            run.update(
                status=status, response=response, error_code=error_code, wait=None, resume_payload=None
            )
            self._write_run(db, run)
            self._event(
                db, run_id, "run.finished", {"status": status, "error_code": error_code}, dedupe="finished"
            )
            return run
