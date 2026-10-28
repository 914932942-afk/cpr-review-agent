"""Conversation binding around the existing twelve nodes; M1-M5 are unchanged."""

from __future__ import annotations

import time
from uuid import NAMESPACE_URL, uuid5

from langgraph.graph import END
from langgraph.types import interrupt

from .context_contracts import (
    ContextConfirmation,
    ContextReceipt,
    ContextSelection,
    ContextSnapshot,
    GoalReference,
)
from .graph import ReviewGraph
from .models import Intent, ReviewGoal
from .review_context import ContextBinder, snapshot_context
from .store import DomainError, digest
from .vision import VisionError, final_tool, media_context


class ContextReviewGraph(ReviewGraph):
    def __init__(self, store, settings, *args):
        self.binder = ContextBinder(store, settings)
        super().__init__(store, settings, *args)

    def n01(self, state):
        run = self.store.get_run(state["run_id"])
        if run["deadline"] is None:
            self.store.update_run(
                run["id"], deadline=time.time() + self.settings.service_budget_seconds, status="running"
            )
        turn = self.store.get("turn", run["turn_id"])
        snapshot = (
            ContextSnapshot.model_validate(turn["context_snapshot"])
            if turn.get("context_snapshot")
            else self.binder.snapshot(
                run, ContextSelection.model_validate(turn.get("context_selection") or {})
            )
        )
        expected = digest(snapshot.model_dump(exclude={"snapshot_id"}, mode="json"))
        if (
            snapshot.snapshot_id != expected
            or snapshot.run_id != run["id"]
            or snapshot.session_id != run["session_id"]
        ):
            raise ValueError("CONTEXT_SNAPSHOT_INVALID")
        self.store.put("turn", {**turn, "context_snapshot": snapshot.model_dump(mode="json")})
        self.store.event(
            run["id"],
            "context.bound",
            {
                "snapshot_id": snapshot.snapshot_id,
                "clip_id": snapshot.clip_id,
                "source_report_id": snapshot.source.report_id if snapshot.source else None,
                "source_status": snapshot.source_status,
            },
            "N01",
            dedupe="context-bound:" + run["id"],
        )
        return self._next(
            state,
            "N02",
            receipts={},
            failed_tools=[],
            phase="request",
            context=snapshot_context(snapshot, self.store.list_objects("preference")),
        )

    def n02(self, state):
        run = self.store.get_run(state["run_id"])
        turn = self.store.get("turn", run["turn_id"])
        context = state["context"]
        snapshot = ContextSnapshot.model_validate(context["binding"])
        selection = ContextSelection.model_validate(turn.get("context_selection") or {})
        confirmed = bool(state.get("form_goal"))
        draft = state.get("form_goal") or turn.get("goal")
        reference = selection.reference
        if draft is None:
            resolution = turn.get("llm_resolution")
            if resolution is None:
                parsed = self.gateway.parse_goal(turn["message"], context)
                resolution = parsed.receipt()
                turn = {**turn, "llm_resolution": resolution}
                self.store.put("turn", turn)
                self.store.event(
                    run["id"],
                    "llm.goal_resolution",
                    {
                        "mode": self.settings.llm_mode,
                        "code": parsed.code,
                        "called": parsed.called,
                        "duration_ms": parsed.duration_ms,
                    },
                    "N02",
                    dedupe="goal-resolution:" + run["id"],
                )
            control = self._guard(state)
            if control:
                return control
            draft = resolution["goal"]
            if resolution.get("reference"):
                reference = GoalReference.model_validate(resolution["reference"])
        if draft is None:
            return self._next(
                state, "N03", context={**context, "clarification_code": "GOAL_CONFIRMATION_REQUIRED"}
            )
        if confirmed:
            reference = GoalReference(
                scope="explicit", source="viewed_report" if snapshot.source else "current_clip"
            )
        goal, reason = self.binder.resolve(snapshot, ReviewGoal.model_validate(draft), reference, confirmed)
        if reason:
            return self._next(state, "N03", context={**context, "clarification_code": reason})
        if goal.intent in {Intent.ANALYSE, Intent.VERIFY} and not run["clip_id"]:
            return self._next(
                state,
                "N10",
                goal=goal.model_dump(mode="json"),
                final_status="rejected",
                error_code="CLIP_REQUIRED",
            )
        decision = {
            "reference": reference.model_dump(),
            "confirmed_by": "form"
            if confirmed
            else ("model" if turn.get("llm_resolution") else "structured"),
        }
        self.store.put(
            "turn", {**turn, "resolved_goal": goal.model_dump(mode="json"), "context_decision": decision}
        )
        self.store.event(
            run["id"],
            "context.resolved",
            {"snapshot_id": snapshot.snapshot_id, "target_label_ids": goal.target_label_ids, **decision},
            "N02",
            dedupe="context-resolved:" + run["id"],
        )
        return self._next(
            state, "N04", goal=goal.model_dump(mode="json"), context={**context, "decision": decision}
        )

    def n03(self, state):
        run_id = state["run_id"]
        snapshot = ContextSnapshot.model_validate(state["context"]["binding"])
        reason = state["context"].get("clarification_code", "GOAL_CONFIRMATION_REQUIRED")
        questions = {
            "CONTEXT_SCOPE_AMBIGUOUS": "Several categories are in scope. Select the category you mean, or choose a full summary.",
            "CONTEXT_SCOPE_UNAVAILABLE": "No confirmed scope exists for this video. Select the categories to review.",
            "CONTEXT_SOURCE_AMBIGUOUS": "Confirm the bound video and report below. To discuss another video, cancel and open that review first.",
            "CONTEXT_SOURCE_UNAVAILABLE": "There is no bound analysis report. Confirm the current video and choose your task.",
        }
        wait_id = str(uuid5(NAMESPACE_URL, f"{run_id}:form:{state['node_seq']}"))
        question = questions.get(
            reason,
            "Natural-language parsing is disabled. Confirm the video, source and categories below."
            if self.settings.llm_mode == "disabled"
            else "Confirm your review goal and source below.",
        )
        self.store.set_wait(
            run_id,
            {
                "kind": "user",
                "wait_kind": "form",
                "wait_id": wait_id,
                "expires_at": time.time() + self.settings.user_wait_seconds,
                "question": question,
                "context_snapshot_id": snapshot.snapshot_id,
                "context_code": reason,
                "source_confirmation_required": True,
                "scope_confirmation_required": reason
                in {"CONTEXT_SCOPE_AMBIGUOUS", "CONTEXT_SCOPE_UNAVAILABLE"},
                "bound_video_name": snapshot.video_name,
                "bound_source_report_id": snapshot.source.report_id if snapshot.source else None,
                "suggested_label_ids": snapshot.active_label_ids,
            },
        )
        answer = interrupt({"kind": "form", "wait_id": wait_id})
        run = self.store.get_run(run_id)
        if run["cancel_requested"]:
            return self._next(
                state, "N12", final_status="cancelled", response=None, error_code="USER_CANCELLED"
            )
        if (run["wait"] or {}).get("expires_at", 0) < time.time():
            return self._next(
                state, "N12", final_status="failed", response=None, error_code="USER_WAIT_EXPIRED"
            )
        if answer.get("kind") != "user_answer" or answer.get("wait_id") != wait_id:
            raise ValueError("Invalid user resume")
        confirmation = ContextConfirmation.model_validate(answer.get("context_confirmation"))
        if confirmation.snapshot_id != snapshot.snapshot_id:
            raise ValueError("CONTEXT_CONFIRMATION_CHANGED")
        goal = ReviewGoal.model_validate(answer["goal"])
        if (
            (run["wait"] or {}).get("scope_confirmation_required")
            and goal.intent != Intent.SUMMARY
            and not goal.target_label_ids
        ):
            raise ValueError("CONTEXT_SCOPE_CONFIRMATION_REQUIRED")
        self.store.clear_wait(run_id)
        return self._next(state, "N02", form_goal=goal.model_dump(mode="json"))

    def n04(self, state):
        goal = ReviewGoal.model_validate(state["goal"])
        run = self.store.get_run(state["run_id"])
        snapshot = ContextSnapshot.model_validate(state["context"]["binding"])
        analysis = goal.intent in {Intent.ANALYSE, Intent.VERIFY}
        receipts = dict(state["receipts"])
        context = {**state["context"], "reused_result": False}
        session = self.store.get("session", run["session_id"])
        if snapshot.clip_id != run["clip_id"] or session["clip_id"] != run["clip_id"]:
            return self._next(state, "N10", error_code="CONTEXT_SESSION_CHANGED", final_status="rejected")
        if self.settings.vision_mode == "real":
            try:
                if analysis or (
                    goal.intent == Intent.EXAMPLE
                    and self.settings.reference_catalog is not None
                    and run["clip_id"]
                ):
                    _, asset, _ = media_context(self.store, self.settings, run["clip_id"])
                    if asset["id"] != snapshot.asset_id or asset["sha256"] != snapshot.video_sha256:
                        raise VisionError("CONTEXT_VIDEO_CHANGED")
                if snapshot.source and goal.intent in {
                    Intent.ANALYSE,
                    Intent.VERIFY,
                    Intent.EXPLAIN,
                    Intent.SUMMARY,
                    Intent.EXAMPLE,
                }:
                    # Never replace a missing/changed bound report with the latest global prediction.
                    previous = self.binder.prediction(snapshot.source, run["clip_id"])
                    if goal.intent != Intent.EXAMPLE:
                        receipts[final_tool(self.settings)] = previous.execution_id
                        context["reused_result"] = True
            except (DomainError, ValueError, VisionError, OSError) as exc:
                self.store.event(
                    run["id"],
                    "context.source_unavailable",
                    {"code": getattr(exc, "code", "CONTEXT_SOURCE_INVALID")},
                    "N04",
                )
                return self._next(
                    state,
                    "N10",
                    error_code="CONTEXT_SOURCE_UNAVAILABLE"
                    if snapshot.source
                    else getattr(exc, "code", "MEDIA_INVALID"),
                    final_status="rejected",
                )
        if goal.intent == Intent.EXAMPLE and self.settings.reference_catalog is not None:
            from .references import prepare_lookup

            lookup = prepare_lookup(self.store, self.settings, run, snapshot, goal.target_label_ids)
            context["reference_lookup"] = lookup
            self.store.event(
                run["id"],
                "reference.lookup",
                {
                    "tool_id": "reference.search",
                    "status": lookup["status"],
                    "requested_label_ids": goal.target_label_ids,
                    "entry_ids": lookup["entry_ids"],
                    "catalog_sha256": lookup["catalog_sha256"],
                },
                "N04",
                dedupe="reference.lookup",
            )
        return self._next(
            state,
            "N05" if analysis else "N10",
            context=context,
            receipts=receipts,
            phase="analysis" if analysis else "response",
        )

    def n11(self, state):
        lookup = state.get("context", {}).get("reference_lookup")
        if lookup is not None:
            from .models import ReviewResponse
            from .references import NOTICE

            available = lookup["status"] == "available"
            response = ReviewResponse(
                kind="facts",
                title="Reference clips",
                message="Dataset-labelled reference clips are ready for comparison."
                if available
                else "No verified reference is available for this request. Your saved findings are unchanged.",
                analysis_status="not_requested",
                limitations=[
                    NOTICE,
                    "Reference retrieval does not run vision or change your practice findings.",
                ],
            )
            return self._next(state, "N12", response=response.model_dump(mode="json"))
        command = super().n11(state)
        response = command.update.get("response")
        if self.settings.vision_mode != "real" or not (response or {}).get("review"):
            return command
        from .grounded_review import build_grounded_review

        turn = self.store.get("turn", self.store.get_run(state["run_id"])["turn_id"])
        intent = (state.get("goal") or {}).get("intent", "analyse")
        resolution = turn.get("explanation_resolution")
        if resolution is None:
            resolution = {"plan": None, "code": "TEMPLATE_PRIMARY_RESULT", "called": False, "duration_ms": 0}
            if intent in {"explain", "summarise"}:
                if turn.get("explanation_attempted"):
                    # A crash after dispatch must not silently issue another paid request.
                    resolution["code"] = "LLM_PLAN_INTERRUPTED"
                elif self.settings.llm_mode == "deepseek":
                    self.store.put("turn", {**turn, "explanation_attempted": True})
                    resolution = self.gateway.plan_explanation(
                        {"intent": intent, "findings": response["review"]["findings"]}
                    ).receipt()
                else:
                    resolution["code"] = "LLM_DISABLED"
            turn = self.store.get("turn", turn["id"])
            self.store.put("turn", {**turn, "explanation_resolution": resolution})
        grounded = build_grounded_review(response, intent, resolution)
        command.update["context"] = {**state.get("context", {}), "grounded_review": grounded}
        self.store.event(
            state["run_id"],
            "explanation.prepared",
            {
                "mode": grounded["mode"],
                "code": resolution["code"],
                "called": resolution["called"],
                "content_sha256": grounded["content_sha256"],
            },
            "N11",
            dedupe="explanation.prepared",
        )
        return command

    def n12(self, state):
        # Keep the original strict response validation and atomic store commit contract.
        from .models import ReviewResponse, ScaffoldResponse

        run = self.store.get_run(state["run_id"])
        goal = ReviewGoal.model_validate(state["goal"]) if state.get("goal") else None
        response = (
            (ReviewResponse if self.settings.vision_mode == "real" else ScaffoldResponse)
            .model_validate(state["response"])
            .model_dump(mode="json")
            if state.get("response")
            else None
        )
        binding = state.get("context", {}).get("binding")
        receipt = None
        if binding:
            snapshot = ContextSnapshot.model_validate(binding)
            turn = self.store.get("turn", run["turn_id"])
            decision = turn.get("context_decision") or {}
            prediction = (response or {}).get("prediction")
            receipt = ContextReceipt(
                snapshot_id=snapshot.snapshot_id,
                session_id=snapshot.session_id,
                clip_id=snapshot.clip_id,
                video_name=snapshot.video_name,
                video_sha256=snapshot.video_sha256,
                source_report_id=snapshot.source.report_id if snapshot.source else None,
                source_execution_id=snapshot.source.execution_id if snapshot.source else None,
                source_status=snapshot.source_status,
                goal_confirmed=bool(decision),
                scope_reference=decision.get("reference", {}).get("scope", "explicit"),
                target_label_ids=(
                    [c["label_id"] for c in response.get("classification_claims", [])]
                    if response and response.get("classification_claims")
                    else (goal.target_label_ids if goal else [])
                ),
                result_execution_id=prediction["execution_id"] if prediction else None,
                reused_result=bool((response or {}).get("reused_result")),
            ).model_dump(mode="json")
        self.store.commit_run(
            run["id"],
            state.get("final_status", "failed"),
            response,
            state.get("error_code"),
            [c.model_dump(mode="json") for c in goal.memory_commands] if goal else [],
            context_receipt=receipt,
            grounded_review=state.get("context", {}).get("grounded_review"),
        )
        return self._next(state, END)
