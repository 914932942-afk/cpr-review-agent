from __future__ import annotations

import time
from typing import Any, TypedDict
from uuid import NAMESPACE_URL, uuid5

from langgraph.errors import GraphBubbleUp
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from .config import Settings
from .evidence import build_review_evidence
from .models import NODE_NAMES, Intent, ReviewGoal, ReviewResponse, ScaffoldResponse, ToolRequest
from .ports import AssessmentPort, ModelGateway, PolicyPort
from .store import Store, digest
from .vision import (
    build_response,
    file_hash,
    final_tool,
    latest_prediction,
    media_context,
    validate_receipt,
    vision_profile,
)

NORMAL_ROUTES = {
    "N01": ["N02"],
    "N02": ["N03", "N04", "N10"],
    "N03": ["N02"],
    "N04": ["N05", "N10"],
    "N05": ["N06"],
    "N06": ["N07", "N10", "N05"],
    "N07": ["N08", "N09"],
    "N08": ["N09"],
    "N09": ["N07", "N05", "N10"],
    "N10": ["N11"],
    "N11": ["N12"],
    "N12": [END],
}


class ReviewState(TypedDict, total=False):
    run_id: str
    node_seq: int
    goal: dict[str, Any] | None
    form_goal: dict[str, Any] | None
    receipts: dict[str, str]
    failed_tools: list[str]
    target_states: dict[str, str]
    plan: list[dict[str, str]]
    step_index: int
    action: str
    execution_id: str
    error_code: str | None
    final_status: str
    response: dict[str, Any] | None
    phase: str
    context: dict[str, Any]
    module_assessment: dict[str, Any]
    scheduling: dict[str, Any]


class ReviewGraph:
    def __init__(
        self,
        store: Store,
        settings: Settings,
        checkpointer,
        gateway: ModelGateway,
        assessment: AssessmentPort,
        policy: PolicyPort,
    ):
        self.store, self.settings = store, settings
        self.gateway, self.assessment, self.policy = gateway, assessment, policy
        builder = StateGraph(ReviewState)
        for node_id in NODE_NAMES:
            destinations = list(
                dict.fromkeys(NORMAL_ROUTES[node_id] + (["N10", "N12"] if node_id != "N12" else []))
            )
            builder.add_node(
                node_id, self._wrap(node_id, getattr(self, node_id.lower())), destinations=tuple(destinations)
            )
        builder.add_edge(START, "N01")
        self.compiled = builder.compile(checkpointer=checkpointer)

    def _next(self, state, goto, **updates):
        return Command(goto=goto, update={"node_seq": state.get("node_seq", 0) + 1, **updates})

    def _guard(self, state):
        run = self.store.get_run(state["run_id"])
        if run["graph_version"] != self.settings.graph_version:
            raise RuntimeError("Graph version mismatch; refusing incompatible checkpoint")
        if run["cancel_requested"]:
            return self._next(
                state, "N12", final_status="cancelled", response=None, error_code="USER_CANCELLED"
            )
        now = time.time()
        if state.get("phase") == "closing":
            if run["closing_deadline"] and now > run["closing_deadline"]:
                return self._next(
                    state, "N12", final_status="failed", response=None, error_code="CLOSING_TIMEOUT"
                )
        elif run["deadline"] and now > run["deadline"]:
            self.store.update_run(run["id"], closing_deadline=now + self.settings.closing_seconds)
            return self._next(state, "N10", phase="closing", error_code="SERVICE_TIMEOUT")

    def _wrap(self, node_id, fn):
        def call(state):
            # Wait nodes must consume their registered interrupt before control checks.
            if node_id not in {"N03", "N08", "N12"}:
                control = self._guard(state)
                if control:
                    return control
            self.store.event(
                state["run_id"],
                "node.enter",
                {"name": NODE_NAMES[node_id]},
                node_id,
                f"node:{node_id}:{state.get('node_seq', 0)}",
            )
            try:
                return fn(state)
            except GraphBubbleUp:
                raise
            except Exception as exc:
                if node_id == "N12":
                    raise  # Never turn a failed database commit into a successful response.
                self.store.event(state["run_id"], "node.error", {"type": type(exc).__name__}, node_id)
                target = "N12" if node_id in {"N10", "N11"} else "N10"
                return self._next(
                    state, target, final_status="failed", error_code="NODE_ERROR", response=None
                )

        return call

    def n01(self, state):
        run = self.store.get_run(state["run_id"])
        if run["deadline"] is None:
            self.store.update_run(
                run["id"], deadline=time.time() + self.settings.service_budget_seconds, status="running"
            )
        recent = [
            t
            for t in self.store.list_objects("turn")
            if t["session_id"] == run["session_id"] and t["id"] != run["turn_id"]
        ][:6]
        context = {
            "clip_id": run["clip_id"],
            "recent_turn_ids": [t["id"] for t in recent],
            "recent_goals": [
                t.get("resolved_goal") or t["goal"] for t in recent if t.get("resolved_goal") or t.get("goal")
            ][:4],
            "preferences": self.store.list_objects("preference"),
        }
        return self._next(state, "N02", receipts={}, failed_tools=[], phase="request", context=context)

    def n02(self, state):
        run = self.store.get_run(state["run_id"])
        turn = self.store.get("turn", run["turn_id"])
        draft = state.get("form_goal") or turn.get("goal")
        if draft is None:
            resolution = turn.get("llm_resolution")
            if resolution is None:
                parsed = self.gateway.parse_goal(turn["message"], state.get("context", {}))
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
        if draft is None:
            return self._next(state, "N03")
        goal = ReviewGoal.model_validate(draft)
        self.store.put("turn", {**turn, "resolved_goal": goal.model_dump(mode="json")})
        if goal.intent in {Intent.ANALYSE, Intent.VERIFY} and not run["clip_id"]:
            return self._next(
                state,
                "N10",
                goal=goal.model_dump(mode="json"),
                final_status="rejected",
                error_code="CLIP_REQUIRED",
            )
        return self._next(state, "N04", goal=goal.model_dump(mode="json"))

    def n03(self, state):
        run_id = state["run_id"]
        wait_id = str(uuid5(NAMESPACE_URL, f"{run_id}:form:{state['node_seq']}"))
        self.store.set_wait(
            run_id,
            {
                "kind": "user",
                "wait_kind": "form",
                "wait_id": wait_id,
                "expires_at": time.time() + self.settings.user_wait_seconds,
                "question": (
                    "Natural-language parsing is disabled. Select a structured goal to continue."
                    if self.settings.llm_mode == "disabled"
                    else "Automatic goal parsing needs your input. Select a structured goal to continue."
                ),
            },
        )
        answer = interrupt({"kind": "form", "wait_id": wait_id, "schema": ReviewGoal.model_json_schema()})
        # This is reached only after Command(resume=...), including cancellation and expiry.
        run = self.store.get_run(run_id)
        if run["cancel_requested"]:
            return self._next(state, "N12", final_status="cancelled", response=None)
        if (run["wait"] or {}).get("expires_at", 0) < time.time():
            return self._next(
                state, "N12", final_status="failed", response=None, error_code="USER_WAIT_EXPIRED"
            )
        if answer.get("kind") != "user_answer" or answer.get("wait_id") != wait_id:
            raise ValueError("Invalid user resume")
        goal = ReviewGoal.model_validate(answer["goal"])
        self.store.clear_wait(run_id)
        return self._next(state, "N02", form_goal=goal.model_dump(mode="json"))

    def n04(self, state):
        goal = ReviewGoal.model_validate(state["goal"])
        # Query only persisted scaffold reports; they are never converted into predictions.
        run = self.store.get_run(state["run_id"])
        self.store.get("session", run["session_id"])
        history = [
            report["id"]
            for report in self.store.list_objects("report")
            if report["session_id"] == run["session_id"]
        ][:4]
        context = {**state.get("context", {}), "history_report_ids": history, "example_ids": []}
        analysis = goal.intent in {Intent.ANALYSE, Intent.VERIFY}
        receipts = state["receipts"]
        if self.settings.vision_mode == "real":
            if analysis:
                try:
                    media_context(self.store, self.settings, run["clip_id"])
                except Exception as exc:
                    return self._next(
                        state,
                        "N10",
                        error_code=getattr(exc, "code", "MEDIA_INVALID"),
                        final_status="rejected",
                    )
            previous = latest_prediction(self.store, self.settings, run["clip_id"])
            if previous:
                receipts = {**receipts, final_tool(self.settings): previous.execution_id}
                context["reused_result"] = True
        return self._next(
            state,
            "N05" if analysis else "N10",
            context=context,
            receipts=receipts,
            phase="analysis" if analysis else "response",
        )

    def n05(self, state):
        goal = ReviewGoal.model_validate(state["goal"])
        if hasattr(self.assessment, "evaluate"):
            result = self.assessment.evaluate(goal, state["receipts"])
            self.store.event(
                state["run_id"],
                "module.call",
                {
                    "module_id": "M3",
                    "implementation_id": self.assessment.implementation,
                    "source_prediction_ids": result.source_prediction_ids,
                },
                "N05",
            )
            return self._next(
                state,
                "N06",
                target_states={str(k): v for k, v in result.states.items()},
                module_assessment=result.model_dump(mode="json"),
            )
        return self._next(state, "N06", target_states=self.assessment.assess(goal, state["receipts"]))

    def n06(self, state):
        goal = ReviewGoal.model_validate(state["goal"])
        decision = None
        if hasattr(self.policy, "decide"):
            decision = self.policy.decide(
                goal,
                state["receipts"],
                state["failed_tools"],
                state.get("module_assessment"),
                self.store.get_run(state["run_id"]),
            )
            action = decision.action
        else:
            action = self.policy.choose(goal, state["receipts"], state["failed_tools"])
        self.store.event(
            state["run_id"],
            "policy.decision",
            {
                "action": action,
                "is_stub": self.settings.vision_mode == "stub",
                "reason": decision.reason
                if decision
                else "Deterministic integration baseline; no M4 gain estimator is connected.",
                "implementation_id": getattr(self.policy, "implementation", None),
                "profile_id": decision.profile_id if decision else None,
                "stop_code": decision.stop_code if decision else None,
                "audit": decision.audit if decision else {},
            },
        )
        if getattr(self.policy, "implementation", None) == "consent-coverage-dag-v2":
            schedule = dict(state.get("scheduling", {}))
            schedule["history"] = [*schedule.get("history", []), decision.model_dump(mode="json")][-6:]
            if action == "STOP":
                schedule["stop"] = decision.model_dump(mode="json")
                updates = {"action": action, "scheduling": schedule}
                if (
                    decision.stop_code == "M4_BUDGET_INSUFFICIENT"
                    and final_tool(self.settings) not in state["receipts"]
                ):
                    updates.update(error_code="M4_BUDGET_INSUFFICIENT", final_status="rejected")
                return self._next(state, "N10", **updates)
            from .m4_controller import missing_tools

            covered = list((state.get("module_assessment") or {}).get("states", {}))
            covered = [
                int(k)
                for k in covered
                if (state.get("module_assessment") or {})["states"][k] != "not_evaluated"
            ]
            missing = missing_tools(
                decision.profile_id, state["receipts"], covered, goal.target_label_ids, goal.allow_partial
            )
            if not missing or any(tool in state["failed_tools"] for tool in missing):
                raise ValueError("M4_INVALID_OR_BLOCKED_PLAN")
            selected = decision.audit.get("selected", {})
            if selected.get("partial_scope"):
                if not goal.allow_partial:
                    raise ValueError("M4_PARTIAL_SCOPE_NOT_AUTHORISED")
                run = self.store.get_run(state["run_id"])
                turn = self.store.get("turn", run["turn_id"])
                self.store.put("turn", {**turn, "m4_committed_partial_route": decision.profile_id})
                schedule["partial_scope_authorised"] = True
            schedule["plan"] = decision.model_dump(mode="json")
            plan = [
                {"tool_id": tool, "execution_id": str(uuid5(NAMESPACE_URL, state["run_id"] + ":" + tool))}
                for tool in missing
            ]
            return self._next(
                state,
                "N07",
                action=action,
                plan=plan,
                step_index=0,
                scheduling=schedule,
                context={**state.get("context", {}), "reused_result": False},
            )
        if action == "STOP":
            return self._next(state, "N10", action=action)
        paths = {
            "run_full_pipeline": ["pose.extract", "skeleton.predict", "rgb.predict", "fusion.predict"],
            "run_video": ["video.analyze"],
            "run_skeleton": ["pose.extract", "skeleton.predict"],
            "run_rgb": ["rgb.predict"],
            "run_fusion": ["pose.extract", "skeleton.predict", "rgb.predict", "fusion.predict"],
        }
        missing = [tool for tool in paths[action] if tool not in state["receipts"]]
        if not missing or any(tool in state["failed_tools"] for tool in missing):
            # Exclusion changes state, preventing an empty planning loop.
            blocked = list(dict.fromkeys(state["failed_tools"] + paths[action]))
            return self._next(state, "N05", failed_tools=blocked)
        plan = [
            {"tool_id": tool, "execution_id": str(uuid5(NAMESPACE_URL, state["run_id"] + ":" + tool))}
            for tool in missing
        ]
        return self._next(state, "N07", action=action, plan=plan, step_index=0)

    def n07(self, state):
        run = self.store.get_run(state["run_id"])
        step = state["plan"][state["step_index"]]
        tool = step["tool_id"]
        bindings = {"clip_id": run["clip_id"]}
        if self.settings.vision_mode == "real":
            _, asset, _ = media_context(self.store, self.settings, run["clip_id"])
            bindings.update(
                video_sha256=asset["sha256"], bundle_sha256=file_hash(self.settings.vision_manifest)
            )
            if self.settings.vision_pipeline == "branches":
                bindings["pipeline_sha256"] = file_hash(self.settings.branch_manifest)
        if tool == "skeleton.predict" or (
            tool == "rgb.predict" and self.settings.vision_pipeline == "branches"
        ):
            bindings["pose_receipt_id"] = state["receipts"]["pose.extract"]
        elif tool == "fusion.predict":
            bindings.update(
                skeleton_receipt_id=state["receipts"]["skeleton.predict"],
                **(
                    {"rgb_receipt_id": state["receipts"]["rgb.predict"]}
                    if "rgb.predict" in state["receipts"]
                    else {}
                ),
            )
        if getattr(self.policy, "implementation", None) == "consent-coverage-dag-v2":
            bindings["route_profile"] = state["scheduling"]["plan"]["profile_id"]
            if bindings["route_profile"] == "body-qualified-v1":
                bindings.pop("rgb_receipt_id", None)
            anchor = state.get("context", {}).get("reuse_source_execution_id")
            if anchor:
                bindings["reuse_source_execution_id"] = anchor
        # Receipt IDs deliberately cannot masquerade as PoseArtifact / Prediction IDs.
        request_hash = digest({"tool_id": tool, "bindings": bindings, "bundle": run["bundle_id"]})
        request = ToolRequest(
            execution_id=step["execution_id"],
            run_id=run["id"],
            clip_id=run["clip_id"],
            tool_id=tool,
            profile_id=vision_profile(self.settings)
            if self.settings.vision_mode == "real"
            else "placeholder-v1",
            input_bindings=bindings,
            request_hash=request_hash,
            attempt_token=str(uuid5(NAMESPACE_URL, step["execution_id"] + ":attempt:1")),
        )
        execution = self.store.create_tool(request.model_dump(mode="json"))
        return self._next(
            state,
            "N09" if execution["status"] in {"succeeded", "failed", "cancelled"} else "N08",
            execution_id=execution["id"],
        )

    def n08(self, state):
        wait_id = str(uuid5(NAMESPACE_URL, state["execution_id"] + ":wait"))
        self.store.set_wait(
            state["run_id"],
            {"kind": "tool", "wait_kind": "tool", "wait_id": wait_id, "execution_id": state["execution_id"]},
        )
        signal = interrupt({"kind": "tool", "wait_id": wait_id, "execution_id": state["execution_id"]})
        control = self._guard(state)
        if control:
            return control
        if signal.get("kind") != "tool_complete" or signal.get("execution_id") != state["execution_id"]:
            raise ValueError("Invalid tool resume")
        self.store.clear_wait(state["run_id"])
        return self._next(state, "N09")

    def n09(self, state):
        execution = self.store.get_tool(state["execution_id"])
        request = execution["request"]
        run = self.store.get_run(state["run_id"])
        if request["run_id"] != run["id"] or request["clip_id"] != run["clip_id"]:
            raise ValueError("Cross-run or cross-clip result")
        if execution["status"] != "succeeded":
            return self._next(
                state, "N05", failed_tools=list(dict.fromkeys(state["failed_tools"] + [request["tool_id"]]))
            )
        output = execution["output"]
        if self.settings.vision_mode == "real" and request["tool_id"] == final_tool(self.settings):
            result = validate_receipt(output, self.store, self.settings)
            if result.execution_id != execution["id"] or result.clip_id != run["clip_id"]:
                raise ValueError("Result execution/clip mismatch")
            self.store.put("analysis", {"id": execution["id"], "prediction": result.model_dump(mode="json")})
        elif self.settings.vision_mode == "real" and self.settings.vision_pipeline == "branches":
            from .branch_vision import validate_branch

            result = validate_branch(output, self.store, self.settings)
            if result.execution_id != execution["id"] or result.clip_id != run["clip_id"]:
                raise ValueError("Result execution/clip mismatch")
        elif not output or output.get("is_stub") is not True or output["execution_id"] != execution["id"]:
            raise ValueError("Invalid placeholder receipt")
        receipts = {**state["receipts"], request["tool_id"]: execution["id"]}
        index = state["step_index"] + 1
        return self._next(
            state,
            "N05"
            if getattr(self.policy, "implementation", None) == "consent-coverage-dag-v2"
            else "N07"
            if index < len(state["plan"])
            else "N05",
            receipts=receipts,
            step_index=index,
        )

    def n10(self, state):
        goal = ReviewGoal.model_validate(state["goal"]) if state.get("goal") else None
        if state.get("error_code"):
            status = state.get("final_status", "failed")
        elif goal and goal.intent in {Intent.ANALYSE, Intent.VERIFY}:
            # Faithful business status: no real analysis = no valid judgment = failed.
            # The UI separately shows that the workflow reached N12 successfully.
            required = final_tool(self.settings) if self.settings.vision_mode == "real" else "fusion.predict"
            status = (
                "completed"
                if (self.settings.vision_mode == "real" and required in state.get("receipts", {}))
                else "failed"
            )
        else:
            status = "completed"
        if (
            self.settings.vision_mode == "real"
            and getattr(self.policy, "implementation", None) == "consent-coverage-dag-v2"
            and goal
            and final_tool(self.settings) in state.get("receipts", {})
            and not state.get("error_code")
        ):
            current = validate_receipt(
                self.store.get_tool(state["receipts"][final_tool(self.settings)])["output"],
                self.store,
                self.settings,
            )
            if not set(goal.target_label_ids) <= set(current.supported_label_ids):
                status = "partially_completed"
        failure = (
            "TOOL_EXECUTION_FAILED"
            if state.get("failed_tools")
            else (
                "NO_EXISTING_ANALYSIS" if self.settings.vision_mode == "real" else "MODULES_NOT_IMPLEMENTED"
            )
        )
        if self.settings.vision_mode == "real" and state.get("failed_tools"):
            tools = self.store.tools_for_run(state["run_id"])
            errors = [tool["error"]["code"] for tool in tools if tool.get("error")]
            failure = errors[-1] if errors else failure
        error = state.get("error_code") or (failure if status == "failed" else None)
        return self._next(
            state,
            "N11",
            final_status=status,
            error_code=error,
            phase="closing" if state.get("phase") == "closing" else "response",
        )

    def n11(self, state):
        goal = ReviewGoal.model_validate(state["goal"]) if state.get("goal") else None
        if self.settings.vision_mode == "real":
            execution_id = state.get("receipts", {}).get(final_tool(self.settings))
            prediction = (
                validate_receipt(self.store.get_tool(execution_id)["output"], self.store, self.settings)
                if execution_id
                else None
            )
            response = build_response(
                goal,
                prediction,
                state.get("error_code"),
                state.get("context", {}).get("reused_result", False),
            )
            response.scheduling = state.get("scheduling")
            from .m5_validation import visual_summary

            response.visual_processing = visual_summary(self.store, self.settings, prediction)
            if response.prediction:
                if (
                    not state.get("module_assessment")
                    and getattr(self.assessment, "implementation", None) == "source-quality-selective-v1"
                ):
                    checked_goal = goal.model_copy(
                        update={"target_label_ids": [c.label_id for c in response.classification_claims]}
                    )
                    checked = self.assessment.evaluate(
                        checked_goal, {final_tool(self.settings): prediction.execution_id}
                    )
                    state = {**state, "module_assessment": checked.model_dump(mode="json")}
                    self.store.event(
                        state["run_id"],
                        "module.call",
                        {
                            "module_id": "M3",
                            "implementation_id": self.assessment.implementation,
                            "source_prediction_ids": checked.source_prediction_ids,
                        },
                        "N11",
                    )
                if state.get("module_assessment"):
                    from .modules.contracts import Assessment

                    response.module_assessment = Assessment.model_validate(state["module_assessment"])
                    response.target_states = {str(k): v for k, v in response.module_assessment.states.items()}
                execution = self.store.get_tool(response.prediction.execution_id)
                response.review = build_review_evidence(
                    response.prediction,
                    response.classification_claims,
                    self.settings.data_dir,
                    execution["run_id"],
                )
            return self._next(state, "N12", response=response.model_dump(mode="json"))
        analysis = bool(goal and goal.intent in {Intent.ANALYSE, Intent.VERIFY})
        kind = "scaffold_receipt" if analysis else "facts"
        title = "Workflow complete · analysis modules not connected" if analysis else "Request processed"
        message = (
            f"The request reached the end of the scaffold workflow. "
            f"{len(state.get('receipts', {}))} placeholder tool calls completed; no CPR judgment was made."
        )
        if state.get("error_code") == "CLIP_REQUIRED":
            kind, title, message = (
                "error",
                "A video is required",
                "Attach a video before requesting analysis.",
            )
        elif state.get("error_code") not in {None, "MODULES_NOT_IMPLEMENTED"}:
            kind, title, message = (
                "error",
                "Run stopped",
                "See the run error code and trace for the stopping reason.",
            )
        elif goal and goal.intent == Intent.MEMORY:
            kind, title, message = (
                "memory_receipt",
                "Preference request processed",
                "Only explicit settings are saved.",
            )
        elif goal and goal.intent == Intent.EXAMPLE:
            message = "No approved CPR example catalogue is connected. No substitute or invented example was returned."
        elif goal and goal.intent in {Intent.EXPLAIN, Intent.SUMMARY}:
            message = "Scaffold reports contain no CPR findings. There is no clinical conclusion to explain."
        elif goal and goal.intent == Intent.FEEDBACK:
            message = "Feedback interpretation is not connected in this scaffold. No prediction or label was changed."
        response = ScaffoldResponse(
            kind=kind,
            title=title,
            message=message,
            analysis_status="not_implemented" if analysis else "not_requested",
            target_states=state.get("target_states", {}),
            limitations=[
                "M1–M5 algorithms are not connected; language parser status is reported separately.",
                "Local SQLite and local executor adapters; no PostgreSQL/Celery deployment yet.",
            ],
        )
        return self._next(state, "N12", response=response.model_dump(mode="json"))

    def n12(self, state):
        goal = ReviewGoal.model_validate(state["goal"]) if state.get("goal") else None
        response = (
            (ReviewResponse if self.settings.vision_mode == "real" else ScaffoldResponse)
            .model_validate(state["response"])
            .model_dump(mode="json")
            if state.get("response")
            else None
        )
        self.store.commit_run(
            state["run_id"],
            state.get("final_status", "failed"),
            response,
            state.get("error_code"),
            [c.model_dump(mode="json") for c in goal.memory_commands] if goal else [],
        )
        return self._next(state, END)
