"""Application adapter for the frozen v15 CLI; never imports the ML stack into FastAPI."""

from __future__ import annotations

import json
import os
import subprocess
import time
from uuid import uuid4

from .config import ROOT, Settings
from .evidence import file_hash, read_bound_artifact
from .models import TERMINAL, ClipPrediction, ReviewResponse, ToolRequest
from .store import Store

LABEL_NAMES = {
    1: "Overlapping hands",
    2: "Clenched hands",
    3: "Single hand",
    4: "Bent arms",
    5: "Tilted arms",
    6: "Jump pressing",
    7: "Squatting",
    8: "Standing",
    9: "Wrong hand position",
    11: "Slow frequency",
    13: "Changing pressing position",
}


class VisionError(RuntimeError):
    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


def final_tool(settings):
    return "fusion.predict" if settings.vision_pipeline == "branches" else "video.analyze"


def vision_profile(settings):
    return (
        "single-view-v15-branches-v1:" + file_hash(settings.branch_manifest)
        if settings.vision_pipeline == "branches"
        else "single-view-v15"
    )


def media_context(store: Store, settings: Settings, clip_id: str):
    clip = store.get("clip", clip_id)
    asset = store.get("asset", clip["asset_id"])
    if asset["source"] == "fixture":
        raise VisionError("SYNTHETIC_FIXTURE_NOT_CPR")
    if clip["start_seconds"] != 0 or clip["end_seconds"] is not None:
        raise VisionError("WHOLE_CLIP_ONLY")
    path = (settings.data_dir / asset["storage_rel"]).resolve()
    if not path.is_relative_to((settings.data_dir / "media").resolve()):
        raise VisionError("STORAGE_BOUNDARY")
    if not path.is_file():
        raise VisionError("MEDIA_UNAVAILABLE")
    if file_hash(path) != asset["sha256"]:
        raise VisionError("MEDIA_CHANGED")
    return clip, asset, path


def validate_receipt(output: dict, store: Store, settings: Settings) -> ClipPrediction:
    result = ClipPrediction.model_validate(output)
    _, asset, _ = media_context(store, settings, result.clip_id)
    if result.asset_id != asset["id"] or result.video_sha256 != asset["sha256"]:
        raise VisionError("RESULT_MEDIA_MISMATCH")
    if result.bundle_sha256 != file_hash(settings.vision_manifest):
        raise VisionError("RESULT_BUNDLE_MISMATCH")
    if result.profile_id != vision_profile(settings):
        raise VisionError("RESULT_PIPELINE_MISMATCH")
    read_bound_artifact(result, settings.data_dir)
    if settings.vision_pipeline == "branches":
        from .branch_vision import validate_fusion

        validate_fusion(result, store, settings)
    return result


def latest_prediction(store: Store, settings: Settings, clip_id: str | None):
    if not clip_id:
        return None
    for obj in store.list_objects("analysis"):
        if obj["prediction"].get("clip_id") == clip_id:
            try:
                return validate_receipt(obj["prediction"], store, settings)
            except (ValueError, VisionError):
                continue  # Missing or incompatible artifacts are never treated as fresh evidence.
    return None


class VideoPolicy:
    """Deterministic integration baseline, not the future M4 learned scheduling algorithm."""

    def __init__(self, settings=None):
        self.settings = settings or Settings()

    def choose(self, goal, receipts, failed):
        tool = final_tool(self.settings)
        if goal.analysis_mode == "existing_only" or tool in receipts or failed:
            return "STOP"
        return "run_fusion" if self.settings.vision_pipeline == "branches" else "run_video"


class VideoAssessment:
    def __init__(self, store: Store, settings=None):
        self.store, self.settings = store, settings or Settings()

    def assess(self, goal, receipts):
        tool = final_tool(self.settings)
        if tool not in receipts:
            return {str(label): "not_evaluated" for label in goal.target_label_ids}
        output = self.store.get_tool(receipts[tool])["output"]
        result = ClipPrediction.model_validate(output)
        return {
            str(label): "model_flagged" if label in result.predicted_label_ids else "not_flagged"
            for label in goal.target_label_ids
        }


def build_response(goal, prediction, error_code=None, reused=False):
    from .models import Intent, ModelClaim

    analysis = bool(goal and goal.intent in {Intent.ANALYSE, Intent.VERIFY})
    show = bool(
        not error_code
        and prediction
        and goal
        and goal.intent in {Intent.ANALYSE, Intent.VERIFY, Intent.EXPLAIN, Intent.SUMMARY}
    )
    targets = (goal.target_label_ids or prediction.supported_label_ids) if show else []
    claims = []
    if show:
        score_map = {s.original_label_id: s.uncalibrated_score for s in prediction.scores}
        claims = [
            ModelClaim(
                label_id=label,
                name=LABEL_NAMES[label],
                state="model_flagged" if label in prediction.predicted_label_ids else "not_flagged",
                uncalibrated_score=score_map[label],
                source_execution_id=prediction.execution_id,
            )
            for label in targets
        ]
    message = "Request processed. No new video analysis was requested."
    if show:
        flagged = [c.name for c in claims if c.state == "model_flagged"]
        message = (
            ("Model flagged: " + ", ".join(flagged) + ".")
            if flagged
            else (
                "No supported error was flagged in the selected scope. This does not establish correct CPR."
            )
        )
        if goal.intent == Intent.VERIFY:
            message += " This uses the same analysis; no independent verification path is connected."
        elif goal.intent == Intent.SUMMARY:
            message = (
                f"Review summary: {len(flagged)} model-flagged categories among {len(claims)} displayed. "
                + message
            )
        elif goal.intent == Intent.EXPLAIN:
            message = "Recorded evidence for the selected categories. " + message
    elif analysis:
        message = "No valid analysis is available. See the error code; no CPR conclusion was produced."
    elif goal and goal.intent == Intent.EXAMPLE:
        message = "No approved example catalogue is connected."
    elif goal and goal.intent in {Intent.EXPLAIN, Intent.SUMMARY}:
        message = "This clip has no compatible saved analysis to explain."
    elif goal and goal.intent == Intent.MEMORY:
        message = "Only the explicitly requested preferences are saved."
    elif goal and goal.intent == Intent.FEEDBACK:
        message = "Feedback processing is not connected. No labels or predictions were changed."
    return ReviewResponse(
        kind="error" if error_code else ("analysis_result" if show else "facts"),
        title="Review unavailable"
        if error_code
        else (
            {
                Intent.EXPLAIN: "Finding evidence",
                Intent.SUMMARY: "Review summary",
                Intent.VERIFY: "Saved-result verification",
            }.get(goal.intent, "Clip-level review")
            if show
            else "Request processed"
        ),
        message=message,
        analysis_status="completed_experimental"
        if show
        else ("unavailable" if analysis else "not_requested"),
        classification_claims=claims,
        target_states={str(c.label_id): c.state for c in claims}
        if show
        else ({str(label): "not_evaluated" for label in goal.target_label_ids} if analysis else {}),
        prediction=prediction if show else None,
        reused_result=reused if show else False,
        limitations=[
            "Experimental single-view training review. Scores are uncalibrated model outputs, not probabilities of correctness.",
            "Evidence covers the whole clip; error onset times, compression depth and measured BPM are not provided.",
            "No supported error flagged is not proof of correct CPR. Camera/profile changes can reduce accuracy.",
            "Report text uses templates. Independent rechecking and the five final module improvements are not all connected.",
        ],
    )


class RealVisionTool:
    def __init__(self, settings: Settings, store: Store, stop_event):
        self.settings, self.store, self.stop_event = settings, store, stop_event

    def execute(self, request: ToolRequest) -> ClipPrediction:
        if request.tool_id != "video.analyze" or request.profile_id != "single-view-v15":
            raise VisionError("TOOL_PROFILE_MISMATCH")
        _, asset, video = media_context(self.store, self.settings, request.clip_id)
        bindings = request.input_bindings
        manifest_hash = file_hash(self.settings.vision_manifest)
        if bindings.get("video_sha256") != asset["sha256"] or bindings.get("bundle_sha256") != manifest_hash:
            raise VisionError("TOOL_INPUT_CHANGED")
        base = self.settings.data_dir / "analysis" / request.execution_id
        base.mkdir(parents=True, exist_ok=True)
        # Resume only complete, hash-validated output. A partial attempt gets a new directory.
        for done in base.glob("*/receipt.json"):
            try:
                cached = validate_receipt(
                    json.loads(done.read_text(encoding="utf-8")), self.store, self.settings
                )
                if cached.execution_id == request.execution_id and cached.clip_id == request.clip_id:
                    return cached
            except (ValueError, VisionError):
                continue
        attempt = base / uuid4().hex
        attempt.mkdir()
        output = attempt / "inference.json"
        env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONIOENCODING": "utf-8"}
        command = [
            str(self.settings.ml_python),
            str(ROOT / "scripts/vision_worker.py"),
            "--video",
            str(video),
            "--output",
            str(output),
            "--manifest",
            str(self.settings.vision_manifest),
            "--lock",
            str(self.settings.data_dir / "vision-gpu.lock"),
        ]
        started = time.monotonic()
        with (attempt / "process.log").open("wb") as log:
            process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, env=env)
            try:
                self.store.event(
                    request.run_id,
                    "vision.started",
                    {
                        "execution_id": request.execution_id,
                        "profile_id": request.profile_id,
                        "bundle_sha256": manifest_hash,
                    },
                )
                while process.poll() is None:
                    run = self.store.get_run(request.run_id)
                    if self.stop_event.is_set() or run["cancel_requested"] or run["status"] in TERMINAL:
                        raise VisionError("VISION_CANCELLED")
                    if run["deadline"] and time.time() > run["deadline"]:
                        raise VisionError("SERVICE_TIMEOUT")
                    if time.monotonic() - started > self.settings.vision_timeout_seconds:
                        raise VisionError("VISION_TIMEOUT")
                    time.sleep(0.1)
            finally:
                if process.poll() is None:
                    process.kill()
                process.wait(timeout=10)
        if process.returncode:
            reason_file = attempt / "worker-error.json"
            reason = json.loads(reason_file.read_text(encoding="utf-8")) if reason_file.exists() else {}
            raise VisionError(reason.get("code", "VISION_PROCESS_FAILED"))
        raw = json.loads(output.read_text(encoding="utf-8"))
        if (
            raw.get("schema") != "cpr-offline-analysis-v2"
            or raw.get("status") != "completed_experimental"
            or raw.get("public_label_verification") is not None
            or raw.get("model_or_threshold_modified") is not False
            or raw.get("video_sha256") != asset["sha256"]
            or raw.get("bundle_sha256") != manifest_hash
        ):
            raise VisionError("INVALID_VISION_OUTPUT")
        receipt = ClipPrediction(
            execution_id=request.execution_id,
            clip_id=request.clip_id,
            asset_id=asset["id"],
            video_sha256=raw["video_sha256"],
            bundle_sha256=raw["bundle_sha256"],
            artifact_rel=output.relative_to(self.settings.data_dir).as_posix(),
            artifact_sha256=file_hash(output),
            decoded_frames=raw["decoded_frames"],
            reported_fps=raw["reported_fps"],
            predicted_label_ids=raw["predicted_original_label_ids"],
            supported_label_ids=raw["supported_label_ids"],
            scores=raw["scores"],
            analysis_seconds=raw["analysis_seconds"],
        )
        (attempt / "receipt.json").write_text(receipt.model_dump_json(indent=2), encoding="utf-8")
        return receipt
