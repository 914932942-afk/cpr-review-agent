"""Four explicit stage adapters sharing a bound actor track; final output stays v15-compatible."""

from __future__ import annotations

import json
import os
import subprocess
import time
from uuid import uuid4

from .config import ROOT
from .evidence import file_hash
from .m4_lineage import reusable
from .models import TERMINAL, BranchReceipt, ClipPrediction
from .vision import VisionError, media_context, validate_receipt, vision_profile

DEPENDENCIES = {
    "pose.extract": {},
    "skeleton.predict": {"pose_receipt_id": "pose.extract"},
    "rgb.predict": {"pose_receipt_id": "pose.extract"},
    "fusion.predict": {"skeleton_receipt_id": "skeleton.predict", "rgb_receipt_id": "rgb.predict"},
}

WORKERS = {
    "scripts/branch_worker.py",
    "scripts/m1_branch_worker.py",
    "scripts/m1_fast_branch_worker.py",
    "scripts/module_slots_worker.py",
}


def pipeline_worker(value):
    name = value.get("worker_entrypoint", "scripts/branch_worker.py")
    if name not in WORKERS or name not in value.get("sources", {}):
        raise VisionError("PIPELINE_WORKER_NOT_PINNED")
    return ROOT / name


def validate_pipeline(settings):
    value = json.loads(settings.branch_manifest.read_text(encoding="utf-8"))
    if value.get("schema") != "cpr-branch-pipeline-manifest-v1" or value.get(
        "model_bundle_sha256"
    ) != file_hash(settings.vision_manifest):
        raise VisionError("PIPELINE_MODEL_MISMATCH")
    pipeline_worker(value)
    if "module_slots" in value:
        from .modules.contracts import CONTRACT_VERSION
        from .modules.registry import build_registry

        if (
            value.get("module_contract_version") != CONTRACT_VERSION
            or value.get("worker_entrypoint") != "scripts/module_slots_worker.py"
        ):
            raise VisionError("MODULE_PROFILE_INVALID")
        registry = build_registry()
        registry.validate_selection(value["module_slots"])
        required = {
            p.relative_to(ROOT).as_posix() for p in (ROOT / "backend/cpr_agent/modules").rglob("*.py")
        }
        required.update(
            [
                "backend/cpr_agent/module_runtime.py",
                "backend/cpr_agent/models.py",
                "experiments/module_slots_v1/slots_pipeline.py",
            ]
        )
        for module_id, implementation in value["module_slots"].items():
            required.update(registry.spec(module_id, implementation).required_files)
        if not required <= value["sources"].keys():
            raise VisionError("MODULE_SOURCE_NOT_PINNED")
    elif value.get("worker_entrypoint") == "scripts/module_slots_worker.py":
        raise VisionError("MODULE_SELECTION_MISSING")
    for name, expected in value["sources"].items():
        source = (ROOT / name).resolve()
        if not source.is_relative_to(ROOT) or not source.is_file() or file_hash(source) != expected:
            raise VisionError("PIPELINE_SOURCE_CHANGED")
    return file_hash(settings.branch_manifest)


def validate_module_receipts(raw, settings, ids, clip_id):
    profile = json.loads(settings.branch_manifest.read_text(encoding="utf-8"))
    if "module_slots" not in profile:
        if raw.get("module_executions"):
            raise VisionError("MODULE_PROFILE_INVALID")
        return False
    from .modules.contracts import CONTRACT_VERSION
    from .modules.registry import build_registry

    executions = raw.get("module_executions", {})
    if raw.get("module_contract_version") != CONTRACT_VERSION or set(executions) != set(ids):
        raise VisionError("MODULE_RECEIPT_MISSING")
    registry = build_registry()
    for module_id in ids:
        item = executions[module_id]
        spec = registry.spec(module_id, profile["module_slots"][module_id])
        binding = item.get("binding", {})
        if (
            item.get("implementation_id") != spec.implementation_id
            or item.get("kind") != spec.kind
            or item.get("contract_version") != CONTRACT_VERSION
            or item.get("module_id") != module_id
            or binding.get("clip_id") != clip_id
            or binding.get("video_sha256") != raw["video_sha256"]
            or binding.get("bundle_sha256") != raw["bundle_sha256"]
            or binding.get("pipeline_sha256") != raw["pipeline_sha256"]
        ):
            raise VisionError("MODULE_RECEIPT_BINDING_MISMATCH")
        if (
            module_id == "M5"
            and raw.get("tool_id") == "rgb.predict"
            and binding.get("actor_source_sha256") != raw["shared_pose_sha256"]
        ):
            raise VisionError("MODULE_ACTOR_BINDING_MISMATCH")
    return any(item["kind"] == "improved" for item in executions.values())


def artifact_path(output, settings):
    path = (settings.data_dir / output.artifact_rel).resolve()
    if not path.is_relative_to((settings.data_dir / "analysis").resolve()) or not path.is_file():
        raise VisionError("BRANCH_ARTIFACT_MISSING")
    if file_hash(path) != output.artifact_sha256 or path.stat().st_size > 1_048_576:
        raise VisionError("BRANCH_ARTIFACT_CHANGED")
    return path


def validate_branch(output, store, settings):
    receipt = BranchReceipt.model_validate(output)
    _, asset, _ = media_context(store, settings, receipt.clip_id)
    if (
        receipt.asset_id != asset["id"]
        or receipt.video_sha256 != asset["sha256"]
        or receipt.bundle_sha256 != file_hash(settings.vision_manifest)
        or receipt.profile_id != vision_profile(settings)
    ):
        raise VisionError("BRANCH_BINDING_MISMATCH")
    path = artifact_path(receipt, settings)
    raw = json.loads(path.read_text(encoding="utf-8"))
    modified = (
        validate_module_receipts(raw, settings, ["M5"], receipt.clip_id)
        if receipt.tool_id == "rgb.predict"
        else False
    )
    request = store.get_tool(receipt.execution_id)["request"]
    expected = {
        "schema": "cpr-decomposed-stage-v1",
        "status": "completed_experimental",
        "tool_id": receipt.tool_id,
        "video_sha256": receipt.video_sha256,
        "bundle_sha256": receipt.bundle_sha256,
        "pipeline_sha256": receipt.profile_id.split(":", 1)[1],
        "labels_used": False,
        "model_or_threshold_modified": modified,
    }
    if (
        any(raw.get(key) != value for key, value in expected.items())
        or request["clip_id"] != receipt.clip_id
        or request["tool_id"] != receipt.tool_id
        or request["execution_id"] != receipt.execution_id
        or request["profile_id"] != receipt.profile_id
    ):
        raise VisionError("BRANCH_CONTRACT_MISMATCH")
    for item in raw.get("files", []):
        file = (path.parent / item["name"]).resolve()
        if file.parent != path.parent or not file.is_file() or file_hash(file) != item["sha256"]:
            raise VisionError("BRANCH_FEATURE_CHANGED")
    if not raw.get("files"):
        raise VisionError("BRANCH_FEATURE_MISSING")
    expected_labels = {
        "pose.extract": [],
        "skeleton.predict": [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 13],
        "rgb.predict": [1, 2, 5, 7, 9],
    }[receipt.tool_id]
    if (
        sorted(raw.get("supported_label_ids", [])) != expected_labels
        or not 8 <= raw.get("decoded_frames", 0) <= 900
        or not 0 < raw.get("reported_fps", 0) <= 240
    ):
        raise VisionError("BRANCH_CONTRACT_MISMATCH")
    required = DEPENDENCIES[receipt.tool_id]
    if required and raw.get("shared_pose_sha256") != raw.get("dependencies", {}).get("pose.extract"):
        raise VisionError("BRANCH_DEPENDENCY_MISMATCH")
    if set(raw.get("dependencies", {})) != set(required.values()):
        raise VisionError("BRANCH_DEPENDENCY_MISMATCH")
    for binding, tool in required.items():
        source = store.get_tool(request["input_bindings"][binding])
        if (
            source["status"] != "succeeded"
            or not reusable(store, settings, request, source)
            or source["request"]["tool_id"] != tool
        ):
            raise VisionError("BRANCH_DEPENDENCY_MISMATCH")
        validated = validate_branch(source["output"], store, settings)
        if validated.clip_id != receipt.clip_id or raw["dependencies"][tool] != validated.artifact_sha256:
            raise VisionError("BRANCH_DEPENDENCY_MISMATCH")
    if receipt.tool_id == "rgb.predict":
        from .m5_validation import validate_visual_archive, validate_visual_metadata

        validate_visual_metadata(raw, settings)
        validate_visual_archive(raw, path)
    return receipt


def validate_fusion(prediction, store, settings):
    validate_pipeline(settings)
    raw = json.loads(artifact_path(prediction, settings).read_text(encoding="utf-8"))
    profile = json.loads(settings.branch_manifest.read_text(encoding="utf-8"))
    if prediction.analysis_route == "body-qualified-v1":
        return validate_body_final(prediction, raw, store, settings, profile)
    if "module_slots" in profile:
        modified = validate_module_receipts(raw, settings, ["M2", "M5"], prediction.clip_id)
        if raw.get("model_or_threshold_modified") != modified:
            raise VisionError("MODULE_RESULT_MODE_MISMATCH")
    request = store.get_tool(prediction.execution_id)["request"]
    sources = []
    if set(raw.get("dependencies", {})) != {"skeleton.predict", "rgb.predict"}:
        raise VisionError("BRANCH_DEPENDENCY_MISMATCH")
    for binding, tool in DEPENDENCIES["fusion.predict"].items():
        execution = store.get_tool(request["input_bindings"][binding])
        receipt = validate_branch(execution["output"], store, settings)
        if (
            execution["status"] != "succeeded"
            or not reusable(store, settings, request, execution)
            or receipt.clip_id != prediction.clip_id
            or receipt.tool_id != tool
            or receipt.artifact_sha256 != raw["dependencies"][tool]
        ):
            raise VisionError("BRANCH_DEPENDENCY_MISMATCH")
        sources.append(json.loads(artifact_path(receipt, settings).read_text(encoding="utf-8")))
    if sources[0]["shared_pose_sha256"] != sources[1]["shared_pose_sha256"]:
        raise VisionError("CROSS_TRACK_FUSION")
    if "module_slots" in profile:
        if any(
            e["binding"].get("actor_source_sha256") != sources[0]["shared_pose_sha256"]
            for e in raw["module_executions"].values()
        ):
            raise VisionError("MODULE_ACTOR_BINDING_MISMATCH")
        if raw["module_executions"]["M2"]["prediction_id"] != prediction.execution_id:
            raise VisionError("MODULE_RESULT_ID_MISMATCH")
        if raw["module_executions"]["M5"] != sources[1]["module_executions"]["M5"]:
            raise VisionError("MODULE_SOURCE_RECEIPT_CHANGED")


class BranchVisionTool:
    def __init__(self, settings, store, stop_event):
        self.settings, self.store, self.stop_event = settings, store, stop_event

    def execute(self, request):
        settings = self.settings
        if request.tool_id not in DEPENDENCIES or request.profile_id != vision_profile(settings):
            raise VisionError("TOOL_PROFILE_MISMATCH")
        pipeline_sha = validate_pipeline(settings)
        _, asset, video = media_context(self.store, settings, request.clip_id)
        if (
            request.input_bindings.get("video_sha256") != asset["sha256"]
            or request.input_bindings.get("bundle_sha256") != file_hash(settings.vision_manifest)
            or request.input_bindings.get("pipeline_sha256") != pipeline_sha
        ):
            raise VisionError("TOOL_INPUT_CHANGED")
        inputs = {}
        dependencies = DEPENDENCIES[request.tool_id]
        if (
            request.tool_id == "fusion.predict"
            and request.input_bindings.get("route_profile") == "body-qualified-v1"
        ):
            dependencies = {"skeleton_receipt_id": "skeleton.predict"}
        for binding, tool in dependencies.items():
            source = self.store.get_tool(request.input_bindings[binding])
            if source["status"] != "succeeded" or not reusable(
                self.store, settings, request.model_dump(), source
            ):
                raise VisionError("BRANCH_DEPENDENCY_MISMATCH")
            receipt = validate_branch(source["output"], self.store, settings)
            if receipt.tool_id != tool or receipt.clip_id != request.clip_id:
                raise VisionError("BRANCH_DEPENDENCY_MISMATCH")
            inputs[tool] = str(artifact_path(receipt, settings))
        inputs["__route_profile__"] = request.input_bindings.get("route_profile", "full-v15")
        base = settings.data_dir / "analysis" / request.execution_id
        base.mkdir(parents=True, exist_ok=True)
        for done in base.glob("*/receipt.json"):
            try:
                cached = json.loads(done.read_text(encoding="utf-8"))
                result = (
                    validate_receipt(cached, self.store, settings)
                    if request.tool_id == "fusion.predict"
                    else validate_branch(cached, self.store, settings)
                )
                if result.execution_id == request.execution_id and result.clip_id == request.clip_id:
                    return result
            except (ValueError, VisionError, OSError):
                continue
        attempt = base / uuid4().hex
        attempt.mkdir()
        output = attempt / "inference.json"
        command = [
            str(settings.ml_python),
            str(pipeline_worker(json.loads(settings.branch_manifest.read_text(encoding="utf-8")))),
            "--video",
            str(video),
            "--output",
            str(output),
            "--manifest",
            str(settings.vision_manifest),
            "--pipeline-manifest",
            str(settings.branch_manifest),
            "--lock",
            str(ROOT.parents[1] / "data/vision-gpu.lock"),
            "--tool",
            request.tool_id,
            "--inputs",
            json.dumps(inputs),
        ]
        if json.loads(settings.branch_manifest.read_text(encoding="utf-8")).get("module_slots") is not None:
            command.extend(["--clip-id", request.clip_id, "--execution-id", request.execution_id])
        started = time.monotonic()
        env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONIOENCODING": "utf-8"}
        with (attempt / "process.log").open("wb") as log:
            process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, env=env)
            try:
                self.store.event(
                    request.run_id,
                    "vision.stage_started",
                    {
                        "execution_id": request.execution_id,
                        "tool_id": request.tool_id,
                        "pipeline_sha256": pipeline_sha,
                    },
                )
                while process.poll() is None:
                    run = self.store.get_run(request.run_id)
                    if self.stop_event.is_set() or run["cancel_requested"] or run["status"] in TERMINAL:
                        raise VisionError("VISION_CANCELLED")
                    if run["deadline"] and time.time() > run["deadline"]:
                        raise VisionError("SERVICE_TIMEOUT")
                    if time.monotonic() - started > settings.vision_timeout_seconds:
                        raise VisionError("VISION_TIMEOUT")
                    time.sleep(0.1)
            finally:
                if process.poll() is None:
                    process.kill()
                process.wait(timeout=10)
        if process.returncode:
            raise VisionError("BRANCH_STAGE_FAILED")
        raw = json.loads(output.read_text(encoding="utf-8"))
        common = dict(
            execution_id=request.execution_id,
            clip_id=request.clip_id,
            asset_id=asset["id"],
            profile_id=request.profile_id,
            video_sha256=asset["sha256"],
            bundle_sha256=file_hash(settings.vision_manifest),
            artifact_rel=output.relative_to(settings.data_dir).as_posix(),
            artifact_sha256=file_hash(output),
            analysis_seconds=time.monotonic() - started,
        )
        if request.tool_id == "fusion.predict":
            # Report total actual tool execution time, not the CPU-only fusion duration.
            stage_seconds = sum(
                self.store.get_tool(request.input_bindings[k])["output"]["analysis_seconds"]
                for k in ["skeleton_receipt_id", "rgb_receipt_id"]
                if k in request.input_bindings
                and self.store.get_tool(request.input_bindings[k])["run_id"] == request.run_id
            )
            pose_id = self.store.get_tool(request.input_bindings["skeleton_receipt_id"])["request"][
                "input_bindings"
            ]["pose_receipt_id"]
            if self.store.get_tool(pose_id)["run_id"] == request.run_id:
                stage_seconds += self.store.get_tool(pose_id)["output"]["analysis_seconds"]
            common["analysis_seconds"] += stage_seconds
            result = ClipPrediction(
                **common,
                decoded_frames=raw["decoded_frames"],
                reported_fps=raw["reported_fps"],
                predicted_label_ids=raw["predicted_original_label_ids"],
                supported_label_ids=raw["supported_label_ids"],
                analysis_route=raw.get("analysis_route", "full-v15"),
                scores=raw["scores"],
            )
            validate_receipt(result.model_dump(), self.store, settings)
        else:
            result = BranchReceipt(**common, tool_id=request.tool_id)
            validate_branch(result.model_dump(), self.store, settings)
        (attempt / "receipt.json").write_text(result.model_dump_json(indent=2), encoding="utf-8")
        return result


def validate_body_final(prediction, raw, store, settings, profile):
    if profile.get("module_slots", {}).get("M4") != "consent-coverage-dag-v2" or profile.get(
        "m4_routes", {}
    ).get("body-qualified-v1", {}).get("supported_label_ids") != [3, 4, 8, 11]:
        raise VisionError("M4_ROUTE_NOT_CERTIFIED")
    request = store.get_tool(prediction.execution_id)["request"]
    if (
        request["input_bindings"].get("route_profile") != "body-qualified-v1"
        or set(raw.get("dependencies", {})) != {"skeleton.predict"}
        or raw.get("module_executions") != {}
        or raw.get("model_or_threshold_modified") is not False
    ):
        raise VisionError("M4_BODY_RESULT_INVALID")
    source = store.get_tool(request["input_bindings"]["skeleton_receipt_id"])
    if not reusable(store, settings, request, source):
        raise VisionError("M4_BODY_SOURCE_NOT_APPROVED")
    receipt = validate_branch(source["output"], store, settings)
    if (
        receipt.clip_id != prediction.clip_id
        or receipt.artifact_sha256 != raw["dependencies"]["skeleton.predict"]
        or receipt.tool_id != "skeleton.predict"
    ):
        raise VisionError("M4_BODY_SOURCE_MISMATCH")
    value = json.loads(artifact_path(receipt, settings).read_text(encoding="utf-8"))
    scores = [s for s in value["scores"] if s["original_label_id"] in [3, 4, 8, 11]]
    flags = [k for k in value["predicted_original_label_ids"] if k in [3, 4, 8, 11]]
    if (
        raw["scores"] != scores
        or prediction.predicted_label_ids != flags
        or raw["decoded_frames"] != value["decoded_frames"]
        or raw["reported_fps"] != value["reported_fps"]
        or raw.get("fullpose_audit") != value.get("fullpose_audit")
        or raw.get("module_contract_version") != "cpr-modules-v1"
    ):
        raise VisionError("M4_BODY_READOUT_CHANGED")
