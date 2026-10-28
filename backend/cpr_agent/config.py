from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[2]


def read_environment(path: Path) -> dict[str, str]:
    """Small literal .env reader: no expansion/execution; process environment has priority."""
    values = {}
    if path.is_file():
        for raw in path.read_text(encoding="utf-8-sig").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key, value = key.strip(), value.strip()
            if not key.startswith(("CPR_", "DEEPSEEK_")):
                continue
            if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
                value = value[1:-1]
            values[key] = value
    values.update({k: v for k, v in os.environ.items() if k.startswith(("CPR_", "DEEPSEEK_"))})
    return values


def project_path(value: str) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else ROOT / path).resolve()


@dataclass(frozen=True)
class Settings:
    data_dir: Path = ROOT / "data"
    stub_delay_seconds: float = 0.25
    service_budget_seconds: float = 120
    user_wait_seconds: float = 86400
    closing_seconds: float = 5
    max_upload_bytes: int = 50 * 1024 * 1024
    owner_id: str = "local-demo"
    bundle_id: str = "scaffold-v0.1"
    graph_version: str = "cpr-review-12-v0.1"
    context_mode: str = "legacy"
    fail_tool: str | None = None  # Test injection, never accepted from an HTTP request.
    vision_mode: str = "stub"
    ml_python: Path | None = None
    vision_manifest: Path = ROOT / "data/public_cpr/web-integration-v1/bundle-manifest.json"
    vision_timeout_seconds: float = 600
    vision_pipeline: str = "monolithic"
    branch_manifest: Path = ROOT / "data/branch-integration-v1/pipeline-manifest.json"
    reference_catalog: Path | None = None
    upload_preflight: bool = False
    llm_mode: str = "disabled"
    deepseek_api_key: str = field(default="", repr=False, compare=False)
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_model: str = "deepseek-flash"
    llm_timeout_seconds: float = 8
    llm_circuit_failures: int = 3
    llm_circuit_cooldown_seconds: float = 60

    def __post_init__(self):
        if self.context_mode not in {"legacy", "bound-v1"}:
            raise ValueError("CPR_CONTEXT_MODE must be legacy or bound-v1")
        if self.context_mode == "bound-v1" and self.graph_version == "cpr-review-12-v0.1":
            object.__setattr__(self, "graph_version", "cpr-review-12-v0.6-grounded")
        if self.vision_pipeline not in {"monolithic", "branches"}:
            raise ValueError("CPR_VISION_PIPELINE must be monolithic or branches")
        if self.vision_pipeline == "branches" and (
            self.vision_mode != "real" or not self.branch_manifest.is_file()
        ):
            raise ValueError("Branch pipeline requires real vision and a branch manifest")
        if self.llm_mode not in {"disabled", "deepseek"}:
            raise ValueError("CPR_LLM_MODE must be disabled or deepseek")
        if not math.isfinite(self.llm_timeout_seconds) or not 0 < self.llm_timeout_seconds <= 30:
            raise ValueError("CPR_LLM_TIMEOUT_SECONDS must be in (0, 30]")
        if (
            self.llm_circuit_failures < 1
            or not math.isfinite(self.llm_circuit_cooldown_seconds)
            or (self.llm_circuit_cooldown_seconds <= 0)
        ):
            raise ValueError("LLM circuit settings must be positive")
        if self.llm_mode == "deepseek":
            url = urlsplit(self.deepseek_base_url)
            if (
                url.scheme != "https"
                or url.hostname != "api.deepseek.com"
                or url.port not in {None, 443}
                or url.username
                or url.password
                or url.query
                or url.fragment
                or url.path.rstrip("/") not in {"", "/v1"}
            ):
                raise ValueError("DeepSeek base URL must be the official HTTPS API endpoint")
        if self.vision_mode not in {"stub", "real"}:
            raise ValueError("CPR_VISION_MODE must be stub or real")
        if self.vision_mode == "real" and (not self.ml_python or not self.ml_python.is_file()):
            raise ValueError("Real vision requires an existing CPR_ML_PYTHON executable")
        if self.vision_mode == "real" and not self.vision_manifest.is_file():
            raise ValueError("Real vision requires a frozen bundle manifest")

    @classmethod
    def from_env(cls, env_path: Path | None = None) -> "Settings":
        values = read_environment(env_path if env_path is not None else ROOT / ".env")
        branch_manifest = project_path(values.get("CPR_BRANCH_MANIFEST", str(cls.branch_manifest)))
        is_slots = branch_manifest.is_file() and "module_slots" in json.loads(
            branch_manifest.read_text(encoding="utf-8")
        )
        return cls(
            data_dir=project_path(values.get("CPR_DATA_DIR", str(ROOT / "data"))),
            stub_delay_seconds=float(values.get("CPR_STUB_DELAY_SECONDS", "0.25")),
            service_budget_seconds=float(values.get("CPR_SERVICE_BUDGET_SECONDS", "120")),
            vision_mode=values.get("CPR_VISION_MODE", "stub"),
            ml_python=project_path(values["CPR_ML_PYTHON"]) if values.get("CPR_ML_PYTHON") else None,
            vision_manifest=project_path(values.get("CPR_VISION_MANIFEST", str(cls.vision_manifest))),
            vision_timeout_seconds=float(values.get("CPR_VISION_TIMEOUT_SECONDS", "600")),
            vision_pipeline=values.get("CPR_VISION_PIPELINE", "monolithic"),
            branch_manifest=branch_manifest,
            context_mode=values.get("CPR_CONTEXT_MODE", "legacy"),
            upload_preflight=values.get("CPR_UPLOAD_PREFLIGHT", "false").lower() == "true",
            graph_version="cpr-review-12-v0.6-grounded"
            if values.get("CPR_CONTEXT_MODE") == "bound-v1"
            else (
                ("cpr-review-12-v0.4-slots" if is_slots else "cpr-review-12-v0.3-branches")
                if values.get("CPR_VISION_PIPELINE") == "branches"
                else "cpr-review-12-v0.2"
            )
            if values.get("CPR_VISION_MODE") == "real"
            else cls.graph_version,
            reference_catalog=project_path(values["CPR_REFERENCE_CATALOG"])
            if values.get("CPR_REFERENCE_CATALOG")
            else None,
            llm_mode=values.get("CPR_LLM_MODE", "disabled"),
            deepseek_api_key=values.get("DEEPSEEK_API_KEY", "").strip(),
            deepseek_base_url=values.get("DEEPSEEK_BASE_URL", cls.deepseek_base_url).strip(),
            deepseek_model=values.get("DEEPSEEK_MODEL", cls.deepseek_model).strip(),
            llm_timeout_seconds=float(values.get("CPR_LLM_TIMEOUT_SECONDS", "8")),
        )
