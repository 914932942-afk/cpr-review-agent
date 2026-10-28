"""Optional LLM organisation, never free-form factual generation or vision inference."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Literal

import httpx
from pydantic import Field

from .models import StrictModel


class ExplanationPlan(StrictModel):
    ordered_label_ids: list[int] = Field(min_length=1, max_length=11)
    lead: Literal["findings", "evidence", "limits"]


@dataclass(frozen=True)
class PlanResolution:
    plan: ExplanationPlan | None
    code: str
    called: bool = False
    duration_ms: float = 0

    def receipt(self):
        return {
            "plan": self.plan.model_dump() if self.plan else None,
            "code": self.code,
            "called": self.called,
            "duration_ms": self.duration_ms,
        }


def validate_plan(value, label_ids):
    plan = ExplanationPlan.model_validate(value, strict=True)
    if len(plan.ordered_label_ids) != len(label_ids) or set(plan.ordered_label_ids) != set(label_ids):
        raise ValueError("Explanation scope changed")
    return plan


def request_plan(gateway, facts) -> PlanResolution:
    """Only category IDs/states and intent leave the machine; no scores, IDs, files or user prose."""
    from .llm import _reject_duplicates

    started = time.monotonic()

    def result(plan, code, called=False):
        return PlanResolution(plan, code, called, round((time.monotonic() - started) * 1000, 2))

    settings = gateway.settings
    if not settings.deepseek_api_key:
        return result(None, "LLM_KEY_MISSING")
    if gateway.status()["circuit_open"]:
        return result(None, "LLM_CIRCUIT_OPEN")
    # Construct a fresh allowlisted payload even if the caller accidentally supplies more fields.
    payload = {
        "intent": facts["intent"],
        "findings": [{"label_id": f["label_id"], "state": f["state"]} for f in facts["findings"]],
        "schema": ExplanationPlan.model_json_schema(),
    }
    try:
        with httpx.Client(
            transport=gateway.transport,
            timeout=settings.llm_timeout_seconds,
            follow_redirects=False,
            trust_env=False,
        ) as client:
            with client.stream(
                "POST",
                settings.deepseek_base_url.rstrip("/") + "/chat/completions",
                headers={"Authorization": "Bearer " + settings.deepseek_api_key},
                json={
                    "model": settings.deepseek_model,
                    "messages": [
                        {
                            "role": "system",
                            "content": (
                                "Organise saved CPR model findings; never assess a person or write factual prose. "
                                "Return JSON only: ordered_label_ids contains every supplied ID exactly once; "
                                "lead is findings, evidence or limits. Prioritise flagged categories. "
                                "Use evidence for explain and findings for summarise. No additional fields. "
                                "The server renders all factual sentences and always includes limitations."
                            ),
                        },
                        {"role": "user", "content": json.dumps(payload)},
                    ],
                    "response_format": {"type": "json_object"},
                    "thinking": {"type": "disabled"},
                    "max_tokens": 240,
                    "stream": False,
                },
            ) as response:
                if response.status_code != 200:
                    return result(None, "LLM_PLAN_HTTP_ERROR", True)
                body = bytearray()
                for chunk in response.iter_bytes():
                    if time.monotonic() - started > settings.llm_timeout_seconds:
                        return result(None, "LLM_PLAN_TIMEOUT", True)
                    body.extend(chunk)
                    if len(body) > 16_384:
                        return result(None, "LLM_PLAN_TOO_LARGE", True)
                envelope = json.loads(body, object_pairs_hook=_reject_duplicates)
        choice = envelope["choices"][0]
        if choice["finish_reason"] != "stop" or choice["message"].get("tool_calls"):
            return result(None, "LLM_PLAN_INCOMPLETE", True)
        content = json.loads(choice["message"]["content"], object_pairs_hook=_reject_duplicates)
        plan = validate_plan(content, [f["label_id"] for f in payload["findings"]])
        return result(plan, "LLM_PLAN_VALIDATED", True)
    except httpx.TimeoutException:
        return result(None, "LLM_PLAN_TIMEOUT", True)
    except httpx.RequestError:
        return result(None, "LLM_PLAN_REQUEST_FAILED", True)
    except (ValueError, KeyError, IndexError, TypeError):
        return result(None, "LLM_PLAN_INVALID", True)
