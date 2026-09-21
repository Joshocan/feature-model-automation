"""LLM client abstractions for the campaign (Phase 5.6, 5.7).

Single ``GenerationLLM`` Protocol. Three implementations:

* :class:`FakeLLM`         — deterministic in-memory, for tests + fixtures
* :class:`OllamaCloudLLM`  — Ollama HTTP for open-weight glm/deepseek
* :class:`OpenAILLM`       — OpenAI REST for gpt-6-astra

Every implementation must record ``finish_reason`` so the loop can flag
truncation (5.7) and refuse to score truncated runs downstream.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Protocol, runtime_checkable

import requests


# ─────────────────────────────────────────────────────────────────────────────
# Data types
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class GenerationRequest:
    prompt: str
    max_output_tokens: int
    temperature: float = 0.2
    reasoning_effort: Optional[str] = None   # "low" | "medium" | "high" | None
    seed: Optional[int] = None
    stop: Optional[list] = None


@dataclass(frozen=True)
class GenerationResponse:
    text: str
    finish_reason: str                       # "stop" | "length" | "error" | ...
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    wall_seconds: float = 0.0
    provider: str = ""
    model_id: str = ""
    raw: Dict[str, Any] = field(default_factory=dict)   # provider-specific payload


@runtime_checkable
class GenerationLLM(Protocol):
    """Runtime-checkable Protocol for generation clients."""

    provider: str
    model_id: str

    def generate(self, request: GenerationRequest) -> GenerationResponse: ...  # noqa: E704


# ─────────────────────────────────────────────────────────────────────────────
# FakeLLM — deterministic, no network
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FakeLLM:
    """Deterministic in-memory LLM for tests and fixtures.

    ``responses`` is consumed in FIFO order. When it's empty, a canonical
    empty feature model is returned so long-running loops don't crash.
    """
    responses: list = field(default_factory=list)
    provider: str = "fake"
    model_id: str = "fake-lm"
    finish_reason: str = "stop"
    seen_requests: list = field(default_factory=list)

    _DEFAULT_FM: str = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<featureModel><struct>'
        '<and abstract="true" mandatory="true" name="{ROOT}"/>'
        '</struct><constraints/></featureModel>'
    )

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        self.seen_requests.append(request)
        text = self.responses.pop(0) if self.responses else self._DEFAULT_FM.replace("{ROOT}", "Root")
        return GenerationResponse(
            text=text,
            finish_reason=self.finish_reason,
            prompt_tokens=len(request.prompt) // 4,
            completion_tokens=len(text) // 4,
            wall_seconds=0.0,
            provider=self.provider,
            model_id=self.model_id,
        )


# ─────────────────────────────────────────────────────────────────────────────
# OllamaCloudLLM — for glm-5.3-flash, deepseek-v4.1-flash
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class OllamaCloudLLM:
    """Chat/generate over Ollama HTTP for cloud-hosted open-weight models."""
    model_id: str                                       # e.g. "glm-5.3-flash"
    host: str = "https://ollama.com"
    provider: str = "ollama_cloud"
    api_key_env: str = "OLLAMA_API_KEY"
    api_key_file: str = ""
    timeout_s: int = 600

    def __post_init__(self) -> None:
        env_host = os.getenv("OLLAMA_LLM_HOST") or os.getenv("OLLAMA_HOST")
        if env_host:
            self.host = env_host.rstrip("/")
        self._api_key = os.getenv(self.api_key_env, "").strip()
        if not self._api_key and self.api_key_file:
            p = Path(self.api_key_file).expanduser()
            if p.exists():
                self._api_key = p.read_text(encoding="utf-8").strip()

    def _headers(self) -> Dict[str, str]:
        h = {"Content-Type": "application/json"}
        if self._api_key:
            h["Authorization"] = f"Bearer {self._api_key}"
        return h

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        payload: Dict[str, Any] = {
            "model": self.model_id,
            "prompt": request.prompt,
            "stream": False,
            "options": {
                "temperature": request.temperature,
                "num_predict": request.max_output_tokens,
            },
        }
        if request.seed is not None:
            payload["options"]["seed"] = request.seed
        if request.reasoning_effort:
            payload["options"]["reasoning_effort"] = request.reasoning_effort
        if request.stop:
            payload["options"]["stop"] = list(request.stop)

        t0 = time.time()
        r = requests.post(
            f"{self.host}/api/generate",
            headers=self._headers(),
            data=json.dumps(payload),
            timeout=self.timeout_s,
        )
        dt = time.time() - t0
        r.raise_for_status()
        data = r.json()
        text = data.get("response", "")
        # Ollama's finish reason surfaces as ``done_reason`` or ``done``
        done_reason = str(data.get("done_reason") or ("stop" if data.get("done") else "unknown"))
        return GenerationResponse(
            text=text,
            finish_reason=done_reason if done_reason != "length" else "length",
            prompt_tokens=data.get("prompt_eval_count"),
            completion_tokens=data.get("eval_count"),
            wall_seconds=round(dt, 2),
            provider=self.provider,
            model_id=self.model_id,
            raw={k: v for k, v in data.items() if k != "response"},
        )


# ─────────────────────────────────────────────────────────────────────────────
# OpenAILLM — for gpt-6-astra
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class OpenAILLM:
    """Chat-completion over OpenAI REST."""
    model_id: str
    provider: str = "openai"
    api_key_env: str = "OPENAI_API_KEY"
    api_key_file: str = ""
    base_url: str = "https://api.openai.com/v1"
    timeout_s: int = 600

    def __post_init__(self) -> None:
        env_url = os.getenv("OPENAI_BASE_URL")
        if env_url:
            self.base_url = env_url.rstrip("/")
        self._api_key = os.getenv(self.api_key_env, "").strip()
        if not self._api_key and self.api_key_file:
            p = Path(self.api_key_file).expanduser()
            if p.exists():
                self._api_key = p.read_text(encoding="utf-8").strip()

    def _headers(self) -> Dict[str, str]:
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}",
        }

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        payload: Dict[str, Any] = {
            "model": self.model_id,
            "messages": [{"role": "user", "content": request.prompt}],
            "max_completion_tokens": request.max_output_tokens,
            "temperature": request.temperature,
        }
        if request.reasoning_effort:
            payload["reasoning_effort"] = request.reasoning_effort
        if request.seed is not None:
            payload["seed"] = request.seed
        if request.stop:
            payload["stop"] = list(request.stop)

        t0 = time.time()
        r = requests.post(
            f"{self.base_url}/chat/completions",
            headers=self._headers(),
            data=json.dumps(payload),
            timeout=self.timeout_s,
        )
        dt = time.time() - t0
        r.raise_for_status()
        data = r.json()
        choice = (data.get("choices") or [{}])[0]
        text = (choice.get("message") or {}).get("content", "")
        finish_reason = str(choice.get("finish_reason") or "unknown")
        usage = data.get("usage") or {}
        return GenerationResponse(
            text=text,
            finish_reason=finish_reason,
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            wall_seconds=round(dt, 2),
            provider=self.provider,
            model_id=self.model_id,
            raw={"id": data.get("id"), "system_fingerprint": data.get("system_fingerprint")},
        )


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────

def make_client(*, provider: str, model_id: str, **kwargs) -> GenerationLLM:
    """Dispatch by provider name (as recorded in experiment.yaml)."""
    provider = provider.lower()
    if provider == "fake":
        return FakeLLM(model_id=model_id, **kwargs)
    if provider == "ollama_cloud":
        return OllamaCloudLLM(model_id=model_id, **kwargs)
    if provider == "openai":
        return OpenAILLM(model_id=model_id, **kwargs)
    raise ValueError(f"unknown provider: {provider!r}")
