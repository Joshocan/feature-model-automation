from __future__ import annotations

import os
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, Optional

import requests
from fame.exceptions import LLMTimeoutError, LLMHTTPError, format_error


@dataclass
class OllamaHTTP:
    """
    HTTP text-generation client for the FAME construction pipelines.

    It retains the Ollama-compatible interface used throughout FAME and also
    supports OpenAI Chat Completions when FAME_LLM_PROVIDER=openai.

    Env:
      - OLLAMA_HOST default http://127.0.0.1:11434
      - OLLAMA_LLM_MODEL default gpt-oss:120b-cloud
    """
    model: str = "gpt-oss:120b-cloud"
    host: str = "http://127.0.0.1:11434"
    timeout_s: int = 500
    api_key: str = ""
    auth_header: str = "Authorization"
    auth_scheme: str = "Bearer"
    retries: int = 3
    retry_delay_s: float = 5.0

    def __post_init__(self) -> None:
        self.provider = os.getenv("FAME_LLM_PROVIDER", "ollama").strip().lower()
        self.temperature = float(os.getenv("FAME_LLM_TEMPERATURE", "0.2"))
        # Prefer LLM-specific host; fallback to shared OLLAMA_HOST
        self.host = os.getenv("OLLAMA_LLM_HOST", os.getenv("OLLAMA_HOST", self.host)).rstrip("/")
        self.model = os.getenv("FAME_LLM_MODEL", os.getenv("OLLAMA_LLM_MODEL", self.model)).strip()
        if self.provider == "openai":
            self.host = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
            self.api_key = os.getenv("OPENAI_API_KEY", "").strip()
            return
        if self.provider == "anthropic":
            self.host = os.getenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com/v1").rstrip("/")
            self.api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
            return
        if self.provider == "gemini":
            self.host = os.getenv("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta").rstrip("/")
            self.api_key = os.getenv("GEMINI_API_KEY", "").strip()
            return
        key = os.getenv("OLLAMA_API_KEY", "").strip()
        key_file = os.getenv("OLLAMA_API_KEY_FILE", "").strip()
        if not key and key_file:
            try:
                key = Path(key_file).expanduser().read_text(encoding="utf-8").strip()
            except Exception:
                key = ""
        self.api_key = key
        self.auth_header = os.getenv("OLLAMA_AUTH_HEADER", self.auth_header).strip() or "Authorization"
        self.auth_scheme = os.getenv("OLLAMA_AUTH_SCHEME", self.auth_scheme).strip()
        self.retries = int(os.getenv("OLLAMA_RETRIES", self.retries))
        self.retry_delay_s = float(os.getenv("OLLAMA_RETRY_DELAY", self.retry_delay_s))

    def generate(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        temperature: float = 0.2,
    ) -> str:
        # The web run configuration is authoritative. CLI callers that do not
        # set FAME_LLM_TEMPERATURE retain the method/config supplied value.
        configured_temperature = os.getenv("FAME_LLM_TEMPERATURE", "").strip()
        if configured_temperature:
            temperature = self.temperature
        if self.provider == "openai":
            return self._generate_openai(prompt, system=system, temperature=temperature)
        if self.provider == "anthropic":
            return self._generate_anthropic(prompt, system=system, temperature=temperature)
        if self.provider == "gemini":
            return self._generate_gemini(prompt, system=system, temperature=temperature)

        url = f"{self.host}/api/generate"
        payload: Dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature},
        }
        if system:
            payload["system"] = system

        headers: Dict[str, str] = {}
        if self.api_key:
            if self.auth_scheme:
                headers[self.auth_header] = f"{self.auth_scheme} {self.api_key}"
            else:
                headers[self.auth_header] = self.api_key
        last_exc = None
        for attempt in range(1, self.retries + 1):
            try:
                r = requests.post(url, json=payload, headers=headers, timeout=self.timeout_s)
                if not r.ok:
                    detail = ""
                    try:
                        detail = r.json().get("error", "")
                    except Exception:
                        detail = r.text
                    raise LLMHTTPError(self.host, self.model, r.status_code, detail)

                data = r.json()
                out = data.get("response", "")
                return (out or "").strip()
            except requests.exceptions.ReadTimeout as e:
                last_exc = LLMTimeoutError(self.host, self.model, self.timeout_s)
            except requests.exceptions.RequestException as e:
                last_exc = LLMHTTPError(self.host, self.model, -1, detail=str(e))
            except LLMHTTPError as e:
                last_exc = e

            if attempt < self.retries:
                time.sleep(self.retry_delay_s)
                continue
            raise last_exc

    def _generate_openai(self, prompt: str, *, system: Optional[str], temperature: float) -> str:
        url = f"{self.host}/chat/completions"
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        payload: Dict[str, Any] = {"model": self.model, "messages": messages, "temperature": temperature}
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        last_exc = None
        for attempt in range(1, self.retries + 1):
            try:
                r = requests.post(url, json=payload, headers=headers, timeout=self.timeout_s)
                if not r.ok:
                    try:
                        detail = r.json().get("error", {}).get("message", "")
                    except Exception:
                        detail = r.text
                    raise LLMHTTPError(self.host, self.model, r.status_code, detail)
                choices = r.json().get("choices", [])
                content = choices[0].get("message", {}).get("content", "") if choices else ""
                if not isinstance(content, str) or not content.strip():
                    raise LLMHTTPError(self.host, self.model, r.status_code, "No text in OpenAI response")
                return content.strip()
            except requests.exceptions.ReadTimeout:
                last_exc = LLMTimeoutError(self.host, self.model, self.timeout_s)
            except requests.exceptions.RequestException as exc:
                last_exc = LLMHTTPError(self.host, self.model, -1, detail=str(exc))
            except LLMHTTPError as exc:
                last_exc = exc
            if attempt < self.retries:
                time.sleep(self.retry_delay_s)
                continue
            raise last_exc


    def _generate_anthropic(self, prompt: str, *, system: Optional[str], temperature: float) -> str:
        url = f"{self.host}/messages"
        messages = [{"role": "user", "content": prompt}]
        # Claude 4.x models (opus-4-7, sonnet-4-6, etc.) deprecated the
        # temperature field — sending it causes a 400. Omit it for these models;
        # retain it for claude-3.x and earlier which still accept it.
        model_lower = self.model.lower()
        supports_temperature = any(f"claude-{v}" in model_lower for v in ("1", "2", "3"))
        max_tokens = int(os.getenv("ANTHROPIC_MAX_TOKENS", "32000"))
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
        }
        if supports_temperature:
            payload["temperature"] = temperature
        if system:
            payload["system"] = system
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        last_exc = None
        for attempt in range(1, self.retries + 1):
            try:
                r = requests.post(url, json=payload, headers=headers, timeout=self.timeout_s)
                if not r.ok:
                    try:
                        detail = r.json().get("error", {}).get("message", "")
                    except Exception:
                        detail = r.text
                    raise LLMHTTPError(self.host, self.model, r.status_code, detail)
                data = r.json()
                stop_reason = data.get("stop_reason", "")
                content_blocks = data.get("content", [])
                text = content_blocks[0].get("text", "") if content_blocks else ""
                if not isinstance(text, str) or not text.strip():
                    raise LLMHTTPError(self.host, self.model, r.status_code, "No text in Anthropic response")
                if stop_reason == "max_tokens":
                    raise LLMHTTPError(self.host, self.model, r.status_code,
                                       f"Output truncated: max_tokens ({max_tokens}) reached before completion. "
                                       "Increase ANTHROPIC_MAX_TOKENS env var.")
                return text.strip()
            except requests.exceptions.ReadTimeout:
                last_exc = LLMTimeoutError(self.host, self.model, self.timeout_s)
            except requests.exceptions.RequestException as exc:
                last_exc = LLMHTTPError(self.host, self.model, -1, detail=str(exc))
            except LLMHTTPError as exc:
                last_exc = exc
            if attempt < self.retries:
                time.sleep(self.retry_delay_s)
                continue
            raise last_exc

    def _generate_gemini(self, prompt: str, *, system: Optional[str], temperature: float) -> str:
        url = f"{self.host}/models/{self.model}:generateContent?key={self.api_key}"
        contents = []
        if system:
            contents.append({"role": "user", "parts": [{"text": system}]})
            contents.append({"role": "model", "parts": [{"text": "Understood."}]})
        contents.append({"role": "user", "parts": [{"text": prompt}]})
        payload: Dict[str, Any] = {
            "contents": contents,
            "generationConfig": {"temperature": temperature},
        }
        headers = {"content-type": "application/json"}
        last_exc = None
        for attempt in range(1, self.retries + 1):
            try:
                r = requests.post(url, json=payload, headers=headers, timeout=self.timeout_s)
                if not r.ok:
                    try:
                        detail = r.json().get("error", {}).get("message", "")
                    except Exception:
                        detail = r.text
                    raise LLMHTTPError(self.host, self.model, r.status_code, detail)
                candidates = r.json().get("candidates", [])
                text = ""
                if candidates:
                    parts = candidates[0].get("content", {}).get("parts", [])
                    text = parts[0].get("text", "") if parts else ""
                if not isinstance(text, str) or not text.strip():
                    raise LLMHTTPError(self.host, self.model, r.status_code, "No text in Gemini response")
                return text.strip()
            except requests.exceptions.ReadTimeout:
                last_exc = LLMTimeoutError(self.host, self.model, self.timeout_s)
            except requests.exceptions.RequestException as exc:
                last_exc = LLMHTTPError(self.host, self.model, -1, detail=str(exc))
            except LLMHTTPError as exc:
                last_exc = exc
            if attempt < self.retries:
                time.sleep(self.retry_delay_s)
                continue
            raise last_exc


def assert_ollama_running(host: Optional[str] = None) -> None:
    if os.getenv("FAME_LLM_PROVIDER", "ollama").strip().lower() in {"openai", "anthropic", "gemini"}:
        return
    h = (host or os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")).rstrip("/")
    try:
        r = requests.get(f"{h}/api/tags", timeout=500)
        if not (200 <= r.status_code < 400):
            raise RuntimeError(f"Ollama not healthy: {r.status_code}")
    except Exception as e:
        raise RuntimeError(
            f"ERROR: Ollama is not reachable at {h}. Start Ollama first.\n"
            f"Details: {e}"
        )
