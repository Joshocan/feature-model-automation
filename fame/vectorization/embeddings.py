from __future__ import annotations

import os
from pathlib import Path
from dataclasses import dataclass
from typing import List, Sequence

import requests


class Embedder:
    def embed_documents(self, texts: Sequence[str]) -> List[List[float]]:
        raise NotImplementedError


@dataclass
class OllamaEmbedder(Embedder):
    """
    Embedding via Ollama server HTTP API.
    Requires Ollama server running (local or remote).

    Env vars supported:
      - OLLAMA_HOST  (default: http://127.0.0.1:11434)
      - OLLAMA_EMBED_MODEL (default: nomic-embed-text)
    """
    model: str = "nomic-embed-text"
    host: str = "http://127.0.0.1:11434"
    timeout_s: int = 120
    api_key: str = ""
    auth_header: str = "Authorization"
    auth_scheme: str = "Bearer"

    def __post_init__(self) -> None:
        env_host = os.getenv("OLLAMA_EMBED_HOST") or os.getenv("OLLAMA_HOST")
        env_model = os.getenv("OLLAMA_EMBED_MODEL")
        if env_host:
            self.host = env_host.rstrip("/")
        if env_model:
            self.model = env_model.strip()
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

    def _embed_one(self, text: str) -> List[float]:
        url = f"{self.host}/api/embeddings"
        payload = {"model": self.model, "prompt": text}
        headers = {}
        if self.api_key:
            if self.auth_scheme:
                headers[self.auth_header] = f"{self.auth_scheme} {self.api_key}"
            else:
                headers[self.auth_header] = self.api_key
        r = requests.post(url, json=payload, headers=headers, timeout=self.timeout_s)
        r.raise_for_status()
        data = r.json()
        emb = data.get("embedding")
        if not isinstance(emb, list):
            raise RuntimeError(f"Ollama returned invalid embedding payload: {data}")
        return [float(x) for x in emb]

    def embed_documents(self, texts: Sequence[str]) -> List[List[float]]:
        # Simple sequential embedding (safe + easy to debug)
        out: List[List[float]] = []
        for t in texts:
            t = (t or "").strip()
            if not t:
                out.append([])
                continue
            out.append(self._embed_one(t))
        return out
