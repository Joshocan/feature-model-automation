"""Embedding client for the iFS 2027 campaign.

The retrieval encoder (``nomic-embed-text`` via Ollama) is trained with mandatory
task prefixes. The brief spells this out (§1): unprefixed queries or documents
degrade silently — same model, worse retrieval, no error signal. To keep the
mistake impossible, this module refuses to embed anything that does not carry
the expected prefix. Callers must invoke :meth:`OllamaEmbedder.embed_documents`
for chunks (auto-prepends ``search_document: ``) and
:meth:`OllamaEmbedder.embed_queries` for queries (auto-prepends
``search_query: ``). The raw text remains canonical; only the *embedded* form
carries the prefix.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import requests

DOC_PREFIX = "search_document: "
QUERY_PREFIX = "search_query: "


class Embedder:
    """Abstract encoder — kept for future substitution (e.g. sentence-transformers)."""

    def embed_documents(self, texts: Sequence[str]) -> List[List[float]]:
        raise NotImplementedError

    def embed_queries(self, texts: Sequence[str]) -> List[List[float]]:
        raise NotImplementedError


@dataclass
class OllamaEmbedder(Embedder):
    """Embed via a running Ollama HTTP server."""

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
        self.auth_header = (
            os.getenv("OLLAMA_AUTH_HEADER", self.auth_header).strip() or "Authorization"
        )
        self.auth_scheme = os.getenv("OLLAMA_AUTH_SCHEME", self.auth_scheme).strip()

    # ------------------------------------------------------------------ low level

    def _headers(self) -> dict:
        h: dict = {}
        if self.api_key:
            h[self.auth_header] = (
                f"{self.auth_scheme} {self.api_key}" if self.auth_scheme else self.api_key
            )
        return h

    def _embed_one(self, prefixed_text: str) -> List[float]:
        url = f"{self.host}/api/embeddings"
        payload = {"model": self.model, "prompt": prefixed_text}
        r = requests.post(url, json=payload, headers=self._headers(), timeout=self.timeout_s)
        r.raise_for_status()
        data = r.json()
        emb = data.get("embedding")
        if not isinstance(emb, list) or not emb:
            raise RuntimeError(f"Ollama returned invalid embedding payload: {data}")
        return [float(x) for x in emb]

    def _embed_batch(self, texts: Sequence[str], *, prefix: str) -> List[List[float]]:
        out: List[List[float]] = []
        for t in texts:
            body = (t or "").strip()
            if not body:
                raise ValueError("Refusing to embed empty text.")
            # Idempotency: if already prefixed correctly, don't double-prefix.
            if not body.startswith(prefix.strip()):
                body = prefix + body
            out.append(self._embed_one(body))
        return out

    # ------------------------------------------------------------------ public API

    def embed_documents(self, texts: Sequence[str]) -> List[List[float]]:
        """Embed chunk texts. Auto-prepends ``search_document: `` if missing."""
        return self._embed_batch(texts, prefix=DOC_PREFIX)

    def embed_queries(self, texts: Sequence[str]) -> List[List[float]]:
        """Embed query texts. Auto-prepends ``search_query: `` if missing."""
        return self._embed_batch(texts, prefix=QUERY_PREFIX)
