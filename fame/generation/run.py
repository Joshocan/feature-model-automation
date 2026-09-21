"""Run configuration + deterministic run_id derivation (Phase 5, 5.T8)."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class RunConfig:
    """All parameters that identify a single run.

    ``run_id`` is a SHA-256 of the config JSON. Changing any field yields a
    different run_id, hence a different results directory (Phase 5.T8: resume
    never overwrites a differently-configured run).
    """
    campaign_id: str
    corpus: str
    ordering_id: str                       # e.g. "primary", "alt_1"
    N: int
    grounding: str                         # "rag" | "nonrag"
    model_id: str
    provider: str
    seed: int
    repetition: int                        # 0-indexed within (config, seed) group
    metamodel_block: bool                  # True in guided arms, False in ablation
    k_doc: Optional[int]                   # None for nonrag
    domain: str
    root_feature: str

    # Frozen artefact hashes (bind the run to a specific protocol state)
    prompt_template_hash: str
    metamodel_hash: str
    chunks_hash: str
    encoder_digest: str

    # Model settings
    max_output_tokens: int
    temperature: float = 0.2
    reasoning_effort: Optional[str] = None
    context_window: int = 1_000_000

    # Free-form provenance
    extra: Dict[str, Any] = field(default_factory=dict)

    def canonical_dict(self) -> Dict[str, Any]:
        """Deterministic JSON-safe dict — used for run_id + snapshot."""
        d = {
            "campaign_id":          self.campaign_id,
            "corpus":               self.corpus,
            "ordering_id":          self.ordering_id,
            "N":                    self.N,
            "grounding":            self.grounding,
            "model_id":             self.model_id,
            "provider":             self.provider,
            "seed":                 self.seed,
            "repetition":           self.repetition,
            "metamodel_block":      self.metamodel_block,
            "k_doc":                self.k_doc,
            "domain":               self.domain,
            "root_feature":         self.root_feature,
            "prompt_template_hash": self.prompt_template_hash,
            "metamodel_hash":       self.metamodel_hash,
            "chunks_hash":          self.chunks_hash,
            "encoder_digest":       self.encoder_digest,
            "max_output_tokens":    self.max_output_tokens,
            "temperature":          self.temperature,
            "reasoning_effort":     self.reasoning_effort,
            "context_window":       self.context_window,
            "extra":                dict(sorted(self.extra.items())),
        }
        return d

    def run_id(self) -> str:
        """First 16 hex chars of the SHA-256 of the canonical config JSON."""
        payload = json.dumps(self.canonical_dict(), sort_keys=True,
                             ensure_ascii=False, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    def config_hash(self) -> str:
        """Alias for :meth:`run_id` — used as the enclosing directory name."""
        return self.run_id()
