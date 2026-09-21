"""Unified N-step generation engine for the iFS 2027 campaign.

Replaces the retired 4-pipeline design (SS/IS × RAG/Non-RAG). Single loop
parameterised by (corpus, ordering, N, grounding, model, seed).

Public API is deliberately narrow: build a :class:`RunConfig`, hand it to
:func:`run_generation`, and every D7/D8/D9/D10 artefact is written under the
run directory.
"""
from .batching import Batch, slice_ordering
from .grounding import GroundingContext, build_grounding
from .llm_client import (
    FakeLLM,
    GenerationLLM,
    GenerationRequest,
    GenerationResponse,
    OllamaCloudLLM,
    OpenAILLM,
    make_client,
)
from .loop import RunAlreadyExists, RunResult, run_generation
from .persistence import RunPaths, atomic_write_json, atomic_write_text
from .prompt_assembly import PromptBundle, render_prompt
from .run import RunConfig
from .token_budget import (
    TokenCounter,
    count_tokens,
    universal_estimate,
)

__all__ = [
    # batching
    "Batch",
    "slice_ordering",
    # grounding
    "GroundingContext",
    "build_grounding",
    # llm_client
    "FakeLLM",
    "GenerationLLM",
    "GenerationRequest",
    "GenerationResponse",
    "OllamaCloudLLM",
    "OpenAILLM",
    "make_client",
    # loop
    "RunAlreadyExists",
    "RunResult",
    "run_generation",
    # persistence
    "RunPaths",
    "atomic_write_json",
    "atomic_write_text",
    # prompt_assembly
    "PromptBundle",
    "render_prompt",
    # run
    "RunConfig",
    # token_budget
    "TokenCounter",
    "count_tokens",
    "universal_estimate",
]
