from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence


@dataclass
class GroundednessMetrics:
    groundedness: float
    relevance: float
    context_utilization: float
    hallucination_proxy: float
    prompt_adherence: float


def evaluate_groundedness(
    answers: Sequence[str],
    contexts: Sequence[str],
    prompts: Sequence[str],
    *,
    provider: str = "openai",
    model: str = "gpt-4.1",
    temperature: float = 0.0,
    max_tokens: int = 1024,
    api_key_env: Optional[str] = None,
    base_url: Optional[str] = None,
) -> GroundednessMetrics:
    """
    Evaluate groundedness-style metrics using TruLens LLM feedback.

    Inputs are parallel lists of answers, contexts, prompts.
    Provider options: openai, anthropic, gemini (OpenAI-compatible base_url may work as well).
    """
    if not (len(answers) == len(contexts) == len(prompts)):
        raise ValueError("answers, contexts, prompts must have the same length")

    try:
        # Newer TruLens (>=2.x): Groundedness lives in feedback.grounded
        from trulens_eval import Feedback
        try:
            from trulens_eval.feedback.grounded import Groundedness  # type: ignore
        except Exception:
            # Older TruLens (<2.x)
            from trulens_eval.feedback import Groundedness  # type: ignore
        from trulens_eval.feedback.provider import OpenAI, Anthropic, Gemini
    except Exception as e:
        raise ImportError(
            "trulens-eval with Groundedness is required. Install/upgrade with: pip install \"trulens-eval>=2.7.0\""
        ) from e

    provider = provider.lower()
    if provider == "openai":
        api_key_env = api_key_env or "OPENAI_API_KEY"
        prov = OpenAI(model_engine=model, api_key_env=api_key_env, api_base=base_url)
    elif provider == "anthropic":
        api_key_env = api_key_env or "ANTHROPIC_API_KEY"
        prov = Anthropic(model=model, api_key_env=api_key_env)
    elif provider == "gemini":
        api_key_env = api_key_env or "GEMINI_API_KEY"
        prov = Gemini(model_name=model, api_key_env=api_key_env)
    else:
        raise ValueError(f"Unsupported provider: {provider}")

    g = Groundedness(groundedness_provider=prov)
    # API compatibility: prefer qa_groundedness_measure_with_cot_reasons if present, else groundedness_measure_with_cot_reasons
    grounded_fn = getattr(g, "qa_groundedness_measure_with_cot_reasons", None) or getattr(
        g, "groundedness_measure_with_cot_reasons", None
    )
    if grounded_fn is None:
        raise ImportError("Groundedness callable not found in installed trulens-eval.")

    grounded_fb = Feedback(grounded_fn, name="groundedness")
    # Provider helper methods are stable across versions
    relevance_fb = Feedback(prov.relevance_with_cot_reasons, name="relevance")
    context_fb = Feedback(prov.context_relevance_with_cot_reasons, name="context_utilization")
    adherence_fb = Feedback(prov.prompt_adherence_with_cot_reasons, name="prompt_adherence")

    grounded_scores: List[float] = []
    relevance_scores: List[float] = []
    context_scores: List[float] = []
    adherence_scores: List[float] = []

    for q, ctx, ans in zip(prompts, contexts, answers):
        record = dict(input=q, context=[ctx], output=ans, temperature=temperature, max_tokens=max_tokens)
        grounded_scores.append(float(grounded_fb(record)))
        relevance_scores.append(float(relevance_fb(record)))
        context_scores.append(float(context_fb(record)))
        adherence_scores.append(float(adherence_fb(record)))

    def _mean(xs: List[float]) -> float:
        return sum(xs) / len(xs) if xs else 0.0

    grounded_mean = _mean(grounded_scores)
    relevance_mean = _mean(relevance_scores)
    context_mean = _mean(context_scores)
    adherence_mean = _mean(adherence_scores)

    return GroundednessMetrics(
        groundedness=round(grounded_mean, 4),
        relevance=round(relevance_mean, 4),
        context_utilization=round(context_mean, 4),
        hallucination_proxy=round(1 - grounded_mean, 4),
        prompt_adherence=round(adherence_mean, 4),
    )
