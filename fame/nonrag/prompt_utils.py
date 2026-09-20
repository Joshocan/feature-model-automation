from __future__ import annotations

from pathlib import Path
from typing import Optional, TYPE_CHECKING, Dict

from fame.utils.dirs import FamePaths, ensure_dir
from .prompting import load_ss_nonrag_prompt, render_ss_nonrag_prompt, serialize_high_level_features
from fame.utils.placeholder_check import assert_no_placeholders

if TYPE_CHECKING:
    from .ss_pipeline import SSNonRagConfig


def build_ss_nonrag_prompt(
    cfg: SSNonRagConfig,
    *,
    context: str,
    paths: FamePaths,
    extra_placeholders: Optional[Dict[str, str]] = None,
    strict: bool = True,
) -> str:
    """
    Render the SS-NonRAG prompt with all placeholders filled.
    """
    tmpl = load_ss_nonrag_prompt(cfg.prompt_path)
    spec_dir = paths.specifications
    xsd_path = cfg.xsd_path or (spec_dir / "feature_model_featureide.xsd")
    metamodel_path = cfg.feature_metamodel_path or (spec_dir / "feature_metamodel_specification.txt")

    xsd_text = xsd_path.read_text(encoding="utf-8") if xsd_path.exists() else ""
    metamodel_text = metamodel_path.read_text(encoding="utf-8") if metamodel_path.exists() else ""
    high_level_xml = serialize_high_level_features(cfg.high_level_features)

    # Core-metamodel prompt placeholders. Empty strings when unset so the
    # renderer's strict-check doesn't fail on prompts that DO declare
    # these placeholders but the caller supplied no file.
    dv_path = getattr(cfg, "domain_vocabulary_path", None)
    ar_path = getattr(cfg, "artefacts_registry_path", None)
    domain_vocabulary_text = (
        Path(dv_path).read_text(encoding="utf-8") if dv_path and Path(dv_path).exists() else ""
    )
    artefacts_registry_text = (
        Path(ar_path).read_text(encoding="utf-8") if ar_path and Path(ar_path).exists() else ""
    )

    return render_ss_nonrag_prompt(
        root_feature=cfg.root_feature,
        domain=cfg.domain,
        context=context,
        prompt_template=tmpl,
        extra_placeholders={
            "CONTEXT": context,
            "XSD_METAMODEL": xsd_text,
            "FEATURE_METAMODEL": metamodel_text,
            "HIGH_LEVEL_FEATURES": high_level_xml,
            "MAX_DEPTH": str(cfg.max_depth) if cfg.max_depth is not None else "",
            "DOMAIN_VOCABULARY": domain_vocabulary_text,
            "ARTEFACTS_REGISTRY": artefacts_registry_text,
            "inputfile": "",
            "mmdir": "",
            **(extra_placeholders or {}),
        },
        strict=strict,
    )


def save_modified_prompt(
    *,
    prompt: str,
    model_safe: str,
    ts: str,
    paths: FamePaths,
    pipeline_type: str = "ss_nonrag",
) -> Path:
    """
    Save the fully rendered prompt into results/modified_prompts.
    """
    ensure_dir(paths.modified_prompts)
    out = paths.modified_prompts / f"{pipeline_type}_{model_safe}_{ts}-prompt.txt"
    out.write_text(prompt, encoding="utf-8")
    return out
