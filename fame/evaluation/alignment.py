"""Occurrence-aware parent, redundancy and reference-attribution diagnostics."""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Mapping

import numpy as np
from lxml import etree

from fame.utils.marker_grammar import parse_marker

FEATURE_TAGS = {"feature", "and", "or", "alt"}


def extract_indexed_nodes(xml_path: Path) -> list[dict]:
    """Preorder feature occurrences, with parent indices rather than names."""
    parser = etree.XMLParser(resolve_entities=False, no_network=True, load_dtd=False)
    root = etree.parse(str(xml_path), parser).getroot()
    struct = root.find("struct")
    if struct is None:
        raise ValueError(f"Missing <struct>: {xml_path}")
    nodes: list[dict] = []

    def walk(node, parent_index):
        index = parent_index
        if node.tag in FEATURE_TAGS:
            name = node.get("name")
            if not name:
                raise ValueError(f"Unnamed feature node: {xml_path}")
            index = len(nodes)
            desc = node.find("description")
            description = (desc.text or "") if desc is not None else ""
            nodes.append(dict(index=index, name=name, parent_index=parent_index,
                              citations=tuple(parse_marker(description).doc_ids)))
        for child in node:
            if child.tag in FEATURE_TAGS:
                walk(child, index)

    for top in struct:
        if top.tag in FEATURE_TAGS:
            walk(top, None)
    return nodes


def evaluate_alignment(similarity, generated: list[dict], reference: list[dict], *,
                       tau: float, generated_self_similarity=None,
                       attribution: Mapping[str, set[str]] | None = None) -> dict:
    """Return raw counts and nullable rates; no ratios from an empty denominator."""
    matrix = np.asarray(similarity, dtype=float)
    if matrix.shape != (len(generated), len(reference)):
        raise ValueError("Similarity matrix and feature occurrences disagree")
    if not generated or not reference:
        raise ValueError("Feature populations must be nonempty")
    best = []
    for i in range(len(generated)):
        j = int(np.argmax(matrix[i]))  # first reference index breaks ties
        best.append(j if matrix[i, j] >= tau else None)

    evaluable = correct = exact_evaluable = exact_correct = 0
    for i, node in enumerate(generated):
        j = best[i]
        parent_i = node["parent_index"]
        if j is None or parent_i is None or best[parent_i] is None:
            continue
        ref_parent = reference[j]["parent_index"]
        if ref_parent is None:
            continue
        evaluable += 1
        correct += int(best[parent_i] == ref_parent)
        if node["name"] == reference[j]["name"]:
            exact_evaluable += 1
            exact_correct += int(node["name"] == reference[j]["name"] and
                                 generated[parent_i]["name"] == reference[ref_parent]["name"])

    name_counts = Counter(node["name"] for node in generated)
    duplicate_surplus = sum(n - 1 for n in name_counts.values() if n > 1)
    near = {0.8: 0, 0.9: 0}
    if generated_self_similarity is not None:
        self_matrix = np.asarray(generated_self_similarity, dtype=float)
        if self_matrix.shape != (len(generated), len(generated)):
            raise ValueError("Generated self-similarity matrix has wrong shape")
        for i in range(len(generated)):
            for j in range(i + 1, len(generated)):
                if generated[i]["name"] == generated[j]["name"]:
                    continue  # exact duplicates are counted separately
                for threshold in near:
                    near[threshold] += int(self_matrix[i, j] >= threshold)

    cited = applicable = agreeing = unmapped = unannotated = 0
    attribution = attribution or {}
    for i, node in enumerate(generated):
        for doc in node["citations"]:
            cited += 1
            j = best[i]
            if j is None:
                unmapped += 1
                continue
            reference_docs = attribution.get(reference[j]["name"])
            if not reference_docs:
                unannotated += 1
                continue
            applicable += 1
            agreeing += int(doc in reference_docs)

    return dict(n_parent_evaluable=evaluable, n_parent_correct=correct,
        parent_match_rate=correct/evaluable if evaluable else None,
        n_exact_parent_evaluable=exact_evaluable,
        n_exact_parent_correct=exact_correct,
        exact_name_parent_match_rate=exact_correct/exact_evaluable if exact_evaluable else None,
        exact_duplicate_surplus=duplicate_surplus,
        near_duplicate_pairs_tau_0_8=near[0.8] if generated_self_similarity is not None else None,
        near_duplicate_pairs_tau_0_9=near[0.9] if generated_self_similarity is not None else None,
        n_citations=cited, n_attribution_applicable=applicable,
        n_attribution_agreeing=agreeing,
        attribution_agreement_rate=agreeing/applicable if applicable else None,
        n_citations_unmapped=unmapped,
        n_citations_reference_unannotated=unannotated)
