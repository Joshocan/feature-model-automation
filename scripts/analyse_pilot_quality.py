#!/usr/bin/env python3
"""Zero-call pilot audits: citations, redundancy, variability, and GT parents."""
from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable
from xml.etree import ElementTree as ET

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from fame.evaluation.coverage import extract_nodes  # noqa: E402
from fame.evaluation.local_encoder import LocalTransformerEncoder, discover_cached_snapshot  # noqa: E402
from fame.evaluation.variability import group_kind_distribution  # noqa: E402


DEFAULT_ANALYSIS = REPO / "results" / "pilot-correctness-analysis-2026-09-23"
DEFAULT_MANIFEST = REPO / "config" / "pilot_correctness_manifest.json"
TRACE_RE = re.compile(r"Trace:\s*\[\s*(rep_\d+(?:\s*,\s*rep_\d+)*)\s*\]\s*\Z")
NEAR_DUPLICATE_TAU = 0.80


def _cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--analysis-dir", type=Path, default=DEFAULT_ANALYSIS)
    p.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    return p.parse_args()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: Iterable[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _base(row: dict[str, Any]) -> dict[str, Any]:
    return {key: row[key] for key in (
        "cohort", "primary", "source_campaign", "run_id", "model_id", "seed",
        "N", "grounding", "completed", "admissible",
    )}


def _parse_trace(text: str) -> list[str]:
    match = TRACE_RE.search(text or "")
    return [part.strip() for part in match.group(1).split(",")] if match else []


def _features(xml_path: Path) -> list[dict[str, Any]]:
    root = ET.parse(xml_path).getroot()
    struct = root.find("struct")
    output: list[dict[str, Any]] = []

    def walk(elem: ET.Element, parent: str | None = None) -> None:
        name = elem.get("name") if elem.tag in {"and", "or", "alt", "feature"} else None
        current_parent = parent
        if name:
            desc = elem.find("description")
            description = (desc.text or "").strip() if desc is not None else ""
            output.append({
                "name": name,
                "parent": parent,
                "tag": elem.tag,
                "mandatory": (elem.get("mandatory") or "").lower() == "true",
                "description": description,
                "citations": _parse_trace(description),
            })
            current_parent = name
        for child in elem:
            if child.tag in {"and", "or", "alt", "feature"}:
                walk(child, current_parent)

    if struct is not None:
        for child in struct:
            walk(child)
    return output


def _first_seen(run_root: Path) -> dict[str, int]:
    seen: dict[str, int] = {}
    for fallback, path in enumerate(sorted((run_root / "fm_iter").glob("step_*.xml"))):
        match = re.search(r"step_(\d+)", path.stem)
        step = int(match.group(1)) if match else fallback
        try:
            for name, _ in extract_nodes(path):
                seen.setdefault(name, step)
        except Exception:
            continue
    return seen


def _citation_first_seen(run_root: Path) -> dict[tuple[str, str], int]:
    """Earliest accepted iteration containing each final-style feature/citation pair."""
    seen: dict[tuple[str, str], int] = {}
    for fallback, path in enumerate(sorted((run_root / "fm_iter").glob("step_*.xml"))):
        match = re.search(r"step_(\d+)", path.stem)
        step = int(match.group(1)) if match else fallback
        try:
            for feature in _features(path):
                for doc_id in feature["citations"]:
                    seen.setdefault((feature["name"], doc_id), step)
        except Exception:
            continue
    return seen


def _context_by_step(run_root: Path) -> tuple[dict[int, set[str]], dict[int, set[str]]]:
    batch: dict[int, set[str]] = {}
    retrieved: dict[int, set[str]] = {}
    path = run_root / "context_log.jsonl"
    if not path.exists():
        return batch, retrieved
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        step = int(row["step_index"])
        batch[step] = set(row.get("batch_doc_ids") or [])
        retrieved[step] = set(row.get("chunk_doc_ids") or [])
    return batch, retrieved


def _load_reference_metadata() -> tuple[set[str], dict[str, set[str]], dict[str, str]]:
    with (REPO / "data/raw/repair/manifest_repair.csv").open(encoding="utf-8") as handle:
        known = {row["doc_id"] for row in csv.DictReader(handle, delimiter=";")}
    attribution: dict[str, set[str]] = defaultdict(set)
    for row in _read_csv(REPO / "data/attribution/repair.csv"):
        attribution[row["gt_feature_id"]].add(row["doc_id"])
    partition = {
        row["gt_feature_id"]: row["partition"]
        for row in _read_csv(REPO / "data/feature_partition/repair.csv")
    }
    return known, attribution, partition


def _best_reference_matches(d15_path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    best: dict[tuple[str, str], dict[str, Any]] = {}
    with d15_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = (row["run_id"], row["generated_feature"])
            similarity = float(row["similarity"])
            if key not in best or similarity > best[key]["similarity"]:
                best[key] = {
                    "gt_feature": row["reference_feature"],
                    "gt_parent": row["reference_parent"] or None,
                    "generated_parent": row["generated_parent"] or None,
                    "similarity": similarity,
                }
    return best


def _best_generated_matches(d15_path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    """Best generated feature for each run/reference feature."""
    best: dict[tuple[str, str], dict[str, Any]] = {}
    with d15_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = (row["run_id"], row["reference_feature"])
            similarity = float(row["similarity"])
            if key not in best or similarity > best[key]["similarity"]:
                best[key] = {
                    "generated_feature": row["generated_feature"],
                    "similarity": similarity,
                }
    return best


def _jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / len(a | b) if a or b else 1.0


def citation_audit(
    inventory: list[dict[str, str]],
    best: dict[tuple[str, str], dict[str, Any]],
    *,
    tau: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    known, attribution, partition = _load_reference_metadata()
    detail: list[dict[str, Any]] = []
    summary: list[dict[str, Any]] = []
    for run in inventory:
        if run["admissible"] != "True":
            continue
        run_root = REPO / run["run_root"]
        features = _features(REPO / run["fm_gen"])
        first_seen = _first_seen(run_root)
        citation_seen = _citation_first_seen(run_root)
        batch_by_step, retrieved_by_step = _context_by_step(run_root)
        citation_sets: list[set[str]] = []
        applicable = agreeing = 0
        l1_total = l1_valid = l2_total = l2_retrieved = l2_batch = 0
        cited_all: set[str] = set()
        for feature in features:
            citations = feature["citations"]
            citation_sets.append(set(citations))
            cited_all.update(citations)
            match = best.get((run["run_id"], feature["name"]))
            matched_gt = match["gt_feature"] if match and match["similarity"] >= tau else None
            is_applicable = bool(matched_gt and partition.get(matched_gt) == "attested")
            for doc_id in citations:
                feature_step = first_seen.get(feature["name"])
                step = citation_seen.get((feature["name"], doc_id), feature_step)
                l1_total += 1
                l1_ok = doc_id in known
                l1_valid += int(l1_ok)
                agreement: bool | None = None
                if is_applicable:
                    applicable += 1
                    agreement = doc_id in attribution.get(matched_gt, set())
                    agreeing += int(agreement)
                retrieved_ok = batch_ok = None
                if step is not None and run["grounding"] == "rag":
                    l2_total += 1
                    retrieved_ok = doc_id in retrieved_by_step.get(step, set())
                    batch_ok = doc_id in batch_by_step.get(step, set())
                    l2_retrieved += int(retrieved_ok)
                    l2_batch += int(batch_ok)
                detail.append({
                    **_base(run),
                    "generated_feature": feature["name"],
                    "generated_parent": feature["parent"] or "",
                    "feature_first_seen_step": feature_step if feature_step is not None else "",
                    "citation_first_seen_step": step if step is not None else "",
                    "cited_doc_id": doc_id,
                    "referential_integrity": l1_ok,
                    "retrieved_at_first_seen": retrieved_ok,
                    "in_batch_at_first_seen": batch_ok,
                    "matched_gt_feature": matched_gt or "",
                    "match_similarity": match["similarity"] if match else "",
                    "gt_partition": partition.get(matched_gt, "") if matched_gt else "",
                    "attribution_applicable": is_applicable,
                    "gt_attribution_agreement": agreement,
                })
        counts = [len(s) for s in citation_sets]
        set_counts = Counter(tuple(sorted(s)) for s in citation_sets)
        pair_jaccards = [
            _jaccard(citation_sets[i], citation_sets[j])
            for i in range(len(citation_sets)) for j in range(i + 1, len(citation_sets))
        ]
        summary.append({
            **_base(run),
            "n_features": len(features),
            "n_citations": sum(counts),
            "n_distinct_cited_docs": len(cited_all),
            "citations_per_feature_mean": statistics.mean(counts) if counts else 0,
            "citations_per_feature_median": statistics.median(counts) if counts else 0,
            "citations_per_feature_min": min(counts) if counts else 0,
            "citations_per_feature_max": max(counts) if counts else 0,
            "n_unique_citation_sets": len(set_counts),
            "dominant_citation_set_share": max(set_counts.values()) / len(features) if features else 0,
            "mean_pairwise_citation_jaccard": statistics.mean(pair_jaccards) if pair_jaccards else 0,
            "L1_valid_rate": l1_valid / l1_total if l1_total else "",
            "L2_retrieved_at_first_seen_rate": l2_retrieved / l2_total if l2_total else "",
            "L2_batch_scope_at_first_seen_rate": l2_batch / l2_total if l2_total else "",
            "attribution_applicable_citations": applicable,
            "gt_attribution_agreement_rate": agreeing / applicable if applicable else "",
        })
    return detail, summary


def _normalise_name(name: str) -> str:
    return "".join(ch.lower() for ch in name if ch.isalnum())


def _permutation_key(name: str) -> str:
    return "_".join(sorted(part.lower() for part in re.split(r"[_\-\s]+", name) if part))


def redundancy_audit(
    inventory: list[dict[str, str]],
    best: dict[tuple[str, str], dict[str, Any]],
    encoder: LocalTransformerEncoder,
    *,
    tau: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    citation_detail, _ = citation_audit(inventory, best, tau=tau)
    supported: dict[tuple[str, str], bool] = defaultdict(bool)
    applicable: set[tuple[str, str]] = set()
    for row in citation_detail:
        key = (row["run_id"], row["generated_feature"])
        if row["attribution_applicable"]:
            applicable.add(key)
            supported[key] = supported[key] or row["gt_attribution_agreement"] is True

    pairs: list[dict[str, Any]] = []
    feature_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for run in inventory:
        if run["admissible"] != "True":
            continue
        features = _features(REPO / run["fm_gen"])
        names = [feature["name"] for feature in features]
        embeddings = encoder.encode(names, normalize_embeddings=True, convert_to_tensor=False)
        similarities = np.asarray(embeddings) @ np.asarray(embeddings).T
        normalised = Counter(_normalise_name(name) for name in names)
        permutations = Counter(_permutation_key(name) for name in names)
        near_indices: set[int] = set()
        n_pairs_04 = n_pairs_08 = n_pairs_09 = 0
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                similarity = float(similarities[i, j])
                n_pairs_04 += int(similarity >= tau)
                n_pairs_08 += int(similarity >= NEAR_DUPLICATE_TAU)
                n_pairs_09 += int(similarity >= 0.90)
                if similarity >= NEAR_DUPLICATE_TAU:
                    near_indices.update((i, j))
                    pairs.append({
                        **_base(run),
                        "feature_a": names[i],
                        "feature_b": names[j],
                        "similarity": similarity,
                        "normalised_equal": _normalise_name(names[i]) == _normalise_name(names[j]),
                        "permutation_equal": _permutation_key(names[i]) == _permutation_key(names[j]),
                    })
        unmatched = no_support = applicable_features = 0
        for index, feature in enumerate(features):
            match = best[(run["run_id"], feature["name"])]
            key = (run["run_id"], feature["name"])
            ref_unmatched = match["similarity"] < tau
            unmatched += int(ref_unmatched)
            is_applicable = key in applicable
            applicable_features += int(is_applicable)
            no_attr_support = is_applicable and not supported[key]
            no_support += int(no_attr_support)
            feature_rows.append({
                **_base(run),
                "generated_feature": feature["name"],
                "best_gt_feature": match["gt_feature"],
                "best_gt_similarity": match["similarity"],
                "reference_unmatched_tau_0_4": ref_unmatched,
                "normalised_name_collision": normalised[_normalise_name(feature["name"])] > 1,
                "permutation_collision": permutations[_permutation_key(feature["name"])] > 1,
                "near_duplicate_member_tau_0_8": index in near_indices,
                "attribution_support_applicable": is_applicable,
                "has_gt_attribution_support": supported[key] if is_applicable else "",
            })
        summaries.append({
            **_base(run),
            "n_features": len(features),
            "n_normalised_name_collision_features": sum(v for v in normalised.values() if v > 1),
            "n_permutation_collision_features": sum(v for v in permutations.values() if v > 1),
            "n_semantic_pairs_tau_0_4": n_pairs_04,
            "n_near_duplicate_pairs_tau_0_8": n_pairs_08,
            "n_near_duplicate_pairs_tau_0_9": n_pairs_09,
            "n_near_duplicate_member_features_tau_0_8": len(near_indices),
            "n_reference_unmatched_tau_0_4": unmatched,
            "reference_unmatched_share_tau_0_4": unmatched / len(features) if features else 0,
            "n_attribution_support_applicable_features": applicable_features,
            "n_without_gt_attribution_support": no_support,
            "no_gt_attribution_support_share": no_support / applicable_features if applicable_features else "",
        })
    return feature_rows, pairs, summaries


def variability_audit(inventory: list[dict[str, str]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    per_run: list[dict[str, Any]] = []
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for run in inventory:
        if run["admissible"] == "True":
            dist = group_kind_distribution(REPO / run["fm_gen"])
            values = {
                "n_internal": dist.n_internal, "n_and": dist.n_and, "n_or": dist.n_or,
                "n_alt": dist.n_alt, "and_prop": dist.and_prop, "or_prop": dist.or_prop,
                "alt_prop": dist.alt_prop, "mandatory_ratio": dist.mandatory_ratio,
                "degenerate": dist.is_degenerate,
            }
            per_run.append({**_base(run), **values})
        grouped[(run["cohort"], run["model_id"], run["grounding"])].append(run)
    indexed = {row["run_id"]: row for row in per_run}
    summary: list[dict[str, Any]] = []
    for (cohort, model, grounding), runs in sorted(grouped.items()):
        scored = [indexed[run["run_id"]] for run in runs if run["run_id"] in indexed]
        mean = lambda key: statistics.mean(float(row[key]) for row in scored) if scored else ""
        n_nondegenerate = sum(not row["degenerate"] for row in scored)
        summary.append({
            "cohort": cohort, "model_id": model, "grounding": grounding,
            "n_planned": len(runs),
            "n_completed": sum(run["completed"] == "True" for run in runs),
            "n_admissible": len(scored),
            "n_non_degenerate": n_nondegenerate,
            "non_degenerate_rate_planned": n_nondegenerate / len(runs),
            "non_degenerate_rate_admissible": n_nondegenerate / len(scored) if scored else "",
            "and_prop_mean": mean("and_prop"), "or_prop_mean": mean("or_prop"),
            "alt_prop_mean": mean("alt_prop"), "mandatory_ratio_mean": mean("mandatory_ratio"),
        })
    return per_run, summary


def parent_audit(
    inventory: list[dict[str, str]],
    best: dict[tuple[str, str], dict[str, Any]],
    *,
    tau: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    per_run: list[dict[str, Any]] = []
    for run in inventory:
        if run["admissible"] != "True":
            continue
        features = _features(REPO / run["fm_gen"])
        eligible = correct = feature_matched = 0
        exact_eligible = exact_correct = 0
        for feature in features:
            child = best[(run["run_id"], feature["name"])]
            if child["similarity"] < tau:
                continue
            feature_matched += 1
            if feature["parent"] is None or child["gt_parent"] is None:
                continue
            parent = best.get((run["run_id"], feature["parent"]))
            if parent is None or parent["similarity"] < tau:
                continue
            eligible += 1
            correct += int(parent["gt_feature"] == child["gt_parent"])
            if feature["name"] == child["gt_feature"]:
                exact_eligible += 1
                exact_correct += int(feature["parent"] == child["gt_parent"])
        per_run.append({
            **_base(run),
            "n_features": len(features),
            "n_semantically_matched_features": feature_matched,
            "n_parent_evaluable": eligible,
            "n_parent_correct": correct,
            "semantic_parent_match_rate": correct / eligible if eligible else "",
            "n_exact_name_parent_evaluable": exact_eligible,
            "n_exact_parent_correct": exact_correct,
            "exact_name_parent_match_rate": exact_correct / exact_eligible if exact_eligible else "",
        })
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in per_run:
        grouped[(row["cohort"], row["model_id"], row["grounding"])].append(row)
    summary = []
    for (cohort, model, grounding), rows in sorted(grouped.items()):
        rates = [float(row["semantic_parent_match_rate"]) for row in rows if row["semantic_parent_match_rate"] != ""]
        summary.append({
            "cohort": cohort, "model_id": model, "grounding": grounding,
            "n_scored": len(rows), "n_parent_evaluable": sum(row["n_parent_evaluable"] for row in rows),
            "n_parent_correct": sum(row["n_parent_correct"] for row in rows),
            "semantic_parent_match_micro": (
                sum(row["n_parent_correct"] for row in rows) / sum(row["n_parent_evaluable"] for row in rows)
                if sum(row["n_parent_evaluable"] for row in rows) else ""
            ),
            "semantic_parent_match_macro": statistics.mean(rates) if rates else "",
            "n_exact_name_parent_evaluable": sum(row["n_exact_name_parent_evaluable"] for row in rows),
            "n_exact_parent_correct": sum(row["n_exact_parent_correct"] for row in rows),
            "exact_name_parent_match_micro": (
                sum(row["n_exact_parent_correct"] for row in rows)
                / sum(row["n_exact_name_parent_evaluable"] for row in rows)
                if sum(row["n_exact_name_parent_evaluable"] for row in rows) else ""
            ),
        })
    return per_run, summary


def parent_sensitivity_audit(
    inventory: list[dict[str, str]],
    best: dict[tuple[str, str], dict[str, Any]],
    reverse_best: dict[tuple[str, str], dict[str, Any]],
    *,
    taus: list[float],
) -> list[dict[str, Any]]:
    """Threshold and reciprocal-match validation of semantic parent match."""
    aggregate: dict[tuple[str, str, str, float, str], dict[str, Any]] = defaultdict(
        lambda: {"n": 0, "correct": 0, "shuffle_expected": 0.0, "run_denominators": []}
    )
    for run in inventory:
        if run["admissible"] != "True":
            continue
        features = _features(REPO / run["fm_gen"])
        for tau in taus:
            for policy in ("independent_max", "reciprocal_nearest"):
                relations: list[tuple[str, str]] = []
                for feature in features:
                    child = best[(run["run_id"], feature["name"])]
                    if child["similarity"] < tau or feature["parent"] is None or child["gt_parent"] is None:
                        continue
                    parent = best.get((run["run_id"], feature["parent"]))
                    if parent is None or parent["similarity"] < tau:
                        continue
                    if policy == "reciprocal_nearest":
                        child_reverse = reverse_best[(run["run_id"], child["gt_feature"])]
                        parent_reverse = reverse_best[(run["run_id"], parent["gt_feature"])]
                        if (
                            child_reverse["generated_feature"] != feature["name"]
                            or parent_reverse["generated_feature"] != feature["parent"]
                        ):
                            continue
                    relations.append((child["gt_parent"], parent["gt_feature"]))
                n = len(relations)
                correct = sum(expected == mapped for expected, mapped in relations)
                expected_counts = Counter(expected for expected, _ in relations)
                mapped_counts = Counter(mapped for _, mapped in relations)
                # Expected number correct after permuting mapped-parent labels within the run.
                shuffled = (
                    sum(expected_counts[key] * mapped_counts[key] for key in set(expected_counts) | set(mapped_counts)) / n
                    if n else 0.0
                )
                key = (run["cohort"], run["model_id"], run["grounding"], tau, policy)
                bucket = aggregate[key]
                bucket["n"] += n
                bucket["correct"] += correct
                bucket["shuffle_expected"] += shuffled
                bucket["run_denominators"].append(n)
    output = []
    for (cohort, model, grounding, tau, policy), values in sorted(aggregate.items()):
        n = values["n"]
        output.append({
            "cohort": cohort, "model_id": model, "grounding": grounding,
            "tau": tau, "matching_policy": policy,
            "n_parent_evaluable": n, "n_parent_correct": values["correct"],
            "parent_match_rate": values["correct"] / n if n else "",
            "shuffled_parent_expected_rate": values["shuffle_expected"] / n if n else "",
            "run_denominators": " | ".join(str(value) for value in values["run_denominators"]),
        })
    return output


def _citation_model_summary(
    detail: list[dict[str, Any]], summaries: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    detail_groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    run_groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in detail:
        detail_groups[(row["cohort"], row["model_id"], row["grounding"])].append(row)
    for row in summaries:
        run_groups[(row["cohort"], row["model_id"], row["grounding"])].append(row)
    output = []
    for key, runs in sorted(run_groups.items()):
        rows = detail_groups[key]
        l1 = [row for row in rows]
        l2 = [row for row in rows if row["retrieved_at_first_seen"] is not None]
        attr = [row for row in rows if row["attribution_applicable"]]
        output.append({
            "cohort": key[0], "model_id": key[1], "grounding": key[2], "n_runs": len(runs),
            "features_mean": statistics.mean(row["n_features"] for row in runs),
            "citations_per_feature_mean": (
                sum(row["n_citations"] for row in runs) / sum(row["n_features"] for row in runs)
            ),
            "distinct_cited_docs_mean": statistics.mean(row["n_distinct_cited_docs"] for row in runs),
            "unique_citation_sets_mean": statistics.mean(row["n_unique_citation_sets"] for row in runs),
            "dominant_citation_set_share_mean": statistics.mean(row["dominant_citation_set_share"] for row in runs),
            "mean_pairwise_citation_jaccard": statistics.mean(row["mean_pairwise_citation_jaccard"] for row in runs),
            "L1_valid_rate": sum(row["referential_integrity"] is True for row in l1) / len(l1) if l1 else "",
            "L2_retrieved_at_citation_first_seen_rate": (
                sum(row["retrieved_at_first_seen"] is True for row in l2) / len(l2) if l2 else ""
            ),
            "attribution_applicable_citations": len(attr),
            "gt_attribution_agreement_rate": (
                sum(row["gt_attribution_agreement"] is True for row in attr) / len(attr) if attr else ""
            ),
        })
    return output


def _redundancy_model_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["cohort"], row["model_id"], row["grounding"])].append(row)
    output = []
    for key, runs in sorted(groups.items()):
        n_features = sum(row["n_features"] for row in runs)
        applicable = sum(row["n_attribution_support_applicable_features"] for row in runs)
        output.append({
            "cohort": key[0], "model_id": key[1], "grounding": key[2], "n_runs": len(runs),
            "features_mean": n_features / len(runs),
            "normalised_name_collision_features_total": sum(row["n_normalised_name_collision_features"] for row in runs),
            "permutation_collision_features_total": sum(row["n_permutation_collision_features"] for row in runs),
            "near_duplicate_pairs_tau_0_8_mean": statistics.mean(row["n_near_duplicate_pairs_tau_0_8"] for row in runs),
            "near_duplicate_member_share_tau_0_8": (
                sum(row["n_near_duplicate_member_features_tau_0_8"] for row in runs) / n_features
                if n_features else ""
            ),
            "reference_unmatched_share_tau_0_4": (
                sum(row["n_reference_unmatched_tau_0_4"] for row in runs) / n_features
                if n_features else ""
            ),
            "attribution_support_applicable_features": applicable,
            "no_gt_attribution_support_share": (
                sum(row["n_without_gt_attribution_support"] for row in runs) / applicable
                if applicable else ""
            ),
        })
    return output


def main() -> int:
    args = _cli()
    analysis = args.analysis_dir.resolve()
    manifest = json.loads(args.manifest.read_text())
    tau = float(manifest["tau_primary"])
    inventory = _read_csv(analysis / "run_inventory.csv")
    best = _best_reference_matches(analysis / "D15_match_scores.csv")
    reverse_best = _best_generated_matches(analysis / "D15_match_scores.csv")

    citation_detail, citation_summary = citation_audit(inventory, best, tau=tau)
    _write_csv(analysis / "citation_audit.csv", citation_detail, [
        "cohort", "primary", "source_campaign", "run_id", "model_id", "seed", "N", "grounding",
        "generated_feature", "generated_parent", "feature_first_seen_step",
        "citation_first_seen_step", "cited_doc_id",
        "referential_integrity", "retrieved_at_first_seen", "in_batch_at_first_seen",
        "matched_gt_feature", "match_similarity", "gt_partition", "attribution_applicable",
        "gt_attribution_agreement",
    ])
    _write_csv(analysis / "citation_summary_by_run.csv", citation_summary, list(citation_summary[0]))
    citation_models = _citation_model_summary(citation_detail, citation_summary)
    _write_csv(analysis / "citation_summary_by_model.csv", citation_models, list(citation_models[0]))

    snapshot = discover_cached_snapshot(manifest["encoder_model"])
    encoder = LocalTransformerEncoder(snapshot)
    redundancy_features, redundancy_pairs, redundancy_summary = redundancy_audit(
        inventory, best, encoder, tau=tau
    )
    _write_csv(analysis / "redundancy_features.csv", redundancy_features, list(redundancy_features[0]))
    _write_csv(analysis / "near_duplicate_pairs_tau_0_8.csv", redundancy_pairs, list(redundancy_pairs[0]))
    _write_csv(analysis / "redundancy_summary_by_run.csv", redundancy_summary, list(redundancy_summary[0]))
    redundancy_models = _redundancy_model_summary(redundancy_summary)
    _write_csv(analysis / "redundancy_summary_by_model.csv", redundancy_models, list(redundancy_models[0]))

    variability_runs, variability_summary = variability_audit(inventory)
    _write_csv(analysis / "variability_by_run.csv", variability_runs, list(variability_runs[0]))
    _write_csv(analysis / "variability_summary_by_model.csv", variability_summary, list(variability_summary[0]))

    parent_runs, parent_summary = parent_audit(inventory, best, tau=tau)
    _write_csv(analysis / "parent_match_by_run.csv", parent_runs, list(parent_runs[0]))
    _write_csv(analysis / "parent_match_summary_by_model.csv", parent_summary, list(parent_summary[0]))
    parent_sensitivity = parent_sensitivity_audit(
        inventory, best, reverse_best, taus=[float(value) for value in manifest["tau_sweep"]]
    )
    _write_csv(
        analysis / "parent_match_sensitivity.csv",
        parent_sensitivity,
        list(parent_sensitivity[0]),
    )
    print(f"quality audits written to {analysis}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
