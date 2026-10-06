"""Structural and SAT diagnostics with explicit metric states; no generation or XML repair."""
from __future__ import annotations

import re
from pathlib import Path

from lxml import etree

FEATURES = {"and", "or", "alt", "feature"}
ARITIES = {"var": 0, "not": 1, "imp": 2, "eq": 2, "conj": 2, "disj": 2}
NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")


def metric(value=None, status="ok", reason="", **counts):
    return dict(value=value, status=status, reason=reason, **counts)


def formula_counts(root):
    """Classify only canonical plain relations; no logical-equivalence guessing."""
    counts = dict(n_requires=0, n_excludes=0, n_other_constraints=0)
    for container in root.findall("constraints"):
        if any(c.tag != "rule" for c in container):
            raise ValueError("Unsupported element in constraints")
    for rule in root.findall("constraints/rule"):
        if len(rule) != 1:
            raise ValueError("Each rule must have exactly one formula root")
        expr = rule[0]
        for node in expr.iter():
            if node.tag not in ARITIES or len(node) != ARITIES[node.tag]:
                raise ValueError(f"Unsupported operator or arity: {node.tag}")
            if node.tag == "var" and not (node.text or "").strip():
                raise ValueError("Empty constraint variable")
            if node.tag != "var" and (node.text or "").strip():
                raise ValueError("Non-variable formula text")
            if (node.tail or "").strip():
                raise ValueError("Text between formula nodes")
        if (rule.text or "").strip():
            raise ValueError("Text directly in rule")
        if expr.tag == "imp" and all(c.tag == "var" for c in expr):
            counts["n_requires"] += 1
        elif expr.tag == "disj" and all(c.tag == "not" and c[0].tag == "var" for c in expr):
            counts["n_excludes"] += 1
        else:
            counts["n_other_constraints"] += 1
    return counts


def feature_shape(top):
    """Feature-edge depth and explicit variability attributes; root depth is zero."""
    nodes, depths, branches = [], [], []

    def walk(node, depth):
        nodes.append(node)
        depths.append(depth)
        children = [c for c in node if c.tag in FEATURES]
        if children:
            branches.append(len(children))
        for child in children:
            walk(child, depth + 1)

    walk(top, 0)
    groups = [n for n in nodes if any(c.tag in FEATURES for c in n)]
    counts = {tag: sum(n.tag == tag for n in groups) for tag in ("and", "or", "alt")}
    mandatory = sum(n.get("mandatory") in ("true", "1") for n in nodes[1:])
    result = {"n_features": metric(len(nodes)), "max_depth": metric(max(depths)),
        "avg_branching": metric(sum(branches) / len(branches), numerator=sum(branches), denominator=len(branches)) if branches else metric(status="not_applicable", reason="No internal feature nodes"),
        "mandatory_ratio": metric(mandatory / (len(nodes)-1), numerator=mandatory, denominator=len(nodes)-1) if len(nodes)>1 else metric(status="not_applicable", reason="No non-root features"),
        "degenerate": metric(counts["or"] == counts["alt"] == mandatory == 0)}
    for tag, count in counts.items():
        result[f"n_{tag}"] = metric(count)
        result[f"{tag}_prop"] = metric(count/len(groups), numerator=count, denominator=len(groups)) if groups else metric(status="not_applicable", reason="No internal feature nodes")
    return result


METRICS = ("parseable", "xsd_valid", "W1", "W2", "W3", "W4", "W5",
    "tree_invariants", "identifier_syntax", "xml_envelope", "structural_conformance",
    "featureide_parse", "satisfiable", "dead_features", "dead_feature_ratio",
    "n_requires", "n_excludes", "n_other_constraints", "n_features", "max_depth",
    "avg_branching", "mandatory_ratio", "degenerate", "n_and", "n_or", "n_alt",
    "and_prop", "or_prop", "alt_prop")


def evaluate_structure(path: Path, xsd: Path, *, expected_root: str | None,
                       run_sat: bool = True) -> dict:
    result = {key: metric(status="ineligible", reason="Requires parseable final XML") for key in METRICS}
    result["W5"] = metric(status="ineligible", reason="Requires parseable constraints")
    result["featureide_parse"] = metric(status="unsupported", reason="No pinned FeatureIDE parser integrated; XSD is not a substitute")
    if not path.is_file():
        for key in METRICS:
            if key != "featureide_parse":
                result[key] = metric(status="missing_artifact", reason="Final XML absent; no checkpoint substitution")
        return result
    try:
        raw = path.read_bytes()
        parser = etree.XMLParser(resolve_entities=False, no_network=True, load_dtd=False)
        root = etree.fromstring(raw, parser)
        if root.getroottree().docinfo.doctype:
            result["parseable"] = metric(status="unsupported", reason="DTD/entity-bearing XML is not evaluated")
            return result
    except etree.XMLSyntaxError as exc:
        result["parseable"] = metric(False, reason=str(exc))
        return result
    except OSError as exc:
        result["parseable"] = metric(status="evaluator_error", reason=str(exc))
        return result
    result["parseable"] = metric(True)
    result["xml_envelope"] = metric(raw.lstrip().startswith(b"<?xml") and raw.rstrip().endswith(b"</featureModel>"))
    try:
        schema = etree.XMLSchema(etree.parse(str(xsd), parser))
        valid = schema.validate(root)
        result["xsd_valid"] = metric(valid, reason="; ".join(str(e) for e in schema.error_log))
    except (OSError, etree.LxmlError) as exc:
        result["xsd_valid"] = metric(status="evaluator_error", reason=str(exc))
    structs = root.findall("struct")
    struct = structs[0] if len(structs) == 1 else None
    tops = [c for c in struct if c.tag in FEATURES] if struct is not None else []
    nodes = [n for n in struct.iter() if n.tag in FEATURES] if struct is not None else []
    names = [n.get("name") for n in nodes]
    result["W1"] = (metric(root.tag == "featureModel" and len(tops) == 1 and tops[0].get("name") == expected_root)
        if expected_root is not None else metric(status="unsupported", reason="Expected root not supplied"))
    result["W2"] = metric(bool(nodes) and all(names) and len(names) == len(set(names)))
    result["identifier_syntax"] = metric(bool(nodes) and all(NAME.fullmatch(n or "") for n in names))
    result["tree_invariants"] = metric(root.tag == "featureModel" and len(tops) == 1 and bool(nodes) and all(
        n.getparent() is struct or n.getparent().tag in FEATURES for n in nodes) and all(
        (not any(c.tag in FEATURES for c in n)) if n.tag == "feature" else any(c.tag in FEATURES for c in n)
        for n in nodes))
    try:
        counts = formula_counts(root)
        result["W3"] = metric(True, reason="Concrete formula arities; canonical plain relations have two operands")
        result["W5"] = metric(True, reason="FeatureIDE translation: each rule has one formula tree; plain requires/excludes use their canonical trees without an additional formula")
        for key, value in counts.items():
            result[key] = metric(value)
    except ValueError as exc:
        result["W3"] = metric(False, reason=str(exc))
        result["W5"] = metric(False, reason=str(exc))
    variables = [(v.text or "").strip() for v in root.findall("constraints//var")]
    result["W4"] = metric(all(v in names for v in variables), reason="" if all(v in names for v in variables) else "Undeclared constraint variable")
    required = [result[k] for k in ("xsd_valid", "W1", "W2", "W3", "W4", "W5", "tree_invariants")]
    result["structural_conformance"] = (metric(False, reason="At least one required structural check failed")
        if any(m["value"] is False for m in required)
        else metric(True) if all(m["value"] is True for m in required)
        else metric(status="unsupported", reason="At least one required structural check is unavailable"))
    concrete_ok = all(result[k]["value"] is True for k in ("xsd_valid", "W2", "W3", "W4", "tree_invariants"))
    if concrete_ok:
        result.update(feature_shape(tops[0]))
    else:
        for key in ("satisfiable", "dead_features", "dead_feature_ratio"):
            result[key] = metric(status="ineligible", reason="Concrete schema/tree/constraint checks did not pass")
        return result
    if not run_sat:
        for key in ("satisfiable", "dead_features", "dead_feature_ratio"):
            result[key] = metric(status="not_applicable", reason="SAT explicitly disabled")
        return result
    try:
        from fame.evaluation.quality_sat import analyze_sat_quality
        sat = analyze_sat_quality(path)
        result["satisfiable"] = metric(sat.satisfiable)
        if sat.satisfiable:
            dead = [n for n in sat.dead_features if n != tops[0].get("name")]
            result["dead_features"] = metric(dead)
            denominator = len(nodes)-1
            result["dead_feature_ratio"] = metric(len(dead)/denominator, numerator=len(dead), denominator=denominator) if denominator else metric(status="not_applicable", reason="No non-root features")
        else:
            result["dead_features"] = metric(status="not_applicable", reason="Unsatisfiable model; no dead-feature rate")
            result["dead_feature_ratio"] = metric(status="not_applicable", reason="Unsatisfiable model")
    except ImportError as exc:
        for key in ("satisfiable", "dead_features", "dead_feature_ratio"):
            result[key] = metric(status="unsupported", reason=str(exc))
    except Exception as exc:
        for key in ("satisfiable", "dead_features", "dead_feature_ratio"):
            result[key] = metric(status="evaluator_error", reason=f"{type(exc).__name__}: {exc}")
    return result
