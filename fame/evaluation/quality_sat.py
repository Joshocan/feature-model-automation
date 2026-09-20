from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional
import xml.etree.ElementTree as ET


@dataclass
class SATQuality:
    satisfiable: Optional[bool]
    dead_features: Optional[list[str]] = None
    core_features: Optional[list[str]] = None
    products_count: Optional[int] = None
    unsat_core_labels: Optional[list[str]] = None
    unsat_reasons: Optional[list[str]] = None


FEATURE_TAGS = {"feature", "and", "or", "alt"}
FORMULA_TAGS = {"var", "not", "imp", "eq", "conj", "disj"}


def _load_sat_backend():
    try:
        from pysat.formula import IDPool
        from pysat.solvers import Solver
    except Exception as e:
        raise ImportError(
            "SAT quality analysis requires python-sat. Install with: pip install python-sat[pblib,aiger]"
        ) from e
    return IDPool, Solver


class _Encoder:
    def __init__(self) -> None:
        IDPool, _ = _load_sat_backend()
        self.pool = IDPool()
        self.clauses: list[list[int]] = []
        self.clause_labels: list[str] = []
        self.features: list[str] = []

    def var(self, name: str) -> int:
        if name not in self.features:
            self.features.append(name)
        return int(self.pool.id(name))

    def aux(self, prefix: str = "aux") -> int:
        return int(self.pool.id(f"{prefix}_{self.pool.top + 1}"))

    def add(self, *lits: int, label: str) -> None:
        self.clauses.append([int(l) for l in lits])
        self.clause_labels.append(label)


def _named_children(node: ET.Element) -> list[ET.Element]:
    return [ch for ch in node if ch.tag in FEATURE_TAGS]


def _child_var(node: ET.Element, enc: _Encoder) -> tuple[Optional[int], str]:
    name = (node.attrib.get("name") or "").strip()
    if name:
        return enc.var(name), name
    return None, ""


def _encode_tree(node: ET.Element, enc: _Encoder, *, parent_lit: Optional[int], parent_name: Optional[str]) -> None:
    node_lit, node_name = _child_var(node, enc)
    current_parent_lit = node_lit if node_lit is not None else parent_lit
    current_parent_name = node_name if node_name else parent_name

    if parent_lit is not None and node_lit is not None:
        enc.add(-node_lit, parent_lit, label=f"Hierarchy: selecting '{node_name}' requires parent '{parent_name}'.")
        if node.attrib.get("mandatory", "").lower() == "true":
            enc.add(-parent_lit, node_lit, label=f"Mandatory relation: selecting '{parent_name}' requires child '{node_name}'.")

    children = _named_children(node)
    child_pairs = []
    for ch in children:
        lit, name = _child_var(ch, enc)
        if lit is not None:
            child_pairs.append((lit, name))

    if current_parent_lit is not None and child_pairs:
        child_lits = [lit for lit, _ in child_pairs]
        child_names = [name for _, name in child_pairs]
        if node.tag == "or":
            enc.add(
                -current_parent_lit,
                *child_lits,
                label=f"OR-group under '{current_parent_name}' requires at least one of {child_names}.",
            )
            for lit, name in child_pairs:
                enc.add(-lit, current_parent_lit, label=f"Hierarchy: selecting '{name}' requires parent '{current_parent_name}'.")
        elif node.tag == "alt":
            enc.add(
                -current_parent_lit,
                *child_lits,
                label=f"ALT-group under '{current_parent_name}' requires at least one of {child_names}.",
            )
            for lit, name in child_pairs:
                enc.add(-lit, current_parent_lit, label=f"Hierarchy: selecting '{name}' requires parent '{current_parent_name}'.")
            for i in range(len(child_pairs)):
                for j in range(i + 1, len(child_pairs)):
                    left_name = child_pairs[i][1]
                    right_name = child_pairs[j][1]
                    enc.add(
                        -child_pairs[i][0],
                        -child_pairs[j][0],
                        label=f"ALT-group under '{current_parent_name}' forbids selecting both '{left_name}' and '{right_name}'.",
                    )

    for ch in children:
        _encode_tree(ch, enc, parent_lit=current_parent_lit, parent_name=current_parent_name)


def _formula_lit(node: ET.Element, enc: _Encoder, *, label: str) -> int:
    tag = node.tag
    if tag == "var":
        name = (node.text or "").strip()
        if not name:
            raise ValueError("Empty <var> in constraint")
        return enc.var(name)
    if tag == "not":
        if len(node) != 1:
            raise ValueError("<not> must have exactly one child")
        child = _formula_lit(node[0], enc, label=label)
        aux = enc.aux("not")
        enc.add(-aux, -child, label=label)
        enc.add(aux, child, label=label)
        return aux
    if tag in {"conj", "disj"}:
        if len(node) < 1:
            raise ValueError(f"<{tag}> must have children")
        lits = [_formula_lit(ch, enc, label=label) for ch in node if ch.tag in FORMULA_TAGS]
        if not lits:
            raise ValueError(f"<{tag}> contains no formula children")
        aux = enc.aux(tag)
        if tag == "conj":
            for lit in lits:
                enc.add(-aux, lit, label=label)
            enc.add(aux, *[-lit for lit in lits], label=label)
        else:
            enc.add(-aux, *lits, label=label)
            for lit in lits:
                enc.add(aux, -lit, label=label)
        return aux
    if tag == "imp":
        if len(node) != 2:
            raise ValueError("<imp> must have exactly two children")
        left = _formula_lit(node[0], enc, label=label)
        right = _formula_lit(node[1], enc, label=label)
        aux = enc.aux("imp")
        enc.add(-aux, -left, right, label=label)
        enc.add(aux, left, label=label)
        enc.add(aux, -right, label=label)
        return aux
    if tag == "eq":
        if len(node) != 2:
            raise ValueError("<eq> must have exactly two children")
        left = _formula_lit(node[0], enc, label=label)
        right = _formula_lit(node[1], enc, label=label)
        aux = enc.aux("eq")
        enc.add(-aux, -left, right, label=label)
        enc.add(-aux, left, -right, label=label)
        enc.add(aux, left, right, label=label)
        enc.add(aux, -left, -right, label=label)
        return aux
    raise ValueError(f"Unsupported constraint tag: <{tag}>")


def _format_rule(rule: ET.Element, index: int) -> str:
    text = " ".join(part.strip() for part in rule.itertext() if part and part.strip())
    if not text:
        text = ET.tostring(rule, encoding="unicode")
    text = " ".join(text.split())
    return f"Constraint rule {index}: {text}"


def _encode_constraints(root: ET.Element, enc: _Encoder) -> None:
    constraints = root.find("constraints")
    if constraints is None:
        return
    for idx, rule in enumerate(constraints.findall("rule"), start=1):
        formula_nodes = [ch for ch in rule if ch.tag in FORMULA_TAGS]
        if not formula_nodes:
            continue
        if len(formula_nodes) != 1:
            raise ValueError("<rule> must contain exactly one formula root")
        label = _format_rule(rule, idx)
        enc.add(_formula_lit(formula_nodes[0], enc, label=label), label=label)


def _resolve_named_root(node: ET.Element) -> ET.Element:
    current = node
    while not (current.attrib.get("name") or "").strip():
        children = [ch for ch in current if ch.tag in FEATURE_TAGS]
        if len(children) != 1:
            break
        current = children[0]
    return current


def _build_cnf(xml_path: Path) -> tuple[_Encoder, list[int]]:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    struct = root.find("struct")
    if struct is None:
        raise ValueError("<struct> not found in feature model")

    top = [ch for ch in struct if ch.tag in FEATURE_TAGS]
    if len(top) != 1:
        raise ValueError("Expected exactly one root feature/group in <struct>")

    enc = _Encoder()
    root_node = _resolve_named_root(top[0])
    root_name = (root_node.attrib.get("name") or "").strip()
    if not root_name:
        raise ValueError("Root feature/group is missing a name")
    root_lit = enc.var(root_name)
    enc.add(root_lit, label=f"Root selection: '{root_name}' must always be selected.")

    _encode_tree(root_node, enc, parent_lit=None, parent_name=None)
    _encode_constraints(root, enc)
    feature_vars = [enc.var(name) for name in enc.features]
    return enc, feature_vars


def _solve(clauses: Iterable[list[int]], assumptions: Optional[list[int]] = None) -> bool:
    _, Solver = _load_sat_backend()
    with Solver(name="g3") as solver:
        for clause in clauses:
            solver.add_clause(list(clause))
        return bool(solver.solve(assumptions=assumptions or []))


def _unsat_core_labels(enc: _Encoder) -> list[str]:
    _, Solver = _load_sat_backend()
    selectors = [enc.aux("sel") for _ in enc.clauses]
    with Solver(name="g3") as solver:
        for selector, clause in zip(selectors, enc.clauses):
            solver.add_clause(list(clause) + [-selector])
        sat = solver.solve(assumptions=selectors)
        if sat:
            return []
        core = set(solver.get_core() or [])
    labels: list[str] = []
    seen: set[str] = set()
    for selector, label in zip(selectors, enc.clause_labels):
        if selector in core and label not in seen:
            seen.add(label)
            labels.append(label)
    return labels


def _count_models(clauses: list[list[int]], feature_vars: list[int]) -> int:
    _, Solver = _load_sat_backend()
    count = 0
    with Solver(name="g3") as solver:
        for clause in clauses:
            solver.add_clause(list(clause))
        while solver.solve():
            count += 1
            model = set(solver.get_model())
            blocking = [(-v if v in model else v) for v in feature_vars]
            solver.add_clause(blocking)
    return count


def analyze_sat_quality(xml_path: str | Path, *, compute_products: bool = False) -> SATQuality:
    xml_path = Path(xml_path).expanduser()
    enc, feature_vars = _build_cnf(xml_path)

    satisfiable = _solve(enc.clauses)
    if not satisfiable:
        reasons = _unsat_core_labels(enc)
        return SATQuality(
            satisfiable=False,
            dead_features=None,
            core_features=None,
            products_count=0 if compute_products else None,
            unsat_core_labels=reasons,
            unsat_reasons=reasons,
        )

    dead: list[str] = []
    core: list[str] = []
    for name in enc.features:
        lit = enc.var(name)
        if not _solve(enc.clauses, assumptions=[lit]):
            dead.append(name)
        if not _solve(enc.clauses, assumptions=[-lit]):
            core.append(name)

    products = None
    if compute_products:
        products = _count_models(enc.clauses, feature_vars)

    return SATQuality(
        satisfiable=True,
        dead_features=sorted(dead),
        core_features=sorted(core),
        products_count=products,
        unsat_core_labels=None,
        unsat_reasons=None,
    )
