# — structural and logical diagnostics

This implements offline evaluation of completed final outputs while retaining all
planned runs from the run inventory. Failed runs receive explicit ineligible metric rows,
never substituted checkpoints or invented semantic/conformance scores.

```bash
./.venv/bin/python scripts/evaluate_structure.py \
  --inventory results/ifs-2027/analysis/inventory-current-v1 \
  --output results/ifs-2027/analysis/structure-current-v1
```

Choose a fresh output directory. The driver verifies the current evaluation
contract hash, final-artifact hashes against the inventory, and schema hashes
against the frozen run configurations. It writes
long-form `metrics.csv`, `metrics.json`, and `summary.json` only under analysis.
It makes no model calls. Every measurement has value, status and reason, with
numerator/denominator where applicable. Unknown is distinct from false or zero.

## Implemented

- Secure XML parsing without external entity resolution; DTD-bearing inputs are
  explicitly unsupported. XML declaration/envelope compliance is separate.
- Frozen-XSD validation (including key/keyref), exact configured root (W1), unique
  names (W2), concrete formula arities (W3), declared constraint references (W4).
- Explicit leaf/group and parent-tree checks, identifier syntax.
- Canonical requires (`imp(var,var)`), excludes (`disj(not(var),not(var))`) and
  other supported formula counts; unsupported rules are not silently discarded.
- Feature-only node counts, root-zero depth, internal branching, group proportions,
  explicit non-root mandatory ratio and inherited structural degeneracy proxy.
- SAT via python-sat's `g3` solver, plus non-root dead-feature lists/ratios for SAT
  models. UNSAT dead-feature rates are undefined. Products are not enumerated.
- Dependency versions and source/inventory/schema hashes in the result manifest.

SAT and shape evaluation use the concrete schema/tree/constraint gate, independently
of the unresolved strict-admissibility definition. Models rejected by that gate are
not scored by SAT. This is a diagnostic population, not a claim of full conformance.
`mandatory="1"` is accepted consistently with the XSD boolean type. Internal nodes
are not automatically classified as abstract.

## Deliberate limitations

W5 uses the documented concrete FeatureIDE mapping: canonical plain relations
contain one formula tree and no additional or mixed formula material. The W5
check overlaps W3/XSD and is not an independent quality signal. A failing
required structural check makes structural conformance false; all passing
checks make it true. Actual FeatureIDE application parsing is still unverified.

Actual FeatureIDE parser acceptance also remains `unsupported`: lxml XSD validation
is not represented as a FeatureIDE invocation. No strict-admissibility headline or
publication-ready flag is produced. D01's primary semantic population is resolved;
D02's parser and strict sensitivity gate remain open. Description/trace correctness
and textual support belong to the later
provenance phase, not to this phase's XML-envelope check.

The earlier `derive_metrics.py` adapter now exports real structural results rather
than nonexistent validator attributes, but its legacy completed-only collection is
not the authoritative campaign workflow. Use this inventory-led entry point.

Evaluator errors cause nonzero exit after diagnostic output. Expected ineligible
and explicitly unsupported metrics are reported without fabricating values.

## Verification

```bash
./.venv/bin/python -m pytest tests/test_structural_evaluation.py \
  tests/test_aggregate.py tests/test_inventory.py tests/test_evaluation_spec.py -q
```

Fixtures cover malformed/missing XML, invalid empty groups, duplicate/wrong-root
features, depth, SAT/UNSAT/dead features, unknown operators and references, canonical
constraint counts, auxiliary-variable collisions, missing schema and empty
denominators. Original campaign artifacts and generation settings are unchanged.
