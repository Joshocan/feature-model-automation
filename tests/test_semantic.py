from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from fame.evaluation.local_encoder import (
    hash_snapshot_files, verify_encoder_identity,
)
from fame.evaluation.semantic import (
    SEMANTIC_METRICS, evaluate_semantic, maximum_threshold_matching, prf_from_similarity,
)


def test_independent_prf_keeps_precision_and_recall_separate() -> None:
    similarity = np.array([
        [0.9, 0.1, 0.0],
        [0.8, 0.2, 0.0],
        [0.1, 0.8, 0.0],
        [0.1, 0.7, 0.0],
    ])
    score = prf_from_similarity(similarity, threshold=0.4)
    assert score["semantic_precision"] == 1.0
    assert score["semantic_recall"] == pytest.approx(2 / 3)
    assert score["semantic_f1"] == pytest.approx(0.8)


def test_one_to_one_matching_penalises_many_to_one_duplicates() -> None:
    similarity = np.array([[0.9, 0.1], [0.8, 0.1], [0.7, 0.1]])
    registered = prf_from_similarity(similarity, threshold=0.4)
    sensitivity = prf_from_similarity(
        similarity, threshold=0.4, matching_policy="one_to_one"
    )
    assert registered["semantic_precision"] == 1.0
    assert sensitivity["semantic_precision"] == pytest.approx(1 / 3)
    assert sensitivity["semantic_recall"] == pytest.approx(1 / 2)


def test_maximum_matching_finds_augmenting_path() -> None:
    similarity = np.array([[0.9, 0.8], [0.85, 0.1]])
    assert set(maximum_threshold_matching(similarity, threshold=0.4)) == {(0, 1), (1, 0)}


# ─────────────────────────────────────────────────────────────────────────────
# — evaluate_semantic envelope tests
# ─────────────────────────────────────────────────────────────────────────────

FM_TEMPLATE = """<?xml version="1.0" encoding="UTF-8"?>
<featureModel>
  <struct>
    <and mandatory="true" name="Root">
      {children}
    </and>
  </struct>
</featureModel>
"""


def _write_fm(path: Path, feature_names: list[str]) -> Path:
    children = "\n      ".join(f'<feature name="{n}"/>' for n in feature_names)
    path.write_text(FM_TEMPLATE.format(children=children), encoding="utf-8")
    return path


class _FakeEncoder:
    """Deterministic name → one-hot embedding. Identical names → sim=1.0."""

    def __init__(self, vocab: list[str]):
        self.vocab = {name: i for i, name in enumerate(vocab)}
        self.dim = len(vocab)

    def encode(self, texts, *, normalize_embeddings=True, convert_to_tensor=False):
        mat = np.zeros((len(texts), self.dim), dtype=float)
        for row, text in enumerate(texts):
            idx = self.vocab.get(text)
            if idx is not None:
                mat[row, idx] = 1.0
        return mat


def test_rho_and_recall_partition_identity(tmp_path):
    fm = _write_fm(tmp_path / 'model.xml', ['A', 'B'])
    kwargs = dict(encoder=_FakeEncoder(['Root','A','B']), tau_primary=0.4,
                  attested=['A','B'], organising=['Root'], reach=['A','B'])
    metrics = evaluate_semantic(fm, fm, **kwargs)['metrics']
    ratio = metrics['rho_T_C']['value']
    assert ratio == 2 / 3
    assert metrics['rho_T_C']['numerator'] == 2
    assert metrics['recall_reach']['value'] == 1
    assert metrics['semantic_recall_total']['value'] == pytest.approx(
        ratio * metrics['recall_reach']['value'] +
        (1-ratio) * metrics['recall_organising']['value'])
    with pytest.raises(ValueError, match='rho must equal'):
        evaluate_semantic(fm, fm, rho=0.1, **kwargs)


def test_semantics_accepts_organising_ancestors_in_reach(tmp_path):
    fm = _write_fm(tmp_path / 'model.xml', ['A', 'B'])
    m = evaluate_semantic(fm, fm, encoder=_FakeEncoder(['Root','A','B']),
                          tau_primary=0.4, attested=['A'], organising=['Root','B'],
                          reach=['Root','A'], rho=2/3)['metrics']
    assert m['reach_size']['value'] == 2
    assert m['recall_reach']['value'] == 1
    assert m['rho_T_C']['value'] == 2/3


def test_evaluate_semantic_missing_gen_reports_missing_artifact(tmp_path: Path) -> None:
    ref = _write_fm(tmp_path / "ref.xml", ["A", "B"])
    outcome = evaluate_semantic(
        tmp_path / "does_not_exist.xml", ref,
        encoder=_FakeEncoder(["Root", "A", "B"]),
        tau_primary=0.4,
    )
    assert all(env["status"] == "missing_artifact" for env in outcome["metrics"].values())
    assert set(outcome["metrics"].keys()) == set(SEMANTIC_METRICS)


def test_evaluate_semantic_perfect_match_hits_all_targets(tmp_path: Path) -> None:
    gen = _write_fm(tmp_path / "gen.xml", ["A", "B"])
    ref = _write_fm(tmp_path / "ref.xml", ["A", "B"])
    encoder = _FakeEncoder(["Root", "A", "B"])
    outcome = evaluate_semantic(
        gen, ref, encoder=encoder, tau_primary=0.4,
        attested=("Root", "A"), organising=("B",), reach=("Root", "A"),
        rho=2 / 3,
    )
    m = outcome["metrics"]
    assert m["semantic_precision"]["value"] == 1.0
    assert m["semantic_recall_total"]["value"] == 1.0
    assert m["semantic_f1_total"]["value"] == pytest.approx(1.0)
    assert m["recall_reach"]["value"] == 1.0
    assert m["recall_attested"]["value"] == 1.0
    assert m["recall_organising"]["value"] == 1.0
    assert m["matched_reference_ids"]["value"] == ["A", "B", "Root"]
    assert m["duplicate_generated_hits"]["value"] == 0
    assert m["rho_T_C"]["value"] == pytest.approx(2 / 3)
    # Attribution agreement needs a cited generated node and reference
    # attribution; a perfect name match alone cannot establish it.
    assert m["attribution_agreement_rate"]["status"] == "not_applicable"
    assert all(env["status"] == "ok" for key, env in m.items()
               if key not in {"attribution_agreement_rate"})


def test_evaluate_semantic_partial_generation_recalls_correctly(tmp_path: Path) -> None:
    # Generated FM covers only "A", missing "B" and "C" from the reference.
    gen = _write_fm(tmp_path / "gen.xml", ["A"])
    ref = _write_fm(tmp_path / "ref.xml", ["A", "B", "C"])
    encoder = _FakeEncoder(["Root", "A", "B", "C"])
    outcome = evaluate_semantic(
        gen, ref, encoder=encoder, tau_primary=0.4,
        attested=("A", "B"), organising=("C",),
        reach=("A",),  # Only A is reachable in this corpus slice.
    )
    m = outcome["metrics"]
    # 2 of 4 reference features matched (Root + A).
    assert m["semantic_recall_total"]["value"] == pytest.approx(2 / 4)
    assert m["semantic_recall_total"]["numerator"] == 2
    assert m["semantic_recall_total"]["denominator"] == 4
    # Precision: 2 of 2 gen features hit (Root + A).
    assert m["semantic_precision"]["value"] == 1.0
    # Reach recall: A matched, denominator=1.
    assert m["recall_reach"]["value"] == 1.0
    # Organising recall: C not matched.
    assert m["recall_organising"]["value"] == 0.0


def test_evaluate_semantic_empty_ref_partition_marks_recall_not_applicable(tmp_path: Path) -> None:
    gen = _write_fm(tmp_path / "gen.xml", ["A"])
    ref = _write_fm(tmp_path / "ref.xml", ["A"])
    outcome = evaluate_semantic(
        gen, ref, encoder=_FakeEncoder(["Root", "A"]),
        tau_primary=0.4, attested=(), organising=(), reach=(),
    )
    m = outcome["metrics"]
    assert m["recall_attested"]["status"] == "not_applicable"
    assert m["recall_organising"]["status"] == "not_applicable"
    assert m["recall_reach"]["status"] == "not_applicable"
    # Precision and total recall stay real numbers.
    assert m["semantic_precision"]["status"] == "ok"
    assert m["semantic_recall_total"]["status"] == "ok"


def test_evaluate_semantic_empty_generated_marks_ineligible(tmp_path: Path) -> None:
    # An FM with only the root and no feature children.
    (tmp_path / "gen.xml").write_text(
        '<?xml version="1.0"?><featureModel><struct/></featureModel>',
        encoding="utf-8",
    )
    ref = _write_fm(tmp_path / "ref.xml", ["A"])
    outcome = evaluate_semantic(
        tmp_path / "gen.xml", ref, encoder=_FakeEncoder(["Root", "A"]),
        tau_primary=0.4,
    )
    m = outcome["metrics"]
    assert m["semantic_precision"]["status"] == "ineligible"
    assert m["semantic_recall_total"]["status"] == "ineligible"
    assert m["n_generated"]["value"] == 0


def test_evaluate_semantic_tau_sweep_returns_thresholded_view(tmp_path: Path) -> None:
    gen = _write_fm(tmp_path / "gen.xml", ["A", "B"])
    ref = _write_fm(tmp_path / "ref.xml", ["A", "B"])
    encoder = _FakeEncoder(["Root", "A", "B"])
    outcome = evaluate_semantic(
        gen, ref, encoder=encoder, tau_primary=0.4,
        tau_sweep=(0.3, 0.4, 0.5, 0.6),
    )
    sweep = outcome["tau_sweep_view"]
    assert set(sweep) == {"precision", "recall_total", "f1_total"}
    # One-hot fake encoder gives sim=1.0 on match, 0.0 elsewhere — every
    # τ ≤ 1.0 should give the same recall/precision.
    for tau in (0.3, 0.4, 0.5, 0.6):
        assert sweep["precision"][tau] == 1.0
        assert sweep["recall_total"][tau] == 1.0


def test_evaluate_semantic_raw_pairs_are_unrounded_and_complete(tmp_path: Path) -> None:
    gen = _write_fm(tmp_path / "gen.xml", ["A"])
    ref = _write_fm(tmp_path / "ref.xml", ["A", "B"])
    outcome = evaluate_semantic(
        gen, ref, encoder=_FakeEncoder(["Root", "A", "B"]),
        tau_primary=0.4,
    )
    # 2 generated (Root + A) × 3 reference (Root + A + B) = 6 raw pairs.
    assert len(outcome["pairs"]) == 6
    # No thresholding applied — sub-τ pairs are preserved.
    assert any(p["similarity"] == 0.0 for p in outcome["pairs"])


# ─────────────────────────────────────────────────────────────────────────────
# — encoder identity (D03)
# ─────────────────────────────────────────────────────────────────────────────

def _fake_snapshot(tmp_path: Path, revision: str, weight_bytes: bytes = b"weights") -> Path:
    snap = tmp_path / "snapshots" / revision
    snap.mkdir(parents=True)
    (snap / "model.safetensors").write_bytes(weight_bytes)
    (snap / "tokenizer.json").write_bytes(b"tokenizer")
    (snap / "config.json").write_bytes(b"{}")
    return snap


def test_verify_encoder_identity_ok_when_revision_matches(tmp_path: Path) -> None:
    snap = _fake_snapshot(tmp_path, "abc123")
    result = verify_encoder_identity(snap, expected_revision="abc123")
    assert result["status"] == "ok"
    assert result["value"] is True
    assert result["revision"] == "abc123"
    expected = hashlib.sha256(b"weights").hexdigest()
    assert result["weights_sha256"]["model.safetensors"] == expected


def test_verify_encoder_identity_flags_revision_mismatch(tmp_path: Path) -> None:
    snap = _fake_snapshot(tmp_path, "abc123")
    result = verify_encoder_identity(snap, expected_revision="def456")
    assert result["status"] == "revision_mismatch"
    assert result["value"] is False
    assert "def456" in result["reason"]


def test_verify_encoder_identity_reports_missing_snapshot(tmp_path: Path) -> None:
    result = verify_encoder_identity(tmp_path / "nope",
                                      expected_revision="abc123")
    assert result["status"] == "missing_artifact"


def test_verify_encoder_identity_missing_weights_marks_missing_artifact(tmp_path: Path) -> None:
    snap = tmp_path / "snapshots" / "abc123"
    snap.mkdir(parents=True)
    (snap / "tokenizer.json").write_bytes(b"tokenizer")
    result = verify_encoder_identity(snap, expected_revision="abc123")
    assert result["status"] == "missing_artifact"
    assert "weight file" in result["reason"]


def test_hash_snapshot_files_reports_missing_as_none(tmp_path: Path) -> None:
    snap = _fake_snapshot(tmp_path, "abc123")
    result = hash_snapshot_files(snap)
    assert result["model.safetensors"] is not None
    assert result["pytorch_model.bin"] is None
