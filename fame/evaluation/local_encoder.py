"""Offline sentence embedding using a pinned local Hugging Face snapshot."""
from __future__ import annotations

import hashlib
from pathlib import Path

WEIGHT_FILENAMES = ("model.safetensors", "pytorch_model.bin")
TOKENIZER_FILENAMES = ("tokenizer.json", "tokenizer_config.json", "vocab.txt",
                        "spiece.model", "sentencepiece.bpe.model")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def hash_snapshot_files(snapshot: Path) -> dict:
    """Return sha256 for every weight/tokenizer file present in ``snapshot``.

    Missing files are recorded as ``None`` rather than dropped so downstream
    verification can distinguish absence from mismatch.
    """
    result: dict[str, str | None] = {}
    for name in (*WEIGHT_FILENAMES, *TOKENIZER_FILENAMES, "config.json"):
        path = snapshot / name
        result[name] = _sha256_file(path) if path.is_file() else None
    return result


def verify_encoder_identity(snapshot: Path, *, expected_revision: str | None,
                             expected_versions: dict | None = None) -> dict:
    """Envelope-form check that the snapshot matches the pinned pilot instrument.

    Returns ``{value, status, reason, revision, weights_sha256, tokenizer_sha256,
    versions, expected_versions}``. ``status`` is one of ``ok``,
    ``missing_artifact``, ``revision_mismatch``, ``version_mismatch``,
    ``unsupported``, ``evaluator_error``.

    ``expected_revision`` is compared to the snapshot directory's basename
    (which is the git-revision hash under the Hugging Face cache convention).
    Version pinning is optional: pass ``{"transformers": "4.57.6", ...}`` to
    enforce; a mismatch flips status to ``version_mismatch`` but keeps
    everything else populated for diagnostics.
    """
    if not snapshot.is_dir():
        return dict(value=False, status="missing_artifact",
                    reason=f"snapshot dir not found: {snapshot}",
                    revision=None, weights_sha256=None, tokenizer_sha256=None,
                    versions=None, expected_versions=expected_versions)
    hashes = hash_snapshot_files(snapshot)
    weight_hashes = {k: v for k, v in hashes.items() if k in WEIGHT_FILENAMES and v is not None}
    tokenizer_hashes = {k: v for k, v in hashes.items() if k in TOKENIZER_FILENAMES and v is not None}
    revision = snapshot.name

    if not weight_hashes:
        return dict(value=False, status="missing_artifact",
                    reason=f"no weight file ({'/'.join(WEIGHT_FILENAMES)}) in {snapshot}",
                    revision=revision, weights_sha256=None,
                    tokenizer_sha256=tokenizer_hashes or None,
                    versions=None, expected_versions=expected_versions)

    versions: dict[str, str | None] = {}
    try:
        import importlib.metadata as _meta
        for pkg in ("transformers", "torch", "tokenizers", "safetensors"):
            try:
                versions[pkg] = _meta.version(pkg)
            except _meta.PackageNotFoundError:
                versions[pkg] = None
    except Exception as exc:
        return dict(value=False, status="evaluator_error",
                    reason=f"could not read package versions: {exc}",
                    revision=revision, weights_sha256=weight_hashes,
                    tokenizer_sha256=tokenizer_hashes,
                    versions=None, expected_versions=expected_versions)

    if expected_revision and revision != expected_revision:
        return dict(value=False, status="revision_mismatch",
                    reason=f"snapshot revision {revision!r} != expected {expected_revision!r}",
                    revision=revision, weights_sha256=weight_hashes,
                    tokenizer_sha256=tokenizer_hashes,
                    versions=versions, expected_versions=expected_versions)

    if expected_versions:
        bad = {k: (versions.get(k), v) for k, v in expected_versions.items()
               if versions.get(k) != v}
        if bad:
            return dict(value=False, status="version_mismatch",
                        reason=f"package versions differ from pilot: {bad}",
                        revision=revision, weights_sha256=weight_hashes,
                        tokenizer_sha256=tokenizer_hashes,
                        versions=versions, expected_versions=expected_versions)

    return dict(value=True, status="ok", reason="",
                revision=revision, weights_sha256=weight_hashes,
                tokenizer_sha256=tokenizer_hashes,
                versions=versions, expected_versions=expected_versions)


def discover_cached_snapshot(model_id: str) -> Path:
    """Resolve a model ID to the newest snapshot already present locally."""
    slug = "models--" + model_id.replace("/", "--")
    root = Path.home() / ".cache" / "huggingface" / "hub" / slug / "snapshots"
    snapshots = sorted(p for p in root.glob("*") if p.is_dir()) if root.exists() else []
    if snapshots:
        return snapshots[-1].resolve()
    raise FileNotFoundError(
        f"no cached snapshot for {model_id}; expected under {root}"
    )


class LocalTransformerEncoder:
    """Sentence-transformers mean pooling without its optional training stack."""

    def __init__(self, snapshot: Path, *, batch_size: int = 64):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True)
        self.model = AutoModel.from_pretrained(str(snapshot), local_files_only=True)
        self.model.eval()

    def encode(
        self,
        texts: list[str],
        *,
        normalize_embeddings: bool = True,
        convert_to_tensor: bool = False,
    ):
        torch = self.torch
        batches = []
        with torch.no_grad():
            for start in range(0, len(texts), self.batch_size):
                encoded = self.tokenizer(
                    texts[start:start + self.batch_size],
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                )
                token_embeddings = self.model(**encoded).last_hidden_state
                mask = encoded["attention_mask"].unsqueeze(-1).expand(token_embeddings.size()).float()
                pooled = (token_embeddings * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
                if normalize_embeddings:
                    pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
                batches.append(pooled.cpu())
        result = torch.cat(batches, dim=0)
        return result if convert_to_tensor else result.numpy()
