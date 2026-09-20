#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import requests

from fame.utils.runtime import workspace
from fame.ingestion.pipeline import ingest_and_prepare
from fame.vectorization.pipeline import index_all_chunks
from fame.services.chroma_service import start_chroma
from fame.services.ollama_service import _ollama_bin, start_ollama, verify_running, pull_models
from fame.nonrag.cli_utils import normalize_dataset_name


DATASET_PROCESSED_SUBDIR = {
    "federation": "federation",
    "repair": "repair",
}


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Run dataset-aware preprocessing for RAG")
    ap.add_argument("--dataset", choices=("federation", "repair", "fed", "federated", "model_repair"), default="", help="Dataset preset for raw/chunk directories and collection prefix")
    ap.add_argument("--raw-dir", default="", help="Explicit input directory containing source PDFs")
    ap.add_argument("--out-dir", default="", help="Explicit output directory for *.chunks.json")
    ap.add_argument("--collection-prefix", default="", help="Explicit Chroma collection prefix")
    ap.add_argument("--batch-size", type=int, default=int(os.getenv("VEC_BATCH_SIZE", "24")), help="Vectorization batch size")
    return ap


def _resolve_paths(paths, dataset: str | None, raw_dir: str | None, out_dir: str | None) -> tuple[Path, Path, str | None]:
    dataset_name = normalize_dataset_name(dataset) if dataset else None
    if raw_dir:
        resolved_raw = Path(raw_dir).expanduser().resolve()
    elif dataset_name:
        resolved_raw = (paths.base_dir / "data" / "raw" / dataset_name).resolve()
    else:
        resolved_raw = paths.raw_data

    if out_dir:
        resolved_out = Path(out_dir).expanduser().resolve()
    elif dataset_name:
        subdir = DATASET_PROCESSED_SUBDIR.get(dataset_name, dataset_name)
        resolved_out = (paths.base_dir / "data" / "processed" / subdir / "chunks").resolve()
    else:
        resolved_out = (paths.processed_data / "chunks").resolve()

    return resolved_raw, resolved_out, dataset_name


def chroma_healthy(host: str, port: int) -> bool:
    endpoints = [
        f"http://{host}:{port}/api/v1/heartbeat",
        f"http://{host}:{port}/api/v2/heartbeat",
        f"http://{host}:{port}/heartbeat",
        f"http://{host}:{port}/api/v1/version",
        f"http://{host}:{port}/",
    ]
    for url in endpoints:
        try:
            r = requests.get(url, timeout=2)
            if 200 <= r.status_code < 400:
                return True
        except Exception:
            pass
    return False


def ensure_chroma_running(chroma_path: Path, host: str, port: int, timeout_s: int) -> int:
    if chroma_healthy(host, port):
        print(f"SUCCESS: Chroma already running at http://{host}:{port}")
        return 0

    print("🔧 Chroma not reachable — starting via chroma_service...")
    pid = start_chroma(
        path=str(chroma_path),
        host=host,
        port=port,
        timeout_s=timeout_s,
        force_restart=False,
    )
    return pid


def ensure_ollama_running(log_dir: Path, timeout_s: int) -> int:
    try:
        verify_running()
        print("SUCCESS: Ollama already running.")
        return 0
    except Exception:
        print("🔧 Ollama not reachable — attempting to start locally...")
        pid = start_ollama(log_dir=str(log_dir), timeout_s=timeout_s, force_restart=False)
        return pid


def ensure_ollama_embed_model(model: str, host: str) -> None:
    host_lower = host.lower()
    is_local = any(h in host_lower for h in ["127.0.0.1", "localhost"])

    if not is_local:
        print(
            "WARN:  Remote Ollama host detected; skipping auto-pull of embedding model.\n"
            f"    Ensure '{model}' is available on {host} or set OLLAMA_EMBED_MODEL to a model present there."
        )
        return

    bin_path = _ollama_bin()
    if not bin_path:
        print(
            "WARN:  Cannot auto-pull embedding model because 'ollama' binary is not found.\n"
            f"    Please pull manually (local host):\n"
            f"      ollama pull {model}\n"
        )
        return

    print(f"⬇️  Ensuring Ollama embedding model is available: {model}")
    pull_models([model])


def main() -> None:
    args = _build_parser().parse_args()
    ws = workspace("vectorize", base_dir=os.getenv("FAME_BASE_DIR"))
    paths = ws.paths
    raw_dir, chunks_out_dir, dataset_name = _resolve_paths(
        paths,
        dataset=args.dataset or None,
        raw_dir=args.raw_dir or None,
        out_dir=args.out_dir or None,
    )

    chroma_host = os.getenv("CHROMA_HOST", "127.0.0.1").strip()
    chroma_port = int(os.getenv("CHROMA_PORT", "8000"))
    chroma_timeout = int(os.getenv("CHROMA_STARTUP_TIMEOUT", "120"))
    chroma_path = Path(os.getenv("CHROMA_PATH", str(paths.vector_db))).expanduser().resolve()

    ollama_timeout = int(os.getenv("OLLAMA_STARTUP_TIMEOUT", "60"))
    ollama_log_dir = Path(os.getenv("OLLAMA_LOG_DIR", str(paths.base_dir / "data" / "ollama"))).expanduser().resolve()
    embed_model = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text").strip()
    collection_prefix = args.collection_prefix or (f"{dataset_name}_" if dataset_name else os.getenv("COLLECTION_PREFIX", ""))

    print("\n==================== PREPROCESSING FOR RAG ====================")
    print(f"BASE_DIR           : {paths.base_dir}")
    print(f"DATASET            : {dataset_name or '(default)'}")
    print(f"RAW_DIR            : {raw_dir}")
    print(f"CHUNKS_OUT_DIR     : {chunks_out_dir}")
    print(f"COLLECTION_PREFIX  : {collection_prefix}")
    print(f"CHROMA_MODE        : {os.getenv('CHROMA_MODE', 'persistent')}")
    print(f"CHROMA_PATH        : {chroma_path}")
    print(f"CHROMA_HOST:PORT   : {chroma_host}:{chroma_port}")
    print(f"OLLAMA_HOST        : {os.getenv('OLLAMA_HOST', 'http://127.0.0.1:11434')}")
    print(f"OLLAMA_EMBED_MODEL : {embed_model}")
    print("===============================================================\n")

    chroma_pid = ensure_chroma_running(chroma_path, chroma_host, chroma_port, chroma_timeout)
    ollama_pid = ensure_ollama_running(ollama_log_dir, ollama_timeout)
    ensure_ollama_embed_model(embed_model, os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434"))

    print("\n🧩 Running ingestion pipeline...")
    ingest_res = ingest_and_prepare(raw_dir=raw_dir, out_dir=chunks_out_dir)
    print(f"SUCCESS: Ingestion done: processed={len(ingest_res['processed'])}, skipped={len(ingest_res['skipped'])}")

    if not ingest_res["processed"]:
        print(f"WARN:  No chunks.json produced. Ensure there is at least one supported file in {raw_dir}.")
        sys.exit(2)

    print("\n🧠 Running vectorization (indexing) pipeline...")
    vec_res = index_all_chunks(
        chunks_dir=chunks_out_dir,
        batch_size=args.batch_size,
        collection_prefix=collection_prefix,
    )
    print("SUCCESS: Vectorization done.")
    print(vec_res)

    print("\nSUCCESS: PREPROCESSING FOR RAG COMPLETE.")
    if chroma_pid:
        print(f"Chroma PID started by this run: {chroma_pid}")
    if ollama_pid:
        print(f"Ollama PID started by this run: {ollama_pid}")


if __name__ == "__main__":
    main()
