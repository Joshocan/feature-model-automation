from __future__ import annotations

import os
import signal
import subprocess
import time
from pathlib import Path
from typing import Optional, List

from fame.utils.runtime import ensure_base_only  # minimal workspace bootstrap


# ----------------------------
# Helpers
# ----------------------------

def _which(cmd: str) -> Optional[str]:
    try:
        out = subprocess.run(["bash", "-lc", f"command -v {cmd}"], capture_output=True, text=True)
        p = out.stdout.strip()
        return p if p else None
    except Exception:
        return None


def _ollama_bin() -> str:
    """
    Resolution order:
      1) OLLAMA_BIN env var
      2) PATH (command -v)
      3) common Homebrew paths (Apple Silicon + Intel)
    """
    env = os.getenv("OLLAMA_BIN", "").strip()
    if env:
        p = Path(env).expanduser()
        if p.exists():
            return str(p)

    p = _which("ollama")
    if p:
        return p

    # common brew locations
    for candidate in ("/opt/homebrew/bin/ollama", "/usr/local/bin/ollama"):
        if Path(candidate).exists():
            return candidate

    return ""


def _run(args: List[str], check: bool = False) -> subprocess.CompletedProcess:
    return subprocess.run(args, check=check, capture_output=True, text=True)


def _base_url() -> str:
    # allow remote mode even if you run locally
    return os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")


def _is_ollama_running() -> bool:
    """
    Check if ollama server responds. Ollama exposes /api/tags when running.
    """
    try:
        r = _run(["curl", "-fsS", f"{_base_url()}/api/tags"], check=False)
        return r.returncode == 0
    except Exception:
        return False


def _pkill_ollama_serve() -> None:
    """
    Kill any running 'ollama serve' processes (best-effort, macOS/Linux).
    """
    if _which("pkill"):
        _run(["pkill", "-f", "ollama serve"], check=False)
        time.sleep(1)
        _run(["pkill", "-f", "ollama serve"], check=False)
    else:
        print("WARN:  pkill not found; cannot auto-stop existing ollama serve processes.")


def stop_existing(pid_file: Path) -> None:
    """
    Stop ollama serve via pidfile, then pkill safety net.
    """
    if pid_file.exists():
        try:
            pid = int(pid_file.read_text().strip())
            print(f"WARN:  Stopping previous Ollama PID from file: {pid}")
            try:
                os.kill(pid, signal.SIGTERM)
                time.sleep(1)
                try:
                    os.kill(pid, 0)
                    os.kill(pid, signal.SIGKILL)
                except OSError:
                    pass
            except OSError:
                pass
        except Exception:
            pass
        finally:
            pid_file.unlink(missing_ok=True)

    _pkill_ollama_serve()


# ----------------------------
# Local mode: manage server
# ----------------------------

def start_ollama(
    log_dir: str,
    timeout_s: int = 60,
    force_restart: bool = True,
) -> int:
    """
    Start ollama server and wait until it responds.
    Returns PID of background process.

    NOTE: If you are using `brew services start ollama`, prefer REMOTE mode and
    do not call this. (Set OLLAMA_MODE=remote)
    """
    ollama_exe = _ollama_bin()
    if not ollama_exe:
        raise RuntimeError(
            "ERROR: Ollama executable not found.\n"
            "You may have only the Python 'ollama' package installed.\n\n"
            "Fix (Homebrew):\n"
            "  brew install ollama\n"
            "  brew services start ollama\n\n"
            "Then verify:\n"
            "  which ollama\n"
            "  ollama --version\n\n"
            "Or set explicitly:\n"
            "  export OLLAMA_BIN=/opt/homebrew/bin/ollama"
        )

    log_path = Path(log_dir).expanduser().resolve()
    log_path.mkdir(parents=True, exist_ok=True)

    log_file = log_path / "ollama_serve.log"
    pid_file = log_path / "ollama_serve.pid"

    if force_restart:
        print("🧹 Cleaning up any existing Ollama server processes...")
        stop_existing(pid_file)
        time.sleep(1)
    else:
        if _is_ollama_running():
            raise RuntimeError(f"Ollama appears to already be running at {_base_url()}")

    log_file.write_text("", encoding="utf-8")

    print(f"Starting Ollama server: {ollama_exe} serve")
    print(f"Logs: {log_file}")

    with log_file.open("a", encoding="utf-8") as lf:
        proc = subprocess.Popen([ollama_exe, "serve"], stdout=lf, stderr=lf)

    pid_file.write_text(str(proc.pid), encoding="utf-8")

    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if proc.poll() is not None:
            tail = _run(["tail", "-n", "200", str(log_file)], check=False).stdout
            raise RuntimeError(
                f"Ollama exited early with code {proc.returncode}\n"
                f"Log file: {log_file}\n"
                f"--- Log tail ---\n{tail}\n"
            )

        if _is_ollama_running():
            print(f"SUCCESS: Ollama is up at {_base_url()}")
            return proc.pid

        time.sleep(1)

    tail = _run(["tail", "-n", "200", str(log_file)], check=False).stdout
    raise RuntimeError(
        f"Timed out waiting for Ollama readiness after {timeout_s}s\n"
        f"Log file: {log_file}\n"
        f"--- Log tail ---\n{tail}\n"
    )


# ----------------------------
# Remote mode: server already running
# ----------------------------

def verify_running() -> None:
    if not _is_ollama_running():
        raise RuntimeError(
            f"ERROR: Ollama is not reachable at {_base_url()}\n"
            "If using Homebrew service:\n"
            "  brew services start ollama\n"
            "Then check:\n"
            "  curl -s http://127.0.0.1:11434/api/tags\n"
            "Or set:\n"
            "  export OLLAMA_HOST=http://<host>:11434"
        )
    print(f"SUCCESS: Ollama reachable at {_base_url()}")


# ----------------------------
# Model management (uses local CLI)
# ----------------------------

def pull_models(models: List[str]) -> None:
    """
    Pull each model via local CLI 'ollama pull <model>'.
    (Pulling requires local binary even in remote mode.)
    """
    ollama_exe = _ollama_bin()
    if not ollama_exe:
        raise RuntimeError("Cannot pull models: local Ollama executable not found (OLLAMA_BIN / PATH).")

    for m in models:
        m = m.strip()
        if not m:
            continue
        print(f"Pulling model: {m}")
        r = _run([ollama_exe, "pull", m], check=False)
        if r.returncode != 0:
            raise RuntimeError(f"Failed to pull model '{m}'.\nSTDERR:\n{r.stderr}\nSTDOUT:\n{r.stdout}")
        print(f"SUCCESS: Model '{m}' pulled (or already present).")


def list_models() -> str:
    ollama_exe = _ollama_bin()
    if not ollama_exe:
        raise RuntimeError("Cannot list models: local Ollama executable not found (OLLAMA_BIN / PATH).")

    r = _run([ollama_exe, "list"], check=False)
    if r.returncode != 0:
        raise RuntimeError(f"Failed to list models.\nSTDERR:\n{r.stderr}\nSTDOUT:\n{r.stdout}")
    return r.stdout


def verify_models_available(required: List[str]) -> None:
    out = list_models()
    print("--- Ollama List Output ---")
    print(out)
    print("--------------------------")

    missing = [m for m in required if m not in out]
    if missing:
        raise RuntimeError(f"ERROR: Missing models in 'ollama list': {missing}")
    print(f"SUCCESS: All required models are available: {required}")


# ----------------------------
# High-level setup
# ----------------------------

def setup_ollama(
    embedding_model: str,
    llm_model: str,
    log_dir: str,
    timeout_s: int = 60,
    force_restart: bool = True,
    mode: str = "remote",
) -> int:
    """
    Setup in either:
      - mode='remote': verify server reachable; optionally pull+verify
      - mode='local' : start server, pull, verify

    Returns PID in local mode; returns 0 in remote mode.
    """
    mode = mode.lower().strip()
    if mode not in ("local", "remote"):
        raise ValueError("mode must be 'local' or 'remote'")

    if mode == "local":
        pid = start_ollama(log_dir=log_dir, timeout_s=timeout_s, force_restart=force_restart)
    else:
        verify_running()
        pid = 0

    # You can choose to skip pulling in remote mode if you want
    pull_models([embedding_model, llm_model])
    verify_models_available([embedding_model, llm_model])

    print(f"SUCCESS: Ollama ready. Models: '{embedding_model}', '{llm_model}'")
    return pid


def stop_ollama(log_dir: str) -> None:
    log_path = Path(log_dir).expanduser().resolve()
    pid_file = log_path / "ollama_serve.pid"
    stop_existing(pid_file)
    print("SUCCESS: Ollama stopped (best-effort).")


# ----------------------------
# CLI entrypoint
# ----------------------------

if __name__ == "__main__":
    # Minimal workspace bootstrap (creates data/raw, prompts, etc.)
    ws = ensure_base_only(base_dir=os.getenv("FAME_BASE_DIR"))
    base_dir = ws.paths.base_dir

    # Env-controlled config
    log_dir = os.getenv("OLLAMA_LOG_DIR", str(base_dir / "data" / "ollama"))
    embedding_model = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text").strip()
    llm_model = os.getenv("OLLAMA_LLM_MODEL", "").strip()

    if not llm_model:
        raise RuntimeError(
            "ERROR: OLLAMA_LLM_MODEL is not set.\n"
            "Example:\n"
            "  export OLLAMA_LLM_MODEL=llama3.1:8b\n"
            "  python -m fame.services.ollama_service"
        )

    timeout_s = int(os.getenv("OLLAMA_STARTUP_TIMEOUT", "60"))
    force_restart = os.getenv("OLLAMA_FORCE_RESTART", "1").strip() not in ("0", "false", "False")
    mode = os.getenv("OLLAMA_MODE", "remote").strip().lower()

    print("Starting Ollama service manager...")
    print(f"Mode            : {mode}")
    print(f"OLLAMA_HOST     : {_base_url()}")
    print(f"Log dir         : {log_dir}")
    print(f"Embedding model : {embedding_model}")
    print(f"LLM model       : {llm_model}")
    print(f"Force restart   : {force_restart}")
    print(f"Ollama bin      : {_ollama_bin() or '(not found)'}")

    pid = setup_ollama(
        embedding_model=embedding_model,
        llm_model=llm_model,
        log_dir=log_dir,
        timeout_s=timeout_s,
        force_restart=force_restart,
        mode=mode,
    )

    if mode == "local":
        print(f"SUCCESS: Ollama ready. PID={pid}")
    else:
        print("SUCCESS: Ollama ready (remote mode).")
