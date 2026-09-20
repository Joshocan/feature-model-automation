from __future__ import annotations

import os
import signal
import subprocess
import time
from pathlib import Path
from typing import Optional

from fame.utils.runtime import workspace, print_created


def _which(cmd: str) -> Optional[str]:
    """Return absolute path to cmd if found on PATH, else None."""
    try:
        out = subprocess.run(["bash", "-lc", f"command -v {cmd}"], capture_output=True, text=True)
        p = out.stdout.strip()
        return p if p else None
    except Exception:
        return None


def _run_quiet(args: list[str]) -> subprocess.CompletedProcess:
    """Run a subprocess without raising; returns CompletedProcess."""
    return subprocess.run(args, check=False, capture_output=True, text=True)


def _read_tail(log_file: Path, n_lines: int = 300) -> str:
    """Return last n_lines of log file (best-effort)."""
    if not log_file.exists():
        return f"(log file not found: {log_file})"
    try:
        out = subprocess.run(["tail", "-n", str(n_lines), str(log_file)], capture_output=True, text=True)
        if out.stdout.strip():
            return out.stdout
        if out.stderr.strip():
            return out.stderr
        return "(log file empty)"
    except Exception as e:
        return f"(failed to read log tail: {e})"


def _is_healthy(host: str, port: int) -> bool:
    """
    Multi-endpoint health probe (Chroma endpoints differ by version).
    Returns True as soon as any endpoint responds with HTTP 2xx/3xx.
    """
    endpoints = [
        f"http://{host}:{port}/api/v1/heartbeat",
        f"http://{host}:{port}/api/v2/heartbeat",
        f"http://{host}:{port}/heartbeat",
        f"http://{host}:{port}/api/v1/version",
        f"http://{host}:{port}/",
    ]
    for url in endpoints:
        try:
            r = subprocess.run(["curl", "-fsS", url], capture_output=True, text=True)
            if r.returncode == 0:
                return True
        except Exception:
            pass
    return False


def _kill_process_on_port(port: int) -> None:
    """
    Kill any process currently listening on the given TCP port (macOS/Linux).
    Uses lsof. Best-effort.
    """
    if not _which("lsof"):
        print("WARN:  lsof not available; cannot kill by port. Skipping.")
        return

    out = _run_quiet(["lsof", "-ti", f"tcp:{port}"])
    pids = [p.strip() for p in out.stdout.splitlines() if p.strip()]

    for pid in pids:
        print(f"WARN:  Killing process on port {port}: PID {pid}")
        _run_quiet(["kill", "-TERM", pid])
    time.sleep(1)

    for pid in pids:
        # If still alive, SIGKILL
        try:
            os.kill(int(pid), 0)
            print(f"WARN:  PID {pid} still alive; SIGKILL")
            _run_quiet(["kill", "-KILL", pid])
        except Exception:
            pass


def stop_existing(pid_file: Path, port: int) -> None:
    """
    Stop any previous Chroma instances:
      1) stop PID in pid_file (if exists)
      2) kill anything listening on the port
      3) kill stray chroma/chromadb run processes (safety net)
    Best-effort and safe to call repeatedly.
    """
    # 1) PID file
    if pid_file.exists():
        try:
            pid = int(pid_file.read_text().strip())
            print(f"WARN:  Stopping previous Chroma PID from file: {pid}")
            try:
                os.kill(pid, signal.SIGTERM)
                time.sleep(1)
                # Still alive? kill
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

    # 2) Port-based kill
    _kill_process_on_port(port)

    # 3) Stray process safety net (best-effort)
    if _which("pkill"):
        _run_quiet(["pkill", "-f", "chroma run"])
        _run_quiet(["pkill", "-f", "chromadb run"])


def start_chroma(
    path: str,
    host: str = "127.0.0.1",
    port: int = 8000,
    timeout_s: int = 120,
    force_restart: bool = True,
) -> int:
    """
    Start Chroma server and wait until it responds on a heartbeat endpoint.

    - If force_restart=True (default), kills any existing instance on the port / pidfile.
    - Returns the PID of the started server process.
    - Raises RuntimeError with log tail on failure.
    """
    chroma_path = Path(path).expanduser().resolve()
    chroma_path.mkdir(parents=True, exist_ok=True)

    log_file = chroma_path / "chroma_server.log"
    pid_file = chroma_path / "chroma_server.pid"

    if force_restart:
        print("🧹 Cleaning up any existing Chroma processes...")
        stop_existing(pid_file, port)
        time.sleep(1)
    else:
        if _is_healthy(host, port):
            raise RuntimeError(f"Chroma appears to already be running at http://{host}:{port} (port in use).")

    # Determine executable preference: chroma first, then chromadb
    chroma_exe = _which("chroma")
    chromadb_exe = _which("chromadb")

    if chroma_exe:
        cmd = ["chroma", "run", "--path", str(chroma_path), "--host", host, "--port", str(port)]
    elif chromadb_exe:
        cmd = ["chromadb", "run", "--path", str(chroma_path), "--host", host, "--port", str(port)]
    else:
        raise RuntimeError(
            "Neither 'chroma' nor 'chromadb' CLI found on PATH.\n"
            "Install (same python env): python3 -m pip install -U chromadb\n"
            "Then verify: which chroma && chroma --help"
        )

    # Fresh log
    log_file.write_text("", encoding="utf-8")

    print("Starting Chroma with command:")
    print("  " + " ".join(cmd))
    print(f"Logs: {log_file}")

    # Start process with stdout+stderr to log
    with log_file.open("a", encoding="utf-8") as lf:
        proc = subprocess.Popen(cmd, stdout=lf, stderr=lf)

    pid_file.write_text(str(proc.pid), encoding="utf-8")

    # Wait for readiness
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        # Died early?
        if proc.poll() is not None:
            tail = _read_tail(log_file, 300)
            msg = (
                f"Chroma exited early with code {proc.returncode}.\n"
                f"Command: {' '.join(cmd)}\n"
                f"Log file: {log_file}\n"
                f"--- Log tail ---\n{tail}\n"
            )
            print(msg)
            raise RuntimeError(msg)

        if _is_healthy(host, port):
            print(f"SUCCESS: Chroma is up at http://{host}:{port}")
            return proc.pid

        time.sleep(1)

    # Timeout
    tail = _read_tail(log_file, 300)
    msg = (
        f"Timed out waiting for Chroma health after {timeout_s}s.\n"
        f"Command: {' '.join(cmd)}\n"
        f"Log file: {log_file}\n"
        f"--- Log tail ---\n{tail}\n"
    )
    print(msg)
    raise RuntimeError(msg)


def stop_chroma(path: str, port: int = 8000) -> None:
    """
    Stop Chroma using pidfile + port kill. Best-effort.
    """
    chroma_path = Path(path).expanduser().resolve()
    pid_file = chroma_path / "chroma_server.pid"
    stop_existing(pid_file, port)


if __name__ == "__main__":
    # Workspace-aware defaults (repo root) + create needed dirs for 'vectorize'
    ws = workspace("vectorize", base_dir=os.getenv("FAME_BASE_DIR"))
    # print_created(ws)  # uncomment for debugging

    # Config via env vars for easy bash integration
    chroma_dir = os.getenv("CHROMA_PATH", str(ws.paths.vector_db))
    host = os.getenv("CHROMA_HOST", "127.0.0.1")
    port = int(os.getenv("CHROMA_PORT", "8000"))
    timeout_s = int(os.getenv("STARTUP_TIMEOUT", "120"))
    force_restart = os.getenv("CHROMA_FORCE_RESTART", "1").strip() not in ("0", "false", "False")

    print("Starting ChromaDB (service manager)...")
    print(f"Path  : {chroma_dir}")
    print(f"Host  : {host}")
    print(f"Port  : {port}")
    print(f"Force restart: {force_restart}")

    pid = start_chroma(chroma_dir, host=host, port=port, timeout_s=timeout_s, force_restart=force_restart)
    print(f"SUCCESS: ChromaDB running. PID={pid}")
    print(f"Try: curl -s http://{host}:{port}/api/v1/heartbeat")
