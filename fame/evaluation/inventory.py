"""Matrix-led, read-only inventory. No validation, scoring or generation calls.

Archived directories are snapshots, not necessarily independent paid attempts:
step-resume archives overlap. Never sum their usage to infer total expenditure.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

from fame.generation.run import RunConfig


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_object(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def run_config(row: dict) -> RunConfig:
    return RunConfig(**{k: v for k, v in row.items() if not k.startswith("_")})


def build_inventory(repo: Path, contract: dict, path_map: dict | None = None) -> dict:
    """Return tables, retaining missing/corrupt runs as rows and explicit issues.

    path_map maps original repository-relative directory prefixes to relocated
    repository-relative prefixes. IDs and original paths are never rewritten.
    """
    repo = repo.resolve()
    mapping = path_map or {}
    runs, snapshots, artifacts, pilots, issues = [], [], [], [], []

    def resolve(name: str) -> Path:
        original = Path(name)
        if original.is_absolute() or ".." in original.parts:
            raise ValueError(f"Non-portable input path: {name}")
        for prefix in sorted(mapping, key=len, reverse=True):
            if name == prefix or name.startswith(prefix + "/"):
                original = Path(mapping[prefix]) / Path(name).relative_to(prefix)
                break
        target = (repo / original).resolve()
        if not target.is_relative_to(repo):
            raise ValueError(f"Mapped path escapes repository: {name}")
        return target

    def issue(kind: str, path: str, run_id: str = "", detail: str = ""):
        issues.append(dict(kind=kind, path=path, run_id=run_id, detail=detail))

    def metadata(original: str, rid: str) -> dict | None:
        path = resolve(original)
        if not path.is_file():
            return None
        try:
            return read_object(path)
        except (ValueError, OSError) as exc:
            issue("invalid_metadata", original, rid, str(exc))
            return None

    def record_files(original: str, rid: str, population: str, snapshot: str):
        directory = resolve(original)
        if not directory.is_dir():
            return
        for path in sorted(directory.rglob("*")):
            if not path.is_file():
                continue
            relative = path.relative_to(directory).as_posix()
            try:
                # Reject symlinks outside the repository as well as path-map escapes.
                if not path.resolve().is_relative_to(repo):
                    raise ValueError("Artifact symlink escapes repository")
                before = path.stat()
                digest = sha256(path)
                after = path.stat()
                if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                    raise ValueError("Artifact changed during inventory; rerun after writers stop")
                artifacts.append(dict(population=population, run_id=rid, snapshot=snapshot,
                    original_path=f"{original}/{relative}",
                    resolved_path=path.relative_to(repo).as_posix(),
                    size_bytes=after.st_size, sha256=digest))
            except (ValueError, OSError) as exc:
                issue("artifact_read_error", str(path), rid, str(exc))

    for name, expected in contract["source_sha256"].items():
        path = resolve(name)
        if not path.is_file():
            issue("missing_source", name)
        elif sha256(path) != expected:
            issue("source_hash_mismatch", name)

    seen = set()
    lane_counts = Counter()
    expected_paths = set()
    for matrix_name in contract["population"]["matrices"]:
        matrix = read_object(resolve(matrix_name))
        lane = matrix.get("lane", "unknown")
        for index, raw in enumerate(matrix["runs"]):
            cfg = run_config(raw)
            rid = cfg.run_id()
            if rid in seen:
                raise ValueError(f"Duplicate planned run ID: {rid}")
            if cfg.campaign_id != contract["campaign_id"]:
                raise ValueError(f"Wrong campaign in matrix: {rid}")
            seen.add(rid)
            lane_counts[lane] += 1
            original = f"results/{cfg.campaign_id}/{cfg.corpus}/{cfg.config_hash()}/{rid}"
            expected_paths.add(resolve(original))
            meta_path = original + "/run_meta.json"
            meta = metadata(meta_path, rid)
            if meta is not None and (not isinstance(meta.get("steps", []), list)
                    or not all(isinstance(s, dict) for s in meta.get("steps", []))):
                issue("invalid_metadata", meta_path, rid, "steps must be a list of objects")
                meta = None
            final_exists = resolve(original + "/fm_gen.xml").is_file()
            has_meta = resolve(meta_path).is_file()
            row = dict(cfg.canonical_dict())
            row.update(run_id=rid, config_hash=cfg.config_hash(), lane=lane,
                arm=cfg.extra.get("arm"), population="main", matrix_path=matrix_name,
                matrix_row=index, original_path=original,
                resolved_path=resolve(original).relative_to(repo).as_posix(),
                run_meta_exists=has_meta, final_exists=final_exists,
                context_log_exists=resolve(original + "/context_log.jsonl").is_file(),
                run_config_exists=resolve(original + "/run_config.json").is_file(),
                iteration_files=len(list(resolve(original + "/fm_iter").glob("step_*.xml"))),
                recorded_completed=None, completed=False, completed_steps=None,
                recorded_terminal_status=None, failed_step=None, failure_reason=None,
                recovery_history=[], archive_snapshot_count=0)
            if meta is None:
                row["inventory_status"] = ("invalid_metadata" if has_meta else
                    "partial" if resolve(original).exists() else "not_started")
            else:
                row.update(recorded_completed=meta.get("completed"),
                    completed_steps=meta.get("completed_steps"),
                    recorded_terminal_status=meta.get("terminal_status"),
                    failed_step=meta.get("failed_step"),
                    recovery_history=meta.get("recovery_history", []))
                identity_ok = meta.get("run_id") == rid and meta.get("config") == cfg.canonical_dict()
                stored_cfg = metadata(original + "/run_config.json", rid)
                if stored_cfg is not None and stored_cfg != cfg.canonical_dict():
                    identity_ok = False
                if not identity_ok:
                    issue("identity_mismatch", meta_path, rid)
                steps = meta.get("steps", [])
                failed = [s for s in steps if s.get("error") or s.get("terminal_failure")]
                row["failure_reason"] = (failed[-1].get("error") or failed[-1].get("terminal_failure")) if failed else None
                accepted = {s.get("step_index") for s in steps if s.get("carry_forward") is True}
                checkpoints = all(resolve(original + f"/fm_iter/step_{j:02d}.xml").is_file() for j in range(cfg.N))
                complete = (meta.get("completed") is True and meta.get("terminal_status") == "completed"
                    and meta.get("completed_steps") == cfg.N and meta.get("planned_steps") == cfg.N
                    and accepted == set(range(cfg.N)) and checkpoints and final_exists and identity_ok)
                row["completed"] = complete
                if meta.get("completed") is True and not complete:
                    issue("completion_inconsistent", original, rid)
                row["inventory_status"] = ("identity_mismatch" if not identity_ok else
                    "completed" if complete else "completion_inconsistent" if meta.get("completed") is True else
                    meta.get("terminal_status") or "partial")
            runs.append(row)
            record_files(original, rid, "main", "selected")
            archive = f"results/recovery_archive/{cfg.campaign_id}/{cfg.corpus}/{rid}"
            if resolve(archive).is_dir():
                for directory in sorted(resolve(archive).iterdir()):
                    if not directory.is_dir():
                        continue
                    source = archive + "/" + directory.name
                    prior = metadata(source + "/run_meta.json", rid)
                    if prior and ((prior.get("run_id") not in (None, rid)) or
                            (prior.get("config") is not None and prior["config"] != cfg.canonical_dict())):
                        issue("archive_identity_mismatch", source, rid)
                    snapshots.append(dict(run_id=rid, snapshot_id=directory.name,
                        original_path=source, resolved_path=directory.relative_to(repo).as_posix(),
                        kind="archive_snapshot_not_independent_repetition",
                        recorded_terminal_status=(prior or {}).get("terminal_status"),
                        completed_steps=(prior or {}).get("completed_steps"),
                        recovery_history=(prior or {}).get("recovery_history", [])))
                    record_files(source, rid, "recovery", directory.name)
                    row["archive_snapshot_count"] += 1

    if len(runs) != contract["population"]["expected_planned_runs"]:
        issue("planned_count_mismatch", "matrices", detail=str(len(runs)))
    if dict(lane_counts) != contract["population"]["expected_by_lane"]:
        issue("lane_count_mismatch", "matrices", detail=str(dict(lane_counts)))
    # Unplanned directories are diagnostic only, never added to the population.
    for corpus in sorted({r["corpus"] for r in runs}):
        base = resolve(f"results/{contract['campaign_id']}/{corpus}")
        for directory in sorted(base.glob("*/*")):
            if directory.is_dir() and directory.resolve() not in expected_paths:
                issue("unplanned_run_directory", directory.relative_to(repo).as_posix())

    manifest_name = contract["population"].get("pilot_manifest")
    if manifest_name:
        manifest = read_object(resolve(manifest_name))
        for source in manifest["sources"]:
            original = source["path"]
            root = resolve(original)
            if not root.is_dir():
                issue("missing_pilot_source", original)
                continue
            found = list(sorted(root.glob("*/*/*/run_meta.json")))
            if not found:
                issue("empty_pilot_source", original)
            for path in found:
                relative = path.parent.relative_to(root).as_posix()
                source_path = original + "/" + relative
                meta = metadata(source_path + "/run_meta.json", path.parent.name)
                pilots.append(dict(population="pilot", cohort=source["cohort"],
                    primary_within_pilot=source["primary"], source_campaign=root.name,
                    run_id=(meta or {}).get("run_id", path.parent.name),
                    original_path=source_path, resolved_path=path.parent.relative_to(repo).as_posix(),
                    recorded_completed=(meta or {}).get("completed"),
                    recorded_terminal_status=(meta or {}).get("terminal_status"),
                    config=(meta or {}).get("config"),
                    membership_basis="observed_metadata_not_main_planned_population"))
                record_files(source_path, pilots[-1]["run_id"], "pilot", source["path"])

    return dict(runs=runs, recovery_snapshots=snapshots, artifacts=artifacts,
        pilots=pilots, issues=issues, summary=dict(planned=len(runs),
            by_lane=dict(lane_counts), by_status=dict(Counter(r["inventory_status"] for r in runs)),
            recovery_snapshots=len(snapshots), pilot_records=len(pilots),
            artifact_files=len(artifacts), issues=len(issues),
            conformance_evaluated=False, semantic_scores_computed=False))
