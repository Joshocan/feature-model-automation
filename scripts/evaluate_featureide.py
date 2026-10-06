#!/usr/bin/env python3
"""Read-only FeatureIDE 3.10.0 compatibility evaluation of completed inventory runs."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile

REPO = Path(__file__).resolve().parents[1]
JAVA_SOURCE = REPO / 'tools/featureide/FeatureIDEParse.java'
JAR_SHA256 = 'b718c2461b4af2eccd64c5afd06ac804554bd723d2685d4521da423c9ef1829a'
PREFIX = 'FEATUREIDE_RESULT '


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def completed_runs(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    seen, selected = set(), []
    for row in rows:
        rid = row.get('run_id')
        if not rid or rid in seen:
            raise ValueError(f'Missing or duplicate run_id: {rid}')
        seen.add(rid)
        if row.get('completed', '').lower() not in {'true', 'false'}:
            raise ValueError(f'{rid}: invalid completed flag')
        if row['completed'].lower() == 'true':
            if not row.get('resolved_path'):
                raise ValueError(f'{rid}: no resolved_path')
            selected.append(row)
    if not selected:
        raise ValueError('No completed runs in inventory')
    return rows, selected


def parse_model(path, classpath, java='java', timeout=30):
    result = dict(parsed_ok=None, status='evaluator_error', reason='',
                  errors=None, warnings=None, loaded_features=None, problems=[])
    if not path.is_file():
        return dict(result, status='missing_artifact', reason=str(path))
    try:
        proc = subprocess.run([java, '-Xmx512m', '-cp', classpath,
                               'FeatureIDEParse', str(path)],
                              capture_output=True, text=True, timeout=timeout)
        lines = [s[len(PREFIX):] for s in proc.stdout.splitlines() if s.startswith(PREFIX)]
        if proc.returncode != 0 or len(lines) != 1:
            raise ValueError(f'Java exit {proc.returncode}: {proc.stderr}\n{proc.stdout}')
        payload = json.loads(lines[0])
        if type(payload.get('parsed_ok')) is not bool:
            raise ValueError('Invalid bridge result')
        result.update(payload, status='ok', reason='')
        result['stderr'] = proc.stderr
    except (OSError, subprocess.TimeoutExpired, ValueError) as exc:
        result['reason'] = str(exc)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inventory', type=Path, required=True, help='Inventory directory or runs.csv')
    p.add_argument('--jar', type=Path, default=REPO / 'tools/featureide/lib/de.ovgu.featureide.lib.fm-v3.10.0.jar')
    p.add_argument('--output', type=Path, required=True, help='Fresh output directory')
    p.add_argument('--java', default='java')
    p.add_argument('--javac', default='javac')
    p.add_argument('--timeout', type=float, default=30)
    p.add_argument('--preflight-only', action='store_true')
    args = p.parse_args()
    inv = args.inventory / 'runs.csv' if args.inventory.is_dir() else args.inventory
    all_rows, selected = completed_runs(inv)
    if args.output.exists():
        p.error('Output already exists; use a fresh versioned directory')
    if args.timeout <= 0:
        p.error('--timeout must be positive')
    if not args.jar.is_file() or digest(args.jar) != JAR_SHA256:
        p.error('Missing or different FeatureIDE JAR. See docs/featureide-evaluation.md')
    summary = dict(featureide_version='3.10.0', jar_sha256=JAR_SHA256,
                   jar_path=str(args.jar.resolve()), timeout_seconds=args.timeout,
                   max_heap_mb=512, java_command=args.java, javac_command=args.javac,
                   inventory_sha256=digest(inv), inventory_path=str(inv.resolve()),
                   driver_sha256=digest(Path(__file__)), adapter_sha256=digest(JAVA_SOURCE),
                   planned_runs=len(all_rows), completed_runs=len(selected),
                   scope='Pinned XML reader acceptance; not Eclipse UI, SAT, or semantic correctness.',
                   preflight_only=args.preflight_only)
    with tempfile.TemporaryDirectory(prefix='featureide-') as tmp:
        build = Path(tmp)
        subprocess.run([args.javac, '-encoding', 'UTF-8', '-cp', str(args.jar.resolve()),
                        '-d', tmp, str(JAVA_SOURCE)], check=True, timeout=60)
        summary['java_version'] = subprocess.run([args.java, '-version'], capture_output=True,
                                                   text=True, check=True).stderr
        classpath = os.pathsep.join([tmp, str(args.jar.resolve())])
        # Known positive/negative controls ensure the bridge actually detects rejection.
        fixtures = {
            'valid': ('<featureModel><struct><and name="Root"><feature name="A"/></and></struct></featureModel>', True),
            'malformed': ('<featureModel><struct>', False),
            'duplicate': ('<featureModel><struct><and name="Root"><feature name="A"/><feature name="A"/></and></struct></featureModel>', False),
        }
        controls = []
        for name, (xml, expected) in fixtures.items():
            path = build / (name + '.xml')
            path.write_text(xml, encoding='utf-8')
            result = parse_model(path, classpath, args.java, args.timeout)
            controls.append(dict(control=name, expected=expected, **result))
            if result['status'] != 'ok' or result['parsed_ok'] != expected:
                raise RuntimeError(f'Preflight failed: {controls[-1]}')
        # References are diagnostics, not assumed universally accepted by FeatureIDE.
        for corpus in ('repair', 'federation'):
            path = REPO / 'data/ground_truth' / (corpus + '.xml')
            controls.append(dict(control=corpus, input_sha256=digest(path),
                                 **parse_model(path, classpath, args.java, args.timeout)))
        summary['controls'] = controls
        if any(r['status'] != 'ok' for r in controls):
            raise RuntimeError(f'Reference preflight runtime error: {controls}')
        args.output.mkdir(parents=True, exist_ok=False)
        results = []
        fields = ['run_id', 'corpus', 'model_id', 'arm', 'N', 'grounding', 'seed',
                  'input_path', 'input_sha256', 'structural__featureide_parse',
                  'structural__featureide_parse__status', 'structural__featureide_parse__reason',
                  'errors', 'warnings', 'loaded_features', 'problems', 'stderr']
        with (args.output / 'featureide_parse.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for index, row in enumerate([] if args.preflight_only else selected, 1):
                path = Path(row['resolved_path']) / 'fm_gen.xml'
                if not path.is_absolute():
                    path = REPO / path
                before = digest(path) if path.is_file() else ''
                result = parse_model(path, classpath, args.java, args.timeout)
                if before and (not path.is_file() or digest(path) != before):
                    result.update(status='evaluator_error', parsed_ok=None, reason='Input changed during evaluation')
                out = {k: row.get(k, '') for k in fields[:7]}
                out.update(input_path=str(path), input_sha256=before,
                           structural__featureide_parse=result['parsed_ok'],
                           structural__featureide_parse__status=result['status'],
                           structural__featureide_parse__reason=result['reason'],
                           errors=result['errors'], warnings=result['warnings'],
                           loaded_features=result['loaded_features'],
                           problems=json.dumps(result['problems']), stderr=result.get('stderr', ''))
                writer.writerow(out)
                f.flush()
                results.append(result)
                print(f'[{index}/{len(selected)}] {row["run_id"]}: {result["status"]} parsed={result["parsed_ok"]}', flush=True)
        summary.update(evaluated=len(results), accepted=sum(r['parsed_ok'] is True for r in results),
                       rejected=sum(r['parsed_ok'] is False for r in results),
                       status_counts=dict(Counter(r['status'] for r in results)),
                       results_sha256=digest(args.output / 'featureide_parse.csv'))
        (args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
        print(json.dumps(summary, indent=2))
        return int(any(r['status'] != 'ok' for r in results))


if __name__ == '__main__':
    raise SystemExit(main())
