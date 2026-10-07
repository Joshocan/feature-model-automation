"""Run offline evaluators into a fresh tree and compare CSV evidence with the archive.

No generation calls or overwrites. Logs, exact commands, environment and comparisons
are retained. This is not hosted CI or permission to publish.
"""
import argparse
import concurrent.futures
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / 'results/ifs-2027/analysis'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def equivalent(a, b):
    if a == b:
        return True
    try:
        x, y = float(a), float(b)
        return math.isfinite(x) and math.isfinite(y) and math.isclose(x, y, abs_tol=1e-6, rel_tol=1e-6)
    except (ValueError, TypeError):
        return False


def compare_csv(old, new):
    result = dict(reference=str(old.relative_to(ROOT)), reproduced=str(new.relative_to(ROOT)),
                  reference_sha256=sha(old), reproduced_sha256=sha(new),
                  ignored_fields=[], rows=0, different_cells=0, examples=[])
    if result['reference_sha256'] == result['reproduced_sha256']:
        result['status'] = 'byte_identical'
        return result
    # Preserve CSV row order; any order-only differences remain visible for review.
    import itertools
    with old.open(newline='') as f, new.open(newline='') as g:
        a, b = csv.DictReader(f), csv.DictReader(g)
        if set(a.fieldnames or []) != set(b.fieldnames or []):
            result.update(status='columns_differ', old_columns=a.fieldnames, new_columns=b.fieldnames)
            return result
        for index, (x, y) in enumerate(itertools.zip_longest(a, b), 2):
            result['rows'] += 1
            if x is None or y is None:
                result['different_cells'] += 1
                continue
            for key in x:
                if not equivalent(x[key], y[key]):
                    result['different_cells'] += 1
                    if len(result['examples']) < 8:
                        result['examples'].append(dict(line=index, column=key, old=x[key][:180], new=y[key][:180]))
    result['status'] = 'equivalent_with_numeric_tolerance' if not result['different_cells'] else 'differences'
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    out = args.output.resolve()
    if out.exists() or not out.is_relative_to(ARCHIVE) or out == ARCHIVE:
        p.error('Choose a NEW subdirectory inside results/ifs-2027/analysis')
    out.mkdir()
    logs = out / 'logs'; logs.mkdir()
    env = dict(os.environ, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
               TOKENIZERS_PARALLELISM='false', PYTHONUNBUFFERED='1', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
    records = []

    def run(name, script, *arguments):
        command = [sys.executable, str(ROOT / 'scripts' / script), *map(str, arguments)]
        start = time.monotonic()
        print('START ' + name, flush=True)
        with (logs / (name + '.log')).open('w') as log:
            proc = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        record = dict(stage=name, command=command, returncode=proc.returncode,
                      seconds=round(time.monotonic() - start, 3))
        records.append(record)
        (out / 'commands.json').write_text(json.dumps(records, indent=2))
        print(f'END {name}: exit={proc.returncode}', flush=True)
        if proc.returncode:
            raise RuntimeError('Stage failed; inspect log: ' + name)

    (out / 'environment.json').write_text(json.dumps(dict(python=sys.version, platform=platform.platform(),
        dependencies=subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True).splitlines(),
        numeric_comparison=dict(abs_tol=1e-6, rel_tol=1e-6),
        source_sha256={str(p.relative_to(ROOT)): sha(p) for folder in ('fame', 'scripts', 'config')
                       for p in (ROOT / folder).rglob('*') if p.is_file() and p.suffix in ('.py', '.json', '.yaml', '.lock')}), indent=2))
    inventory = out / 'inventory-current-v1'
    run('inventory', 'inventory_campaign.py', '--output', inventory)
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        jobs = [pool.submit(run, name, script, '--inventory', inventory, '--output', out / target)
                for name, script, target in [
                    ('structure', 'evaluate_structure.py', 'structure-current-v1'),
                    ('semantic', 'evaluate_semantic.py', 'semantic-current-v3'),
                    ('provenance', 'evaluate_provenance.py', 'provenance-current-v2'),
                    ('featureide', 'evaluate_featureide.py', 'featureide-current-v1')]]
        for job in concurrent.futures.as_completed(jobs):
            job.result()
    pairs = out / 'semantic-current-v3/pairs.csv'
    run('aggregate', 'aggregate_campaign.py', '--structural', out / 'structure-current-v1',
        '--semantic', out / 'semantic-current-v3', '--provenance', out / 'provenance-current-v2',
        '--output', out / 'aggregate-current-v4')
    run('tau', 'tau_rescore.py', '--pairs', pairs, '--tau', .3, .4, .5, .6,
        '--policy', 'independent_max', 'one_to_one', 'one_to_one_max_weight', '--output', out / 'tau-matching-v2')
    run('siblings', 'evaluate_siblings.py', '--pairs', pairs, '--output', out / 'sibling-current-v1')
    wide = out / 'aggregate-current-v4/wide.csv'
    run('matching', 'compare_matching.py', '--wide', wide, '--tau', out / 'tau-matching-v2/tau_sweep.csv',
        '--output', out / 'matching-comparison-v2')
    for family in ('A_grounding', 'B_granularity', 'C_ablation'):
        for strict in (False, True):
            key = family + ('_strict' if strict else '')
            target = 'family-' + family + ('-strict' if strict else '') + '-v1'
            run(key, 'analyse_family.py', '--wide', wide, '--family', ROOT / f'config/analysis/families/{key}.json',
                '--output', out / target)
    run('figures', 'plot_paper_figures.py', '--analysis', out, '--output', out / 'paper-figures-v4')
    run('verify', 'verify_evaluation_outputs.py', '--inventory', inventory,
        '--structural', out / 'structure-current-v1', '--semantic', out / 'semantic-current-v3',
        '--provenance', out / 'provenance-current-v2', '--aggregate', out / 'aggregate-current-v4')
    comparisons = []
    for new in sorted(out.glob('*/*.csv')):
        old = ARCHIVE / new.relative_to(out)
        if old.is_file():
            comparisons.append(compare_csv(old, new))
    (out / 'comparison.json').write_text(json.dumps(comparisons, indent=2))
    print(json.dumps({'comparisons': len(comparisons), 'differences': sum(r['status'] in ('differences', 'columns_differ') for r in comparisons)}, indent=2))


if __name__ == '__main__':
    main()
