"""Reproduce expert tables in a temporary checkout without private workbooks."""
import argparse
import csv
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.restore_research_data import safe_path, sha


def verify(archive):
    manifest_path = archive / 'artifact-manifest.json'
    manifest = json.loads(manifest_path.read_text())
    sources = {source: record for record in manifest['files']
               for source in [record['source'], *record.get('restore_sources', [])]}
    required = ['results/expert-study/public-inputs/' + name for name in
                ('ratings.csv', 'citation_judgements.csv', 'model_selection.csv', 'citation_items.csv')]
    required += ['results/ifs-2027/analysis/reproduction-2026-10-07/' + name for name in
                 ('aggregate-current-v4/wide.csv', 'matching-comparison-v2/matching_comparison.csv',
                  'sibling-current-v1/sibling_metrics.csv')]
    required.append('results/expert-study/analysis/astra-n1-rag-evaluation-v1.csv')
    with (archive/'expert-study/inputs/model_selection.csv').open(newline='') as f:
        required += [r['xml_path'] for r in csv.DictReader(f)]
    with tempfile.TemporaryDirectory(prefix='expert-public-reproduction-') as folder:
        dest = Path(folder).resolve()
        hashes = []
        for name in required:
            record = sources[name]
            source = safe_path(archive, record['path'])
            if sha(source) != record['sha256']:
                raise ValueError('Archive input hash mismatch')
            target = safe_path(dest, name)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            hashes.append(dict(archive_path=record['path'],sha256=record['sha256']))
        output = dest/'results/expert-study/reproduced'
        command = [sys.executable, str(ROOT/'scripts/expert_analysis.py'), '--root', str(dest),
                   '--public-inputs', str(dest/'results/expert-study/public-inputs'), '--output', str(output)]
        process = subprocess.run(command, capture_output=True, text=True)
        if process.returncode:
            raise ValueError('Public expert reproduction failed: ' + process.stderr)
        comparisons = []
        for reference in sorted((archive/'expert-study/anonymised-results').glob('*.csv')):
            if reference.name in ('input_hashes.csv','software_versions.csv'):
                continue
            with reference.open(newline='') as f, (output/reference.name).open(newline='') as g:
                equal = list(csv.DictReader(f)) == list(csv.DictReader(g))
            comparisons.append(dict(file=reference.name,identical_rows=equal))
        if len(comparisons) != 18 or not all(r['identical_rows'] for r in comparisons):
            raise ValueError('Public expert tables differ from the archived analysis')
        with (output/'sanity_checks.csv').open(newline='') as f:
            checks = list(csv.DictReader(f))
        if len(checks) != 82 or any(r['passed'] != 'True' for r in checks):
            raise ValueError('Expert sanity checks did not all pass')
        return dict(private_workbooks_present=False, temporary_checkout=True,
                    source_manifest_sha256=sha(manifest_path), script_sha256=sha(ROOT/'scripts/expert_analysis.py'),
                    sanity_checks_passed=len(checks), analysis_csvs_identical=len(comparisons),
                    comparisons=comparisons, inputs=hashes,
                    excluded_comparisons=['input_hashes.csv','software_versions.csv'],
                    exclusion_reason='Public CSV inputs replace private workbooks; implementation hash records the new reader.')


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():
        p.error('Choose a new verification report path')
    report=verify(a.archive.resolve())
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(report,indent=2)+'\n')
    print('82 checks passed; 18 analysis CSVs match in a temporary checkout with no private workbooks.')
