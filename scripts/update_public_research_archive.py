"""Stage the final analysis evidence with descriptive archive paths.

No scoring or provider calls. Original software outputs are read-only. The
manifest retains source paths and hashes; public derivatives are labelled.
Default is a read-only plan. --apply updates only the specified data repository.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path
import re
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.prepare_data_release import digest, read, refresh_metadata, release_files

TEMPLATES = ROOT / 'docs/data-release/public-archive'
DATE = re.compile(r'[-_]?20\d{2}[-_]\d{2}[-_]\d{2}')


def public_path(path):
    """Rename archive locations only, never identifiers or evidence contents."""
    result = []
    for part in Path(path).parts:
        value = DATE.sub('', part)
        value = re.sub('reanalysis', 'verification', value, flags=re.I)
        if value == 'pilot':
            value = 'pilot-initial'
        result.append(value)
    return '/'.join(result)


def csv_bytes(rows):
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode('utf-8')


def csv_read(path):
    with path.open(newline='', encoding='utf-8-sig') as stream:
        return list(csv.DictReader(stream))


def public_astra_csv(source):
    rows = csv_read(source)
    for row in rows:
        for key in ('input_sha256', 'evaluation_source_sha256'):
            values = json.loads(row[key])
            normal = {}
            for path, value in values.items():
                if '/feature-model-automation/' in path:
                    path = path.split('/feature-model-automation/', 1)[1]
                if path.startswith('/'):
                    raise ValueError('Unrecognised personal path in Astra provenance')
                normal[path] = value
            row[key] = json.dumps(normal, sort_keys=True)
    return csv_bytes(rows)


def additions():
    """Explicit selections, with all final analysis files accounted for."""
    pairs = []
    def folder(source, target):
        for p in sorted((ROOT / source).rglob('*')):
            if p.is_file() and not p.name.startswith('.'):
                pairs.append((p, f'{target}/{p.relative_to(ROOT / source).as_posix()}'))
    folder('results/ifs-2027/analysis/cross-reference-v1', 'analysis/negative-control')
    folder('results/expert-study/analysis-final', 'expert-study/anonymised-results')
    folder('results/ifs-2027/analysis/paper-figures-rs-v2', 'paper/figures/publication')
    mapping = {
        'table3_consistent.csv': 'attribution-agreement/table3.csv',
        'attribution_run_scores.csv': 'attribution-agreement/run_scores.csv',
        'published_table3_comparison.csv': 'attribution-agreement/baseline_definition_comparison.csv',
        'task1_restricted.csv': 'attribution-agreement/high_confidence_alignments.csv',
        'task2_recall_summary.csv': 'reference-partitions/recall_summary.csv',
        'task2_run_recall.csv': 'reference-partitions/run_recall.csv',
        'task3_sat_summary.csv': 'logical-validity/satisfiability_summary.csv',
        'task3_sat_runs.csv': 'logical-validity/satisfiability_runs.csv',
        'task3_unsat_diagnoses.csv': 'logical-validity/unsatisfiable_causes.csv',
        'task3_w_checks.csv': 'logical-validity/wellformedness_runs.csv',
        'task3_w_summary.csv': 'logical-validity/wellformedness_summary.csv',
        'input_hashes.json': 'verification/input_hashes.json',
        'status.json': 'verification/status.json',
        'stepA_search_scope.json': 'verification/baseline_source_search_scope.json',
        'stepA_search_hits.csv': 'verification/baseline_source_search.csv',
    }
    source = ROOT / 'results/reanalysis_2026-10-15-final'
    if {p.name for p in source.iterdir() if p.is_file()} != set(mapping):
        raise ValueError('Final verification file set changed; review archive mapping')
    pairs += [(source / name, 'analysis/' + target) for name, target in mapping.items()]
    pairs += [(ROOT / 'results/reanalysis_2026-10-15' / name,
               'analysis/conformance-denominators/' + target) for name, target in (
        ('task4_conformance_denominators.csv', 'condition_counts.csv'),
        ('task4_conformance_differences.csv', 'condition_differences.csv'),
        ('input_hashes.json', 'input_hashes.json'))]
    pairs += [(ROOT / 'results/reanalysis_2026-10-15.md', 'analysis/verification/analysis-history.md'),
              (ROOT / 'results/expert-study/analysis/astra-n1-rag-evaluation-v1.csv',
               'expert-study/automated-scores/astra-n1.csv')]
    folder('results/ifs-2027/analysis/reproduction-2026-10-07',
           'protocol/reproduction/automated-evaluation')
    return pairs


def verify_existing(root):
    for line in (root / 'checksums.sha256').read_text().splitlines():
        expected, name = line.split('  ', 1)
        p = root / name
        if not p.is_file() or digest(p) != expected:
            raise ValueError('Existing archive checksum mismatch: ' + name)


def prepare(root, apply=False):
    root = root.resolve()
    if root == ROOT or not (root / 'artifact-manifest.json').is_file():
        raise ValueError('Specify the existing research-data repository, not the software checkout')
    manifest = read(root / 'artifact-manifest.json')
    if manifest.get('public_analysis_update'):
        raise ValueError('Public update already applied; inspect it instead of overwriting it')
    verify_existing(root)
    print('Existing checksum catalogue verified', flush=True)
    originals = manifest['files']
    all_paths = [p.relative_to(root).as_posix() for p in release_files(root)]
    names = {name: public_path(name) for name in all_paths}
    if len(set(names.values())) != len(names):
        raise ValueError('Descriptive naming creates a collision')
    copied = additions()
    for source, target in copied:
        if not source.is_file() or DATE.search(target) or 'reanalysis' in target:
            raise ValueError('Missing source or unsuitable public name: ' + target)
        if (root / target).exists():
            raise ValueError('New evidence target already exists: ' + target)
    report = dict(renamed_files=sum(a != b for a,b in names.items()),
                  selected_new_files=len(copied), apply=apply,
                  publication_approved=False)
    if not apply:
        return report
    backup = root / '.release-history/before-public-analysis'
    backup.mkdir(parents=True, exist_ok=False)
    for p in release_files(root):
        rel = p.relative_to(root)
        # Original evidence need not be duplicated; only documentation/metadata
        # rewritten by this operation is backed up.
        if rel.as_posix() not in {r['path'] for r in originals}:
            dst = backup / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, dst)
    for old, new in names.items():
        if old != new:
            target = root / new
            target.parent.mkdir(parents=True, exist_ok=True)
            (root / old).rename(target)
    for r in originals:
        old = r['path']
        r['path'] = names.get(old, old)
        if old != r['path']:
            r['previous_public_path'] = old
    for r in manifest.get('derived_files', []):
        r['path'] = names.get(r['path'], r['path'])
    # Remove only empty containers left by renaming; no evidence is deleted.
    old_parents = {p for old,new in names.items() if old != new for p in (root/old).parents
                   if p.is_relative_to(root) and p != root}
    for p in sorted(old_parents, key=lambda p: len(p.parts), reverse=True):
        if p.is_dir() and not any(p.iterdir()):
            p.rmdir()
    records = originals
    by_hash = {}
    for r in records:
        by_hash.setdefault(r['sha256'], r)
    public_expert = set()
    source_targets = {s for r in records for s in [r['source'], *r.get('restore_sources', [])]}
    identical = 0
    for source, target in copied:
        h = digest(source)
        rel = source.relative_to(ROOT).as_posix()
        if rel in source_targets:
            raise ValueError('Source already archived: ' + rel)
        source_targets.add(rel)
        # Reproduction snapshots often duplicate the original CSV bytes. Retain
        # separate restore identities without adding almost a gigabyte twice.
        if 'reproduction-2026-10-07/' in rel and h in by_hash:
            by_hash[h].setdefault('restore_sources', []).append(rel)
            identical += 1
            continue
        dest = root / target
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, dest)
        records.append(dict(source=rel, path=target, sha256=h, bytes=dest.stat().st_size))
        if target == 'expert-study/automated-scores/astra-n1.csv':
            dest.write_bytes(public_astra_csv(source))
            records[-1].update(original_source_sha256=h, sha256=digest(dest),bytes=dest.stat().st_size,
                transformation='Only provenance JSON path keys made software-relative; numeric scores and recorded input hashes unchanged')
        by_hash.setdefault(h, records[-1])
        if target.startswith('expert-study/'):
            public_expert.add(target)

    def document(target, content):
        p = root / target
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding='utf-8')
        if target.startswith('expert-study/'):
            public_expert.add(target)

    def derivative(target, payload, restore_source, origin):
        p = root / target
        if p.exists():
            raise FileExistsError(target)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(payload)
        records.append(dict(source=restore_source, path=target, sha256=digest(p), bytes=len(payload),
                            derivative_of=origin, transformation='public response input; see expert-study/README.md'))
        public_expert.add(target)

    gitignore = (root/'.gitignore').read_text() if (root/'.gitignore').exists() else ''
    if '.release-history/' not in gitignore:
        document('.gitignore', gitignore + '\n# Local recoverable archive-preparation history, never publish.\n.release-history/\n')

    base = ROOT / 'results/expert-study'
    for name in ('ratings.csv', 'citation_judgements.csv'):
        p = base / 'analysis-final' / name
        derivative('expert-study/inputs/' + name, p.read_bytes(),
                   'results/expert-study/public-inputs/' + name, p.relative_to(ROOT).as_posix())
    selection = csv_read(base / 'selection-v3/expert_sample_frozen.csv')
    for r in selection:
        path = r['xml_path'].replace('\\', '/')
        if '/results/' in path:
            path = 'results/' + path.split('/results/', 1)[1]
        if not path.startswith('results/') or digest(ROOT / path) != r['xml_sha256']:
            raise ValueError('Selected XML identity mismatch')
        r['xml_path'] = path
    derivative('expert-study/inputs/model_selection.csv', csv_bytes(selection),
               'results/expert-study/public-inputs/model_selection.csv', 'frozen model selection; absolute paths normalised')
    key = csv_read(base / 'citations-v3/citation_key_AUTHOR_ONLY.csv')
    derivative('expert-study/inputs/citation_items.csv', csv_bytes(key),
               'results/expert-study/public-inputs/citation_items.csv', 'frozen citation sampling key; no rater identifiers')
    for template, target in (
        ('README.md', 'README.md'), ('analysis-README.md', 'analysis/README.md'),
        ('expert-README.md', 'expert-study/README.md'),
        ('reproduction-README.md', 'protocol/reproduction/README.md'),
        ('figures-README.md', 'paper/figures/README.md')):
        document(target, (TEMPLATES / template).read_text())
    for folder, title, body in (
        ('analysis/negative-control', 'Cross-reference negative control',
         '`scores.csv` contains 500 completed runs × six thresholds × two policies. '
         'Own/cross scores are paired within each run. `cell_summary.csv` and '
         '`headline_rankings.csv` summarise the same saved scores. See ../README.md and ../../protocol/reproduction/README.md.'),
        ('analysis/attribution-agreement', 'Attribution agreement',
         'Use table3.csv for the consistent citation-weighted definition; run_scores.csv '
         'retains the per-run values. high_confidence_alignments.csv restricts to ≥0.8 '
         'or unique exact names. baseline_definition_comparison.csv explains earlier published values. See ../README.md.'),
        ('analysis/reference-partitions', 'Reference-partition recall',
         'run_recall.csv retains each run/policy/partition; recall_summary.csv gives '
         'the mean and whole-run bootstrap intervals. See ../README.md for denominators.'),
        ('analysis/logical-validity', 'Logical validity',
         'satisfiability_runs.csv and satisfiability_summary.csv distinguish conformant '
         'models and SAT-test eligibility. unsatisfiable_causes.csv records counterfactual '
         'constraint/mandatory-flag checks, not repaired experiment outputs. W-condition checks are separate.'),
        ('analysis/conformance-denominators', 'Conformance denominators',
         'condition_counts.csv gives planned, completed and conformant counts. '
         'condition_differences.csv contrasts conditional and unconditional rates; '
         'the reported Delta-Conf uses conformant/completed. Zero-completion cells are undefined.'),
        ('analysis/verification', 'Analysis provenance',
         'status.json records the successful final verification. analysis-history.md preserves '
         'the earlier failed baseline check followed by its resolution. Source-search records '
         'retain original software paths. Use the thematic tables, not historical checks, for reporting.'),
        ('expert-study/anonymised-results', 'Expert analysis outputs',
         'Read summary.md and ../README.md. Tables 4 and 12 are table4.csv and '
         'table12_percentiles.csv. Original response CSVs are also supplied under ../inputs/.'),
        ('expert-study/inputs', 'Public expert-analysis inputs',
         'Four CSVs reproduce the numeric assessment without returned workbooks. '
         'Blank Q5 means NJ; Cannot judge remains missing. Read ../README.md before use.'),
        ('expert-study/protocol', 'Expert assessment protocol',
         'The current public analysis definitions and replication command are in ../README.md. '
         'The companion software docs/expert-evaluation-strategy.md documents study preparation.'),
        ('expert-study/selection', 'Expert model selection',
         'The selected ten codes, run IDs and XML hashes are in ../inputs/model_selection.csv. '
         'Citation item strata are in ../inputs/citation_items.csv. These keys reveal model identities.'),
        ('expert-study/blank-instrument', 'Blank instrument',
         'The companion software supplies data/Expert-evaluation-form.xlsx. '
         'It is not needed to reproduce the returned numeric responses supplied as public CSVs. '
         'No returned workbook is distributed here.'),
    ):
        document(folder + '/README.md', '# ' + title + '\n\n' + body + '\n')
    path_rows = [dict(previous_path=a, archive_path=b) for a,b in names.items() if a != b]
    (root / 'protocol/archive-path-map.csv').write_bytes(csv_bytes(path_rows))
    expert_manifest = dict(files=sorted(public_expert),
        scope='explicit public numeric data and documentation; excludes original forms, comments and identity keys',
        author_publication_approval=False)
    document('protocol/public-expert-files.json', json.dumps(expert_manifest, indent=2) + '\n')
    # Update scope, but never mark rights/privacy/manuscript gates approved.
    approval = (root / 'RELEASE_APPROVAL.md').read_text()
    approval = approval.replace('Updated: 2026-10-06. Scope: automated experiments and evaluation, excluding expert\nresponses and expert analysis.',
        'Scope: automated experiments, evaluation and anonymised expert-assessment inputs/results.')
    document('RELEASE_APPROVAL.md', approval)
    document('CHANGELOG.md', (root / 'CHANGELOG.md').read_text() +
        '\n## Public analysis archive update\n\nAdded anonymised expert inputs/results, cross-reference controls, '
        'attribution/partition/SAT/conformance analyses, updated figures and reproduction evidence. '
        'Public paths now use descriptive names; original source identifiers/hashes remain. '
        'Human publication approvals have not been inferred.\n')
    document('DATA_DICTIONARY.md', (root / 'DATA_DICTIONARY.md').read_text() +
        '\n## Public analysis additions\n\nSee analysis/README.md for the current thematic tables and '
        'expert-study/README.md for response codes and missingness. Public expert inputs preserve '
        'numeric responses and exclude comments/identity metadata. `restore_sources` in the artifact '
        'manifest provides additional byte-identical software locations. `derivative_of` identifies '
        'normalised public inputs rather than claiming unchanged private workbooks.\n')
    document('paper/tables/README.md', '# Paper table sources\n\n'
        '- Matching-policy reversal: analysis/matching-sensitivity/matching-comparison-v2/headline_rankings.csv.\n'
        '- Provenance Table 3: analysis/attribution-agreement/table3.csv.\n'
        '- Expert Tables 4 and 12: expert-study/anonymised-results/table4.csv and table12_percentiles.csv.\n'
        '- Grounding/ablation conformance: analysis/conformance-denominators/.\n'
        '- Partition recall and logical validity: analysis/reference-partitions/ and analysis/logical-validity/.\n'
        '- Negative controls: analysis/negative-control/.\n\n'
        'Paths above are relative to the archive root. Reconcile rounding, populations and numbering '
        'against the final manuscript in paper/reconciliation.md. campaign-outcomes.csv is a '
        'supporting completion table, not a complete replacement for the paper tables.\n')
    manifest['files'] = records
    manifest['release_scope'] = 'automated evidence and anonymised expert-assessment inputs/results; author approval pending'
    manifest['excluded'] = [x for x in manifest['excluded'] if not x.startswith('expert-study')]
    manifest['excluded'].append('original expert workbooks, participant identity keys, emails and free-text comments')
    manifest['public_analysis_update'] = dict(script_sha256=digest(Path(__file__)),
        identical_reproduction_aliases=identical, source_outputs_unchanged=True,
        expert_response_input_mode='public CSVs', publication_approved=False)
    document('artifact-manifest.json', json.dumps(manifest, indent=2) + '\n')
    validation = read(root / 'release-validation.json')
    validation['release_scope'] = manifest['release_scope']
    validation['publication_ready'] = False
    validation['public_analysis_update'] = manifest['public_analysis_update']
    document('release-validation.json', json.dumps(validation, indent=2) + '\n')
    schemas = []
    for r in records:
        p = root / r['path']
        if p.suffix == '.csv':
            delimiter = ';' if '/corpus-manifests/' in r['path'] else ','
            with p.open(newline='', encoding='utf-8-sig') as stream:
                schemas.append(dict(path=r['path'],sha256=r['sha256'],delimiter=delimiter,
                                    columns=next(csv.reader(stream,delimiter=delimiter), [])))
    document('protocol/tabular-schemas.json', json.dumps(schemas, indent=2) + '\n')
    refresh_metadata(root)
    verify_existing(root)
    for r in records:
        if digest(root / r['path']) != r['sha256']:
            raise ValueError('Evidence changed: ' + r['path'])
    report.update(identical_reproduction_aliases=identical, manifest_sha256=digest(root/'artifact-manifest.json'),
                  original_and_added_records=len(records))
    return report


def normalise_expert_paths(root):
    root = root.resolve()
    verify_existing(root)
    manifest = read(root/'artifact-manifest.json')
    name = 'expert-study/automated-scores/astra-n1.csv'
    record = next(r for r in manifest['files'] if r['path'] == name)
    source = ROOT/record['source']
    if digest(root/name) not in (digest(source), record['sha256']):
        raise ValueError('Unexpected Astra source content')
    (root/name).write_bytes(public_astra_csv(source))
    record.update(original_source_sha256=digest(source),sha256=digest(root/name),bytes=(root/name).stat().st_size,
        transformation='Only provenance JSON path keys made software-relative; numeric scores and recorded input hashes unchanged')
    (root/'artifact-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    schemas=read(root/'protocol/tabular-schemas.json')
    for r in schemas:
        if r['path']==name:r['sha256']=record['sha256']
    (root/'protocol/tabular-schemas.json').write_text(json.dumps(schemas,indent=2)+'\n')
    refresh_metadata(root)
    return dict(normalised_file=name,original_source_unchanged=True,expert_reverification_required=True)


def finalize(root, verification):
    """Attach verified public replication evidence and reconcile current scope."""
    root = root.resolve()
    verify_existing(root)
    evidence = read(verification)
    if evidence.get('sanity_checks_passed') != 82 or evidence.get('analysis_csvs_identical') != 18:
        raise ValueError('Public expert reproduction did not pass')
    manifest = read(root/'artifact-manifest.json')
    if not manifest.get('public_analysis_update'):
        raise ValueError('Apply the public archive update first')
    for row in evidence['inputs']:
        if digest(root/row['archive_path']) != row['sha256']:
            raise ValueError('An input changed after public reproduction')
    target = root/'protocol/reproduction/expert-public-input-check.json'
    if target.exists():
        backup=root/'.release-history'/('expert-verification-'+digest(target)+'.json')
        backup.parent.mkdir(parents=True,exist_ok=True)
        if not backup.exists():shutil.copyfile(target,backup)
    shutil.copyfile(verification, target)
    (root/'RELEASE_CHECKLIST.md').write_text((TEMPLATES/'release-checklist.md').read_text())
    (root/'README.md').write_text((TEMPLATES/'README.md').read_text())
    (root/'analysis/README.md').write_text((TEMPLATES/'analysis-README.md').read_text())
    (root/'expert-study/README.md').write_text((TEMPLATES/'expert-README.md').read_text())
    p = root/'protocol/privacy-and-blinding.md'
    text = p.read_text().replace('out of this non-expert release', 'out of this public release')
    text = text.replace('No participant results are being released now. Any later participant-level data\nrelease needs its own review of what participants agreed could be shared;',
        'Anonymised numeric participant responses and analyses are now staged for release.\nPublication still needs review of what participants agreed could be shared;')
    p.write_text(text)
    p = root/'protocol/deviations.md'
    text = p.read_text().replace('Excluded from this release scope; neither pending responses nor expert correctness conclusions are supplied.',
        'Anonymised numeric inputs and analyses are staged under expert-study/. Public CSV reproduction passes the supplied checks; original forms/comments/identity keys are excluded. Publication approval remains separate.')
    p.write_text(text)
    p = root/'protocol/software-reproduction.md'
    text = p.read_text().replace('Status: NOT PERFORMED by release-document preparation.',
        'Status: recorded automated reproduction and public expert-input reproduction are included; the final updated software tag still needs approval.')
    text = text.replace('Software commit/tag: pending. Environment lock: pending. Data manifest hash tested:\npending. Commands/logs: pending. Differences/tolerances: pending. Reviewer/date:\npending. Result: pending.',
        'Historical automated reproduction: protocol/reproduction/automated-evaluation/ '
        '(environment.json, commands.json, logs/ and comparison.json): 17 successful stages, '
        '42 byte-identical CSV comparisons. Public expert-input reproduction: '
        'protocol/reproduction/expert-public-input-check.json: 82 checks and 18 matching analysis CSVs.\n\n'
        'Final updated software commit/tag, final archive manifest reviewed, and researcher '
        'reviewer/date/sign-off remain pending. Neither historical execution nor a source hash '
        'alone substitutes for the final release approval.')
    p.write_text(text)
    manifest['public_analysis_update']['finalizer_sha256'] = digest(Path(__file__))
    manifest['public_analysis_update']['public_expert_verification'] = 'protocol/reproduction/expert-public-input-check.json'
    (root/'artifact-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    validation = read(root/'release-validation.json')
    validation['public_analysis_update'] = manifest['public_analysis_update']
    validation['total_evidence_records'] = len(manifest['files'])
    validation['public_expert_reproduction'] = dict(sanity_checks_passed=82,matching_analysis_csvs=18,
                                                  private_workbooks_required=False)
    validation['publication_ready'] = False
    (root/'release-validation.json').write_text(json.dumps(validation,indent=2)+'\n')
    refresh_metadata(root)
    verify_existing(root)
    files = release_files(root)
    invalid = [str(p.relative_to(root)) for p in files
               if DATE.search(p.relative_to(root).as_posix()) or 'reanalysis' in p.relative_to(root).as_posix().lower()]
    if invalid:
        raise ValueError('Date or reanalysis remains in a public filename: ' + str(invalid))
    return dict(files_verified=len(files),manifest_sha256=digest(root/'artifact-manifest.json'),
                public_paths_descriptive=True,publication_approved=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repository', type=Path, required=True)
    parser.add_argument('--apply', action='store_true')
    parser.add_argument('--finalize-with', type=Path, help='Attach a successful public expert verification report and refresh release documents')
    parser.add_argument('--normalise-expert-paths',action='store_true',help='Create the public Astra score derivative with software-relative provenance keys')
    args = parser.parse_args()
    if sum(bool(x) for x in (args.finalize_with,args.apply,args.normalise_expert_paths))>1:
        parser.error('Select only one update action')
    result = (normalise_expert_paths(args.repository) if args.normalise_expert_paths else
              finalize(args.repository,args.finalize_with) if args.finalize_with else prepare(args.repository,args.apply))
    print(json.dumps(result, indent=2))
