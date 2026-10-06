"""Prepare and verify a non-expert research-data release candidate (no API calls).

Original archived artifacts are immutable. Generated catalogues are machine-readable
JSON/CSV exports, not edited spreadsheets. Publication is deliberately not performed.
"""
import argparse
from collections import Counter, defaultdict
import csv
import gzip
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parents[1]
DATE = '2026-10-06'
SECRET = re.compile(rb'(?:sk-(?:proj-|svcacct-)?[A-Za-z0-9_-]{24,}|gh[pousr]_[A-Za-z0-9]{30,}|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----)')


def digest(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def read(p):
    return json.loads(p.read_text())


def write(root, name, value):
    p = root / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value, indent=2) + '\n' if not isinstance(value, str) else value)


def csv_export(root, name, rows, fields):
    p = root / name
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def release_files(root):
    return sorted(p for p in root.rglob('*') if p.is_file()
                  and not any(part in ('.git', 'bundles', 'reorganisation-backup', 'release-preparation-backup', 'expert-study')
                              for part in p.relative_to(root).parts)
                  and not p.name.startswith('~$') and p.name != '.DS_Store')


def normal_source(value):
    marker = 'feature-model-automation/'
    return value.split(marker, 1)[1] if marker in value else value


def refresh_metadata(root):
    manifest = read(root / 'artifact-manifest.json')
    original_paths = {r['path'] for r in manifest['files']}
    manifest['derived_files'] = [dict(path=p.relative_to(root).as_posix(), bytes=p.stat().st_size,
        sha256=digest(p), provenance='non-expert release preparation; see recovery and lineage reports')
        for p in release_files(root) if p.relative_to(root).as_posix() not in original_paths
        and p.name not in ('artifact-manifest.json', 'checksums.sha256')]
    manifest['release_preparer_sha256'] = digest(Path(__file__))
    write(root, 'artifact-manifest.json', manifest)
    write(root, 'checksums.sha256', ''.join(f'{digest(p)}  {p.relative_to(root).as_posix()}\n'
          for p in release_files(root) if p.name != 'checksums.sha256'))


def recover_prompts(root):
    report = read(root/'release-validation.json')
    missing = {r['prompt_hash'] for r in report['prompt_coverage'] if not r['prompt_snapshot_present']}
    revisions = subprocess.check_output(['git','log','--format=%H','--all','--',
                                        'prompts/fm_prompt_template.txt'],cwd=ROOT,text=True).split()
    restored=[]
    for rev in revisions:
        data=subprocess.check_output(['git','show',rev+':prompts/fm_prompt_template.txt'],cwd=ROOT)
        h=hashlib.sha256(data).hexdigest()
        if h not in missing:continue
        rel='protocol/prompt-snapshots/versions/'+h+'/fm_prompt_template.txt'
        p=root/rel;p.parent.mkdir(parents=True,exist_ok=True)
        if p.exists() and p.read_bytes()!=data:raise ValueError('Historical snapshot collision')
        p.write_bytes(data)
        restored.append(dict(path=rel,sha256=h,git_commit=rev,git_path='prompts/fm_prompt_template.txt'))
        missing.remove(h)
    recovered_hashes={r['sha256'] for r in restored}
    for r in report['prompt_coverage']:
        if r['prompt_hash'] in recovered_hashes:r['prompt_snapshot_present']=True
    lineage=read(root/'protocol/analysis-lineage.json');lineage['prompt_coverage']=report['prompt_coverage']
    write(root,'protocol/analysis-lineage.json',lineage)
    write(root,'release-validation.json',report)
    write(root,'protocol/prompt-recovery.json',dict(recovered=restored,still_missing=sorted(missing),
        search_scope='all local git revisions touching prompts/fm_prompt_template.txt; no reconstruction from prose',
        consolidated_pilot_snapshot='checked separately; matches current main prompt, not missing variants'))
    refresh_metadata(root)
    print('Recovered prompts:',len(restored),'Still missing:',len(missing),flush=True)


def prepare(root):
    manifest = read(root / 'artifact-manifest.json')
    original = [r for r in manifest['files'] if r.get('source')]
    for r in original:
        if digest(root / r['path']) != r['sha256']:
            raise ValueError('Archived artifact changed: ' + r['path'])
    print(f'Verified {len(original)} original artifacts', flush=True)
    backup = root / 'protocol/release-preparation-backup'
    if backup.exists():
        raise FileExistsError('Preparation already started; inspect existing backup before rerunning')
    backup.mkdir(parents=True)
    for name in ('README.md', 'DATA_DICTIONARY.md', 'CHANGELOG.md', 'RELEASE_CHECKLIST.md',
                 '.gitignore', 'artifact-manifest.json', 'checksums.sha256'):
        shutil.copy2(root / name, backup / name)
    bysource = {r['source']: r for r in original}
    current = {}
    failures = []
    for p in sorted((root / 'experiments/main-campaign').rglob('run_meta.json')):
        m = read(p); key = (m['config']['corpus'], m['run_id'])
        if key in current:
            raise ValueError('Duplicate current run ' + str(key))
        current[key] = (p, m)
        final = p.parent / 'fm_gen.xml'
        if bool(m['completed']) != final.exists():
            failures.append('Completion/final XML mismatch: ' + m['run_id'])
    matrix_ids = []
    for lane in ('open_weight', 'astra'):
        matrix = read(root / f'protocol/campaign-matrices/frozen/run_matrix_{lane}.json')
        matrix_ids += [(r['corpus'], hashlib.sha256(json.dumps(r, sort_keys=True,
                        ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()[:16])
                       for r in matrix['runs']]
    if len(matrix_ids) != len(set(matrix_ids)) or set(matrix_ids) != set(current):
        failures.append('Frozen matrix IDs do not exactly match current run IDs')
    inventory = read(root / 'analysis/inventory/inventory-current-v1/runs.json')
    if {(r['corpus'], r['run_id']) for r in inventory} != set(current):
        failures.append('Inventory does not cover the current matrix')
    # Index all existing archived snapshots. UUID order is not execution chronology.
    attempts = []
    for d in sorted((root / 'recovery/archived-attempts').glob('*/*/*/*')):
        if not d.is_dir():
            continue
        campaign, corpus, rid, aid = d.parts[-4:]
        p = d / 'run_meta.json'; m = read(p) if p.exists() else {}
        pair = current.get((corpus, rid)); now = pair[1] if pair else {}
        if m and (m.get('run_id') != rid or m['config']['corpus'] != corpus):
            failures.append('Recovery identity mismatch: ' + str(d.relative_to(root)))
        attempts.append(dict(campaign_id=campaign, corpus=corpus, run_id=rid,
            archive_snapshot_id=aid, archived_metadata=str(p.relative_to(root)) if p.exists() else '',
            archived_started_at_utc=m.get('started_at_utc'), archived_ended_at_utc=m.get('ended_at_utc'),
            archived_terminal_status=m.get('terminal_status'), archived_completed_steps=m.get('completed_steps'),
            archived_failed_step=m.get('failed_step'), archived_metadata_sha256=digest(p) if p.exists() else '',
            current_metadata=str(pair[0].relative_to(root)) if pair else '',
            current_terminal_status=now.get('terminal_status'), current_completed=now.get('completed'),
            link_basis='campaign/corpus/run_id',
            recovery_history_json=json.dumps(now.get('recovery_history', []), sort_keys=True)))
        if not pair:
            failures.append('Recovery snapshot without current run: ' + rid)
    if attempts:
        csv_export(root, 'recovery/attempt-index.csv', attempts, list(attempts[0]))
    write(root, 'recovery/README.md', '''# Recovery records

attempt-index.csv contains one row per existing archived snapshot directory, not
one independent experimental repetition. It links to current metadata by campaign,
corpus and run_id. Snapshot UUIDs are not chronological ordering. UTC fields come
from stored run metadata; no archive-creation time is inferred from file timestamps.
Repeated snapshots can represent checkpoint backups rather than fresh model calls.
recovery_history_json preserves recorded resume events; an empty list does not prove
no restart occurred. Missing values mean not recorded. Never sum duplicated step
usage across checkpoint snapshots as independent expenditure.
''')
    # Preserve machine-readable provenance and unresolved labels, without endorsing them.
    lineage = []
    for p in sorted((root / 'analysis').rglob('summary.json')):
        s = read(p)
        lineage.append(dict(path=str(p.relative_to(root)), sha256=digest(p),
            recorded_publication_ready=s.get('publication_ready'), unresolved=s.get('unresolved', []),
            implementation_sha256=s.get('implementation_sha256', {}),
            source_sha256=s.get('source_sha256', {})))
    checks = []
    aggregate = read(root / 'analysis/aggregate/aggregate-current-v4/summary.json')
    for tag, source in aggregate['sources'].items():
        rec = bysource.get(source)
        ok = bool(rec and rec['sha256'] == aggregate['source_sha256'][tag])
        checks.append(dict(kind='aggregate_input', source=source, matches=ok))
        if not ok:
            failures.append('Aggregate input hash mismatch: ' + source)
    figure = read(root / 'paper/figures/paper-figures-v4/figure_manifest.json')
    for tag, info in figure['inputs'].items():
        source = normal_source(info['path']); rec = bysource.get(source)
        ok = bool(rec and rec['sha256'] == info['sha256'])
        checks.append(dict(kind='figure_input', source=source, matches=ok))
        if not ok:
            failures.append('Figure input hash mismatch: ' + source)
    for name, h in figure['outputs'].items():
        if digest(root / 'paper/figures/paper-figures-v4' / name) != h:
            failures.append('Figure output mismatch: ' + name)
    # Compare prompt/schema identifiers with available bytes; missing pilot versions stay visible.
    hashes = {r['sha256'] for r in original if r['path'].startswith('protocol/prompt-snapshots/')}
    prompt_coverage = []
    for population, folder in [('main', 'experiments/main-campaign'), ('pilots', 'experiments/pilots'),
                               ('astra_extension', 'experiments/expert-astra-extension')]:
        counter = Counter()
        for p in (root / folder).rglob('run_meta.json'):
            c = read(p).get('config', {})
            counter[(c.get('prompt_template_hash', ''), c.get('metamodel_hash', ''))] += 1
        for (prompt, schema), n in sorted(counter.items()):
            prompt_coverage.append(dict(population=population, runs=n, prompt_hash=prompt,
                schema_hash=schema, prompt_snapshot_present=bool(prompt and any(h.startswith(prompt) for h in hashes)),
                schema_snapshot_present=bool(schema and any(h.startswith(schema) for h in hashes))))
    write(root, 'protocol/analysis-lineage.json', dict(summaries=lineage, input_checks=checks, prompt_coverage=prompt_coverage))
    # Exact run-level outcome table, not a guessed replacement for manuscript tables.
    counts = defaultdict(Counter)
    for p, m in current.values():
        c = m['config']; key = (c['model_id'], c.get('extra', {}).get('arm', 'unknown'), c['corpus'])
        counts[key]['planned'] += 1
        counts[key]['completed' if m['completed'] else 'not_completed'] += 1
    table = [dict(model_id=k[0], arm=k[1], corpus=k[2], planned=v['planned'],
                  completed=v['completed'], not_completed=v['not_completed']) for k,v in sorted(counts.items())]
    csv_export(root, 'paper/tables/campaign-outcomes.csv', table, list(table[0]))
    write(root, 'paper/tables/table-manifest.json', dict(status='release-support table, not manuscript table-number assignment',
        tables=[dict(path='paper/tables/campaign-outcomes.csv',
                     source='experiments/main-campaign/**/run_meta.json',
                     aggregation='count by model_id, config.extra.arm and corpus; completed boolean',
                     generator='scripts/prepare_data_release.py', sha256=digest(root/'paper/tables/campaign-outcomes.csv'))],
        existing_table_sources=['analysis/aggregate/aggregate-current-v4/cell_summary.csv',
          'analysis/matching-sensitivity/matching-comparison-v2/', 'analysis/statistical-tests/'],
        limitation='Final manuscript was not supplied; table numbering/rounding and claim reconciliation require author review.'))
    write(root, 'paper/tables/README.md', '''# Tables

campaign-outcomes.csv is a release-support table of all planned main-campaign run
outcomes. It counts completion, not conformance. table-manifest.json records its
derivation and points to other tabular results. Final manuscript table numbers,
filters and rounding are not invented: reconcile them with the final paper.
''')
    # Header catalogue: no values are coerced and large pair matrices are not loaded.
    schemas = []
    for rec in original:
        p = root / rec['path']
        if p.suffix != '.csv':
            continue
        delimiter = ';' if '/corpus-manifests/' in rec['path'] else ','
        with p.open(encoding='utf-8-sig', newline='') as f:
            header = next(csv.reader(f, delimiter=delimiter), [])
        schemas.append(dict(path=rec['path'], sha256=rec['sha256'], delimiter=delimiter, columns=header))
    write(root, 'protocol/tabular-schemas.json', schemas)
    absolute_paths = []; suspects = []
    for rec in original:
        p = root / rec['path']
        if p.suffix.lower() not in ('.json', '.jsonl', '.csv', '.txt', '.xml', '.md', '.yaml'):
            continue
        personal = False; secret = False; tail = b''
        with p.open('rb') as f:
            for b in iter(lambda: f.read(1024*1024), b''):
                joined = tail+b
                personal |= b'/Users/' in joined or b'/home/' in joined
                secret |= bool(SECRET.search(joined))
                tail = b[-256:]
        if personal: absolute_paths.append(rec['path'])
        if secret: suspects.append(rec['path'])
    if suspects:
        failures.append('Potential credentials detected; inspect findings before bundling')
    write(root, 'protocol/privacy-scan.json', dict(scope='pattern scan of archived text; not a complete privacy/rights audit',
        suspected_credentials=suspects, files_with_personal_absolute_paths=absolute_paths,
        redactions_performed=False, expert_data_included=False))
    report = dict(date=DATE, artifact_integrity='all original hashes verified',
        original_files=len(original), main_runs=len(current), completed=sum(m['completed'] for _,m in current.values()),
        archived_snapshots=len(attempts), recovery_run_ids=len({r['run_id'] for r in attempts}),
        matrix_inventory_check='passed' if not failures else 'inspect failures',
        failures=failures, input_checks=checks, prompt_coverage=prompt_coverage,
        publication_ready=False, remaining_gates=[
            'Approved authorship, dataset licence and third-party rights',
            'Privacy review of preserved personal paths and generated quotations',
            'Expert blinding/publication timing approval; expert analysis excluded',
            'Final manuscript table/claim reconciliation',
            'Pinned software release and clean-environment reproduction test',
            'Historical prompt versions absent from snapshot catalogue, if any; disclose rather than regenerate',
            'Author approval of strict-admissibility scope and remaining methodological decisions'])
    write(root, 'release-validation.json', report)
    # Human documentation is kept in version-controlled templates in the software repo.
    templates = ROOT / 'docs/data-release'
    for p in templates.rglob('*.md'):
        write(root, p.relative_to(templates).as_posix(), p.read_text())
    with (root / 'CHANGELOG.md').open('a') as f:
        f.write('\n## Non-expert release preparation — 2026-10-06\n\n'
                '- Added recovery index, run-outcome table, schema catalogue, lineage and privacy checks.\n'
                '- Original scientific artifacts remain byte-identical; no metric rerun or expert analysis.\n')
    with (root / '.gitignore').open('a') as f:
        f.write('\n/protocol/release-preparation-backup/\n')
    original_paths = {r['path'] for r in original}
    derived = [dict(path=p.relative_to(root).as_posix(), bytes=p.stat().st_size, sha256=digest(p),
                    provenance='non-expert release preparation; not original experiment output')
               for p in release_files(root) if p.relative_to(root).as_posix() not in original_paths
               and p.name not in ('artifact-manifest.json', 'checksums.sha256')]
    manifest['derived_files'] = derived
    manifest['release_scope'] = 'automated experiments and evaluation only; expert assessment excluded'
    manifest['release_preparer_sha256'] = digest(Path(__file__))
    write(root, 'artifact-manifest.json', manifest)
    checksummed = [p for p in release_files(root) if p.name != 'checksums.sha256']
    write(root, 'checksums.sha256', ''.join(f'{digest(p)}  {p.relative_to(root).as_posix()}\n' for p in checksummed))
    print(json.dumps(report, indent=2), flush=True)
    return not failures


def bundles(root):
    out = root / 'bundles/nonexpert-candidate-2026-10-06'
    if out.exists():
        raise FileExistsError(out)
    # Check the checksum list before packaging; no silent use of edited data.
    expected = {}
    for line in (root/'checksums.sha256').read_text().splitlines():
        h,n = line.split('  ',1)
        if digest(root/n) != h:raise ValueError('Stale checksum: '+n)
        expected[n]=h
    expected['checksums.sha256']=digest(root/'checksums.sha256')
    files=release_files(root)
    if {p.relative_to(root).as_posix() for p in files} != set(expected):
        raise ValueError('Unmanifested release files; regenerate checksums first')
    if read(root/'release-validation.json')['failures']:
        raise ValueError('Technical validation failures must be resolved before bundling')
    out.mkdir(parents=True)
    groups=defaultdict(list)
    for p in files:
        rel=p.relative_to(root)
        group=rel.parts[0] if rel.parts[0] in ('experiments','analysis','recovery','paper') else 'metadata-inputs'
        groups[group].append(p)
    result=[]
    for group, paths in sorted(groups.items()):
        archive=out/(group+'.tar.gz')
        with archive.open('wb') as raw, gzip.GzipFile(filename='',mode='wb',fileobj=raw,mtime=0,compresslevel=6) as gz:
            with tarfile.open(fileobj=gz,mode='w|') as tar:
                for p in paths:
                    name=p.relative_to(root).as_posix()
                    info=tar.gettarinfo(str(p),arcname=name)
                    info.uid=info.gid=info.mtime=0;info.uname=info.gname='';info.mode=0o644
                    with p.open('rb') as f:tar.addfile(info,f)
        seen=set()
        with tarfile.open(archive,'r|gz') as tar:
            for member in tar:
                if not member.isfile() or member.name not in expected:raise ValueError('Unexpected archive entry')
                h=hashlib.sha256();f=tar.extractfile(member)
                for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
                if h.hexdigest()!=expected[member.name]:raise ValueError('Bundle checksum mismatch')
                seen.add(member.name)
        if seen!={p.relative_to(root).as_posix() for p in paths}:raise ValueError('Missing bundle entries')
        result.append(dict(file=archive.name,bytes=archive.stat().st_size,sha256=digest(archive),files=len(paths)))
        print('Built and verified '+archive.name,flush=True)
    write(out,'bundle-manifest.json',dict(status='local candidate; NOT publication-approved',
        scope='non-expert; excludes backups, expert-study and Git',
        artifact_manifest_sha256=digest(root/'artifact-manifest.json'),bundles=result))
    write(out,'SHA256SUMS',''.join(f"{r['sha256']}  {r['file']}\n" for r in result)+
          f"{digest(out/'bundle-manifest.json')}  bundle-manifest.json\n")


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--repository',type=Path,required=True)
    ap.add_argument('--bundle-only',action='store_true')
    ap.add_argument('--recover-prompts',action='store_true')
    args=ap.parse_args()
    if args.recover_prompts:recover_prompts(args.repository.resolve())
    elif args.bundle_only:bundles(args.repository.resolve())
    else:prepare(args.repository.resolve())
