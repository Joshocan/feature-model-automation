import hashlib
import json

from scripts.update_public_research_archive import public_path, public_astra_csv, csv_bytes
from scripts.prepare_data_release import release_files
from scripts.restore_research_data import restore


def test_descriptive_paths_keep_run_ids_and_versions():
    assert public_path('experiments/pilots/pilot-2026-09-22-glm/run01/fm_gen.xml') == 'experiments/pilots/pilot-glm/run01/fm_gen.xml'
    assert public_path('analysis/reanalysis_2026-10-15-final/task1.csv') == 'analysis/verification-final/task1.csv'
    assert public_path('analysis/semantic-current-v3/metrics.csv') == 'analysis/semantic-current-v3/metrics.csv'


def test_astra_derivative_changes_only_provenance_path_keys(tmp_path):
    import csv
    import io
    path=tmp_path/'source.csv'
    mapping={'/Users/private/dev/feature-model-automation/results/run.xml':'abc123'}
    row=dict(f1='0.682',input_sha256=json.dumps(mapping),evaluation_source_sha256=json.dumps(mapping))
    original=csv_bytes([row]);path.write_bytes(original)
    derivative=list(csv.DictReader(io.StringIO(public_astra_csv(path).decode())))[0]
    assert derivative['f1']==row['f1']
    assert json.loads(derivative['input_sha256'])=={'results/run.xml':'abc123'}
    assert '/Users/' not in str(derivative)
    assert path.read_bytes()==original


def test_only_allowlisted_expert_files_packaged(tmp_path):
    (tmp_path/'expert-study').mkdir()
    (tmp_path/'protocol').mkdir()
    (tmp_path/'expert-study/private.xlsx').write_bytes(b'private')
    (tmp_path/'expert-study/ratings.csv').write_text('rater,score\nExpert A,3\n')
    assert not any(p.parent.name == 'expert-study' for p in release_files(tmp_path))
    (tmp_path/'protocol/public-expert-files.json').write_text(json.dumps({'files':['expert-study/ratings.csv']}))
    assert [p.name for p in release_files(tmp_path) if p.parent.name == 'expert-study'] == ['ratings.csv']


def test_identical_evidence_restores_to_both_source_locations(tmp_path):
    archive=tmp_path/'archive';archive.mkdir()
    payload=b'x,y\n1,2\n'
    (archive/'scores.csv').write_bytes(payload)
    h=hashlib.sha256(payload).hexdigest()
    manifest={'files':[{'path':'scores.csv','source':'results/original.csv',
        'restore_sources':['results/reproduced.csv'],'sha256':h}]}
    raw=json.dumps(manifest).encode()
    (archive/'artifact-manifest.json').write_bytes(raw)
    (archive/'checksums.sha256').write_text(f'{h}  scores.csv\n{hashlib.sha256(raw).hexdigest()}  artifact-manifest.json\n')
    dest=tmp_path/'restored'
    result=restore(archive,dest,apply=True)
    assert result['restorable_data_files']==2
    assert (dest/'results/original.csv').read_bytes()==payload
    assert (dest/'results/reproduced.csv').read_bytes()==payload
