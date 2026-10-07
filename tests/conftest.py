"""Self-contained test inputs; never depend on private processed corpus files."""
import json
import shutil
import pytest


@pytest.fixture(autouse=True)
def synthetic_chunk_hashes(monkeypatch, request, tmp_path):
    if request.module.__name__.split('.')[-1] not in {
        'test_campaign_matrix', 'test_campaign_runner', 'test_provider_recovery',
        'test_analysis_followup',
    }:
        return
    from scripts import build_run_matrix as matrix
    root = tmp_path / 'matrix-inputs'
    for name in ['config/experiment.yaml', 'prompts/fm_prompt_template.txt',
                 'prompts/feature-model-schema.xsd', 'data/orderings.json',
                 'data/encoder_versions.txt']:
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(matrix.REPO / name, target)
    orderings = json.loads((root / 'data/orderings.json').read_text())
    for corpus in ('repair', 'federation'):
        target = root / f'data/processed/{corpus}/chunks.jsonl'
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(''.join(json.dumps({'doc_id': doc, 'text': 'synthetic',
            'chunk_id': f'{doc}-{i}'}) + '\n'
            for doc in orderings[corpus]['primary']['order'] for i in range(4)))
    monkeypatch.setattr(matrix, 'REPO', root)
