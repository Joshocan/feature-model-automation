"""Offline tests: HTTP retries and preservation of recovered attempts."""
import json
from unittest.mock import Mock

import pytest
import requests

from fame.generation.llm_client import OpenAILLM, OllamaCloudLLM, GenerationRequest
from fame.generation.persistence import RunPaths
from scripts.campaign import (_mark_completed, _archive_provider_error,
                              _run_config_from_row, _recoverable_provider_error,
                              RunAlreadyExists)
from scripts.build_run_matrix import build_matrix


def response(code):
    r = requests.Response()
    r.status_code = code
    r._content = (b'{"choices":[{"message":{"content":"<x/>"},"finish_reason":"stop"}]}'
                  if code == 200 else b'gateway error')
    return r


@pytest.mark.parametrize('code', [408, 429, 500, 502, 503, 504])
def test_transient_retries(monkeypatch, code):
    post = Mock(side_effect=[response(code), response(200)])
    monkeypatch.setattr('fame.generation.llm_client.requests.post', post)
    monkeypatch.setattr('fame.generation.llm_client.time.sleep', lambda _: None)
    result = OpenAILLM('test').generate(GenerationRequest('test', 100))
    assert post.call_count == 2
    assert result.raw['fame_provider_attempts'] == 2
    assert len(result.raw['fame_retry_events']) == 1


@pytest.mark.parametrize('code,attempts', [(400, 1), (401, 1), (502, 3)])
def test_bounded_retries(monkeypatch, code, attempts):
    post = Mock(return_value=response(code))
    monkeypatch.setattr('fame.generation.llm_client.requests.post', post)
    monkeypatch.setattr('fame.generation.llm_client.time.sleep', lambda _: None)
    with pytest.raises(requests.HTTPError) as exc:
        OpenAILLM('test').generate(GenerationRequest('test', 100))
    assert post.call_count == attempts
    assert exc.value.fame_attempts == attempts


@pytest.mark.parametrize('lane,error', [
    ('astra', 'RuntimeError: OpenAI API HTTP 502: non-JSON error response'),
    ('open_weight', 'ChunkedEncodingError: Connection broken: IncompleteRead'),
])
def test_recovery_preserves_original_and_excludes_other_outcomes(tmp_path, lane, error):
    rows = build_matrix('test-recovery', only_enabled=True, lane=lane)['runs'][:5]
    statuses = ['provider_error', 'malformed_xml', 'truncated_output', 'completed', None]
    original = None
    for row, status in zip(rows, statuses):
        if status is None:
            continue
        cfg = _run_config_from_row(row)
        paths = RunPaths.for_run(results_root=tmp_path, campaign_id=row['campaign_id'],
                                corpus=row['corpus'], config_hash=cfg.config_hash(),
                                run_id=cfg.run_id())
        paths.ensure_dirs()
        paths.run_meta.write_text(json.dumps({'terminal_status': status,
            'completed': status == 'completed',
            'steps': [{'error': error}]}))
        if status == 'provider_error':
            original = paths
            (paths.root / 'evidence.txt').write_text('preserve me')
    _mark_completed(rows, tmp_path, retry_provider_errors=True)
    assert [r['_completed'] for r in rows] == [False, True, True, True, True]
    archive = _archive_provider_error(rows[0], tmp_path)
    assert (archive / 'evidence.txt').read_text() == 'preserve me'
    assert not original.root.exists()
    with pytest.raises(RunAlreadyExists):
        _archive_provider_error(rows[1], tmp_path)
    assert not _recoverable_provider_error({'terminal_status': 'provider_error',
        'steps': [{'error': 'OpenAI API HTTP 429: quota exceeded'}]})


@pytest.mark.parametrize('exhausted', [False, True])
def test_ollama_interrupted_response_retry(monkeypatch, exhausted):
    broken = requests.exceptions.ChunkedEncodingError('Connection broken: IncompleteRead')
    ok = response(200)
    ok._content = b'{"response":"<x/>","done_reason":"stop","eval_count":4}'
    post = Mock(side_effect=[broken, broken, broken] if exhausted else [broken, ok])
    monkeypatch.setattr('fame.generation.llm_client.requests.post', post)
    monkeypatch.setattr('fame.generation.llm_client.time.sleep', lambda _: None)
    client = OllamaCloudLLM(model_id='glm-5.3-flash:cloud')
    request = GenerationRequest('test', 100)
    if exhausted:
        with pytest.raises(requests.exceptions.ChunkedEncodingError) as exc:
            client.generate(request)
        assert exc.value.fame_attempts == 3
        assert len(exc.value.fame_retry_events) == 2
        assert post.call_count == 3
    else:
        result = client.generate(request)
        assert result.text == '<x/>'
        assert result.raw['fame_provider_attempts'] == 2
        assert result.raw['fame_retry_events'][0]['error_type'] == 'ChunkedEncodingError'
        assert post.call_count == 2


@pytest.mark.parametrize('error', [
    'RuntimeError: text mentioning ChunkedEncodingError: is not a transport exception',
    'OSError: [Errno 28] No space left on device',
    'RuntimeError: HTTP 401: authentication failed',
])
def test_unrelated_errors_not_selected(error):
    assert not _recoverable_provider_error({'terminal_status': 'provider_error',
                                           'steps': [{'error': error}]})
