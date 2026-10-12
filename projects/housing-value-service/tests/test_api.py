import json
from pathlib import Path
from fastapi.testclient import TestClient
import pytest
from housing.api import create_app

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def example():
    return json.loads((ROOT / 'examples/request.json').read_text())


@pytest.fixture
def client(monkeypatch):
    monkeypatch.delenv('API_KEY', raising=False)
    with TestClient(create_app(ROOT / 'artifacts/model.json')) as c:
        yield c


def test_health_and_readiness(client):
    assert client.get('/health').json()['status'] == 'alive'
    assert client.get('/ready').json()['status'] == 'ready'


def test_prediction_trace_matches_calculation(client, example):
    example['include_trace'] = True
    response = client.post('/v1/predict', json=example)
    assert response.status_code == 200
    body = response.json()
    trace = body['trace_first_record']
    summed = trace['bias_usd'] + sum(x['contribution_usd'] for x in trace['features'])
    assert summed == pytest.approx(trace['raw_value_usd'], abs=1e-7)
    assert body['predictions'][0]['value'] == pytest.approx(max(summed, 0), abs=.01)
    assert body['unit'] == 'USD'
    assert response.headers['x-request-id'] == body['request_id']
    assert float(response.headers['x-response-time-ms']) >= 0


@pytest.mark.parametrize('field,value', [('longitude', 0), ('latitude', 100), ('households', 0),
    ('housing_median_age', -1), ('median_income', 'NaN'), ('ocean_proximity', 'ACCRA'),
    ('total_bedrooms', 999999), ('median_house_value', 100000)])
def test_invalid_records_rejected(client, example, field, value):
    example['records'][0][field] = value
    assert client.post('/v1/predict', json=example).status_code == 422


@pytest.mark.parametrize('size', [0, 51])
def test_batch_limit(client, example, size):
    assert client.post('/v1/predict', json={'records': example['records'] * size}).status_code == 422


def test_missing_bedrooms_are_imputed(client, example):
    example['records'][0]['total_bedrooms'] = None
    r = client.post('/v1/predict', json=example)
    assert r.status_code == 200
    assert 'imputed' in r.json()['predictions'][0]['warnings'][0]


def test_authentication_when_configured(client, example, monkeypatch):
    monkeypatch.setenv('API_KEY', 'test-secret')
    assert client.post('/v1/predict', json=example).status_code == 401
    assert client.get('/metrics').status_code == 401
    assert client.post('/v1/predict', json=example, headers={'X-API-Key': 'test-secret'}).status_code == 200
    assert client.get('/health').status_code == 200


def test_metrics_count_served_predictions(client, example):
    client.post('/v1/predict', json=example)
    text = client.get('/metrics').text
    assert 'housing_predictions_total 1' in text
    assert 'path="/v1/predict",status="200"' in text


def test_web_demo_and_model_metadata(client):
    assert 'Housing Value Lab' in client.get('/').text
    metadata = client.get('/v1/model').json()
    assert metadata['target']['unit'] == 'USD'
    assert metadata['history']
    assert client.get('/openapi.json').status_code == 200


def test_missing_artifact_prevents_startup(tmp_path):
    with pytest.raises(FileNotFoundError):
        with TestClient(create_app(tmp_path / 'missing.json')):
            pass


def test_oversized_body_rejected(client):
    assert client.post('/v1/predict', content=b'x' * 65537).status_code == 413
