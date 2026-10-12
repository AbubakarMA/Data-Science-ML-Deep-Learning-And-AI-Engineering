"""Check a running HTTP service without third-party HTTP packages."""

import argparse
import json
import time
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]


def smoke(base_url, api_key=None, expected_version=None):
    url = base_url.rstrip('/')
    with urlopen(url + '/ready', timeout=10) as response:
        assert response.status == 200
        ready = json.load(response)
    expected_version = expected_version or json.loads((ROOT / 'artifacts/model.json').read_text())['model_version']
    assert ready['model_version'] == expected_version, 'The server is not serving the expected release'
    body = (ROOT / 'examples/request.json').read_bytes()
    headers = {'Content-Type': 'application/json'}
    if api_key:
        headers['X-API-Key'] = api_key
    started = time.perf_counter()
    with urlopen(Request(url + '/v1/predict', data=body, headers=headers), timeout=10) as response:
        result = json.load(response)
        assert response.status == 200
        assert response.headers['X-Request-ID'] == result['request_id']
    assert result['model_version'] == ready['model_version']
    assert result['unit'] == 'USD'
    assert len(result['predictions']) == 1
    assert result['predictions'][0]['value'] >= 0
    print(json.dumps({'ready': ready, 'prediction': result['predictions'][0],
                      'client_latency_ms': round((time.perf_counter() - started) * 1000, 2)}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('base_url', nargs='?', default='http://127.0.0.1:8000')
    parser.add_argument('--api-key')
    parser.add_argument('--expected-version')
    args = parser.parse_args()
    smoke(args.base_url, args.api_key, args.expected_version)
