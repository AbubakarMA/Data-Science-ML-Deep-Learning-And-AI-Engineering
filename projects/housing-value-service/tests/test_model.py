import copy
import json
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from housing.artifact import Predictor
from housing.preprocessing import Preprocessor
from housing.scratch import ScratchRegressor, objective_and_gradients
from housing.training import split_indices, EXPECTED_DATA_SHA

ROOT = Path(__file__).resolve().parents[1]


def test_gradients_match_finite_differences():
    x = np.array([[1., 2.], [-1., 0.], [3., -2.]])
    y = np.array([2., 1., -1.])
    w, b, l2, eps = np.array([0.4, -0.3]), 0.2, 0.01, 1e-6
    _, dw, db = objective_and_gradients(x, y, w, b, l2)
    for i in range(len(w)):
        plus, minus = w.copy(), w.copy()
        plus[i] += eps; minus[i] -= eps
        numeric = (objective_and_gradients(x, y, plus, b, l2)[0] -
                   objective_and_gradients(x, y, minus, b, l2)[0]) / (2 * eps)
        assert dw[i] == pytest.approx(numeric, rel=1e-6)
    numeric_b = (objective_and_gradients(x, y, w, b + eps, l2)[0] -
                 objective_and_gradients(x, y, w, b - eps, l2)[0]) / (2 * eps)
    assert db == pytest.approx(numeric_b, rel=1e-6)


def test_optimizer_matches_independent_closed_form():
    rng = np.random.default_rng(7)
    x = rng.normal(size=(80, 3)); y = x @ np.array([2., -1., 0.5]) + 3
    model = ScratchRegressor(l2=0.02).fit(x, y)
    design = np.column_stack([x, np.ones(len(x))])
    penalty = np.diag([len(x) * .02] * 3 + [0])
    expected = np.linalg.solve(design.T @ design + penalty, design.T @ y)
    assert np.allclose(model.weights_, expected[:3], atol=1e-6)
    assert model.bias_ == pytest.approx(expected[-1], abs=1e-6)


def test_preprocessing_is_fitted_only_on_training_data():
    data = pd.read_csv(ROOT / 'data/housing.csv')
    prep = Preprocessor().fit(data.iloc[:1000])
    state_before = prep.to_dict()
    unseen = data.iloc[1000:1002].copy()
    unseen['total_bedrooms'] = np.nan
    unseen['median_income'] = 1e6
    result = prep.transform(unseen)
    assert np.isfinite(result).all()
    assert prep.to_dict() == state_before
    assert prep.medians_[4] == pytest.approx(data.iloc[:1000]['total_bedrooms'].median())


def test_artifact_roundtrip_and_tamper_rejection(tmp_path):
    payload = json.loads((ROOT / 'artifacts/model.json').read_text())
    records = json.loads((ROOT / 'examples/request.json').read_text())['records']
    path = tmp_path / 'model.json'; path.write_text(json.dumps(payload))
    assert np.array_equal(Predictor(payload).predict(records), Predictor.load(path).predict(records))
    changed = copy.deepcopy(payload); changed['state']['bias'] += 1
    with pytest.raises(ValueError, match='checksum'):
        Predictor(changed)


def test_reproducible_nonoverlapping_splits():
    config = json.loads((ROOT / 'config.json').read_text())
    train, validation, test = split_indices(20640, config)
    assert [len(x) for x in [train, validation, test]] == [12384, 4128, 4128]
    assert len(set(train) | set(validation) | set(test)) == 20640
    assert not set(train) & set(validation)
    assert not set(train) & set(test)
    assert not set(validation) & set(test)
    assert np.array_equal(train, split_indices(20640, config)[0])


def test_released_model_passed_quality_gate():
    payload = json.loads((ROOT / 'artifacts/model.json').read_text())
    assert payload['state']['data_sha256'] == EXPECTED_DATA_SHA
    assert all(payload['evaluation']['validation_gate'].values())
    val = payload['evaluation']['validation']
    assert val['scratch_ridge']['rmse_usd'] < val['mean_baseline']['rmse_usd'] * .8


def test_released_artifact_matches_training_source():
    payload = json.loads((ROOT / 'artifacts/model.json').read_text())
    names = ['preprocessing.py', 'scratch.py', 'training.py', 'artifact.py']
    digest = hashlib.sha256(b''.join((ROOT / 'housing' / name).read_bytes() for name in names)).hexdigest()
    assert payload['source_sha256'] == digest


def test_invalid_training_data_rejected():
    with pytest.raises(ValueError):
        ScratchRegressor().fit(np.array([[np.inf]]), np.array([1.]))
