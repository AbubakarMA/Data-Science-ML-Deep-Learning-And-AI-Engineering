import json
from pathlib import Path
import pandas as pd
import pytest
from housing.drift import check_drift
from housing.training import split_indices

ROOT = Path(__file__).resolve().parents[1]


def test_shifted_income_triggers_review():
    data = pd.read_csv(ROOT / 'data/housing.csv')
    config = json.loads((ROOT / 'config.json').read_text())
    indices = split_indices(len(data), config)[0]
    training = data.iloc[indices].copy()
    reference = json.loads((ROOT / 'artifacts/drift-reference.json').read_text())
    stable = check_drift(training, reference)
    assert all(x['status'] == 'stable' for x in stable['features'])
    training['median_income'] += 20
    shifted = check_drift(training, reference)
    income = next(x for x in shifted['features'] if x['feature'] == 'median_income')
    assert income['status'] == 'review'


def test_insufficient_drift_sample_rejected():
    with pytest.raises(ValueError, match='100'):
        check_drift(pd.DataFrame({'median_income': [1, 2]}), {})
