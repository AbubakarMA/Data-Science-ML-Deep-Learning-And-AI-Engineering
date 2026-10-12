"""Data audit, deterministic splits, training, validation gate and release."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from housing.artifact import state_digest
from housing.preprocessing import Preprocessor, RAW_FEATURES
from housing.scratch import ScratchRegressor

ROOT = Path(__file__).resolve().parents[1]
TARGET = "median_house_value"
EXPECTED_DATA_SHA = "8a3727f4cf54ac1a327f69b1d5b4db54c5834ea81c6e4efc0d163300022a685e"


def metrics(y, predictions):
    errors = np.maximum(predictions, 0) - y
    mse = float(np.mean(errors ** 2))
    denominator = float(np.sum((y - y.mean()) ** 2))
    return {"mae_usd": float(np.mean(np.abs(errors))), "rmse_usd": float(np.sqrt(mse)),
            "r2": float(1 - np.sum(errors ** 2) / denominator)}


def split_indices(n, config):
    indices = np.random.default_rng(config["random_seed"]).permutation(n)
    a = int(n * config["train_fraction"])
    b = a + int(n * config["validation_fraction"])
    return indices[:a], indices[a:b], indices[b:]


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    os.replace(temp, path)


def train(data_path=None, output_dir=None, config_path=None):
    data_path = Path(data_path or ROOT / "data/housing.csv")
    output_dir = Path(output_dir or ROOT / "artifacts")
    config = json.loads(Path(config_path or ROOT / "config.json").read_text())
    data_sha = hashlib.sha256(data_path.read_bytes()).hexdigest()
    if data_sha != EXPECTED_DATA_SHA:
        raise ValueError("Dataset checksum changed; review and approve the new snapshot")
    frame = pd.read_csv(data_path)
    required = RAW_FEATURES + [TARGET, "ocean_proximity"]
    if list(frame.columns) != required:
        # Source order differs; require the same exact set and reject extra columns.
        if set(frame.columns) != set(required):
            raise ValueError("Unexpected data schema")
    if len(frame) != 20640 or frame.duplicated().any():
        raise ValueError("Unexpected row count or duplicate records")
    if frame[TARGET].isna().any() or (frame[TARGET] <= 0).any():
        raise ValueError("Targets must be present and positive")
    for name in RAW_FEATURES:
        if np.isinf(frame[name].to_numpy(dtype=float)).any():
            raise ValueError("Infinite source feature")

    train_idx, val_idx, test_idx = split_indices(len(frame), config)
    training, validation, test = frame.iloc[train_idx], frame.iloc[val_idx], frame.iloc[test_idx]
    prep = Preprocessor().fit(training)
    x_train, x_val = prep.transform(training), prep.transform(validation)
    y_train = training[TARGET].to_numpy(dtype=float)
    y_val = validation[TARGET].to_numpy(dtype=float)
    model = ScratchRegressor(config["learning_rate"], config["epochs"], config["l2"])
    model.fit(x_train, y_train / 100000)
    baseline = float(y_train.mean())
    baseline_validation = metrics(y_val, np.full(len(y_val), baseline))
    scratch_validation = metrics(y_val, model.predict(x_val) * 100000)
    # sklearn Ridge minimizes sum squared error + alpha * ||w||^2.
    # Our objective uses mean squared error + l2 * ||w||^2, hence alpha=n*l2.
    reference = Ridge(alpha=len(y_train) * config["l2"]).fit(x_train, y_train / 100000)
    reference_validation = metrics(y_val, reference.predict(x_val) * 100000)
    gate = {
        "rmse_beats_baseline": scratch_validation["rmse_usd"] <= baseline_validation["rmse_usd"] * config["max_validation_rmse_ratio"],
        "mae_beats_baseline": scratch_validation["mae_usd"] <= baseline_validation["mae_usd"] * config["max_validation_mae_ratio"],
        "finite_metrics": all(np.isfinite(v) for v in scratch_validation.values()),
    }
    if not all(gate.values()):
        raise ValueError(f"Candidate rejected by validation gate: {gate}")

    # Evaluate test data only after the candidate passes the fixed validation gate.
    x_test, y_test = prep.transform(test), test[TARGET].to_numpy(dtype=float)
    report = {
        "validation": {"mean_baseline": baseline_validation, "scratch_ridge": scratch_validation,
                       "sklearn_ridge": reference_validation},
        "test": {"mean_baseline": metrics(y_test, np.full(len(y_test), baseline)),
                 "scratch_ridge": metrics(y_test, model.predict(x_test) * 100000),
                 "sklearn_ridge": metrics(y_test, reference.predict(x_test) * 100000)},
        "validation_gate": gate,
        "reference_mean_absolute_prediction_difference_usd": float(np.mean(np.abs(
            model.predict(x_val) - reference.predict(x_val))) * 100000),
    }
    state = {"preprocessing": prep.to_dict(), "weights": model.weights_.tolist(), "bias": model.bias_,
             "data_sha256": data_sha, "training_config": config}
    digest = state_digest(state)
    code_hash = hashlib.sha256(b"".join((ROOT / "housing" / name).read_bytes()
        for name in ["preprocessing.py", "scratch.py", "training.py", "artifact.py"])).hexdigest()
    payload = {
        "schema_version": 1, "model_version": digest[:12], "state_sha256": digest, "state": state,
        "created_at": datetime.now(timezone.utc).isoformat(), "source_sha256": code_hash,
        "target": {"name": TARGET, "unit": "USD", "meaning": "District median house value in historical California census data"},
        "evaluation": report,
        "split_counts": {"train": len(train_idx), "validation": len(val_idx), "test": len(test_idx)},
        "split_indices_sha256": {key: hashlib.sha256(value.astype('<i8').tobytes()).hexdigest()
                                 for key, value in [("train", train_idx), ("validation", val_idx), ("test", test_idx)]},
        "data_audit": {"rows": len(frame), "missing": frame.isna().sum().to_dict(), "duplicates": 0},
        "runtime": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__},
    }
    release = output_dir / "releases" / payload["model_version"]
    atomic_json(release / "model.json", payload)
    atomic_json(release / "loss-history.json", model.loss_history_)
    # Active artifact is promoted only after evaluation, never from an API request.
    atomic_json(output_dir / "model.json", payload)
    atomic_json(ROOT / "reports/metrics.json", report)
    example = frame.iloc[int(val_idx[0])][RAW_FEATURES + ["ocean_proximity"]].to_dict()
    example = {k: None if pd.isna(v) else v for k, v in example.items()}
    atomic_json(ROOT / "examples/request.json", {"records": [example]})
    reference_stats = {}
    for name in RAW_FEATURES:
        values = training[name].dropna().to_numpy(dtype=float)
        interior = np.unique(np.quantile(values, np.linspace(0, 1, 11)[1:-1]))
        boundaries = np.concatenate(([-np.inf], interior, [np.inf]))
        counts = np.histogram(values, boundaries)[0]
        reference_stats[name] = {"interior_edges": interior.tolist(), "fractions": (counts / counts.sum()).tolist()}
    atomic_json(output_dir / "drift-reference.json", reference_stats)
    print(json.dumps({"model_version": payload["model_version"], **report}, indent=2))
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data")
    parser.add_argument("--output")
    args = parser.parse_args()
    train(args.data, args.output)
