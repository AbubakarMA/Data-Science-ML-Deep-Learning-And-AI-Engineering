"""Portable JSON model artifacts, without pickle execution at load time."""

import hashlib
import json
from pathlib import Path
import numpy as np
from housing.preprocessing import Preprocessor, input_frame


def state_digest(state):
    canonical = json.dumps(state, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(canonical.encode()).hexdigest()


class Predictor:
    def __init__(self, payload):
        if payload["schema_version"] != 1:
            raise ValueError("Unsupported artifact schema")
        if state_digest(payload["state"]) != payload["state_sha256"]:
            raise ValueError("Artifact checksum mismatch")
        self.payload = payload
        self.preprocessor = Preprocessor.from_dict(payload["state"]["preprocessing"])
        self.weights = np.asarray(payload["state"]["weights"], dtype=float)
        self.bias = float(payload["state"]["bias"])
        if len(self.weights) != len(self.preprocessor.feature_names_):
            raise ValueError("Model and preprocessing dimensions do not match")
        if not np.isfinite(self.weights).all() or not np.isfinite(self.bias):
            raise ValueError("Invalid model parameters")

    @classmethod
    def load(cls, path):
        return cls(json.loads(Path(path).read_text()))

    def predict(self, records):
        x = self.preprocessor.transform(input_frame(records))
        raw = (x @ self.weights + self.bias) * 100000
        # Serving and evaluation use this same nonnegative target policy.
        return np.maximum(raw, 0)

    def warnings(self, record):
        warnings = []
        for name, (lo, hi) in self.preprocessor.raw_ranges_.items():
            value = record.get(name)
            if value is not None and not lo <= value <= hi:
                warnings.append(f"{name} is outside the observed training range")
        if record.get("total_bedrooms") is None:
            warnings.append("total_bedrooms was imputed with the training median")
        return warnings
