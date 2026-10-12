"""Batch PSI screening; an alert requests investigation, never auto-retraining."""

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def check_drift(records, reference):
    if len(records) < 100:
        raise ValueError("Use at least 100 observations for this teaching monitor")
    results = []
    for name, stats in reference.items():
        values = records[name].dropna().to_numpy(dtype=float)
        if not len(values) or not np.isfinite(values).all():
            results.append({"feature": name, "status": "invalid_data", "psi": None})
            continue
        edges = np.concatenate(([-np.inf], stats["interior_edges"], [np.inf]))
        counts = np.histogram(values, edges)[0]
        observed = counts / counts.sum()
        expected = np.asarray(stats["fractions"])
        observed = np.maximum(observed, 1e-6); observed /= observed.sum()
        expected = np.maximum(expected, 1e-6); expected /= expected.sum()
        psi = float(np.sum((observed - expected) * np.log(observed / expected)))
        results.append({"feature": name, "psi": psi,
                        "status": "review" if psi >= 0.2 else "stable"})
    return {"rows": len(records), "features": results,
            "action": "Investigate data and obtain labels; do not automatically promote a retrained model.",
            "threshold_note": "0.2 is a project demonstration threshold, not a universal statistical rule."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("csv")
    args = parser.parse_args()
    reference = json.loads((ROOT / "artifacts/drift-reference.json").read_text())
    print(json.dumps(check_drift(pd.read_csv(args.csv), reference), indent=2))
