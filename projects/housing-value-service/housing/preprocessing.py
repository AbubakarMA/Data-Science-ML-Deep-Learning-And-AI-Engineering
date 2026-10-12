"""Training-only imputation, feature engineering, clipping and standardization."""

import numpy as np
import pandas as pd

RAW_FEATURES = ["longitude", "latitude", "housing_median_age", "total_rooms",
                "total_bedrooms", "population", "households", "median_income"]
DERIVED_FEATURES = ["rooms_per_household", "bedrooms_per_room", "people_per_household"]


class Preprocessor:
    def _numeric(self, records):
        raw = records[RAW_FEATURES].to_numpy(dtype=float)
        if np.isinf(raw).any():
            raise ValueError("Infinite inputs are not allowed")
        filled = np.where(np.isnan(raw), self.medians_, raw)
        cols = {name: filled[:, i] for i, name in enumerate(RAW_FEATURES)}
        derived = np.column_stack([
            cols["total_rooms"] / np.maximum(cols["households"], 1),
            cols["total_bedrooms"] / np.maximum(cols["total_rooms"], 1),
            cols["population"] / np.maximum(cols["households"], 1),
        ])
        return np.column_stack([filled, derived])

    def fit(self, records):
        raw = records[RAW_FEATURES].to_numpy(dtype=float)
        if not len(raw) or np.isinf(raw).any():
            raise ValueError("Training data must be nonempty and finite or missing")
        self.medians_ = np.nanmedian(raw, axis=0)
        if not np.isfinite(self.medians_).all():
            raise ValueError("A feature has no usable training values")
        numeric = self._numeric(records)
        # Learn these limits on training records only, never validation or test.
        self.lower_ = np.quantile(numeric, 0.005, axis=0)
        self.upper_ = np.quantile(numeric, 0.995, axis=0)
        clipped = np.clip(numeric, self.lower_, self.upper_)
        self.means_ = clipped.mean(axis=0)
        self.scales_ = clipped.std(axis=0)
        self.scales_ = np.where(self.scales_ < 1e-12, 1, self.scales_)
        self.categories_ = sorted(records["ocean_proximity"].unique().tolist())
        # Drop the first category; avoid redundancy with the model intercept.
        self.feature_names_ = RAW_FEATURES + DERIVED_FEATURES + [
            "ocean_proximity=" + c for c in self.categories_[1:]
        ]
        self.raw_ranges_ = {
            c: [float(records[c].min()), float(records[c].max())] for c in RAW_FEATURES
        }
        return self

    def transform(self, records):
        categories = records["ocean_proximity"].astype(str)
        unknown = set(categories) - set(self.categories_)
        if unknown:
            raise ValueError(f"Unknown ocean proximity: {sorted(unknown)}")
        numeric = self._numeric(records)
        standardized = (np.clip(numeric, self.lower_, self.upper_) - self.means_) / self.scales_
        encoded = np.column_stack([(categories == c).to_numpy(dtype=float)
                                   for c in self.categories_[1:]])
        return np.column_stack([standardized, encoded])

    def to_dict(self):
        return {"medians": self.medians_.tolist(), "lower": self.lower_.tolist(),
                "upper": self.upper_.tolist(), "means": self.means_.tolist(),
                "scales": self.scales_.tolist(), "categories": self.categories_,
                "feature_names": self.feature_names_, "raw_ranges": self.raw_ranges_}

    @classmethod
    def from_dict(cls, state):
        obj = cls()
        for name in ["medians", "lower", "upper", "means", "scales"]:
            setattr(obj, name + "_", np.asarray(state[name], dtype=float))
        for name in ["categories", "feature_names", "raw_ranges"]:
            setattr(obj, name + "_", state[name])
        return obj


def input_frame(records):
    return pd.DataFrame(records, columns=RAW_FEATURES + ["ocean_proximity"])
