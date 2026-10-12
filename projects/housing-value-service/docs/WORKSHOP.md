# Build and understand the full project today

The goal is a working system you can explain and rebuild. Run the supplied version
first, then implement and change small pieces yourself. Keep a log of what changed,
what you expected and what actually happened. This is a five-hour workshop with
short breaks; live deployment depends on repository and workspace access.

## 1 Define the product and inspect data

Allow 35 minutes. Read the target definition in the README and open the CSV with pandas.
Print shape, dtypes, missing values, target quantiles and ocean-proximity counts.

```python
import pandas as pd
df = pd.read_csv('data/housing.csv')
print(df.shape)
print(df.isna().sum())
print(df['median_house_value'].describe())
print(df['ocean_proximity'].value_counts())
```

Explain why this target differs from our fictional GHS house-price exercise. Why
is floor area absent? Why do district totals need a clear definition? Identify the
missing feature and two capped variables. Write a three-sentence project brief.

Checkpoint: you can state inputs, output, unit, intended use and a meaningful baseline.

## 2 Trace preprocessing and prevent leakage

Allow 40 minutes. Read `Preprocessor.fit()` and `transform()`. Identify each value
learned from training and each deterministic transformation.

The order is: median imputation, derived ratios, training-quantile clipping,
training-mean standardization, then category encoding. For a numeric feature,
standardization is `(clipped_value - training_mean) / training_standard_deviation`.

Implement median replacement for one column independently. Then implement rooms
per household with a safe denominator. Explain why fitting the scaler on all
records makes the final evaluation less credible. Run the preprocessing test.

```powershell
python -m pytest tests/test_model.py -k preprocessing
```

Checkpoint: you can list the saved preprocessing statistics and show that transforming
an unseen record does not change them. Take a 10-minute break.

## 3 Rebuild the gradient calculation

Allow 50 minutes. Read the mathematics in `ARCHITECTURE.md`. Explain the dimensions
of X, y, weights and predictions. Trace `objective_and_gradients()` line by line.

Before vectorizing, calculate one example's gradient with a loop. Then write a
loop version that adds gradients for all records and compare it with `X.T @ errors`.
Explain why the average contributes a division by n and why the regularization
term adds `2 * l2 * weights`. The intercept is not regularized.

```powershell
python -m pytest tests/test_model.py -k 'gradients or optimizer'
python -m housing.training
```

Record validation and test MAE separately. Explain why the sklearn alpha is
`number_of_training_records * l2`. The finite-difference test checks derivatives
independently; the closed-form test checks the optimizer against a different method.

Checkpoint: you can write the prediction, objective and both gradients without copying.

## 4 Understand model release and inference

Allow 30 minutes. Open the released JSON. Find weight values, feature names,
medians, model version, dataset checksum and evaluation results.

Create a prediction with the model directly:

```python
import json
from housing.artifact import Predictor
model = Predictor.load('artifacts/model.json')
request = json.load(open('examples/request.json'))
print(model.predict(request['records']))
```

Confirm the same weights and preprocessing are used after loading. Copy a model
artifact to a temporary file, change its bias without updating its checksum and
confirm loading is rejected. Explain why training should not happen per request.

Checkpoint: you can load a released model and predict without fitting. Take a break.

## 5 Build and exercise the API

Allow 45 minutes. Start `python -m housing.server` and open the browser demo.
Use `/docs` to submit a valid request with `include_trace: true`.
Find the transformed features, their weights and contributions. Sum contributions
and bias; compare with the raw value and the response's nonnegative prediction.

Send an invalid longitude, zero households, an unknown category, a missing bedroom
count and a batch of two records. Explain the difference between validation failure
and imputation. Set `API_KEY` temporarily and demonstrate a 401 followed by a valid
authorized request. Read the relevant tests rather than only their pass count.

Checkpoint: you can describe request validation, model loading, inference, units,
errors and response IDs. Your endpoint must not call training.

## 6 Package and test a release

Allow 35 minutes. Read the Dockerfile line by line. Explain the base runtime,
dependencies, copied artifacts, non-root user, readiness check and exec-form CMD.
Run `docker compose up --build` if Docker is available on your computer.

Open the demo, then run the HTTP smoke script. A container image is a packaged
application; deployment is running a release in a target environment. CI runs
training, gates, tests, image build and smoke tests before a release is accepted.

Checkpoint: you can rebuild and run the image and identify the exact model version
it serves. If Docker is unavailable locally, inspect the actual remote CI logs.

## 7 Operate the service and review the release

Allow 30 minutes. Read `/metrics` after making predictions. Identify the request
count, prediction count and duration sum. Follow a response request ID to the logs.
Do not infer latency percentiles from a duration sum alone.

Run the drift command. Then copy 500 records into a separate file, increase their
median income deliberately and compare the monitor's output. A changed distribution
is a signal to investigate, not an automatic instruction to replace the model.

Review the model card, test report and CI result. After deploying a preview, run
the smoke script against its public URL. Practise switching to a retained artifact
locally and restarting. Read the runbook for source/image rollback in deployment.

Checkpoint: you can diagnose a failed request and restore a known-good release.

## Finish with evidence

Keep your product brief, preprocessing explanation, handwritten gradient, evaluation
log, successful API response, Docker/CI result and rollback notes. These demonstrate
what you understood and built. If any check fails, record the full error and your
hypothesis; fixing it is part of the project.

At the end, explain what happens inside our fit() and predict(), why data splits
matter, what an artifact contains, and how code becomes a monitored service.
