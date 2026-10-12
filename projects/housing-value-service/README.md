# Housing Value Lab

An end-to-end learning project that turns NumPy gradient descent into a validated,
versioned prediction service. It continues the small house-price exercise with
20,640 real historical California census district records.

**Target:** a district's median house value in US dollars. It does not predict the
current price of an individual Ghanaian or Californian house.

## What is implemented

- Data snapshot validation and a deterministic 60/20/20 training/validation/test split.
- Training-only missing-value imputation, engineered ratios, outlier clipping,
  standardization and categorical encoding.
- Ridge regression implemented with NumPy, including gradients and weight updates.
- A mean baseline and an equivalent scikit-learn Ridge comparison.
- A validation gate before releasing a candidate; final held-out evaluation.
- JSON model artifacts with a state checksum and immutable version paths.
- A FastAPI service, batch requests, input validation and optional API-key access.
- An interactive browser demo, readiness checks, JSON request logs and metrics.
- A batch drift monitor that flags changes for investigation.
- A Docker image definition, Compose setup and a GitHub CI workflow that builds
  and runs the image after training and tests pass.
- A Render deployment configuration and rollback runbook.

## Verified model results

| Model | Test MAE in USD | Test RMSE in USD | Test R squared |
|---|---:|---:|---:|
| Training-mean baseline | 91,442 | 116,028 | -0.0001 |
| NumPy ridge regression | 47,642 | 65,523 | 0.6811 |
| scikit-learn Ridge comparison | 47,646 | 65,518 | 0.6811 |

The NumPy model reduces test MAE by about 47.9% versus the baseline. Its validation
predictions differ from the equivalent library model by about USD 438 on average;
finite gradient descent is close to, but does not exactly reach, the library solver.
No per-prediction uncertainty interval is claimed.

Read `reports/metrics.json` for the full recorded values and `docs/MODEL-CARD.md`
for the evaluation scope and limitations.

## Run locally

Use Python 3.12. Clone the project branch, then open its project folder:

```powershell
git clone --branch feature/housing-value-service-20261012 https://github.com/AbubakarMA/Data-Science-ML-Deep-Learning-And-AI-Engineering.git
cd Data-Science-ML-Deep-Learning-And-AI-Engineering/projects/housing-value-service
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.txt
python -m pytest
python -m housing.server
```

On macOS/Linux, use `python3.12 -m venv .venv` and `source .venv/bin/activate`.
If PowerShell activation is blocked, run `.\.venv\Scripts\python.exe` directly;
you do not need to change the machine's execution policy.

Open `http://localhost:8000` for the demo or `http://localhost:8000/docs` for the API.
The checked-in artifact lets you start serving without retraining. Install the
dependencies while online; after that, training, tests and serving use local files.

## Train and trace the model

```powershell
python -m housing.training
python scripts/smoke.py http://localhost:8000
```

Training is a separate action. API requests load and use the already released
model; they never call `fit()`. `config.json` contains the fixed training settings
and validation thresholds. Editing those settings creates a new experiment.
Repeatedly consulting the test set would make it unsuitable as a final evaluation;
use validation for model decisions and obtain a fresh final test set when needed.

The request example lives in `examples/request.json`. On Windows:

```powershell
$body = Get-Content examples/request.json -Raw
Invoke-RestMethod -Method Post -Uri http://localhost:8000/v1/predict -ContentType application/json -Body $body
```

Set `include_trace` to true to inspect transformed features, weights and dollar
contributions for the first record. Their sum plus the intercept produces the
raw prediction; the serving policy floors negative values at zero.

## Docker

With Docker running on your computer:

```powershell
docker compose up --build
```

The image runs as a non-root user, serves the prevalidated model, and honors `PORT`.
It does not download data or train at startup. `compose.yaml` binds the demo to
localhost, uses a read-only filesystem and drops Linux capabilities.

The Docker executable is not present in the assistant's local execution environment.
The GitHub CI job is the container-build and container-smoke-test gate; see
`docs/VERIFICATION.md` for the actual verification status.

## Release and deployment

The remote CI workflow is placed at repository root:
`.github/workflows/housing-service-ci.yml`. Its copy in this project documents the
same workflow. It runs training, the validation gate, tests, Docker build and an
HTTP smoke test, then keeps evaluation evidence.

`render.yaml` uses paths relative to the parent repository and sets a readiness
endpoint and deployment-after-checks behavior. Use a free Render Docker web
service pointing at the project branch, Docker context `projects/housing-value-service`
and Dockerfile `projects/housing-value-service/Dockerfile` for a preview release.
Merging or deploying the main branch is a separate release decision.

## Work through the project today

Start with `docs/WORKSHOP.md`. Complete each checkpoint by running the code and
explaining the result. Use `docs/ARCHITECTURE.md` to connect the files and
`docs/RUNBOOK.md` to practise failures and rollback.
