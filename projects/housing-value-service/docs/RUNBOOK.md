# Runbook

## Check service health

`GET /health` confirms the process responds. `GET /ready` confirms a released
artifact loaded successfully and includes its model version. Deployment health
checks should use readiness. A missing or corrupted artifact prevents startup.

```powershell
python scripts/smoke.py http://localhost:8000
```

After deployment, replace the base URL with the actual service URL. Do not mark
a release complete until readiness and an actual prediction both pass.

## Investigate errors

Find the request ID from the response header or body. Check the structured logs
for that ID, status and duration. A 422 response points to invalid input; 401
means the configured key is missing or wrong; 413 means a declared body is too
large. Startup errors commonly indicate missing artifacts or dependency problems.

The browser may load while a prediction is rejected. Check its error message and
compare the request with `examples/request.json`. Keep API keys in environment
variables, not source files. Set `API_KEY` for a protected demonstration and pass
the key in the `X-API-Key` header. Health endpoints stay accessible to the platform.

## Retrain and release

1. Review the proposed data snapshot and schema; update its approved checksum deliberately.
2. Freeze the configuration before evaluating the final test set.
3. Run training. A failed validation gate must not replace the active artifact.
4. Review validation, final evaluation and model limitations; run tests.
5. Build and smoke-test the exact image that will be released.
6. Deploy a preview version, check readiness and prediction, then promote deliberately.

Do not repeatedly tune against the final test set. If it has become part of your
development loop, obtain a new final evaluation dataset before making new claims.

## Roll back

For local serving, stop the process and set `MODEL_PATH` to an earlier retained
`artifacts/releases/<version>/model.json`, then restart and smoke-test it.
Pass `--expected-version <version>` to the smoke script when deliberately checking
a retained release instead of the active artifact.

On Render, deploy the earlier known-good source commit/image and repeat the
readiness and prediction checks. Keep the artifact and its matching preprocessing
together. Rollback is a release action; it is not another fit() call.

## Observe changing data

```powershell
python -m housing.drift data/housing.csv
```

This batch compares numeric input distributions with bins learned on training
records. PSI at least 0.2 triggers review in this demonstration. It is a chosen
project threshold, not a universal rule or proof of model degradation.
Investigate source changes and data validation, then evaluate with new labels.
Do not automatically retrain or release a replacement from a drift alert.

## Current infrastructure boundaries

The demo is one process. It has no distributed rate limiter, durable request store,
scheduled training orchestrator or hosted metrics database. Authentication can
be enabled with an environment variable; the default local demonstration is open.
An internet deployment needs platform HTTPS and should add a proxy-level body
limit and rate limiting appropriate to its traffic. These are follow-on hardening
tasks, not reasons to add unnecessary infrastructure to the first learning release.
