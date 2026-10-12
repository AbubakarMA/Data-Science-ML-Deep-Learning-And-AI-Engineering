# Verification evidence

## Checks completed in the project workspace

- Downloaded the real data snapshot and verified its SHA-256, 20,640 records,
  expected schema, 207 missing bedroom values and no duplicated records.
- Trained the NumPy model and passed the fixed validation gate before final test reporting.
- Repeated training in a clean environment using pinned dependencies; reproduced
  model version `24a9c8e5b5fc` and the recorded evaluation results.
- Passed 27 initial tests in both the initial and clean environments, then added
  a source-provenance check for 28 total tests. The tests cover
  numerical gradients, an independent closed-form optimizer comparison,
  training-only preprocessing, artifacts, splits, API behavior and drift.
- Started the actual HTTP server and passed readiness and prediction smoke checks.
  The example response was USD 255,597.69 with the expected model version.

## Checks delegated to the remote CI runner

The local workspace has no Docker executable. Its operating-system socket
restrictions also prevented Chromium from launching, so desktop/mobile browser
behavior has not been claimed as locally verified.

The GitHub workflow runs Docker build, an HTTP smoke test inside the image and
a real browser interaction at desktop and mobile widths. It keeps screenshots,
the model artifact and evaluation report as evidence. Inspect the workflow result
before accepting the container release.

The first full CI run passed all stages:
https://github.com/AbubakarMA/Data-Science-ML-Deep-Learning-And-AI-Engineering/actions/runs/38234436489

That run demonstrated a cross-hardware content-hash difference after retraining.
The updated workflow keeps the reviewed artifact unchanged, trains a separate
candidate, and checks the exact served version. Its final CI result is recorded
in the pull request. Browser screenshots from the initial run were also visually
reviewed at desktop and mobile widths.

Live deployment requires a confirmed Render workspace. After creation, verify
the actual URL with readiness and a prediction; a configured deployment is not
the same as a verified live service.
