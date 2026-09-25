# Project Status, Work Completed, and Recommended Next Steps

**Team:** `4BD478E9`  
**Purpose:** Competition entry for predicting whether a synthetic financial alert is escalated (`eskalatsiya = 1`) or dismissed (`eskalatsiya = 0`).  
**Last documented review:** 25 September 2026

## Executive summary

This repository now contains a reproducible training and submission pipeline, a final notebook, submission-generation and validation utilities, automated tests, a record of the controlled model experiments, and an aggregate-only Streamlit EDA website. The latest full test run completed with **109 passed**. The website starts locally at `http://localhost:8501/` and returned HTTP 200 during review.

Two important caveats remain:

1. The website is **not deployed publicly**. `localhost` is reachable only on the machine running it, so organizers cannot use that address.
2. The selected model's pooled forward-only OOF ROC-AUC is **0.578205** over 11,982 scored rows and 28 features. This is an internal estimate, not a hidden-test score or a guarantee of a competitive leaderboard position. It is the clearest current modeling risk.

No result can guarantee championship acceptance or a win. Eligibility also depends on organizer rules, a valid official submission, and access to the required public EDA URL.

## Competition task and project boundaries

The model estimates an escalation probability for each test signal using its associated historical transaction records. Multiple transactions may map to a single signal, so the pipeline summarizes transaction history at signal level. It applies an as-of-signal rule: transaction records later than the signal timestamp are excluded from behavior features.

The competition description says the data are synthetic and transformed/localized for the hackathon. The EDA report repeats that provenance and shows aggregate summaries; it does not serve predictions or expose individual signal identifiers. The prediction CSV and EDA website are separate required deliverables.

## Work completed

### 1. Project structure and reproducibility

- Reviewed the project layout, README, requirements, notebooks, feature and validation code, artifacts, optimization log, EDA site, and automated tests.
- The main reproducible workflow is documented in the root `README.md`. It describes environment creation, dependency installation, fresh-kernel notebook execution, submission creation and validation, tests, the optimization sweep, and local EDA startup.
- The final notebook is `notebooks/final_reproducible_solution.ipynb`. It reads raw competition files, audits the inputs, filters transactions by signal date, creates features, validates and trains the model, evaluates OOF predictions, predicts test probabilities, and documents the output location. Feature artifacts can be cached; the notebook documents how to force a rebuild.
- A previous clean-run inventory is recorded in `artifacts/final_reproduction_inventory.md`, including output dimensions and checksums. That inventory records 14,000 training signals, 6,000 test signals, 28 features, five forward-only folds, 11,982 OOF rows, and a 6,000-row submission.

### 2. Model and experiments

- The controlled optimization sprint tested documented, training-only changes on shared chronological folds. Hypotheses and results are recorded in `experiments/FINAL_OPTIMIZATION_SPRINT.md` and `experiments/controlled_optimization/`.
- The final notebook uses shallow XGBoost (500 estimators, depth 2, learning rate 0.025, minimum child weight 15, subsample and column sample 0.8, L2 5, seed 42) on 28 features.
- OOF ROC-AUC is 0.578205. The previous baseline equal-weight ensemble scored 0.551687 on the same OOF framework. This comparison supports the selected internal model; neither score reveals hidden-test performance.
- The final estimator is trained from labeled training data. The hidden test set is used to produce probabilities, not to choose features or tune model settings.

### 3. Submission pipeline

- `src/submission/make_submission.py` creates `artifacts/predictions/team_4BD478E9.csv` using the configured team ID and test signal order.
- `src/submission/validate_submission.py` checks the required two columns and order (`signal_id,ehtimollik`), one row per test ID, exact ID matching, uniqueness, nonmissing real-valued probabilities within `[0,1]`, parseability, no index column, and sample-submission compatibility.
- `tests/test_submission.py` covers the submission requirements. The reproduction inventory records 20 submission tests passed and a 6,000-row output.
- The inventory gives the last recorded prediction range as 0.046405 to 0.451806. Recheck the generated CSV and validator output after any regeneration; do not assume old checksums or values still apply after changing inputs or dependencies.

### 4. EDA website

- `eda_site/app.py` presents the 14 requested explanatory sections: problem, structure, quality and time, target distribution, transaction activity, directions, transaction types, amount distribution, recent behavior, target-group comparisons, observations, feature decisions, validation/model, and conclusion.
- `eda_site/assets/eda_summary.json` is a precomputed aggregate artifact for faster page loads. `eda_site/build_summary.py` rebuilds it from the local project data and artifacts.
- The interface was redesigned with a dark research-report navigation panel, a high-contrast hero, responsive spacing, cards for metrics and charts, restrained colors, and a simpler visual hierarchy.
- Local verification in this review showed the updated title and hero, aggregate counts, synthetic-data note, no visible Streamlit exception, and an HTTP 200 response at `http://localhost:8501/`.
- **Public deployment is still pending.** The app's Deploy button or a local URL is not proof of deployment. The site must be published and the resulting public URL opened successfully before reporting it as public.

### 5. Bugs fixed in the latest review

- Fixed transaction-gap aggregation so grouped timestamps produce one row per signal and retain the signal ID column.
- Fixed cross-dataset audit fingerprinting: it previously referenced the wrong test fingerprint variable. Added safe zero-denominator handling and avoided mutating the raw transaction DataFrames during audit.
- Fixed safe division for scalar inputs, including zero denominators, as well as vector inputs.
- Fixed window feature date merges when a transaction table already contained the signal date.
- Replaced the broken, last-signal-only velocity calculations with per-signal rates and 7-day versus 30-day count/direction/amount ratios, using only transactions at or before the signal time.
- Aligned train and test feature matrices when categories appear in one split but not the other.
- Made stratified fold creation respect minority-class size. Five stratified folds cannot be formed with only two examples in the minority class; the implementation now uses the feasible number and reports the reduction.
- Corrected time-aware validation to use expanding past-only training history rather than training on future observations. The small fixture now produces four valid folds because one observation is needed to seed the training history.
- Updated two legacy test expectations that contradicted the documented ratio definition and the number of mathematically feasible folds.
- Re-ran the full suite: **109 passed**. Python compilation passed for the changed implementation files. Four date-format warnings and 25 scikit-learn deprecation warnings remain; they did not fail tests.

The validation changes above repair the reusable legacy validation module. The final competition notebook's fixed persisted folds and selected model settings were not intentionally changed during the UI redesign and bug-fix pass.

## Current deliverables and paths

| Deliverable | Location | Current state |
|---|---|---|
| Final reproducible notebook | `notebooks/final_reproducible_solution.ipynb` | Present; prior clean run documented |
| Official-format team CSV | `artifacts/predictions/team_4BD478E9.csv` | Present; previous validation documented; rerun validator before handoff |
| CSV generation | `src/submission/make_submission.py` | Present |
| CSV validation | `src/submission/validate_submission.py` | Present |
| Automated tests | `tests/` | Latest run: 109 passed |
| EDA Streamlit app | `eda_site/app.py` | Redesigned; locally verified |
| EDA aggregate artifact | `eda_site/assets/eda_summary.json` | Present; rebuild if raw data/model artifacts change |
| Clean reproduction inventory | `artifacts/final_reproduction_inventory.md` | Present; reflects its recorded run, not necessarily future regenerated bytes |
| Public website URL | None yet | **Not deployed** |

## Recommended work plan

Complete the remaining work in this order. Each step has a concrete check so the final handoff is reviewable.

### Priority 1 — Publish and verify the EDA website

The competition requires a working public URL. This is the most obvious outstanding deliverable gap.

1. Confirm the repository contains only files that are allowed to be public. Do not publish raw competition data, private/local paths, credentials, or organizer-only materials. Review the aggregate JSON and source tree as part of this check.
2. Push the project to an approved public Git host, or use another host that can serve the Streamlit app. Keep `eda_site/requirements.txt` and the aggregate JSON available to the deployment build.
3. Set `eda_site/app.py` as the Streamlit entry point and deploy. The organizer must not need access to a private account to load the report.
4. Open the resulting deployed URL in a signed-out/private browser session. Verify the app loads, charts render, all 14 sections can be navigated, no local filesystem path is visible, and the public page has no runtime errors.
5. Record the exact checked URL, date, and result in the README and this status document. Until this check succeeds, label the site “not deployed.”

**Acceptance gate:** an external browser can open the exact public URL without organizer-only access and view the EDA report.

### Priority 2 — Reproduce the final prediction artifact from a clean environment

Reconfirm that the notebook, script, and current files agree before the competition handoff.

From the project root in PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
$env:JUPYTER_DATA_DIR = Join-Path (Get-Location) '.jupyter\data'
$env:JUPYTER_CONFIG_DIR = Join-Path (Get-Location) '.jupyter\config'
$env:JUPYTER_RUNTIME_DIR = Join-Path (Get-Location) '.jupyter\runtime'
$env:IPYTHONDIR = Join-Path (Get-Location) '.jupyter\ipython'
$env:JUPYTER_PATH = Join-Path (Get-Location) 'share\jupyter'
New-Item -ItemType Directory -Force -Path $env:JUPYTER_DATA_DIR,$env:JUPYTER_CONFIG_DIR,$env:JUPYTER_RUNTIME_DIR,$env:IPYTHONDIR | Out-Null
.venv\Scripts\python.exe -m ipykernel install --prefix (Get-Location) --name python3 --display-name "Python 3 (WIUT reproduction)"
.venv\Scripts\python.exe -m jupyter nbconvert --to notebook --execute --inplace notebooks/final_reproducible_solution.ipynb
.venv\Scripts\python.exe -m src.submission.make_submission --team-id 4BD478E9
.venv\Scripts\python.exe -m src.submission.validate_submission --submission artifacts/predictions/team_4BD478E9.csv
.venv\Scripts\python.exe -m pytest -q
```

The notebook may take several minutes because the raw Parquet tables are large. Use the documented feature cache for routine runs; set `FORCE_REBUILD_FEATURES = True` in the notebook configuration to rebuild cached features from raw files. Preserve notebook outputs from the fresh-kernel run only if they accurately correspond to that run.

**Acceptance gate:** fresh-kernel notebook execution succeeds; the generator creates the team CSV; validator confirms exact test ID coverage and required schema; all tests pass. Save the new run's versions, metrics, row counts, file hashes, and validation result in the artifact inventory.

### Priority 3 — Verify official competition submission

1. Read the current competition submission instructions and confirm the expected filename, team ID, column names, and upload method.
2. Upload `artifacts/predictions/team_4BD478E9.csv` only to the official competition destination.
3. Confirm the platform accepted it and save the receipt/status. A locally valid CSV does not prove platform acceptance.
4. If the platform rejects it, use the exact rejection message to fix format or eligibility issues, regenerate, validate, and resubmit according to the competition rules.

**Acceptance gate:** the official platform shows the submission as accepted/valid. Do not call a local test “official acceptance.”

### Priority 4 — Address model performance carefully

The current OOF ROC-AUC of 0.578205 is modest. Improve it only through permitted, reproducible training-data experiments.

1. Confirm that the persisted competition folds match the intended real-world time boundary and that every validation row is scored exactly once after the warm-up block.
2. Inspect fold AUC, prevalence, dates, and error patterns to see whether one period drives the score or the model is consistently weak.
3. Form a small number of hypotheses from transaction behavior (for example, robust recency/change summaries, missing-history indicators, stable transaction-gap summaries, or monotonic/simple baselines) and record each before running it.
4. Compare all variants using the same fixed folds, feature generation, and ROC-AUC calculation. Record per-fold results, pooled OOF score, feature count, parameters, and runtime. Retain changes only with repeatable improvement or a clearly documented stability/diversity benefit.
5. Keep the hidden test set out of model selection. Do not repeatedly submit to the leaderboard as a substitute for validation.
6. If a candidate wins, rerun the final notebook and submission validation from a clean environment, update the README, EDA modeling summary if needed, and inventory, then regenerate the final CSV.

**Acceptance gate:** every retained modeling change has a reproducible experiment record and improves or clearly stabilizes forward-only validation. No plan can guarantee a win; leaderboard performance is unknown until a valid competition submission is scored.

### Priority 5 — Final organizer-readiness review

- Confirm there are no credentials, private user files, raw data accidentally staged for publication, local machine paths, or organizer-only content in public deliverables.
- Confirm the EDA site explicitly says the data are synthetic and does not expose individual records.
- Check README commands from a clean environment and ensure the Windows and Unix command notes remain accurate.
- Confirm model metrics are consistently described as OOF validation metrics, never as hidden-test performance.
- Confirm the final notebook, CSV, tests, EDA URL, and artifact inventory all refer to the same team ID and model version.

**Final completion checklist:**

- [ ] Public EDA URL opens successfully without organizer access.
- [ ] Final notebook runs top to bottom in a fresh kernel.
- [ ] Final submission CSV is regenerated from that run and locally validated.
- [ ] Full automated test suite passes.
- [ ] Official competition platform accepts the CSV.
- [ ] README and artifact inventory record the exact final commands, metrics, URLs, outputs, and validation evidence.

## Useful commands

Run the full test suite:

```powershell
.venv\Scripts\python.exe -m pytest -q
```

Run just submission checks:

```powershell
.venv\Scripts\python.exe -m pytest -q tests/test_submission.py
```

Start the EDA site locally (this is for testing, not public deployment):

```powershell
.venv\Scripts\python.exe -m pip install -r eda_site/requirements.txt
.venv\Scripts\python.exe eda_site/build_summary.py
.venv\Scripts\python.exe -m streamlit run eda_site/app.py
```

Reproduce the controlled training-only optimization sweep:

```powershell
.venv\Scripts\python.exe experiments\controlled_optimization.py
```

## Related documentation

- `README.md` — project overview and primary reproduction instructions.
- `eda_site/README.md` — local startup and deployment guidance.
- `experiments/FINAL_OPTIMIZATION_SPRINT.md` — hypotheses, validation results, and model keep/reject decisions.
- `artifacts/final_reproduction_inventory.md` — last recorded clean-run dimensions, checksums, and validation details.
- `notebooks/final_reproducible_solution.ipynb` — final end-to-end modeling workflow.

