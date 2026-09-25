# WIUT HACKATHON FinTech: leakage-safe classification and EDA

For a detailed record of completed work, current risks, verification evidence, and a prioritized next-step plan, see [`PROJECT_STATUS_AND_NEXT_STEPS.md`](PROJECT_STATUS_AND_NEXT_STEPS.md).

This repository contains a reproducible competition pipeline for predicting whether a financial signal is escalated (`eskalatsiya`) or dismissed. It also includes a public-facing explanatory EDA report; the website displays aggregate analysis and does not serve predictions.

## Data and project layout

Raw inputs are expected under `data/raw/`: `train_signals.csv`, `test_signals.csv`, `train_transactions.parquet`, `test_transactions.parquet`, and the competition sample submission CSV. The main reproducible solution is `notebooks/final_reproducible_solution.ipynb`; source utilities are under `src/`, tests under `tests/`, and generated outputs under `artifacts/`. The site is in `eda_site/`.

The final notebook reads the raw files directly, audits signal and transaction dates, filters transactions to those occurring on or before each signal timestamp, generates 28 per-signal aggregate features, and evaluates a fixed shallow XGBoost model using five expanding, forward-only OOF folds after a chronological warm-up period. The selected configuration was the strongest controlled-sprint candidate (OOF ROC-AUC 0.578205); this is an internal validation score, not a hidden-test score. No organizer-only data or private/local machine paths are required by the EDA site.

## Reproduce the final competition result

Use Python 3.11 or newer (the clean reproduction run used Python 3.12). From the repository root, create an isolated environment and install the project requirements. The commands below use Windows PowerShell paths; on macOS/Linux, replace `.venv\Scripts\python.exe` with `.venv/bin/python`.

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
.venv\Scripts\python.exe -m src.submission.make_submission
.venv\Scripts\python.exe -m src.submission.validate_submission
.venv\Scripts\python.exe -m pytest -q
```

The `nbconvert --execute` command starts a fresh kernel and executes every cell from top to bottom. The notebook reports its Python/library versions, data audit, temporal rule, fold-level and OOF ROC-AUC, final predictor weight, feature count, and prediction output. Its validation is chronological: the earliest date block is warm-up only, then each scored fold trains on earlier dates and validates on the next date block. The warm-up rows are intentionally absent from OOF scoring.

The first notebook run computes the signal-level feature tables from raw transactions and caches them as Parquet under `artifacts/final_solution/features/`. A source-file manifest invalidates the cache when its inputs change. To force a rebuild, set `FORCE_REBUILD_FEATURES = True` in the notebook configuration cell or remove that cache directory. The raw transaction Parquet files are large, so feature generation can take several minutes and needs sufficient disk space.

The notebook writes OOF predictions and metrics to `artifacts/final_solution/` and test probabilities to `artifacts/predictions/final_submission.csv`. The submission utility reads `team.id` from `config.yaml`, preserves the exact order in `test_signals.csv`, and writes `artifacts/predictions/team_<TEAM_ID>.csv` (currently `team_4BD478E9.csv`). The validator uses the same configured ID by default. It validates the exact two-column schema, IDs, row count, uniqueness, probability range, and sample-submission structure. CLI options can override the prediction input, test IDs, sample file, team ID, and output directory; run `python -m src.submission.make_submission --help` for details.

## Tests

Run the entire suite with:

```powershell
.venv\Scripts\python.exe -m pytest -q
```

Run the official-submission-specific checks alone with:

```powershell
.venv\Scripts\python.exe -m pytest -q tests/test_submission.py
```

## Controlled optimization sprint

The fixed-fold, training-only challenger sweep can be reproduced with:

```powershell
.venv\Scripts\python.exe experiments\controlled_optimization.py
```

It reuses the exact persisted chronological folds, does not load test data, and writes per-experiment hypotheses, configurations, fold/OOF scores, and predictions under `experiments/controlled_optimization/`. See [`experiments/FINAL_OPTIMIZATION_SPRINT.md`](experiments/FINAL_OPTIMIZATION_SPRINT.md) for results and keep/reject decisions. The final reproducible notebook now uses the winning 28-feature shallow XGBoost candidate; the experiment log retains all keep/reject comparisons.

## Run the EDA website locally

The Streamlit site has its own deployment requirements. Install them into the same environment, build/update its aggregate-only summary, and launch it from the repository root:

```powershell
.venv\Scripts\python.exe -m pip install -r eda_site/requirements.txt
.venv\Scripts\python.exe eda_site/build_summary.py
.venv\Scripts\python.exe -m streamlit run eda_site/app.py
```

Open the local URL printed by Streamlit. `eda_site/assets/eda_summary.json` contains aggregate chart data and OOF metric summaries and is used for fast app startup; rebuild it when the raw data or model artifacts change. The site identifies the competition data as synthetic. The site has been verified locally, but is not deployed publicly. Deployment guidance is in [`eda_site/README.md`](eda_site/README.md).

## Reproduction artifacts

The latest clean run inventory, including output dimensions, verification results, and SHA-256 checksums, is recorded in [`artifacts/final_reproduction_inventory.md`](artifacts/final_reproduction_inventory.md).
