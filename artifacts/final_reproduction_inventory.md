# Team 4BD478E9 final project artifact inventory

Run date: 2026-09-24

## Final package checks
- Environment: Python 3.12.14, XGBoost 3.4.1, pandas 3.0.6, scikit-learn 1.9.1.
- Notebook: 24 cells, 11 code cells; executed top-to-bottom in a fresh kernel; cell errors: 0.
- Data: 14,000 labeled training signals; 6,000 hidden-test signals; 6,987,663 training transactions and 3,027,575 test transactions.
- Model features: 28; selected model: xgboost_shallow; OOF ROC-AUC 0.578205492; five forward-only folds; warm-up excluded; OOF rows 11,982.
- Submission: `artifacts/predictions/team_4BD478E9.csv`, 6,000 rows, columns `signal_id,ehtimollik`, unique IDs: 6,000, probability range [0.046405, 0.451806]. Submission validator and 20 submission tests passed.
- Full test suite: 95 passed, 14 failed in pre-existing audit/basic-feature/validation/window-feature utilities.
- EDA website: Streamlit started locally at `http://localhost:8501/`; health endpoint returned `ok`; synthetic-data provenance and OOF model results are shown. Not deployed publicly.
- Controlled optimization: 22 candidates recorded; no test data loaded. The shallow XGBoost challenger was integrated after obtaining the strongest same-fold OOF performance.

## Key output checksums

| Relative path | Size (bytes) | SHA-256 |
|---|---:|---|
| `README.md` | 6,345 | `da237d53da38e5f552529014d57f910b60053beb376a580a8d08d68481fd21cf` |
| `config.yaml` | 3,803 | `d6efdf9ccbef86cd8a5884eede1c5e0cb7d3878534d9cff89cf374204b60ff50` |
| `requirements.txt` | 449 | `c8f42c9bb342d5c9d319099879c614c53bc5f55ce2b56b0f2b3ece6e5d8eb0cf` |
| `notebooks/final_reproducible_solution.ipynb` | 49,122 | `ed6a6f25f88791b54ad58d52e82879c0b2e8aedab3f10bcaee38ae02675075bd` |
| `src/submission/make_submission.py` | 6,829 | `78d9cb89d53a79f48f95df254ae6aba2b52a111247178e3caff92d5e52fac9ba` |
| `src/submission/validate_submission.py` | 5,615 | `abab3688741696ea7dac3917a72f449c8ff9cdbe9318ca0e27008b6ba02ab3e0` |
| `tests/test_submission.py` | 5,219 | `c96a1a8bdef9ece2ed1009c609dbea64ae2ba43270cc6ef235a7770d4e535f7a` |
| `artifacts/predictions/final_submission.csv` | 184,149 | `4d12d8f4a6c9600faa404208aa8a13554c3fabaf99467b046e246e18aa2035c2` |
| `artifacts/predictions/team_4BD478E9.csv` | 179,349 | `b1dde3ec1df5e7761d11c2227ab65e15ae286e148779d09a8f3bcd6bf6adf50b` |
| `artifacts/final_solution/features/manifest.json` | 1,211 | `a7fff32e0fc07a941fd4acddcf9c910958c9bcbf1ac6ed6cb4259ecced74ac06` |
| `artifacts/final_solution/features/train_features.parquet` | 2,168,442 | `761b5e1e1a5de9d86b20d9511e87f841587e45aac1dd31f863f4d5d01bc8d61b` |
| `artifacts/final_solution/features/test_features.parquet` | 946,460 | `6e5651c0486625d0855cf6447112973fc6393ecc7a6f9874616d0efea66b8de3` |
| `artifacts/final_solution/oof_predictions.parquet` | 114,991 | `06db17f5e083dbbfe9e45fb114ca6b0791e4e92838f0c0a950685a4d38c077cd` |
| `artifacts/final_solution/oof_metrics.csv` | 55 | `37d20b3a21ddc9bf679346e154ea9fb27d1bb6ff009ab8ebf24d19624d8396a9` |
| `artifacts/final_solution/fold_metrics.csv` | 399 | `796e9603bd3689b3582e8158def694aa956069fb145d1a9799e1c36023df40a8` |
| `eda_site/app.py` | 17,537 | `65e482e1887644f5b9b3497833c47d92dfe330c31993a300505331925bb969b0` |
| `eda_site/build_summary.py` | 11,243 | `091c34fd1a7018d58ea639b6e2e4653fdfc31493f6a4d234861daa19d66f57ee` |
| `eda_site/assets/eda_summary.json` | 24,974 | `a4bf2ab549e1c8536008c07482f958a2610a7ee0df762c0dfc993ec7a18b72a4` |
| `eda_site/requirements.txt` | 67 | `84f9cce84944547f8d29b28117def278dc0b794ce9b05b8dd67a941e57491335` |
| `eda_site/README.md` | 1,938 | `37b9b81d1a3eb32f8dd43f0f81a6dc71c62e2f86e5506404f2220a392ad4eed4` |
| `experiments/controlled_optimization.py` | 22,893 | `67c161fffb70e3733aeb0e0391b1f9d1d1e39e4c889a66fbf516f849587b43cb` |
| `experiments/FINAL_OPTIMIZATION_SPRINT.md` | 8,746 | `9e47d3ca05c8e18c1cb007d3088449a438a7f8023dd017737f1922787d480337` |
| `experiments/controlled_optimization/results.csv` | 16,433 | `a75ee88440ebc37b0cd1419726e6f4e676e1dce9b5899da25aafa775e7d4ac8b` |
| `experiments/controlled_optimization/oof_predictions.parquet` | 1,909,857 | `22f0daacf7ddeadc01ad439726c08ee45d4bb745e6e542c8610d380c46054129` |
| `experiments/controlled_optimization/metadata.json` | 1,596 | `ef5f6901485bed889043d3e172a3d6a19f4fa1623e7f13747d9e84a7db4f503a` |
| `experiments/controlled_optimization/baseline_fold_metrics.csv` | 546 | `4a4b1d0e35292a0fc5186780be08883cf9a1d15c1e80ee4d20b266a95e603011` |
