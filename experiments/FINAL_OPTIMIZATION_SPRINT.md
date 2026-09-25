# Controlled final optimization sprint

## Protocol

- Baseline: final notebook feature table (28 features), persisted OOF predictions, fixed five-fold expanding chronological validation, seed 42.
- The exact stored fold indices were reused; SHA-256: `f8bbe25b08b667469f4a7a46266e57d56384af8a9520f3c226bdd8c7f04017f6`. Each scored experiment covers the same 11,982 validation rows; the initial warm-up block remains unscored.
- Feature candidates were generated from training signals and training transactions only, after applying the as-of-signal cutoff. The runner does not load test data.
- ROC-AUC on these repeated experiment comparisons is exploratory and can be optimistic from trying alternatives. No test-set fitting, test-score selection, or hidden-leaderboard optimization was performed.
- Reproduce the full run from repository root with `python experiments/controlled_optimization.py` after installing `requirements.txt`. The fixed model configurations, seeds, fold-index hash, and candidate feature groups are in the runner and `metadata.json`.

## Results and decisions

The baseline’s final equal-weight ExtraTrees/HistGradientBoosting ensemble OOF ROC-AUC is **0.551687**. The strongest tested challenger is shallow XGBoost on the original 28 features: **0.578205** pooled OOF ROC-AUC (+0.026518), with a higher AUC than the baseline ensemble in all five folds. It also improves on the baseline HistGradientBoosting model in all five folds. This is an OOF validation estimate, not a test-set score. At sprint close the incumbent remained unchanged; in the follow-on packaging step for team 4BD478E9, this challenger was adopted as the final notebook predictor and used for submission generation.

| Experiment | OOF ROC-AUC | Fold AUCs / blend weights | Decision |
|---|---:|---|---|
| `baseline_extra_trees` | 0.540623 | `baseline fold scores in controlled_optimization/baseline_fold_metrics.csv` | Reference; unchanged. |
| `baseline_hist_gradient_boosting` | 0.550711 | `baseline fold scores in controlled_optimization/baseline_fold_metrics.csv` | Reference; unchanged. |
| `windows_14_60_180d` | 0.552772 | `[0.532468178, 0.535992987, 0.561507917, 0.587817055, 0.59950271]` | Keep feature definitions in experiment log; modest standalone gain with baseline HGB. |
| `velocity_change` | 0.554362 | `[0.539510921, 0.534820372, 0.558391491, 0.587183154, 0.612909363]` | Keep feature definitions in experiment log; modest standalone gain with baseline HGB. |
| `transaction_gap_stats` | 0.546695 | `[0.545418179, 0.52857925, 0.548705299, 0.59671007, 0.594268424]` | Reject as standalone HGB feature addition; below baseline HGB. |
| `transaction_sequence` | 0.554951 | `[0.560163341, 0.546448864, 0.54762959, 0.590474757, 0.609398884]` | Keep feature definitions in experiment log; modest standalone gain with baseline HGB. |
| `combined_feature_groups` | 0.562401 | `[0.557589242, 0.543329042, 0.567763232, 0.586853659, 0.621551559]` | Promising with HistGradientBoosting (+0.0117 vs its baseline); not needed by champion. |
| `subset_activity_no_amount` | 0.514737 | `[0.515783125, 0.501074589, 0.514720203, 0.53212658, 0.53800978]` | Reject; substantial loss. |
| `subset_without_calendar` | 0.549699 | `[0.529008301, 0.534843306, 0.552891418, 0.596879835, 0.605259762]` | Reject; no gain. |
| `hgb_smoother` | 0.560378 | `[0.544039198, 0.547347005, 0.560033531, 0.59726536, 0.610796789]` | Keep only as recorded parameter comparison; below XGBoost champion. |
| `hgb_more_regularized` | 0.559013 | `[0.551726708, 0.535469194, 0.561540806, 0.591427281, 0.616026112]` | Keep only as recorded parameter comparison; below XGBoost champion. |
| `extra_trees_robust` | 0.544460 | `[0.545226861, 0.539976178, 0.541524282, 0.554409293, 0.542069495]` | Reject; below baseline ExtraTrees. |
| `lightgbm_diverse` | 0.569775 | `[0.560782019, 0.561125044, 0.560534886, 0.575961932, 0.610477504]` | Retain as diversity reference; lower than XGBoost challenger. |
| `xgboost_diverse` | 0.574200 | `[0.566858233, 0.558793871, 0.565889365, 0.588399107, 0.622104103]` | Retain as family-diversity reference; beaten by shallow XGBoost. |
| `xgboost_enriched` | 0.577737 | `[0.577934808, 0.55559415, 0.569970801, 0.591481639, 0.622469709]` | Keep as close comparator; does not beat shallow XGBoost and adds 32 features. |
| `xgboost_shallow` | 0.578205 | `[0.570862248, 0.556754927, 0.568971298, 0.602231198, 0.624287813]` | Keep as sprint champion: +0.0265 vs baseline equal-weight ensemble; beats it in every fold. |
| `forward_blend_hgb_lightgbm_diverse` | 0.565897 | `weights [0.5, 0.75, 0.75, 0.75, 0.75]` | Reject for deployment; forward-only blend did not beat its challenger alone. |
| `forward_blend_hgb_xgboost_diverse` | 0.570141 | `weights [0.5, 0.75, 0.75, 0.75, 0.75]` | Reject for deployment; forward-only blend did not beat its challenger alone. |
| `forward_blend_hgb_xgboost_enriched` | 0.573357 | `weights [0.5, 0.75, 0.75, 0.75, 0.75]` | Reject for deployment; forward-only blend did not beat its challenger alone. |
| `forward_blend_hgb_xgboost_shallow` | 0.573057 | `weights [0.5, 0.75, 0.75, 0.75, 0.75]` | Reject for deployment; forward-only blend did not beat its challenger alone. |
| `forward_blend_hgb_hgb_smoother` | 0.559201 | `weights [0.5, 0.75, 0.75, 0.75, 0.75]` | Reject for deployment; forward-only blend did not beat its challenger alone. |
| `forward_blend_hgb_hgb_more_regularized` | 0.557109 | `weights [0.5, 0.75, 0.75, 0.75, 0.75]` | Reject for deployment; forward-only blend did not beat its challenger alone. |

## Hypotheses, configurations, and full predictions

The `results.csv` file records every tested hypothesis, model configuration, fixed-fold count, feature set, feature count, fold AUCs, and OOF AUC. `oof_predictions.parquet` stores the per-signal OOF prediction columns and leaves warm-up rows missing. `metadata.json` records the seed, row and feature counts, exact fold hashes, generated feature definitions, and `test_data_loaded: false`.

### Main findings

- **Alternative windows:** 14-, 60-, and 180-day windows lift baseline HGB pooled OOF AUC to 0.552772. Useful but small; adding them to the combined feature experiment does not displace the simpler 28-feature XGBoost candidate.
- **Velocity/change:** four normalized recent-versus-prior count/amount rate ratios produce 0.554362 with baseline HGB.
- **Transaction gaps:** mean/median/spread/quantile/CV and short/long-gap shares score 0.546695 with baseline HGB, so do not retain as a standalone addition.
- **Sequence:** final direction/type, last amount relative to mean, transition counts, and switch rates score 0.554951 with baseline HGB.
- **Combined feature groups:** windows + velocity + gaps + sequence score 0.562401 with baseline HGB, a meaningful improvement over its 0.550711 baseline. XGBoost with this 60-feature set scores 0.577737, slightly below its 28-feature shallow variant.
- **Hyperparameters:** HGB candidates score 0.560378 and 0.559013. Shallow XGBoost (500 trees, depth 2, learning rate 0.025, min child weight 15, L2 5) scores 0.578205.
- **Diversity/blending:** LightGBM scores 0.569775; depth-3 XGBoost scores 0.574200. A causal forward-only blend uses weights selected from prior folds on a fixed `{0.25, 0.50, 0.75}` grid; the shallow-XGBoost/HGB blend scores 0.573057, below shallow XGBoost alone. No blend is retained.
- **Feature subsets:** removing amount information collapses HGB to 0.514737; dropping only calendar fields scores 0.549699. Neither subset is retained.
- **Target encoding:** skipped. `signal_id` is unique per signal and is not a repeated relational entity key; encoding it would memorize IDs. Transaction type is a small categorical transaction attribute, not a relational key.

## Scope and artifacts

At the end of the controlled sprint, the incumbent notebook and submission were unchanged. In the subsequent competition packaging step, the best challenger was integrated into `notebooks/final_reproducible_solution.ipynb` and used to generate the team submission. The baseline predictions and metrics remain preserved in this experiment directory for comparison.

- Runner: [`controlled_optimization.py`](controlled_optimization.py)
- Full metric/config/hypothesis log: [`controlled_optimization/results.csv`](controlled_optimization/results.csv)
- OOF predictions: [`controlled_optimization/oof_predictions.parquet`](controlled_optimization/oof_predictions.parquet)
- Baseline fold scores: [`controlled_optimization/baseline_fold_metrics.csv`](controlled_optimization/baseline_fold_metrics.csv)
- Fold/source metadata: [`controlled_optimization/metadata.json`](controlled_optimization/metadata.json)
