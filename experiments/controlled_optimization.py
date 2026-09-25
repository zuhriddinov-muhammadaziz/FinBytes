"""Small, reproducible optimization sprint on the final solution's fixed folds.

All candidate selection uses training OOF only. Test data are never loaded here.
Run from the repository root with: python experiments/controlled_optimization.py
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "raw"
ARTIFACTS = ROOT / "artifacts" / "final_solution"
FEATURES = ARTIFACTS / "features"
OUT = ROOT / "experiments" / "controlled_optimization"
SEED = 42

BASE_HGB = dict(max_iter=250, learning_rate=0.07, max_leaf_nodes=31,
                l2_regularization=1.0, random_state=SEED)
BASE_ET = dict(n_estimators=300, max_features=0.8, min_samples_leaf=5,
               class_weight="balanced", n_jobs=-1, random_state=SEED)


def load_fixed_folds(n_rows: int) -> list[dict]:
    definitions = json.loads((ARTIFACTS / "fold_definitions.json").read_text(encoding="utf-8"))
    folds = [{"fold": row["fold"], "train_idx": np.asarray(row["train_idx"], dtype=int),
              "validation_idx": np.asarray(row["validation_idx"], dtype=int)} for row in definitions]
    covered = np.concatenate([f["validation_idx"] for f in folds])
    if len(np.unique(covered)) != len(covered) or covered.min() < 0 or covered.max() >= n_rows:
        raise ValueError("Saved validation folds are invalid for the training rows")
    return folds


def extra_transaction_features(signals: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """Build calendar-window, transaction-gap, and sequence summaries from train only."""
    sid, signal_date = "signal_id", "signal_sanasi"
    tx = pd.read_parquet(DATA / "train_transactions.parquet",
                         columns=[sid, "tranzaksiya_vaqti", "kirim_chiqim", "tranzaksiya_turi", "miqdor_indeksi"])
    tx["tranzaksiya_vaqti"] = pd.to_datetime(tx["tranzaksiya_vaqti"], errors="coerce")
    tx["miqdor_indeksi"] = pd.to_numeric(tx["miqdor_indeksi"], errors="coerce")
    tx = tx.merge(signals[[sid, signal_date]], on=sid, how="left", validate="many_to_one")
    tx = tx.loc[tx["tranzaksiya_vaqti"].notna() &
                 (tx["tranzaksiya_vaqti"] <= tx[signal_date])].copy()
    date_gap = tx[signal_date] - tx["tranzaksiya_vaqti"]
    tx["_age_days"] = date_gap.dt.total_seconds() / 86400
    groups: dict[str, list[str]] = {"windows": [], "velocity": [], "gaps": [], "sequence": []}
    features = pd.DataFrame({sid: signals[sid].to_numpy()})

    # Hypothesis: intermediate/long windows reveal stable behavior missed by 7/30/90 days.
    grouped = tx.groupby(sid, sort=False, observed=True)
    for days in (14, 60, 180):
        recent = tx.loc[tx["_age_days"] < days]
        by_signal = recent.groupby(sid, sort=False, observed=True)
        names = [f"transaction_count_{days}d", f"amount_sum_{days}d"]
        small = pd.DataFrame({names[0]: by_signal.size(), names[1]: by_signal["miqdor_indeksi"].sum()})
        features = features.merge(small, left_on=sid, right_index=True, how="left", validate="one_to_one")
        groups["windows"].extend(names)

    # Hypothesis: the latest transaction rate and volume change relative to prior windows.
    base_counts = grouped.size()
    _ = base_counts  # Retain grouping semantics explicitly; rate features use cached base windows below.
    for name, numerator, denominator, scale in [
        ("count_rate_ratio_7d_vs_30d", "transaction_count_7d", "transaction_count_30d", 23 / 7),
        ("count_rate_ratio_30d_vs_90d", "transaction_count_30d", "transaction_count_90d", 60 / 30),
        ("amount_rate_ratio_7d_vs_30d", "amount_sum_7d", "amount_sum_30d", 23 / 7),
        ("amount_rate_ratio_30d_vs_90d", "amount_sum_30d", "amount_sum_90d", 60 / 30),
    ]:
        groups["velocity"].append(name)
        # Derived after joining with the existing feature table below.

    # Gap and sequence summaries need a deterministic event order per signal.
    tx.sort_values([sid, "tranzaksiya_vaqti"], kind="mergesort", inplace=True)
    by_signal = tx.groupby(sid, sort=False, observed=True)
    tx["_gap_days"] = by_signal["tranzaksiya_vaqti"].diff().dt.total_seconds() / 86400
    gaps = tx.groupby(sid, sort=False, observed=True)["_gap_days"].agg(
        gap_mean_days="mean", gap_median_days="median", gap_std_days="std",
        gap_min_days="min", gap_q25_days=lambda x: x.quantile(.25),
        gap_q75_days=lambda x: x.quantile(.75), gap_max_days="max")
    gap_extra = pd.DataFrame({
        "gap_cv": gaps["gap_std_days"] / gaps["gap_mean_days"].replace(0, np.nan),
        "gap_short_share": tx.assign(_short=tx["_gap_days"].le(1)).groupby(sid, sort=False)["_short"].mean(),
        "gap_long_share": tx.assign(_long=tx["_gap_days"].ge(30)).groupby(sid, sort=False)["_long"].mean(),
    })
    gap_features = gaps.join(gap_extra)
    gap_features = gap_features.reindex(columns=list(gaps.columns) + list(gap_extra.columns))
    features = features.merge(gap_features, left_on=sid, right_index=True, how="left", validate="one_to_one")
    groups["gaps"].extend(gap_features.columns.tolist())

    tx["_previous_direction"] = by_signal["kirim_chiqim"].shift()
    tx["_previous_type"] = by_signal["tranzaksiya_turi"].shift()
    tx["_direction_switch"] = tx["kirim_chiqim"].ne(tx["_previous_direction"]) & tx["_previous_direction"].notna()
    tx["_type_switch"] = tx["tranzaksiya_turi"].ne(tx["_previous_type"]) & tx["_previous_type"].notna()
    tx["_direction_transition"] = tx["_previous_direction"].astype("string") + ">" + tx["kirim_chiqim"].astype("string")
    tx["_type_transition"] = tx["_previous_type"].astype("string") + ">" + tx["tranzaksiya_turi"].astype("string")
    last = tx.groupby(sid, sort=False, observed=True).tail(1).set_index(sid)
    seq = pd.DataFrame(index=last.index)
    for value in ("kirim", "chiqim"):
        seq[f"last_direction_{value}"] = last["kirim_chiqim"].eq(value).astype("int8")
    for value in ("karta", "bank_otkazmasi", "naqd", "xalqaro"):
        seq[f"last_type_{value}"] = last["tranzaksiya_turi"].eq(value).astype("int8")
    seq["last_amount"] = last["miqdor_indeksi"]
    seq["last_amount_to_mean_ratio"] = last["miqdor_indeksi"] / grouped["miqdor_indeksi"].mean().replace(0, np.nan)
    seq["direction_switch_rate"] = tx.groupby(sid, sort=False)["_direction_switch"].mean()
    seq["type_switch_rate"] = tx.groupby(sid, sort=False)["_type_switch"].mean()
    seq["direction_transition_count"] = tx.groupby(sid, sort=False)["_direction_transition"].nunique()
    seq["type_transition_count"] = tx.groupby(sid, sort=False)["_type_transition"].nunique()
    seq_names = seq.columns.tolist()
    features = features.merge(seq, left_on=sid, right_index=True, how="left", validate="one_to_one")
    groups["sequence"].extend(seq_names)

    # Preserve one row per train signal and fill only missing aggregate values.
    features = features.set_index(sid).reindex(signals[sid]).reset_index()
    features = features.replace([np.inf, -np.inf], np.nan)
    return features, groups


def make_matrix(base: pd.DataFrame, extra: pd.DataFrame, names: list[str]) -> pd.DataFrame:
    all_features = base.merge(extra, on="signal_id", how="left", validate="one_to_one", sort=False)
    selected = all_features[names].copy().replace([np.inf, -np.inf], np.nan).fillna(0)
    return selected.astype("float32")


def evaluate(name: str, hypothesis: str, X: pd.DataFrame, y: pd.Series, folds: list[dict], factory,
             configs: dict, preds_out: dict, results: list[dict]) -> None:
    pred = np.full(len(y), np.nan, dtype="float64")
    fold_scores = []
    for fold in folds:
        tr, va = fold["train_idx"], fold["validation_idx"]
        model = factory()
        model.fit(X.iloc[tr], y.iloc[tr])
        classes = list(model.classes_)
        pred[va] = model.predict_proba(X.iloc[va])[:, classes.index(1)]
        fold_scores.append(roc_auc_score(y.iloc[va], pred[va]))
    mask = np.isfinite(pred)
    score = roc_auc_score(y.iloc[mask], pred[mask])
    preds_out[name] = pred
    results.append({"experiment": name, "hypothesis": hypothesis, "oof_roc_auc": score,
                    "mean_fold_auc": float(np.mean(fold_scores)), "fold_auc_std": float(np.std(fold_scores, ddof=0)),
                    "fold_scores": json.dumps([round(float(v), 9) for v in fold_scores]),
                    "n_features": X.shape[1], "feature_set": ",".join(X.columns),
                    "configuration": json.dumps(configs, sort_keys=True), "folds": len(folds),
                    "oof_rows": int(mask.sum())})
    print(f"{name:32s} OOF={score:.6f} folds={','.join(f'{v:.4f}' for v in fold_scores)} features={X.shape[1]}", flush=True)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    signals = pd.read_csv(DATA / "train_signals.csv", parse_dates=["signal_sanasi"])
    y = signals["eskalatsiya"].astype("int8")
    folds = load_fixed_folds(len(signals))
    base = pd.read_parquet(FEATURES / "train_features.parquet")
    feature_names = [c for c in base.columns if c != "signal_id"]
    if base["signal_id"].tolist() != signals["signal_id"].tolist():
        raise ValueError("Feature cache row order no longer matches raw training IDs")
    extra, groups = extra_transaction_features(signals)
    # Existing OOF is loaded for exact reference and blending only; align through persisted fold row indices.
    baseline = pd.read_parquet(ARTIFACTS / "oof_predictions.parquet")
    covered = np.sort(np.concatenate([f["validation_idx"] for f in folds]))
    if len(baseline) != len(covered) or not np.array_equal(baseline["eskalatsiya"].to_numpy(), y.iloc[covered].to_numpy()):
        raise ValueError("Saved baseline OOF cannot be aligned to the fixed fold definitions")
    if not np.array_equal(pd.to_datetime(baseline["signal_date"]).to_numpy(), signals["signal_sanasi"].iloc[covered].to_numpy()):
        raise ValueError("Saved baseline OOF dates do not align to raw training rows")

    pred_store: dict[str, np.ndarray] = {}
    results: list[dict] = []
    base_oof = {}
    for col in ["extra_trees", "hist_gradient_boosting"]:
        p = np.full(len(y), np.nan)
        p[covered] = baseline[col].to_numpy()
        pred_store[f"baseline_{col}"] = p
        base_oof[col] = p
        results.append({"experiment": f"baseline_{col}", "hypothesis": "Reference existing final-pipeline OOF result; unchanged fixed folds.",
                        "oof_roc_auc": roc_auc_score(y.iloc[covered], p[covered]), "mean_fold_auc": np.nan,
                        "fold_auc_std": np.nan, "fold_scores": "baseline fold scores in artifacts/final_solution/fold_metrics.csv",
                        "n_features": len(feature_names), "feature_set": "baseline_28",
                        "configuration": "existing final notebook configuration", "folds": len(folds), "oof_rows": len(covered)})

    # Feature-window hypothesis.
    window_names = groups["windows"]
    X_window = make_matrix(base, extra, feature_names + window_names)
    evaluate("windows_14_60_180d", "Intermediate/longer recency windows capture behavior not represented by 7/30/90-day aggregates.",
             X_window, y, folds, lambda: HistGradientBoostingClassifier(**BASE_HGB), BASE_HGB, pred_store, results)

    # Change/velocity hypothesis using established 7/30/90 day windows.
    velocity = pd.DataFrame({"signal_id": base["signal_id"]})
    for name, numerator, denominator, scale in [
        ("count_rate_ratio_7d_vs_30d", "transaction_count_7d", "transaction_count_30d", 23 / 7),
        ("count_rate_ratio_30d_vs_90d", "transaction_count_30d", "transaction_count_90d", 60 / 30),
        ("amount_rate_ratio_7d_vs_30d", "amount_sum_7d", "amount_sum_30d", 23 / 7),
        ("amount_rate_ratio_30d_vs_90d", "amount_sum_30d", "amount_sum_90d", 60 / 30),
    ]:
        a = base[numerator].fillna(0).to_numpy(dtype=float)
        b = base[denominator].fillna(0).to_numpy(dtype=float)
        # Normalize recent and prior per-day activity; zero historical activity remains zero after safe division.
        prior = np.maximum(b - a, 0.0)
        velocity[name] = np.divide(a / (7 if "7d" in numerator else 30), prior / scale,
                                   out=np.zeros(len(a)), where=prior > 0)
    vel_names = velocity.columns.drop("signal_id").tolist()
    X_velocity = make_matrix(base, velocity, feature_names + vel_names)
    evaluate("velocity_change", "Recent per-day transaction and amount intensity relative to the preceding part of each existing window highlights acceleration or deceleration.",
             X_velocity, y, folds, lambda: HistGradientBoostingClassifier(**BASE_HGB), BASE_HGB, pred_store, results)

    # Gap statistics hypothesis.
    X_gaps = make_matrix(base, extra, feature_names + groups["gaps"])
    evaluate("transaction_gap_stats", "Irregular spacing, burstiness, and long inactivity gaps may distinguish escalation behavior.",
             X_gaps, y, folds, lambda: HistGradientBoostingClassifier(**BASE_HGB), BASE_HGB, pred_store, results)

    # Transaction-sequence hypothesis.
    X_sequence = make_matrix(base, extra, feature_names + groups["sequence"])
    evaluate("transaction_sequence", "Final direction/type and transition rates may add sequence information beyond unordered counts.",
             X_sequence, y, folds, lambda: HistGradientBoostingClassifier(**BASE_HGB), BASE_HGB, pred_store, results)

    # Combined candidate checks whether independently weak feature groups complement one another.
    all_extra = window_names + vel_names + groups["gaps"] + groups["sequence"]
    X_all = make_matrix(base, pd.concat([extra, velocity.drop(columns="signal_id")], axis=1), feature_names + all_extra)
    evaluate("combined_feature_groups", "The window, velocity, gap, and sequence groups provide complementary gains when modeled together.",
             X_all, y, folds, lambda: HistGradientBoostingClassifier(**BASE_HGB), BASE_HGB, pred_store, results)

    # Robust feature subset hypotheses.
    count_activity = [c for c in feature_names if ("count" in c or c in {"active_days", "incoming_share", "outgoing_share", "days_since_last_transaction", "history_span_days", "signal_month", "signal_day_of_week"})]
    no_calendar = [c for c in feature_names if c not in {"signal_month", "signal_day_of_week"}]
    evaluate("subset_activity_no_amount", "Counts, activity, and recency may be more robust than noisy amount magnitude statistics.",
             make_matrix(base, pd.DataFrame({"signal_id": base["signal_id"]}), count_activity), y, folds,
             lambda: HistGradientBoostingClassifier(**BASE_HGB), BASE_HGB, pred_store, results)
    evaluate("subset_without_calendar", "Removing global calendar fields tests whether gains come from activity rather than date memorization.",
             make_matrix(base, pd.DataFrame({"signal_id": base["signal_id"]}), no_calendar), y, folds,
             lambda: HistGradientBoostingClassifier(**BASE_HGB), BASE_HGB, pred_store, results)

    # Small, predeclared model-parameter checks.
    hgb_params = [
        ("hgb_smoother", dict(max_iter=300, learning_rate=.05, max_leaf_nodes=15, l2_regularization=2.0, random_state=SEED),
         "A smaller tree and stronger regularization may stabilize chronological drift."),
        ("hgb_more_regularized", dict(max_iter=300, learning_rate=.05, max_leaf_nodes=31, l2_regularization=5.0, random_state=SEED),
         "Stronger L2 regularization may improve temporal robustness without reducing useful capacity."),
    ]
    for name, params, hypothesis in hgb_params:
        evaluate(name, hypothesis, base[feature_names].fillna(0).astype("float32"), y, folds,
                 lambda params=params: HistGradientBoostingClassifier(**params), params, pred_store, results)
    et_params = dict(n_estimators=300, max_features=.6, min_samples_leaf=10,
                     class_weight="balanced", n_jobs=-1, random_state=SEED)
    evaluate("extra_trees_robust", "Fewer candidate features and larger leaves may lower variance under changing date blocks.",
             base[feature_names].fillna(0).astype("float32"), y, folds,
             lambda: ExtraTreesClassifier(**et_params), et_params, pred_store, results)

    # Diversity candidates: one LightGBM and one XGBoost configuration, on unchanged features/folds.
    try:
        from lightgbm import LGBMClassifier
        params = dict(n_estimators=350, learning_rate=.03, num_leaves=15, max_depth=-1,
                      min_child_samples=30, colsample_bytree=.8, subsample=.8,
                      reg_lambda=2.0, class_weight="balanced", n_jobs=-1,
                      random_state=SEED, verbosity=-1)
        evaluate("lightgbm_diverse", "A regularized boosting family may add ranking diversity to the two sklearn tree baselines.",
                 base[feature_names].fillna(0).astype("float32"), y, folds,
                 lambda: LGBMClassifier(**params), params, pred_store, results)
    except ImportError:
        results.append({"experiment": "lightgbm_diverse", "hypothesis": "Check LightGBM model-family diversity.", "status": "skipped: dependency missing"})
    try:
        from xgboost import XGBClassifier
        params = dict(n_estimators=350, max_depth=3, learning_rate=.03, min_child_weight=10,
                      subsample=.8, colsample_bytree=.8, reg_lambda=3.0,
                      eval_metric="logloss", n_jobs=-1, random_state=SEED)
        evaluate("xgboost_diverse", "A shallow regularized boosting family may add complementary OOF ranking errors.",
                 base[feature_names].fillna(0).astype("float32"), y, folds,
                 lambda: XGBClassifier(**params), params, pred_store, results)
        # Follow-up hypotheses predeclared from the first-pass signal: use the enriched features
        # with the best diverse family, then check one shallower, more regularized XGBoost setting.
        evaluate("xgboost_enriched", "The strongest first-pass model family may benefit from the complementary recency, velocity, gap, and sequence features.",
                 X_all, y, folds, lambda: XGBClassifier(**params), params, pred_store, results)
        shallow_params = dict(n_estimators=500, max_depth=2, learning_rate=.025, min_child_weight=15,
                              subsample=.8, colsample_bytree=.8, reg_lambda=5.0,
                              eval_metric="logloss", n_jobs=-1, random_state=SEED)
        evaluate("xgboost_shallow", "A shallower tree with slower learning and stronger regularization may be more stable across forward periods.",
                 base[feature_names].fillna(0).astype("float32"), y, folds,
                 lambda: XGBClassifier(**shallow_params), shallow_params, pred_store, results)
    except ImportError:
        results.append({"experiment": "xgboost_diverse", "hypothesis": "Check XGBoost model-family diversity.", "status": "skipped: dependency missing"})

    # Leakage-safe blend diagnostic: fixed weight grid, OOF only, never test-score tuned.
    fold_for_row = np.full(len(y), -1, dtype=int)
    for fold in folds:
        fold_for_row[fold["validation_idx"]] = fold["fold"]
    blend_candidates = [k for k in ("lightgbm_diverse", "xgboost_diverse", "xgboost_enriched",
                                   "xgboost_shallow", "hgb_smoother", "hgb_more_regularized") if k in pred_store]
    blend_records = []
    for candidate in blend_candidates:
        candidate_pred = pred_store[candidate]
        # Forward meta-evaluation: choose weights only from earlier OOF folds, then score the next fold.
        forward_pred, forward_y, selected_weights = [], [], []
        for fold in folds:
            prior_mask = (fold_for_row >= 1) & (fold_for_row < fold["fold"])
            validation_idx = fold["validation_idx"]
            if prior_mask.sum() >= 100 and y.iloc[prior_mask].nunique() == 2:
                weight_grid = (.25, .5, .75)
                chosen = max(weight_grid, key=lambda w: roc_auc_score(
                    y.iloc[prior_mask], w * candidate_pred[prior_mask] + (1 - w) * base_oof["hist_gradient_boosting"][prior_mask]))
            else:
                chosen = .5
            mix = chosen * candidate_pred[validation_idx] + (1 - chosen) * base_oof["hist_gradient_boosting"][validation_idx]
            forward_pred.extend(mix.tolist())
            forward_y.extend(y.iloc[validation_idx].tolist())
            selected_weights.append(round(float(chosen), 3))
        blend_records.append({"experiment": f"forward_blend_hgb_{candidate}",
                              "hypothesis": "A blend weight selected on prior chronological OOF folds may exploit diversity without using later-fold labels.",
                              "oof_roc_auc": roc_auc_score(forward_y, forward_pred),
                              "mean_fold_auc": np.nan, "fold_auc_std": np.nan,
                              "fold_scores": json.dumps(selected_weights), "n_features": np.nan,
                              "feature_set": "OOF blend only", "configuration": json.dumps({"candidate_weight_grid": [.25,.5,.75], "weights_by_fold": selected_weights}),
                              "folds": len(folds), "oof_rows": len(forward_y)})
    results.extend(blend_records)

    # Save complete OOF records for audit/reuse; warm-up rows remain NaN by design.
    prediction_frame = pd.DataFrame({"signal_id": signals["signal_id"], "target": y})
    for name, values in pred_store.items():
        prediction_frame[name] = values
    prediction_frame.to_parquet(OUT / "oof_predictions.parquet", index=False)
    results_df = pd.DataFrame(results)
    results_df.to_csv(OUT / "results.csv", index=False)
    fold_hash = hashlib.sha256((ARTIFACTS / "fold_definitions.json").read_bytes()).hexdigest()
    metadata = {
        "seed": SEED, "fixed_fold_definitions_sha256": fold_hash,
        "fold_numbers": [f["fold"] for f in folds], "validation_indices_sha256": hashlib.sha256(
            np.concatenate([f["validation_idx"] for f in folds]).astype("int64").tobytes()).hexdigest(),
        "train_signal_rows": len(signals), "base_feature_count": len(feature_names),
        "experiment_count": len(results_df), "test_data_loaded": False,
        "feature_groups": groups, "feature_generation": "as-of-signal train transactions only; timestamp stable-sorted for sequence summaries",
        "runtime_seconds": round(time.time() - START, 3),
    }
    (OUT / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("\nResults written to", OUT.relative_to(ROOT))
    print(results_df[["experiment", "oof_roc_auc", "mean_fold_auc", "fold_auc_std"]].to_string(index=False))


START = time.time()
if __name__ == "__main__":
    main()
