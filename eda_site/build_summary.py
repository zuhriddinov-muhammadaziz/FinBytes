"""Build a compact, aggregate-only EDA artifact from the project data."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / "assets" / "eda_summary.json"
ARTIFACTS = ROOT / "artifacts" / "final_solution"


def _date_counts(series: pd.Series) -> list[dict]:
    vals = series.dropna().dt.to_period("M").value_counts().sort_index()
    return [{"month": str(k), "count": int(v)} for k, v in vals.items()]


def _distribution(series: pd.Series, bins: int = 30) -> list[dict]:
    s = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if s.empty:
        return []
    if s.nunique() <= 1:
        return [{"bin": str(s.iloc[0]), "count": int(len(s))}]
    counts, edges = np.histogram(s.clip(upper=s.quantile(.99)), bins=bins)
    return [{"bin": f"{edges[i]:.3g}–{edges[i+1]:.3g}", "count": int(counts[i])} for i in range(len(counts))]


def _model_summary() -> dict:
    """Expose only aggregate OOF metrics; never include signal IDs or row predictions."""
    metrics_path = ARTIFACTS / "oof_metrics.csv"
    folds_path = ARTIFACTS / "fold_metrics.csv"
    predictions_path = ARTIFACTS / "oof_predictions.parquet"
    features_path = ARTIFACTS / "features" / "train_features.parquet"
    result = {"model": None, "oof_roc_auc": None, "oof_rows": None,
              "feature_count": None, "fold_count": 0, "fold_auc": [],
              "baseline_models": []}
    if metrics_path.exists():
        metrics = pd.read_csv(metrics_path)
        if {"model", "oof_roc_auc"}.issubset(metrics.columns) and not metrics.empty:
            best = metrics.loc[metrics["oof_roc_auc"].idxmax()]
            result["model"] = str(best["model"])
            result["oof_roc_auc"] = round(float(best["oof_roc_auc"]), 6)
    if folds_path.exists():
        folds = pd.read_csv(folds_path)
        selected = folds[folds["model"] == result["model"]] if result["model"] else folds.iloc[0:0]
        result["fold_count"] = int(selected["fold"].nunique()) if not selected.empty else 0
        result["fold_auc"] = [{"fold": int(r.fold), "roc_auc": round(float(r.roc_auc), 6)}
                               for r in selected.itertuples()]
    if predictions_path.exists():
        result["oof_rows"] = int(len(pd.read_parquet(predictions_path, columns=["signal_date"])))
    if features_path.exists():
        result["feature_count"] = max(len(pd.read_parquet(features_path, columns=None).columns) - 1, 0)
    sprint_path = ROOT / "experiments" / "controlled_optimization" / "results.csv"
    if sprint_path.exists():
        sprint = pd.read_csv(sprint_path)
        baselines = sprint[sprint["experiment"].isin(["baseline_extra_trees", "baseline_hist_gradient_boosting"])]
        result["baseline_models"] = [{"model": str(r.experiment).replace("baseline_", ""),
                                       "oof_roc_auc": round(float(r.oof_roc_auc), 6)}
                                      for r in baselines.itertuples()]
    return result


def main() -> None:
    d = ROOT / "data" / "raw"
    train = pd.read_csv(d / "train_signals.csv", parse_dates=["signal_sanasi"])
    test = pd.read_csv(d / "test_signals.csv", parse_dates=["signal_sanasi"])
    tx = pd.read_parquet(d / "train_transactions.parquet")
    train_tx_rows = len(tx)
    tx_test = pd.read_parquet(d / "test_transactions.parquet")
    test_tx_rows = len(tx_test)
    tx_cols = list(tx.columns)
    signal_id, signal_date, target = "signal_id", "signal_sanasi", "eskalatsiya"
    tx_date = next((c for c in tx.columns if c.lower() in {"tranzaksiya_vaqti", "transaction_time", "transaction_date", "timestamp"}), None)
    if tx_date is None:
        tx_date = next((c for c in tx.columns if any(k in c.lower() for k in ("vaqti", "date", "time", "timestamp"))), None)
    if signal_id not in tx.columns or tx_date is None:
        raise ValueError(f"Cannot identify transaction key/date columns. Available fields: {tx_cols}")
    tx[tx_date] = pd.to_datetime(tx[tx_date], errors="coerce")
    train[signal_date] = pd.to_datetime(train[signal_date], errors="coerce")
    joined = tx.merge(train[[signal_id, signal_date, target]], on=signal_id, how="left", validate="many_to_one")
    linked = joined[signal_date].notna()
    dated = joined[tx_date].notna()
    future = joined[linked & dated & (joined[tx_date] > joined[signal_date])]
    knowncols = {c.lower(): c for c in tx.columns}
    amount_col = next((knowncols[k] for k in ("amount", "summa", "miqdor", "miqdor_indeksi", "transaction_amount") if k in knowncols), None)
    direction_col = next((knowncols[k] for k in ("direction", "yo'nalish", "yunalish", "kirim_chiqim", "kiruvchi_chiquvchi", "transaction_direction") if k in knowncols), None)
    type_col = next((knowncols[k] for k in ("type", "transaction_type", "tranzaksiya_turi", "turi") if k in knowncols), None)
    valid = joined[linked & dated & (joined[tx_date] <= joined[signal_date])].copy()
    agg = {"transaction_count": (signal_id, "size"), "first_transaction": (tx_date, "min"), "last_transaction": (tx_date, "max")}
    if amount_col:
        valid[amount_col] = pd.to_numeric(valid[amount_col], errors="coerce")
        agg.update(total_amount=(amount_col, "sum"), mean_amount=(amount_col, "mean"))
    per = valid.groupby([signal_id], as_index=False).agg(**agg)
    per = train[[signal_id, target, signal_date]].merge(per, on=signal_id, how="left")
    per["transaction_count"] = per.transaction_count.fillna(0)
    per["month"] = per[signal_date].dt.to_period("M").astype(str)
    target_counts = train[target].value_counts().sort_index()
    month_target = train.groupby(train[signal_date].dt.to_period("M").astype(str))[target].agg(["size", "mean"]).reset_index()
    txn_month = _date_counts(valid[tx_date])
    amount = valid[amount_col].dropna() if amount_col else pd.Series(dtype=float)
    quality = {col: {"nulls": int(tx[col].isna().sum()), "null_percent": round(float(tx[col].isna().mean() * 100), 3), "distinct": int(tx[col].nunique(dropna=True))} for col in tx_cols}
    directions = []
    if direction_col:
        direction_labels = {"kirim": "Incoming (kirim)", "chiqim": "Outgoing (chiqim)"}
        directions = [{"category": direction_labels.get(str(k).lower(), str(k)), "count": int(v)} for k, v in valid[direction_col].fillna("Missing").astype(str).value_counts().items()]
    types = []
    if type_col:
        types = [{"category": str(k), "count": int(v)} for k, v in valid[type_col].fillna("Missing").astype(str).value_counts().head(12).items()]
    future_signals = int(future[signal_id].nunique())
    compare = {}
    for y, label in ((0, "dismissed"), (1, "escalated")):
        g = per[per[target] == y]
        compare[label] = {
            "signals": int(len(g)), "transactions_per_signal_mean": round(float(g.transaction_count.mean()), 4),
            "transactions_per_signal_median": round(float(g.transaction_count.median()), 4),
            "amount_per_signal_mean": round(float(g.total_amount.mean()), 4) if amount_col else None,
        }
    result = {
        "generated_from_aggregates": True,
        "dataset": {"train_signals": len(train), "test_signals": len(test), "train_transaction_rows": train_tx_rows, "test_transaction_rows": test_tx_rows,
                    "train_date_min": str(train[signal_date].min().date()), "train_date_max": str(train[signal_date].max().date()),
                    "test_date_min": str(test[signal_date].min().date()), "test_date_max": str(test[signal_date].max().date()),
                    "train_signal_columns": list(train.columns), "test_signal_columns": list(test.columns), "transaction_columns": tx_cols},
        "quality": {"train_signal_duplicate_ids": int(train[signal_id].duplicated().sum()), "test_signal_duplicate_ids": int(test[signal_id].duplicated().sum()),
                    "transaction_signal_ids_unmatched": int((~tx[signal_id].isin(train[signal_id])).sum()), "transaction_missing_signal_id": int(tx[signal_id].isna().sum()),
                    "transaction_missing_date": int(tx[tx_date].isna().sum()), "linked_dated_transactions": int((linked & dated).sum()),
                    "future_transactions": int(len(future)), "future_transaction_percent": round(100 * len(future) / max(int((linked & dated).sum()), 1), 3),
                    "signals_with_future_transactions": future_signals,
                    "future_by_signal_percent": round(100 * future_signals / max(len(train), 1), 3),
                    "transaction_column_quality": quality},
        "target": {"counts": [{"class": int(k), "label": "Escalated" if int(k) == 1 else "Dismissed", "count": int(v)} for k, v in target_counts.items()],
                   "monthly": [{"month": str(r[signal_date]), "signals": int(r["size"]), "escalation_rate": round(float(r["mean"]), 5)} for _, r in month_target.iterrows()]},
        "activity": {"transactions_by_month": txn_month, "transaction_count_distribution": _distribution(per.transaction_count),
                     "transactions_per_signal": {"mean": round(float(per.transaction_count.mean()), 4), "median": round(float(per.transaction_count.median()), 4),
                                                 "p90": round(float(per.transaction_count.quantile(.9)), 4), "max": int(per.transaction_count.max()),
                                                 "zero_signal_percent": round(float(per.transaction_count.eq(0).mean() * 100), 3)},
                     "directions": directions, "types": types},
        "amounts": {"column": amount_col, "is_index": amount_col == "miqdor_indeksi", "count": int(amount.count()), "mean": round(float(amount.mean()), 4) if len(amount) else None,
                    "median": round(float(amount.median()), 4) if len(amount) else None, "p95": round(float(amount.quantile(.95)), 4) if len(amount) else None,
                    "distribution": _distribution(amount), "mean_per_signal_distribution": _distribution(per["mean_amount"]) if amount_col else [],
                    "amount_per_signal_distribution": _distribution(per["total_amount"]) if amount_col else []},
        "recent": {"date_range_days": int((train[signal_date].max() - train[signal_date].min()).days),
                   "latest_month_signals": int(train[train[signal_date].dt.to_period("M") == train[signal_date].max().to_period("M")].shape[0]),
                   "daily": [{"date": str(k), "signals": int(v)} for k, v in train.groupby(train[signal_date].dt.date).size().tail(90).items()]},
        "target_comparison": compare,
        "method": {"temporal_rule": "For transaction behavior sections, include only dated transactions at or before the linked signal date.",
                   "validation": "Five expanding chronological folds; each validation block follows its training data. The initial date block is warm-up only.",
                   "model_evaluation": _model_summary()}
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote aggregate summary: {OUT} ({OUT.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
