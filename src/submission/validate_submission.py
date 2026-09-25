"""Strict validation for competition submission files."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SUBMISSION_COLUMNS = ["signal_id", "ehtimollik"]


def _read_frame(value: pd.DataFrame | str | Path, name: str) -> pd.DataFrame:
    if isinstance(value, pd.DataFrame):
        return value.copy()
    try:
        return pd.read_csv(value)
    except Exception as exc:
        raise ValueError(f"Could not parse {name} as a CSV file: {exc}") from exc


def validate_submission(
    submission: pd.DataFrame | str | Path,
    test_signals: pd.DataFrame | str | Path,
    sample_submission: pd.DataFrame | str | Path | None = None,
    *,
    id_column: str = "signal_id",
    prediction_column: str = "ehtimollik",
) -> dict[str, Any]:
    """Validate schema, full ID coverage/order, probabilities, and optional sample format.

    Paths are read back with pandas, so the same checks also verify CSV parseability.
    Raises ``ValueError`` with a specific reason when a requirement is violated.
    """
    expected_columns = [id_column, prediction_column]
    frame = _read_frame(submission, "submission")
    signals = _read_frame(test_signals, "test_signals")

    if frame.columns.tolist() != expected_columns:
        raise ValueError(f"Submission columns must be exactly {expected_columns} in that order; got {frame.columns.tolist()}.")
    if id_column not in signals.columns:
        raise ValueError(f"test_signals is missing the required ID column {id_column!r}.")
    if signals[id_column].isna().any():
        raise ValueError("test_signals contains missing IDs.")
    if signals[id_column].duplicated().any():
        raise ValueError("test_signals contains duplicate IDs.")

    expected_ids = signals[id_column].tolist()
    actual_ids = frame[id_column]
    if len(frame) != len(expected_ids):
        raise ValueError(f"Submission row count must equal test_signals ({len(expected_ids)}); got {len(frame)}.")
    if actual_ids.isna().any():
        raise ValueError("Submission contains missing IDs.")
    if actual_ids.duplicated().any():
        raise ValueError("Submission contains duplicate IDs.")

    expected_set, actual_set = set(expected_ids), set(actual_ids.tolist())
    missing_ids = expected_set - actual_set
    unknown_ids = actual_set - expected_set
    if missing_ids or unknown_ids:
        details = []
        if missing_ids:
            details.append(f"missing {len(missing_ids)} test signal ID(s)")
        if unknown_ids:
            details.append(f"contains {len(unknown_ids)} unknown signal ID(s)")
        raise ValueError("Submission ID mismatch: " + "; ".join(details) + ".")
    if actual_ids.tolist() != expected_ids:
        raise ValueError("Submission IDs must match test_signals.csv exactly and in its order.")

    try:
        predictions = pd.to_numeric(frame[prediction_column], errors="raise").to_numpy()
    except (TypeError, ValueError) as exc:
        raise ValueError("Predictions must be numeric real values.") from exc
    if np.iscomplexobj(predictions):
        raise ValueError("Predictions must be real-valued, not complex.")
    predictions = predictions.astype("float64", copy=False)
    if not np.isfinite(predictions).all():
        raise ValueError("Predictions must not contain missing or non-finite values.")
    if ((predictions < 0) | (predictions > 1)).any():
        raise ValueError("Predictions must all be within [0, 1].")

    if sample_submission is not None:
        sample = _read_frame(sample_submission, "sample_submission")
        if sample.columns.tolist() != expected_columns:
            raise ValueError(f"sample_submission columns must be exactly {expected_columns}; got {sample.columns.tolist()}.")
        if len(sample) != len(expected_ids):
            raise ValueError("sample_submission row count does not match test_signals.")
        if sample[id_column].isna().any() or sample[id_column].duplicated().any():
            raise ValueError("sample_submission contains missing or duplicate IDs.")
        if sample[id_column].tolist() != expected_ids:
            raise ValueError("sample_submission IDs must match test_signals exactly and in order.")

    return {
        "valid": True,
        "rows": len(frame),
        "columns": expected_columns,
        "unique_ids": int(actual_ids.nunique()),
        "prediction_min": float(predictions.min()) if len(predictions) else None,
        "prediction_max": float(predictions.max()) if len(predictions) else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submission", default=None,
                        help="Defaults to artifacts/predictions/team_<team.id>.csv from config.yaml")
    parser.add_argument("--test-signals", default="data/raw/test_signals.csv")
    parser.add_argument("--sample-submission", default="data/raw/sample_submission (3).csv")
    args = parser.parse_args()
    submission_path = args.submission
    if submission_path is None:
        import yaml
        config = yaml.safe_load(Path("config.yaml").read_text(encoding="utf-8"))
        team_id = config.get("team", {}).get("id")
        if not team_id or team_id == "TEAM_ID":
            parser.error("Set team.id in config.yaml or pass --submission.")
        submission_path = f"artifacts/predictions/team_{team_id}.csv"
    report = validate_submission(submission_path, args.test_signals, args.sample_submission)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
