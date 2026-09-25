"""Create a deterministic, validated WIUT competition submission CSV."""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from .validate_submission import SUBMISSION_COLUMNS, validate_submission


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _read_frame(value: pd.DataFrame | str | Path, name: str) -> pd.DataFrame:
    if isinstance(value, pd.DataFrame):
        return value.copy()
    try:
        return pd.read_csv(value)
    except Exception as exc:
        raise ValueError(f"Could not parse {name} as a CSV file: {exc}") from exc


def _prediction_values(
    predictions: pd.DataFrame | pd.Series | np.ndarray | list[float],
    expected_ids: list[Any],
    *,
    id_column: str,
    prediction_column: str,
) -> np.ndarray:
    if isinstance(predictions, pd.DataFrame):
        if prediction_column not in predictions.columns:
            raise ValueError(f"Predictions are missing column {prediction_column!r}.")
        prediction_frame = predictions.copy()
        if id_column in prediction_frame.columns:
            ids = prediction_frame[id_column]
            if ids.isna().any():
                raise ValueError("Prediction input contains missing IDs.")
            if ids.duplicated().any():
                raise ValueError("Prediction input contains duplicate IDs.")
            got, expected = set(ids.tolist()), set(expected_ids)
            if got - expected:
                raise ValueError(f"Prediction input contains {len(got - expected)} unknown ID(s).")
            if expected - got:
                raise ValueError(f"Prediction input is missing {len(expected - got)} test ID(s).")
            prediction_frame = prediction_frame.set_index(id_column).reindex(expected_ids)
        values = prediction_frame[prediction_column].to_numpy()
    elif isinstance(predictions, pd.Series):
        values = predictions.to_numpy()
    else:
        values = np.asarray(predictions)

    if values.ndim != 1:
        raise ValueError("Predictions must be a one-dimensional sequence.")
    if len(values) != len(expected_ids):
        raise ValueError(f"Expected {len(expected_ids)} predictions; received {len(values)}.")
    try:
        values = pd.to_numeric(pd.Series(values), errors="raise").to_numpy()
    except (TypeError, ValueError) as exc:
        raise ValueError("Predictions must be numeric real values.") from exc
    if np.iscomplexobj(values):
        raise ValueError("Predictions must be real-valued, not complex.")
    values = values.astype("float64", copy=False)
    if not np.isfinite(values).all():
        raise ValueError("Predictions must not contain missing or non-finite values.")
    if ((values < 0) | (values > 1)).any():
        raise ValueError("Predictions must all be within [0, 1].")
    return values


def create_submission(
    test_signals: pd.DataFrame | str | Path,
    predictions: pd.DataFrame | pd.Series | np.ndarray | list[float],
    output_dir: str | Path,
    team_id: str,
    sample_submission: pd.DataFrame | str | Path | None = None,
    *,
    id_column: str = "signal_id",
    prediction_column: str = "ehtimollik",
) -> Path:
    """Write ``team_<TEAM_ID>.csv`` in test-file order and validate the CSV round-trip."""
    if not isinstance(team_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", team_id):
        raise ValueError("team_id must contain only letters, numbers, underscores, or hyphens and start alphanumeric.")
    signals = _read_frame(test_signals, "test_signals")
    if id_column not in signals.columns:
        raise ValueError(f"test_signals is missing the required ID column {id_column!r}.")
    expected_ids = signals[id_column].tolist()
    if not expected_ids:
        raise ValueError("test_signals contains no rows.")
    if signals[id_column].isna().any() or signals[id_column].duplicated().any():
        raise ValueError("test_signals IDs must be complete and unique.")

    values = _prediction_values(
        predictions,
        expected_ids,
        id_column=id_column,
        prediction_column=prediction_column,
    )
    submission = pd.DataFrame({id_column: expected_ids, prediction_column: values}, columns=SUBMISSION_COLUMNS)
    validate_submission(submission, signals, sample_submission, id_column=id_column, prediction_column=prediction_column)

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    output_path = directory / f"team_{team_id}.csv"
    temporary_path = output_path.with_name(output_path.name + ".tmp")
    try:
        submission.to_csv(temporary_path, index=False)
        # Validate what consumers will parse from disk before publishing the final path.
        validate_submission(temporary_path, signals, sample_submission, id_column=id_column, prediction_column=prediction_column)
        temporary_path.replace(output_path)
        validate_submission(output_path, signals, sample_submission, id_column=id_column, prediction_column=prediction_column)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
    return output_path


def _load_config(config_path: str | Path) -> dict[str, Any]:
    with Path(config_path).open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", default="artifacts/predictions/final_submission.csv", help="CSV with ehtimollik and optionally signal_id columns")
    parser.add_argument("--test-signals", default="data/raw/test_signals.csv")
    parser.add_argument("--sample-submission", default="data/raw/sample_submission (3).csv")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--team-id", default=None, help="Defaults to team.id in config.yaml")
    parser.add_argument("--output-dir", default="artifacts/predictions")
    args = parser.parse_args()

    config = _load_config(args.config)
    team_id = args.team_id or config.get("team", {}).get("id")
    if not team_id or team_id == "TEAM_ID":
        parser.error("Set team.id in config.yaml or pass --team-id.")
    predictions = pd.read_csv(args.predictions)
    if "ehtimollik" in predictions.columns:
        prediction_data: pd.DataFrame | pd.Series = predictions
    else:
        # A single-column prediction CSV is interpreted in test_signals order.
        prediction_data = predictions.iloc[:, 0] if predictions.shape[1] == 1 else predictions
    output_path = create_submission(
        args.test_signals,
        prediction_data,
        args.output_dir,
        team_id,
        args.sample_submission,
    )
    print(f"Validated submission created: {output_path}")


if __name__ == "__main__":
    main()
