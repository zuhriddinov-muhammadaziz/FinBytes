"""Automated tests for official competition submission creation and validation."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.submission.make_submission import create_submission
from src.submission.validate_submission import validate_submission


@pytest.fixture
def signals():
    return pd.DataFrame({"signal_id": ["SG_10", "SG_02", "SG_31"]})


@pytest.fixture
def sample(signals):
    return pd.DataFrame({"signal_id": signals.signal_id, "ehtimollik": [0.0] * len(signals)})


def test_writer_creates_expected_name_schema_and_round_trip(tmp_path, signals, sample):
    path = create_submission(signals, np.array([0.2, 0.5, 1.0]), tmp_path, "TEAM_42", sample)
    assert path.name == "team_TEAM_42.csv"
    parsed = pd.read_csv(path)
    assert parsed.columns.tolist() == ["signal_id", "ehtimollik"]
    assert parsed.signal_id.tolist() == signals.signal_id.tolist()
    assert len(parsed) == len(signals)
    assert validate_submission(path, signals, sample)["valid"] is True
    assert parsed.ehtimollik.tolist() == [0.2, 0.5, 1.0]


def test_writer_reorders_id_bearing_predictions_deterministically(tmp_path, signals, sample):
    predictions = pd.DataFrame({"signal_id": ["SG_31", "SG_10", "SG_02"], "ehtimollik": [0.9, 0.1, 0.3]})
    path = create_submission(signals, predictions, tmp_path, "TEAM_42", sample)
    first = pd.read_csv(path)
    second_path = create_submission(signals, predictions, tmp_path, "TEAM_42", sample)
    second = pd.read_csv(second_path)
    assert first.signal_id.tolist() == signals.signal_id.tolist()
    assert first.ehtimollik.tolist() == [0.1, 0.3, 0.9]
    pd.testing.assert_frame_equal(first, second)


def test_no_index_column_written(tmp_path, signals):
    path = create_submission(signals, [0.1, 0.2, 0.3], tmp_path, "TEAM_42")
    assert pd.read_csv(path).columns.tolist() == ["signal_id", "ehtimollik"]


@pytest.mark.parametrize("predictions, message", [
    ([0.1, np.nan, 0.3], "missing or non-finite"),
    ([0.1, np.inf, 0.3], "missing or non-finite"),
    ([-0.01, 0.2, 0.3], r"within \[0, 1\]"),
    ([0.1, 1.01, 0.3], r"within \[0, 1\]"),
    ([0.1, "not-a-number", 0.3], "numeric real values"),
    ([0.1, 0.2], "Expected 3 predictions"),
])
def test_writer_rejects_invalid_prediction_vectors(tmp_path, signals, predictions, message):
    with pytest.raises(ValueError, match=message):
        create_submission(signals, predictions, tmp_path, "TEAM_42")


@pytest.mark.parametrize("bad_submission, message", [
    (pd.DataFrame({"signal_id": ["SG_10", "SG_02", "SG_31"], "ehtimollik": [0.1, 0.2, 0.3], "extra": [1, 2, 3]}), "exactly"),
    (pd.DataFrame({"signal_id": ["SG_10", "SG_10", "SG_31"], "ehtimollik": [0.1, 0.2, 0.3]}), "duplicate"),
    (pd.DataFrame({"signal_id": ["SG_10", "SG_02", "SG_99"], "ehtimollik": [0.1, 0.2, 0.3]}), "unknown"),
    (pd.DataFrame({"signal_id": ["SG_02", "SG_10", "SG_31"], "ehtimollik": [0.1, 0.2, 0.3]}), "order"),
    (pd.DataFrame({"signal_id": ["SG_10", "SG_02", "SG_31"], "ehtimollik": [0.1, np.nan, 0.3]}), "missing or non-finite"),
])
def test_validator_rejects_invalid_submission(signals, bad_submission, message):
    with pytest.raises(ValueError, match=message):
        validate_submission(bad_submission, signals)


def test_validator_rejects_wrong_row_count(signals):
    submission = pd.DataFrame({"signal_id": ["SG_10", "SG_02"], "ehtimollik": [0.1, 0.2]})
    with pytest.raises(ValueError, match="row count"):
        validate_submission(submission, signals)


def test_validator_rejects_sample_structure_mismatch(signals, sample):
    bad_sample = sample.rename(columns={"ehtimollik": "prediction"})
    submission = pd.DataFrame({"signal_id": signals.signal_id, "ehtimollik": [0.1, 0.2, 0.3]})
    with pytest.raises(ValueError, match="sample_submission columns"):
        validate_submission(submission, signals, bad_sample)


def test_validator_rejects_sample_id_order_mismatch(signals, sample):
    bad_sample = sample.iloc[::-1].reset_index(drop=True)
    submission = pd.DataFrame({"signal_id": signals.signal_id, "ehtimollik": [0.1, 0.2, 0.3]})
    with pytest.raises(ValueError, match="sample_submission IDs"):
        validate_submission(submission, signals, bad_sample)


def test_validator_rejects_missing_or_duplicate_test_ids(sample):
    submission = sample.copy()
    with pytest.raises(ValueError, match="test_signals contains duplicate"):
        validate_submission(submission, pd.DataFrame({"signal_id": ["SG_10", "SG_10", "SG_31"]}))


def test_invalid_team_id_cannot_escape_output_directory(tmp_path, signals):
    with pytest.raises(ValueError, match="team_id"):
        create_submission(signals, [0.1, 0.2, 0.3], tmp_path, "../outside")


def test_path_input_checks_parseable_csv(tmp_path, signals, sample):
    path = create_submission(signals, [0.05, 0.55, 0.95], tmp_path, "TEAM_42", sample)
    report = validate_submission(path, signals, sample)
    assert report == {
        "valid": True,
        "rows": 3,
        "columns": ["signal_id", "ehtimollik"],
        "unique_ids": 3,
        "prediction_min": 0.05,
        "prediction_max": 0.95,
    }
