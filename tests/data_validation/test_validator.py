"""Tests for the customer churn data-validation module."""

from pathlib import Path

import pandas as pd
import pytest

from src.data_validation.validator import (
    DataValidationError,
    EXPECTED_COLUMNS,
    EXPECTED_ROW_COUNT,
    EXPECTED_TARGET_DISTRIBUTION,
    EXPECTED_TARGET_VALUES,
    load_dataset,
    validate_customer_ids,
    validate_duplicate_rows,
    validate_numeric_columns,
    validate_required_values,
    validate_row_count,
    validate_schema,
    validate_target,
    validate_total_charges,
)


DATASET_PATH = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "raw"
    / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
)


@pytest.fixture
def dataset():
    return load_dataset(DATASET_PATH)


def test_dataset_loads(dataset):
    assert isinstance(dataset, pd.DataFrame)


def test_schema(dataset):
    validate_schema(dataset)
    assert list(dataset.columns) == EXPECTED_COLUMNS


def test_row_count(dataset):
    validate_row_count(dataset)
    assert len(dataset) == EXPECTED_ROW_COUNT


def test_customer_ids_are_unique(dataset):
    validate_customer_ids(dataset)
    assert dataset["customerID"].duplicated().sum() == 0


def test_no_duplicate_rows(dataset):
    validate_duplicate_rows(dataset)
    assert dataset.duplicated().sum() == 0


def test_required_values(dataset):
    validate_required_values(dataset)


def test_numeric_columns(dataset):
    validate_numeric_columns(dataset)


def test_total_charges_anomaly(dataset):
    validate_total_charges(dataset)

    converted = pd.to_numeric(dataset["TotalCharges"], errors="coerce")

    assert converted.isna().sum() == 11
    assert (dataset.loc[converted.isna(), "tenure"] == 0).all()
    assert (dataset.loc[converted.isna(), "Churn"] == "No").all()


def test_target_values(dataset):
    validate_target(dataset)

    assert set(dataset["Churn"].unique()) == EXPECTED_TARGET_VALUES


def test_target_distribution(dataset):
    validate_target(dataset)

    assert dataset["Churn"].value_counts().to_dict() == (
        EXPECTED_TARGET_DISTRIBUTION
    )


def test_schema_rejects_missing_column(dataset):
    invalid = dataset.drop(columns=["gender"])

    with pytest.raises(DataValidationError):
        validate_schema(invalid)


def test_duplicate_customer_id_is_rejected(dataset):
    invalid = dataset.copy()
    invalid.loc[1, "customerID"] = invalid.loc[0, "customerID"]

    with pytest.raises(DataValidationError):
        validate_customer_ids(invalid)


def test_invalid_target_is_rejected(dataset):
    invalid = dataset.copy()
    invalid.loc[0, "Churn"] = "UNKNOWN"

    with pytest.raises(DataValidationError):
        validate_target(invalid)
