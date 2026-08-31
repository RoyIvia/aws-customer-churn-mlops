"""Tests for train, validation, and test dataset splitting."""

from pathlib import Path

from src.data_split.splitter import (
    RANDOM_STATE,
    split_dataset,
    validate_split_distribution,
)
from src.data_validation.validator import load_dataset


DATASET_PATH = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "raw"
    / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
)


def test_split_sizes():
    df = load_dataset(DATASET_PATH)

    train, validation, test = split_dataset(df)

    assert len(train) + len(validation) + len(test) == len(df)

    assert len(train) == 4225
    assert len(validation) == 1409
    assert len(test) == 1409


def test_split_is_reproducible():
    df = load_dataset(DATASET_PATH)

    train_1, validation_1, test_1 = split_dataset(df)
    train_2, validation_2, test_2 = split_dataset(df)

    assert train_1["customerID"].tolist() == train_2["customerID"].tolist()
    assert validation_1["customerID"].tolist() == validation_2["customerID"].tolist()
    assert test_1["customerID"].tolist() == test_2["customerID"].tolist()


def test_split_preserves_target_distribution():
    df = load_dataset(DATASET_PATH)

    train, validation, test = split_dataset(df)

    validate_split_distribution(
        df,
        train,
        validation,
        test,
    )


def test_split_has_no_overlapping_customers():
    df = load_dataset(DATASET_PATH)

    train, validation, test = split_dataset(df)

    train_ids = set(train["customerID"])
    validation_ids = set(validation["customerID"])
    test_ids = set(test["customerID"])

    assert train_ids.isdisjoint(validation_ids)
    assert train_ids.isdisjoint(test_ids)
    assert validation_ids.isdisjoint(test_ids)


def test_random_state_is_fixed():
    assert RANDOM_STATE == 42
