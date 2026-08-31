"""Tests for the customer churn preprocessing pipeline."""

from pathlib import Path

import pandas as pd

from src.preprocessing.preprocessor import (
    CATEGORICAL_FEATURES,
    ID_COLUMN,
    NUMERIC_FEATURES,
    TARGET_COLUMN,
    build_preprocessor,
    clean_data,
    load_raw_data,
    prepare_features,
)


DATASET_PATH = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "raw"
    / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
)


def test_raw_dataset_loads():
    df = load_raw_data(DATASET_PATH)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 7043


def test_total_charges_is_converted_to_numeric():
    df = load_raw_data(DATASET_PATH)
    cleaned = clean_data(df)

    assert pd.api.types.is_numeric_dtype(cleaned["TotalCharges"])


def test_customer_id_is_removed_from_model_features():
    df = load_raw_data(DATASET_PATH)
    cleaned = clean_data(df)

    assert ID_COLUMN not in cleaned.columns


def test_features_and_target_are_separated():
    df = load_raw_data(DATASET_PATH)

    features, target = prepare_features(df)

    assert TARGET_COLUMN not in features.columns
    assert len(features) == len(target)
    assert set(target.unique()) == {0, 1}


def test_expected_feature_columns_are_present():
    df = load_raw_data(DATASET_PATH)

    features, _ = prepare_features(df)

    expected = set(NUMERIC_FEATURES + CATEGORICAL_FEATURES)

    assert set(features.columns) == expected


def test_preprocessor_transforms_features():
    df = load_raw_data(DATASET_PATH)

    features, target = prepare_features(df)

    preprocessor = build_preprocessor()
    transformed = preprocessor.fit_transform(features)

    assert transformed.shape[0] == len(target)
    assert transformed.shape[1] > len(features.columns)
    assert not pd.isna(transformed).any()
