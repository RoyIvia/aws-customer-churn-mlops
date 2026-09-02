"""Tests for the baseline model training module."""

from pathlib import Path

import pandas as pd
from xgboost import XGBClassifier

from src.training.train import (
    calculate_class_weight,
    load_dataset,
    save_model,
    split_features_target,
    train_model,
)


def test_load_training_dataset():
    path = Path("data/processed/train/data.csv")

    dataframe = load_dataset(path)

    assert len(dataframe) == 4225
    assert "Churn" in dataframe.columns


def test_split_features_and_target():
    dataframe = pd.DataFrame(
        {
            "feature_a": [1, 2, 3],
            "feature_b": [4, 5, 6],
            "Churn": [0, 1, 0],
        }
    )

    features, target = split_features_target(dataframe)

    assert list(features.columns) == ["feature_a", "feature_b"]
    assert target.tolist() == [0, 1, 0]


def test_calculate_class_weight():
    target = pd.Series([0, 0, 0, 1])

    weight = calculate_class_weight(target)

    assert weight == 3.0


def test_train_model():
    X_train = pd.DataFrame(
        {
            "feature_a": [0, 1, 0, 1, 0, 1],
            "feature_b": [1, 1, 0, 0, 1, 0],
        }
    )

    y_train = pd.Series([0, 1, 0, 1, 0, 1])

    model = train_model(X_train, y_train)

    assert isinstance(model, XGBClassifier)

    predictions = model.predict(X_train)

    assert len(predictions) == len(y_train)


def test_save_model(tmp_path):
    X_train = pd.DataFrame(
        {
            "feature_a": [0, 1, 0, 1],
            "feature_b": [1, 1, 0, 0],
        }
    )

    y_train = pd.Series([0, 1, 0, 1])

    model = train_model(X_train, y_train)

    model_path = tmp_path / "model.joblib"

    save_model(model, model_path)

    assert model_path.exists()
    assert model_path.stat().st_size > 0
