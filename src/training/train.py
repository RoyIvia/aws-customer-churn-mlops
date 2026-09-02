"""Train the baseline customer churn classification model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from xgboost import XGBClassifier


TARGET_COLUMN = "Churn"
RANDOM_STATE = 42

DEFAULT_TRAIN_PATH = Path("data/processed/train/data.csv")
DEFAULT_VALIDATION_PATH = Path("data/processed/validation/data.csv")
DEFAULT_MODEL_PATH = Path("artifacts/model.joblib")
DEFAULT_METADATA_PATH = Path("artifacts/training_metadata.json")


def load_dataset(path: Path) -> pd.DataFrame:
    """Load a processed dataset from CSV."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    dataframe = pd.read_csv(path)

    if TARGET_COLUMN not in dataframe.columns:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' not found in {path}"
        )

    return dataframe


def split_features_target(
    dataframe: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """Separate model features from the target."""
    features = dataframe.drop(columns=[TARGET_COLUMN])
    target = dataframe[TARGET_COLUMN].astype(int)

    if features.empty:
        raise ValueError("No model features were found.")

    return features, target


def calculate_class_weight(target: pd.Series) -> float:
    """Calculate the positive-class weight for the training data."""
    negative_count = int((target == 0).sum())
    positive_count = int((target == 1).sum())

    if positive_count == 0:
        raise ValueError("Training data contains no positive-class samples.")

    return negative_count / positive_count


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> XGBClassifier:
    """Train the baseline XGBoost classifier."""
    scale_pos_weight = calculate_class_weight(y_train)

    model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    return model


def save_model(model: XGBClassifier, path: Path) -> None:
    """Persist the trained model."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def save_metadata(
    path: Path,
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    model: XGBClassifier,
) -> None:
    """Persist metadata describing the training run."""
    _, y_train = split_features_target(train_df)

    metadata = {
        "model_type": "XGBClassifier",
        "task": "binary_classification",
        "target_column": TARGET_COLUMN,
        "positive_class": 1,
        "negative_class": 0,
        "random_state": RANDOM_STATE,
        "training_rows": len(train_df),
        "validation_rows": len(validation_df),
        "feature_count": len(model.feature_names_in_),
        "features": list(model.feature_names_in_),
        "class_distribution": {
            "negative": int((y_train == 0).sum()),
            "positive": int((y_train == 1).sum()),
        },
        "hyperparameters": {
            "n_estimators": 200,
            "max_depth": 4,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "scale_pos_weight": calculate_class_weight(y_train),
        },
    }

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as metadata_file:
        json.dump(metadata, metadata_file, indent=2)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train the baseline customer churn model."
    )

    parser.add_argument(
        "--train-data",
        type=Path,
        default=DEFAULT_TRAIN_PATH,
        help="Path to the processed training dataset.",
    )

    parser.add_argument(
        "--validation-data",
        type=Path,
        default=DEFAULT_VALIDATION_PATH,
        help="Path to the processed validation dataset.",
    )

    parser.add_argument(
        "--model-output",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Path for the trained model artifact.",
    )

    parser.add_argument(
        "--metadata-output",
        type=Path,
        default=DEFAULT_METADATA_PATH,
        help="Path for training metadata.",
    )

    return parser.parse_args()


def main() -> None:
    """Execute the training workflow."""
    args = parse_args()

    train_df = load_dataset(args.train_data)
    validation_df = load_dataset(args.validation_data)

    X_train, y_train = split_features_target(train_df)
    X_validation, _ = split_features_target(validation_df)

    if list(X_train.columns) != list(X_validation.columns):
        raise ValueError(
            "Training and validation feature columns do not match."
        )

    model = train_model(X_train, y_train)

    save_model(model, args.model_output)
    save_metadata(
        args.metadata_output,
        train_df,
        validation_df,
        model,
    )

    print("Training completed successfully.")
    print(f"Training rows: {len(train_df)}")
    print(f"Validation rows: {len(validation_df)}")
    print(f"Features: {len(X_train.columns)}")
    print(f"Model artifact: {args.model_output}")
    print(f"Training metadata: {args.metadata_output}")


if __name__ == "__main__":
    main()
