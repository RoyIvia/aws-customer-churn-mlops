"""SageMaker Processing entry point for model evaluation."""

import argparse
import json
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


TARGET_COLUMN = "Churn"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Evaluate a trained customer churn model."
    )

    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Path to the SageMaker model artifact.",
    )

    parser.add_argument(
        "--validation-data",
        type=Path,
        required=True,
        help="Path to the validation CSV.",
    )

    parser.add_argument(
        "--evaluation-output",
        type=Path,
        required=True,
        help="Directory for evaluation results.",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Classification probability threshold.",
    )

    return parser.parse_args()


def extract_model_artifact(
    model_path: Path,
    output_dir: Path,
) -> Path:
    """Extract the SageMaker model artifact."""

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    if tarfile.is_tarfile(model_path):
        with tarfile.open(
            model_path,
            mode="r:gz",
        ) as archive:
            archive.extractall(output_dir)

        candidates = list(
            output_dir.rglob("xgboost-model")
        )

        if candidates:
            return candidates[0]

        raise FileNotFoundError(
            "Could not find 'xgboost-model' "
            "inside SageMaker model artifact."
        )

    return model_path


def load_model(model_path: Path) -> xgb.Booster:
    """Load the trained XGBoost model."""

    extracted_path = extract_model_artifact(
        model_path=model_path,
        output_dir=model_path.parent / "extracted_model",
    )

    model = xgb.Booster()

    model.load_model(
        str(extracted_path)
    )

    return model


def load_validation_data(
    validation_path: Path,
) -> tuple[pd.DataFrame, pd.Series]:
    """Load validation features and target."""

    dataset = pd.read_csv(validation_path)

    if TARGET_COLUMN not in dataset.columns:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' "
            f"not found in validation data."
        )

    target = dataset[TARGET_COLUMN]

    features = dataset.drop(
        columns=[TARGET_COLUMN]
    )

    return features, target


def predict(
    model: xgb.Booster,
    features: pd.DataFrame,
) -> np.ndarray:
    """Generate probability predictions."""

    matrix = xgb.DMatrix(features)

    return model.predict(matrix)


def calculate_metrics(
    target: pd.Series,
    probabilities: np.ndarray,
    threshold: float,
) -> dict:
    """Calculate classification metrics."""

    predictions = (
        probabilities >= threshold
    ).astype(int)

    return {
        "accuracy": float(
            accuracy_score(
                target,
                predictions,
            )
        ),
        "precision": float(
            precision_score(
                target,
                predictions,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                target,
                predictions,
                zero_division=0,
            )
        ),
        "f1": float(
            f1_score(
                target,
                predictions,
                zero_division=0,
            )
        ),
        "roc_auc": float(
            roc_auc_score(
                target,
                probabilities,
            )
        ),
        "confusion_matrix": (
            confusion_matrix(
                target,
                predictions,
            ).tolist()
        ),
        "threshold": float(threshold),
    }


def save_evaluation(
    metrics: dict,
    output_path: Path,
) -> None:
    """Save evaluation metrics as JSON."""

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metrics,
            file,
            indent=2,
        )


def save_predictions(
    target: pd.Series,
    probabilities: np.ndarray,
    threshold: float,
    output_path: Path,
) -> None:
    """Save validation predictions."""

    predictions = (
        probabilities >= threshold
    ).astype(int)

    result = pd.DataFrame(
        {
            "actual": target.to_numpy(),
            "probability": probabilities,
            "prediction": predictions,
        }
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_csv(
        output_path,
        index=False,
    )


def main() -> None:
    """Evaluate the trained model."""

    args = parse_args()

    print("==> Loading trained model...")

    model = load_model(
        args.model
    )

    print("    ✓ Model loaded")

    print()
    print("==> Loading validation data...")

    features, target = load_validation_data(
        args.validation_data
    )

    print(
        f"    ✓ Validation records: "
        f"{len(features):,}"
    )

    print(
        f"    ✓ Features: "
        f"{len(features.columns)}"
    )

    print()
    print("==> Generating predictions...")

    probabilities = predict(
        model,
        features,
    )

    print("    ✓ Predictions generated")

    print()
    print("==> Calculating evaluation metrics...")

    metrics = calculate_metrics(
        target=target,
        probabilities=probabilities,
        threshold=args.threshold,
    )

    for metric_name in (
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
    ):
        print(
            f"    {metric_name}: "
            f"{metrics[metric_name]:.4f}"
        )

    print()
    print("==> Saving evaluation results...")

    evaluation_path = (
        args.evaluation_output
        / "evaluation.json"
    )

    predictions_path = (
        args.evaluation_output
        / "predictions.csv"
    )

    save_evaluation(
        metrics,
        evaluation_path,
    )

    save_predictions(
        target=target,
        probabilities=probabilities,
        threshold=args.threshold,
        output_path=predictions_path,
    )

    print(
        f"    ✓ {evaluation_path}"
    )

    print(
        f"    ✓ {predictions_path}"
    )

    print()
    print("==> Evaluation complete.")


if __name__ == "__main__":
    main()
