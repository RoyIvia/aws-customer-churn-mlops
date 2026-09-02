"""Evaluate the customer churn model and apply a quality gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


TARGET_COLUMN = "Churn"
DEFAULT_VALIDATION_PATH = Path("data/processed/validation/data.csv")
DEFAULT_MODEL_PATH = Path("artifacts/model.joblib")
DEFAULT_METRICS_PATH = Path("artifacts/evaluation_metrics.json")
DEFAULT_PREDICTIONS_PATH = Path("artifacts/validation_predictions.csv")

# Initial baseline quality-gate thresholds.
MIN_F1 = 0.60
MIN_RECALL = 0.70
MIN_ROC_AUC = 0.75


def load_dataset(path: Path) -> pd.DataFrame:
    """Load the evaluation dataset."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    dataframe = pd.read_csv(path)

    if TARGET_COLUMN not in dataframe.columns:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' not found in {path}"
        )

    return dataframe


def load_model(path: Path):
    """Load a serialized model."""
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}")

    return joblib.load(path)


def validate_features(
    dataframe: pd.DataFrame,
    model,
) -> pd.DataFrame:
    """Validate that evaluation features match the trained model."""
    features = dataframe.drop(columns=[TARGET_COLUMN])

    expected_features = list(model.feature_names_in_)
    actual_features = list(features.columns)

    if actual_features != expected_features:
        raise ValueError(
            "Evaluation features do not match the trained model."
        )

    return features


def calculate_metrics(
    y_true: pd.Series,
    probabilities,
    threshold: float = 0.5,
) -> dict:
    """Calculate binary classification metrics."""
    probabilities = np.asarray(probabilities)
    predictions = (probabilities >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    ).ravel()

    return {
        "threshold": threshold,
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(
            precision_score(y_true, predictions, zero_division=0)
        ),
        "recall": float(
            recall_score(y_true, predictions, zero_division=0)
        ),
        "f1": float(
            f1_score(y_true, predictions, zero_division=0)
        ),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "confusion_matrix": {
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
        },
    }


def apply_quality_gate(metrics: dict) -> dict:
    """Determine whether the model meets minimum quality requirements."""
    checks = {
        "f1": metrics["f1"] >= MIN_F1,
        "recall": metrics["recall"] >= MIN_RECALL,
        "roc_auc": metrics["roc_auc"] >= MIN_ROC_AUC,
    }

    passed = all(checks.values())

    return {
        "status": "PASS" if passed else "FAIL",
        "passed": passed,
        "checks": checks,
        "thresholds": {
            "min_f1": MIN_F1,
            "min_recall": MIN_RECALL,
            "min_roc_auc": MIN_ROC_AUC,
        },
    }


def save_metrics(
    path: Path,
    metrics: dict,
    quality_gate: dict,
) -> None:
    """Save evaluation metrics and quality-gate results."""
    path.parent.mkdir(parents=True, exist_ok=True)

    output = {
        "metrics": metrics,
        "quality_gate": quality_gate,
    }

    with path.open("w", encoding="utf-8") as metrics_file:
        json.dump(output, metrics_file, indent=2)


def save_predictions(
    path: Path,
    y_true: pd.Series,
    probabilities,
    threshold: float,
) -> None:
    """Save actual values, probabilities, and predictions."""
    path.parent.mkdir(parents=True, exist_ok=True)

    probabilities = np.asarray(probabilities)
    predictions = (probabilities >= threshold).astype(int)

    output = pd.DataFrame(
        {
            "actual_churn": y_true.astype(int),
            "churn_probability": probabilities,
            "predicted_churn": predictions,
        }
    )

    output.to_csv(path, index=False)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate the customer churn model."
    )

    parser.add_argument(
        "--validation-data",
        type=Path,
        default=DEFAULT_VALIDATION_PATH,
    )

    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL_PATH,
    )

    parser.add_argument(
        "--metrics-output",
        type=Path,
        default=DEFAULT_METRICS_PATH,
    )

    parser.add_argument(
        "--predictions-output",
        type=Path,
        default=DEFAULT_PREDICTIONS_PATH,
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
    )

    return parser.parse_args()


def main() -> None:
    """Execute model evaluation."""
    args = parse_args()

    validation_df = load_dataset(args.validation_data)
    model = load_model(args.model)

    X_validation = validate_features(validation_df, model)
    y_validation = validation_df[TARGET_COLUMN].astype(int)

    probabilities = model.predict_proba(X_validation)[:, 1]

    metrics = calculate_metrics(
        y_validation,
        probabilities,
        args.threshold,
    )

    quality_gate = apply_quality_gate(metrics)

    save_metrics(
        args.metrics_output,
        metrics,
        quality_gate,
    )

    save_predictions(
        args.predictions_output,
        y_validation,
        probabilities,
        args.threshold,
    )

    print("Model evaluation completed.")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1:        {metrics['f1']:.4f}")
    print(f"ROC-AUC:   {metrics['roc_auc']:.4f}")
    print(f"Quality gate: {quality_gate['status']}")
    print(f"Metrics: {args.metrics_output}")
    print(f"Predictions: {args.predictions_output}")

    if not quality_gate["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
