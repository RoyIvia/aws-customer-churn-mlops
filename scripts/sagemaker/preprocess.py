"""SageMaker Processing entry point for dataset preprocessing."""

import argparse
from pathlib import Path

import pandas as pd

from src.pipeline.prepare_dataset import prepare_dataset


TARGET_COLUMN = "Churn"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Prepare customer churn datasets for SageMaker."
    )

    parser.add_argument(
        "--input-data",
        type=Path,
        required=True,
        help="Path to the raw input dataset.",
    )

    parser.add_argument(
        "--processed-dir",
        type=Path,
        required=True,
        help="Directory for processed datasets.",
    )

    parser.add_argument(
        "--preprocessor-output",
        type=Path,
        required=True,
        help="Path for the fitted preprocessor.",
    )

    return parser.parse_args()


def make_sagemaker_xgboost_compatible(
    processed_dir: Path,
) -> None:
    """Move the target column to the first position for XGBoost."""

    for split_name in ("train", "validation", "test"):
        dataset_path = (
            processed_dir
            / split_name
            / "data.csv"
        )

        dataset = pd.read_csv(dataset_path)

        if TARGET_COLUMN not in dataset.columns:
            raise ValueError(
                f"Target column '{TARGET_COLUMN}' "
                f"not found in {dataset_path}"
            )

        columns = [
            TARGET_COLUMN,
            *[
                column
                for column in dataset.columns
                if column != TARGET_COLUMN
            ],
        ]

        dataset = dataset[columns]

        dataset.to_csv(
            dataset_path,
            index=False,
        )

        print(
            f"    ✓ SageMaker format applied: "
            f"{dataset_path}"
        )


def main() -> None:
    """Run dataset preprocessing."""

    args = parse_args()

    prepare_dataset(
        raw_dataset=args.input_data,
        processed_dir=args.processed_dir,
        preprocessor_path=args.preprocessor_output,
    )

    print()
    print(
        "==> Preparing datasets for SageMaker XGBoost..."
    )

    make_sagemaker_xgboost_compatible(
        args.processed_dir
    )

    print()
    print("==> SageMaker preprocessing complete.")


if __name__ == "__main__":
    main()
