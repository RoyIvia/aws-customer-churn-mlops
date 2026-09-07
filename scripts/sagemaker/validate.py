"""SageMaker Processing entry point for dataset validation."""

import argparse
from pathlib import Path

from src.data_validation.validator import validate_dataset


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Validate the customer churn dataset."
    )

    parser.add_argument(
        "--input-data",
        type=Path,
        required=True,
        help="Path to the dataset to validate.",
    )

    return parser.parse_args()


def main() -> None:
    """Validate the dataset."""

    args = parse_args()

    print("==> Validating dataset...")
    print(f"    Input: {args.input_data}")

    dataset = validate_dataset(args.input_data)

    print(f"    ✓ Dataset validated successfully")
    print(f"    ✓ Records: {len(dataset):,}")
    print(f"    ✓ Columns: {len(dataset.columns)}")


if __name__ == "__main__":
    main()
