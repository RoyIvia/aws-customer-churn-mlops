"""Train, validation, and test dataset splitting."""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.data_validation.validator import validate_dataset


TARGET_COLUMN = "Churn"

TEST_SIZE = 0.20
VALIDATION_SIZE = 0.20
RANDOM_STATE = 42


def split_dataset(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split the dataset into train, validation, and test sets.

    The split is stratified on the Churn target to preserve the
    class distribution across all datasets.
    """

    train_validation, test = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df[TARGET_COLUMN],
    )

    validation_relative_size = VALIDATION_SIZE / (1 - TEST_SIZE)

    train, validation = train_test_split(
        train_validation,
        test_size=validation_relative_size,
        random_state=RANDOM_STATE,
        stratify=train_validation[TARGET_COLUMN],
    )

    return train, validation, test


def validate_split_distribution(
    original: pd.DataFrame,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
) -> None:
    """Validate that target distributions remain approximately consistent."""

    original_distribution = original[TARGET_COLUMN].value_counts(
        normalize=True
    )

    for name, dataset in [
        ("train", train),
        ("validation", validation),
        ("test", test),
    ]:
        distribution = dataset[TARGET_COLUMN].value_counts(
            normalize=True
        )

        for target_value in original_distribution.index:
            difference = abs(
                distribution[target_value]
                - original_distribution[target_value]
            )

            if difference > 0.02:
                raise ValueError(
                    f"{name} target distribution differs too much "
                    f"from the original dataset for '{target_value}'."
                )


if __name__ == "__main__":
    dataset_path = (
        Path(__file__).resolve().parents[2]
        / "data"
        / "raw"
        / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
    )

    dataset = validate_dataset(dataset_path)

    train, validation, test = split_dataset(dataset)

    validate_split_distribution(
        dataset,
        train,
        validation,
        test,
    )

    print("Dataset splitting completed successfully.")
    print()
    print(f"Total records:      {len(dataset):,}")
    print(f"Training records:   {len(train):,}")
    print(f"Validation records: {len(validation):,}")
    print(f"Test records:       {len(test):,}")
    print()

    print("Target distribution:")
    print()

    for name, subset in [
        ("Train", train),
        ("Validation", validation),
        ("Test", test),
    ]:
        churn_rate = (
            subset[TARGET_COLUMN]
            .eq("Yes")
            .mean()
            * 100
        )

        print(
            f"{name:<12} "
            f"rows={len(subset):,} "
            f"churn={churn_rate:.2f}%"
        )
