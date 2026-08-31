"""Prepare leakage-safe train, validation, and test datasets."""

from pathlib import Path

import joblib
import pandas as pd

from src.data_split.splitter import split_dataset
from src.data_validation.validator import validate_dataset
from src.preprocessing.preprocessor import (
    build_preprocessor,
    prepare_features,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DATASET = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
)

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

PREPROCESSOR_PATH = ARTIFACTS_DIR / "preprocessor.joblib"


def transform_split(
    df: pd.DataFrame,
    preprocessor,
    fit: bool = False,
) -> pd.DataFrame:
    """Prepare and transform one dataset split."""

    features, target = prepare_features(df)

    if fit:
        transformed = preprocessor.fit_transform(features)
    else:
        transformed = preprocessor.transform(features)

    feature_names = preprocessor.get_feature_names_out()

    result = pd.DataFrame(
        transformed,
        columns=feature_names,
        index=df.index,
    )

    result["Churn"] = target

    return result


def prepare_dataset() -> None:
    """Validate, split, transform, and persist the dataset."""

    print("==> Loading and validating raw dataset...")

    raw_df = validate_dataset(RAW_DATASET)

    print(f"    ✓ {len(raw_df):,} records validated")

    print()
    print("==> Splitting dataset...")

    train_df, validation_df, test_df = split_dataset(raw_df)

    print(f"    ✓ Training:   {len(train_df):,} rows")
    print(f"    ✓ Validation: {len(validation_df):,} rows")
    print(f"    ✓ Test:       {len(test_df):,} rows")

    print()
    print("==> Building preprocessor...")

    preprocessor = build_preprocessor()

    print("    ✓ Preprocessor created")

    print()
    print("==> Fitting preprocessor on training data only...")

    train_output = transform_split(
        train_df,
        preprocessor,
        fit=True,
    )

    print(
        f"    ✓ Training transformed: "
        f"{train_output.shape}"
    )

    print()
    print("==> Transforming validation data...")

    validation_output = transform_split(
        validation_df,
        preprocessor,
        fit=False,
    )

    print(
        f"    ✓ Validation transformed: "
        f"{validation_output.shape}"
    )

    print()
    print("==> Transforming test data...")

    test_output = transform_split(
        test_df,
        preprocessor,
        fit=False,
    )

    print(
        f"    ✓ Test transformed: "
        f"{test_output.shape}"
    )

    print()
    print("==> Persisting processed datasets...")

    for split_name, dataset in [
        ("train", train_output),
        ("validation", validation_output),
        ("test", test_output),
    ]:
        output_dir = PROCESSED_DIR / split_name
        output_dir.mkdir(parents=True, exist_ok=True)

        output_path = output_dir / "data.csv"

        dataset.to_csv(
            output_path,
            index=False,
        )

        print(f"    ✓ {output_path}")

    print()
    print("==> Persisting fitted preprocessor...")

    ARTIFACTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        preprocessor,
        PREPROCESSOR_PATH,
    )

    print(f"    ✓ {PREPROCESSOR_PATH}")

    print()
    print("==> Dataset preparation complete.")
    print()
    print(f"Training shape:   {train_output.shape}")
    print(f"Validation shape: {validation_output.shape}")
    print(f"Test shape:       {test_output.shape}")
    print(
        "Feature count:    "
        f"{len(preprocessor.get_feature_names_out())}"
    )


if __name__ == "__main__":
    prepare_dataset()
