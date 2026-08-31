"""Data-quality validation for the Telco Customer Churn dataset."""

from pathlib import Path

import pandas as pd


EXPECTED_COLUMNS = [
    "customerID",
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
    "Churn",
]

EXPECTED_ROW_COUNT = 7043

REQUIRED_NUMERIC_COLUMNS = [
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
]

EXPECTED_TARGET_VALUES = {"Yes", "No"}

EXPECTED_TARGET_DISTRIBUTION = {
    "No": 5174,
    "Yes": 1869,
}


class DataValidationError(Exception):
    """Raised when the source dataset fails validation."""


def load_dataset(path: str | Path) -> pd.DataFrame:
    """Load the raw customer churn dataset."""
    path = Path(path)

    if not path.exists():
        raise DataValidationError(f"Dataset not found: {path}")

    return pd.read_csv(path)


def validate_schema(df: pd.DataFrame) -> None:
    """Validate that the dataset contains the expected columns."""
    actual_columns = list(df.columns)

    if actual_columns != EXPECTED_COLUMNS:
        raise DataValidationError(
            "Schema mismatch.\n"
            f"Expected: {EXPECTED_COLUMNS}\n"
            f"Actual:   {actual_columns}"
        )


def validate_row_count(df: pd.DataFrame) -> None:
    """Validate the expected number of records."""
    if len(df) != EXPECTED_ROW_COUNT:
        raise DataValidationError(
            f"Expected {EXPECTED_ROW_COUNT} rows, found {len(df)}."
        )


def validate_customer_ids(df: pd.DataFrame) -> None:
    """Validate customer IDs are present and unique."""
    if df["customerID"].isna().any():
        raise DataValidationError("customerID contains missing values.")

    duplicates = df["customerID"].duplicated().sum()

    if duplicates:
        raise DataValidationError(
            f"Found {duplicates} duplicate customer IDs."
        )


def validate_duplicate_rows(df: pd.DataFrame) -> None:
    """Validate that complete duplicate records do not exist."""
    duplicates = df.duplicated().sum()

    if duplicates:
        raise DataValidationError(
            f"Found {duplicates} duplicate rows."
        )


def validate_required_values(df: pd.DataFrame) -> None:
    """Validate required fields are not null."""
    null_counts = df.isnull().sum()
    invalid = null_counts[null_counts > 0]

    if not invalid.empty:
        raise DataValidationError(
            f"Missing values detected:\n{invalid.to_string()}"
        )


def validate_numeric_columns(df: pd.DataFrame) -> None:
    """Validate columns expected to be numeric."""
    for column in REQUIRED_NUMERIC_COLUMNS:
        if not pd.api.types.is_numeric_dtype(df[column]):
            raise DataValidationError(
                f"{column} must be numeric. "
                f"Found dtype: {df[column].dtype}"
            )


def validate_total_charges(df: pd.DataFrame) -> None:
    """Validate and report the known TotalCharges anomaly."""
    converted = pd.to_numeric(df["TotalCharges"], errors="coerce")

    invalid_count = converted.isna().sum()

    if invalid_count != 11:
        raise DataValidationError(
            "Unexpected TotalCharges anomaly count. "
            f"Expected 11 non-numeric/blank values, found {invalid_count}."
        )

    affected = df.loc[converted.isna()]

    if not (affected["tenure"] == 0).all():
        raise DataValidationError(
            "All invalid TotalCharges records are expected to have tenure=0."
        )

    if not (affected["Churn"] == "No").all():
        raise DataValidationError(
            "All invalid TotalCharges records are expected to have Churn=No."
        )


def validate_target(df: pd.DataFrame) -> None:
    """Validate target values and class distribution."""
    actual_values = set(df["Churn"].dropna().unique())

    if actual_values != EXPECTED_TARGET_VALUES:
        raise DataValidationError(
            "Unexpected target values. "
            f"Expected: {EXPECTED_TARGET_VALUES}; "
            f"Found: {actual_values}"
        )

    distribution = df["Churn"].value_counts().to_dict()

    if distribution != EXPECTED_TARGET_DISTRIBUTION:
        raise DataValidationError(
            "Unexpected target distribution. "
            f"Expected: {EXPECTED_TARGET_DISTRIBUTION}; "
            f"Found: {distribution}"
        )


def validate_dataset(path: str | Path) -> pd.DataFrame:
    """Run all raw-dataset validation checks."""
    df = load_dataset(path)

    validate_schema(df)
    validate_row_count(df)
    validate_customer_ids(df)
    validate_duplicate_rows(df)
    validate_required_values(df)
    validate_numeric_columns(df)
    validate_total_charges(df)
    validate_target(df)

    return df


if __name__ == "__main__":
    dataset_path = (
        Path(__file__).resolve().parents[2]
        / "data"
        / "raw"
        / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
    )

    validate_dataset(dataset_path)

    print("Data validation passed successfully.")
