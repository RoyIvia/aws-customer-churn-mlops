"""Preprocessing and feature engineering for customer churn modeling."""

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ID_COLUMN = "customerID"
TARGET_COLUMN = "Churn"

NUMERIC_FEATURES = [
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
]

CATEGORICAL_FEATURES = [
    "gender",
    "Partner",
    "Dependents",
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
]


def load_raw_data(path: str | Path) -> pd.DataFrame:
    """Load the raw customer churn dataset."""
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    return pd.read_csv(path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Perform deterministic cleaning before feature transformation."""
    cleaned = df.copy()

    # TotalCharges is stored as an object in the source dataset.
    # Blank values correspond to customers with zero tenure.
    cleaned["TotalCharges"] = pd.to_numeric(
        cleaned["TotalCharges"],
        errors="coerce",
    )

    # Remove the customer identifier from model features.
    cleaned = cleaned.drop(columns=[ID_COLUMN])

    return cleaned


def split_features_target(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """Separate model features from the binary churn target."""
    features = df.drop(columns=[TARGET_COLUMN])
    target = df[TARGET_COLUMN].map({"No": 0, "Yes": 1})

    if target.isna().any():
        raise ValueError("Target contains unexpected values.")

    return features, target


def build_preprocessor() -> ColumnTransformer:
    """Build the feature transformation pipeline."""
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent"),
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            (
                "categorical",
                categorical_pipeline,
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
    )


def prepare_features(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """Clean raw data and separate features from target."""
    cleaned = clean_data(df)
    return split_features_target(cleaned)


if __name__ == "__main__":
    dataset_path = (
        Path(__file__).resolve().parents[2]
        / "data"
        / "raw"
        / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
    )

    raw_df = load_raw_data(dataset_path)
    features, target = prepare_features(raw_df)

    preprocessor = build_preprocessor()
    transformed = preprocessor.fit_transform(features)

    print("Preprocessing completed successfully.")
    print(f"Input rows: {len(raw_df):,}")
    print(f"Input features: {features.shape[1]}")
    print(f"Transformed features: {transformed.shape[1]}")
    print(f"Target values: {sorted(target.unique())}")
