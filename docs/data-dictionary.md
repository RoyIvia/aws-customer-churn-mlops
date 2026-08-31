# Customer Churn Data Dictionary

## 1. Dataset Overview

This project uses the **Telco Customer Churn** dataset as the initial source dataset for developing and evaluating the customer churn prediction system.

The dataset contains historical customer-level information covering demographics, subscribed services, contract details, billing information, tenure, and customer churn outcomes.

Each row represents one customer.

### Dataset Statistics

| Attribute | Value |
|---|---:|
| Records | 7,043 |
| Fields | 21 |
| Target variable | `Churn` |
| Positive class | `Yes` |
| Negative class | `No` |
| Churn rate | 26.54% |
| Non-churn rate | 73.46% |
| Duplicate customer IDs | 0 |
| Duplicate rows | 0 |

The raw dataset is stored at:

`data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv`

The raw dataset is treated as immutable. Data cleaning, transformation, and feature engineering will be performed by the preprocessing pipeline and written to the processed-data layer.

---

## 2. Data Source

The dataset was obtained from Kaggle.

The downloaded archive was:

`archive.zip`

The archive contained:

`WA_Fn-UseC_-Telco-Customer-Churn.csv`

The archive was extracted into:

`data/raw/`

The raw dataset and archive are excluded from Git through `.gitignore`.

The project does not commit customer-level source data to the Git repository.

---

## 3. Initial Dataset Validation

The dataset was inspected locally before preprocessing.

The following checks were performed:

- File presence verification
- Record count validation
- Column count validation
- Schema inspection
- Data-type inspection
- Null-value inspection
- Duplicate customer ID inspection
- Duplicate row inspection
- Target distribution analysis
- Non-numeric value inspection

### Validation Results

| Validation | Result |
|---|---:|
| Records | 7,043 |
| Columns | 21 |
| Duplicate customer IDs | 0 |
| Duplicate rows | 0 |
| Conventional null values | 0 |
| Blank `TotalCharges` values | 11 |
| Non-numeric `TotalCharges` values | 11 |

---

## 4. Identifier

| Field | Raw Type | Description | ML Usage |
|---|---|---|---|
| `customerID` | object | Unique identifier assigned to each customer | Excluded |

`customerID` is retained for traceability and customer-level analysis.

It will not be used as a predictive feature because it does not represent a meaningful customer characteristic.

---

## 5. Demographic Features

| Field | Raw Type | Description | ML Usage |
|---|---|---|---|
| `gender` | object | Customer gender | Candidate feature |
| `SeniorCitizen` | int64 | Indicates whether the customer is a senior citizen | Candidate feature |
| `Partner` | object | Indicates whether the customer has a partner | Candidate feature |
| `Dependents` | object | Indicates whether the customer has dependents | Candidate feature |

### `SeniorCitizen`

The raw dataset represents `SeniorCitizen` as an integer:

| Value | Meaning |
|---:|---|
| `0` | Not a senior citizen |
| `1` | Senior citizen |

The preprocessing pipeline may represent this as a binary feature.

---

## 6. Tenure

| Field | Raw Type | Description | ML Usage |
|---|---|---|---|
| `tenure` | int64 | Number of months the customer has remained subscribed | Candidate feature |

`tenure` represents the duration of the customer's relationship with the service provider in months.

It is a numerical feature.

---

## 7. Service Features

| Field | Raw Type | Description | ML Usage |
|---|---|---|---|
| `PhoneService` | object | Indicates whether the customer has phone service | Candidate feature |
| `MultipleLines` | object | Indicates whether the customer has multiple telephone lines | Candidate feature |
| `InternetService` | object | Type of internet service subscribed to by the customer | Candidate feature |
| `OnlineSecurity` | object | Indicates whether online security service is subscribed to | Candidate feature |
| `OnlineBackup` | object | Indicates whether online backup service is subscribed to | Candidate feature |
| `DeviceProtection` | object | Indicates whether device protection service is subscribed to | Candidate feature |
| `TechSupport` | object | Indicates whether technical support service is subscribed to | Candidate feature |
| `StreamingTV` | object | Indicates whether the customer subscribes to streaming TV services | Candidate feature |
| `StreamingMovies` | object | Indicates whether the customer subscribes to streaming movie services | Candidate feature |

Some service fields contain values such as:

- `No`
- `Yes`
- `No internet service`
- `No phone service`

These represent meaningful categorical states and should not automatically be treated as missing values.

---

## 8. Contract and Billing Features

| Field | Raw Type | Description | ML Usage |
|---|---|---|---|
| `Contract` | object | Customer contract type | Candidate feature |
| `PaperlessBilling` | object | Indicates whether the customer uses paperless billing | Candidate feature |
| `PaymentMethod` | object | Payment method used by the customer | Candidate feature |
| `MonthlyCharges` | float64 | Customer's monthly service charge | Candidate feature |
| `TotalCharges` | object | Total charges accumulated by the customer | Candidate feature after transformation |

### Contract

The `Contract` field represents the customer's contractual arrangement.

Known categories include:

- Month-to-month
- One year
- Two year

### Payment Method

The `PaymentMethod` field represents the customer's payment method.

Known categories include:

- Electronic check
- Mailed check
- Bank transfer
- Credit card

The preprocessing and validation pipeline will verify the actual categories present in the dataset.

---

## 9. `TotalCharges` Data-Quality Issue

`TotalCharges` is conceptually a numerical field, but the raw CSV is loaded with this column as an `object` type.

The initial validation identified 11 records where `TotalCharges` is blank or whitespace.

### Validation Results

| Condition | Result |
|---|---:|
| Blank/whitespace `TotalCharges` values | 11 |
| Non-numeric `TotalCharges` values | 11 |
| Affected records with `tenure = 0` | 11 |
| Affected records with `Churn = No` | 11 |

The affected customer records were inspected directly.

All 11 records have:

`tenure = 0`

and:

`Churn = No`

### Preprocessing Decision

The raw source values will not be modified.

During preprocessing:

1. Leading and trailing whitespace will be removed.
2. `TotalCharges` will be converted to a numeric representation.
3. Blank values will initially be represented as missing values.
4. The relationship between `TotalCharges` and `tenure` will be validated.
5. The final treatment of the 11 affected records will be implemented explicitly in the preprocessing pipeline and covered by automated tests.

The data dictionary records the observed issue here; the preprocessing implementation will define and enforce the actual remediation logic.

---

## 10. Target Variable

| Field | Raw Type | Description | ML Usage |
|---|---|---|---|
| `Churn` | object | Indicates whether the customer churned | Target |

The target contains two classes:

- `Yes`
- `No`

The intended binary representation is:

| Raw value | Encoded value |
|---|---:|
| `No` | 0 |
| `Yes` | 1 |

The encoding will be implemented centrally in the preprocessing pipeline so that training and inference use the same transformation.

---

## 11. Target Distribution

The raw dataset contains:

| Churn | Records | Percentage |
|---|---:|---:|
| `No` | 5,174 | 73.46% |
| `Yes` | 1,869 | 26.54% |
| **Total** | **7,043** | **100%** |

The dataset therefore exhibits class imbalance.

A model that predicts `No` for every customer would achieve approximately 73.46% accuracy while failing to identify any customers who churn.

Therefore, accuracy will not be used as the sole model-selection criterion.

Model evaluation will consider metrics including:

- Precision
- Recall
- F1 score
- ROC-AUC
- PR-AUC
- Confusion matrix

The final evaluation strategy will be defined as part of the model-development phase.

---

## 12. Feature Classification

### Identifier

- `customerID`

### Categorical Features

- `gender`
- `Partner`
- `Dependents`
- `PhoneService`
- `MultipleLines`
- `InternetService`
- `OnlineSecurity`
- `OnlineBackup`
- `DeviceProtection`
- `TechSupport`
- `StreamingTV`
- `StreamingMovies`
- `Contract`
- `PaperlessBilling`
- `PaymentMethod`

### Binary Feature

- `SeniorCitizen`

### Numerical Features

- `tenure`
- `MonthlyCharges`
- `TotalCharges` after preprocessing

### Target

- `Churn`

The final feature-treatment strategy will be implemented in the preprocessing pipeline and validated through automated tests.

---

## 13. Source Schema

The raw dataset contains the following 21 fields:

1. `customerID`
2. `gender`
3. `SeniorCitizen`
4. `Partner`
5. `Dependents`
6. `tenure`
7. `PhoneService`
8. `MultipleLines`
9. `InternetService`
10. `OnlineSecurity`
11. `OnlineBackup`
12. `DeviceProtection`
13. `TechSupport`
14. `StreamingTV`
15. `StreamingMovies`
16. `Contract`
17. `PaperlessBilling`
18. `PaymentMethod`
19. `MonthlyCharges`
20. `TotalCharges`
21. `Churn`

---

## 14. Data Leakage Considerations

Only information that would realistically be available when a churn prediction is generated should be used as a predictive feature.

The project will avoid using information that becomes available only after the churn event or after the prediction decision.

The preprocessing and feature-engineering pipeline must therefore ensure that:

- Target information is not included in model features.
- `customerID` is not used as a predictive feature.
- Test data is not used during model selection.
- Preprocessing transformations are fitted only on the appropriate training data.
- Post-churn information is excluded.
- Training and inference use consistent feature transformations.

Data leakage checks will form part of the testing and model-development process.

---

## 15. Raw Data Preservation

The raw dataset will remain unchanged after acquisition.

The intended data flow is:

Kaggle Dataset
      |
      v
data/raw/
      |
      | Read-only source
      v
Data Validation
      |
      v
Preprocessing
      |
      v
data/processed/

The same design will later be implemented in AWS:

External Dataset
      |
      v
Amazon S3 - Raw Data
      |
      v
Amazon SageMaker Processing
      |
      +-- Data Validation
      |
      +-- Data Cleaning
      |
      +-- Feature Engineering
      |
      v
Amazon S3 - Processed Data

The AWS implementation will separate raw and processed data to support reproducibility, lineage, governance, and controlled downstream consumption.

---

## 16. Data Lineage

The project will maintain traceability between:

- Original dataset
- Raw stored copy
- Data-validation results
- Processed dataset
- Training dataset
- Trained model
- Model evaluation
- Registered model
- Deployed model
- Model monitoring data
- Future retraining datasets

The eventual MLOps implementation will extend this lineage into AWS services and model artifacts.

The objective is to make it possible to determine the relationship between the source data, processing steps, model versions, deployments, and monitoring outcomes.

The intended lineage is:

Data Source
    |
    v
Raw Dataset
    |
    v
Validation
    |
    v
Processing
    |
    v
Training Dataset
    |
    v
Model Training
    |
    v
Model Evaluation
    |
    v
Model Registry
    |
    v
Deployment
    |
    v
Monitoring
    |
    v
Retraining

---

## 17. Data Governance Considerations

Although this dataset is a publicly available demonstration dataset, the project architecture is designed around production-oriented data-governance principles.

These include:

- Separating raw and processed data
- Restricting data access through IAM
- Applying least-privilege access controls
- Avoiding unnecessary exposure of customer identifiers
- Keeping raw source data immutable
- Maintaining data and model lineage
- Versioning source code and configuration through Git
- Validating data before model training
- Preventing credentials and secrets from entering source control
- Maintaining reproducibility of data-processing steps
- Separating development artifacts from production artifacts

The same principles will be applied when the pipeline is implemented on AWS.

---

## 18. Source-to-Model Field Mapping

| Source Field | Intended Treatment |
|---|---|
| `customerID` | Retain for traceability; exclude from model |
| `gender` | Categorical encoding |
| `SeniorCitizen` | Binary representation |
| `Partner` | Categorical encoding |
| `Dependents` | Categorical encoding |
| `tenure` | Numeric |
| `PhoneService` | Categorical encoding |
| `MultipleLines` | Categorical encoding |
| `InternetService` | Categorical encoding |
| `OnlineSecurity` | Categorical encoding |
| `OnlineBackup` | Categorical encoding |
| `DeviceProtection` | Categorical encoding |
| `TechSupport` | Categorical encoding |
| `StreamingTV` | Categorical encoding |
| `StreamingMovies` | Categorical encoding |
| `Contract` | Categorical encoding |
| `PaperlessBilling` | Categorical encoding |
| `PaymentMethod` | Categorical encoding |
| `MonthlyCharges` | Numeric |
| `TotalCharges` | Numeric conversion and validation |
| `Churn` | Binary target encoding |

The final transformations will be implemented in version-controlled preprocessing code rather than manually performed against the raw CSV.

---

## 19. Data Validation Baseline

The following baseline has been established from the raw dataset.

### Structural Validation

| Check | Result |
|---|---:|
| Expected records | 7,043 |
| Actual records | 7,043 |
| Expected columns | 21 |
| Actual columns | 21 |
| Duplicate customer IDs | 0 |
| Duplicate rows | 0 |

### Missing and Invalid Data

| Check | Result |
|---|---:|
| Conventional null values | 0 |
| Blank `TotalCharges` values | 11 |
| Non-numeric `TotalCharges` values | 11 |

### Target Validation

| Churn | Records | Percentage |
|---|---:|---:|
| `No` | 5,174 | 73.46% |
| `Yes` | 1,869 | 26.54% |

This baseline will be used to develop automated data-validation tests.

---

## 20. Current Data Status

The dataset has been acquired and locally validated at a basic structural and data-quality level.

### Completed

- Dataset acquisition
- Kaggle account setup
- Dataset download
- Archive extraction
- Raw data directory creation
- File presence verification
- Schema inspection
- Record count validation
- Column count validation
- Data-type inspection
- Null-value inspection
- Duplicate-record inspection
- Duplicate-customer inspection
- Target-distribution analysis
- `TotalCharges` anomaly identification
- Initial data-quality assessment
- Data dictionary documentation

### Not Yet Completed

- Automated data-validation script
- Automated data-quality tests
- Formal preprocessing pipeline
- Feature encoding
- Train/validation/test split
- Feature engineering
- Model training
- Model evaluation
- SageMaker Processing
- SageMaker Training
- Model Registry
- Model deployment
- Endpoint testing
- Model monitoring
- Data-drift monitoring
- Model-quality monitoring
- Automated retraining

These components will be implemented incrementally and tracked through Git commits.

---

## 21. Project Implementation Direction

The project will progress from local development and validation toward a production-oriented AWS MLOps implementation.

The high-level progression is:

Local Development
       |
       v
Data Validation
       |
       v
Data Preprocessing
       |
       v
Feature Engineering
       |
       v
Model Development
       |
       v
Model Evaluation
       |
       v
AWS Data Pipeline
       |
       v
SageMaker Processing
       |
       v
SageMaker Training
       |
       v
Model Evaluation / Quality Gate
       |
       v
SageMaker Model Registry
       |
       v
Model Deployment
       |
       v
Monitoring
       |
       v
Drift / Quality Detection
       |
       v
Retraining

The local environment is primarily used for development, validation, testing, and reproducibility.

The production-oriented machine-learning workflow will ultimately run using AWS managed services.

---

## 22. Reproducibility

The project uses Git to version:

- Source code
- Configuration
- Infrastructure definitions
- Pipeline definitions
- Tests
- Documentation
- Automation scripts

Raw and processed datasets are excluded from Git.

The development environment uses Python 3.12 and a project-local virtual environment:

`.venv/`

The project includes a bootstrap script at:

`scripts/bootstrap.sh`

The bootstrap script validates prerequisites, verifies Python 3.12, creates required project directories, and creates the Python virtual environment when required.

Dependencies are declared in:

`requirements.txt`

The environment can therefore be recreated without manually repeating the entire local setup process.

---

## 23. Initial Repository Data Structure

The relevant project structure is:

aws-customer-churn-mlops/
|
+-- architecture/
|
+-- cicd/
|
+-- data/
|   +-- raw/
|   +-- processed/
|   +-- external/
|
+-- deployment/
|
+-- docs/
|   +-- data-dictionary.md
|
+-- experiments/
|
+-- infrastructure/
|
+-- model/
|
+-- monitoring/
|
+-- notebooks/
|
+-- pipelines/
|
+-- scripts/
|   +-- bootstrap.sh
|
+-- security/
|
+-- src/
|
+-- tests/
|
+-- .gitignore
+-- LICENSE
+-- Makefile
+-- README.md
+-- requirements.txt

This structure separates application code, machine-learning artifacts, infrastructure, testing, deployment, monitoring, documentation, and data layers.

---

## 24. Next Implementation Step

The next stage is to convert the manual data-quality checks performed during initial exploration into a version-controlled automated data-validation component.

The validation component should programmatically verify:

1. Expected schema
2. Required columns
3. Record count expectations
4. Duplicate customer IDs
5. Duplicate records
6. Missing values
7. Invalid numeric values
8. Target values
9. Target-class distribution
10. Data-type expectations
11. `TotalCharges` data-quality conditions

The resulting validation logic will become part of the MLOps pipeline rather than remaining as manually executed terminal commands.

The implementation should be designed so that the same validation logic can later be executed within an AWS SageMaker Processing job.
