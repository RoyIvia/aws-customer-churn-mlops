# Data Dictionary

## 1. Dataset Overview

The project uses historical customer-level data from a subscription-based service.

Each row represents one customer and contains demographic, service, contract, billing, and tenure information.

The dataset is used to train a supervised binary classification model that predicts customer churn.

---

## 2. Identifier

| Field        | Type   | Description                                 | ML Usage |
| ------------ | ------ | ------------------------------------------- | -------- |
| `customerID` | String | Unique identifier assigned to each customer | Excluded |

`customerID` is retained for customer-level identification and traceability but is not used as a predictive feature.

---

## 3. Demographic Features

| Field           | Type           | Description                                        | ML Usage          |
| --------------- | -------------- | -------------------------------------------------- | ----------------- |
| `gender`        | Categorical    | Customer gender                                    | Candidate feature |
| `SeniorCitizen` | Binary/Integer | Indicates whether the customer is a senior citizen | Candidate feature |
| `Partner`       | Categorical    | Indicates whether the customer has a partner       | Candidate feature |
| `Dependents`    | Categorical    | Indicates whether the customer has dependents      | Candidate feature |

---

## 4. Service Features

| Field             | Type        | Description                                                  | ML Usage          |
| ----------------- | ----------- | ------------------------------------------------------------ | ----------------- |
| `PhoneService`    | Categorical | Indicates whether the customer has phone service             | Candidate feature |
| `MultipleLines`   | Categorical | Indicates whether the customer has multiple telephone lines  | Candidate feature |
| `InternetService` | Categorical | Type of internet service subscribed to by the customer       | Candidate feature |
| `OnlineSecurity`  | Categorical | Indicates whether online security service is subscribed to   | Candidate feature |
| `TechSupport`     | Categorical | Indicates whether technical support service is subscribed to | Candidate feature |

---

## 5. Contract and Billing Features

| Field              | Type        | Description                                           | ML Usage          |
| ------------------ | ----------- | ----------------------------------------------------- | ----------------- |
| `Contract`         | Categorical | Customer contract type                                | Candidate feature |
| `PaperlessBilling` | Categorical | Indicates whether the customer uses paperless billing | Candidate feature |
| `PaymentMethod`    | Categorical | Payment method used by the customer                   | Candidate feature |
| `MonthlyCharges`   | Numeric     | Customer's monthly service charge                     | Candidate feature |
| `TotalCharges`     | Numeric     | Total charges accumulated by the customer             | Candidate feature |

---

## 6. Tenure Feature

| Field    | Type    | Description                                           | ML Usage          |
| -------- | ------- | ----------------------------------------------------- | ----------------- |
| `tenure` | Numeric | Number of months the customer has remained subscribed | Candidate feature |

---

## 7. Target Variable

| Field   | Type        | Description                            | ML Usage |
| ------- | ----------- | -------------------------------------- | -------- |
| `Churn` | Categorical | Indicates whether the customer churned | Target   |

Expected values:

* `Yes`
* `No`

The target will be transformed into a binary representation during preprocessing:

```text
Yes → 1
No  → 0
```

---

## 8. Feature Categories

The candidate predictive variables can therefore be grouped as follows.

### Demographic

* `gender`
* `SeniorCitizen`
* `Partner`
* `Dependents`

### Service

* `PhoneService`
* `MultipleLines`
* `InternetService`
* `OnlineSecurity`
* `TechSupport`

### Contract and Billing

* `Contract`
* `PaperlessBilling`
* `PaymentMethod`
* `MonthlyCharges`
* `TotalCharges`

### Customer Tenure

* `tenure`

---

## 9. Excluded Fields

### `customerID`

`customerID` is excluded from model training because it is an identifier and does not represent a meaningful customer behavior characteristic.

It may be retained in inference outputs or prediction records to associate predictions with customers.

---

## 10. Data Quality Considerations

The preprocessing pipeline must validate:

* Missing values
* Invalid categorical values
* Invalid numeric values
* Numeric fields incorrectly represented as strings
* Duplicate customer records
* Unexpected target values
* Inconsistent categorical representations
* Potential outliers in numeric variables

Particular attention should be given to `TotalCharges`, which may require conversion from string representation to a numeric data type.

---

## 11. Data Leakage Considerations

Features must represent information that would have been available when the prediction was generated.

The following principles apply:

1. The target variable `Churn` must not be included as an input feature.
2. `customerID` must not be used as a predictive feature.
3. Post-churn information must not be included.
4. Any feature derived from the target must be excluded.
5. Preprocessing parameters must be learned from the training dataset and applied to validation and test data.

---

## 12. Data Splitting

The dataset will be divided into:

* Training dataset
* Validation dataset
* Test dataset

The test dataset will remain isolated until final model evaluation.

Stratification should be considered because churn is a binary target and class imbalance may exist.

---

## 13. Feature Engineering

Initial implementation will use the existing customer attributes as candidate features.

Feature engineering may subsequently include:

* Encoding categorical variables
* Converting numeric fields to appropriate data types
* Handling missing values
* Scaling numeric variables where required by the selected algorithm
* Creating derived customer-behavior features where justified

Feature engineering decisions must be validated against the data and monitored for leakage.

---

## 14. Data Contract

The preprocessing pipeline should produce a consistent schema suitable for downstream SageMaker processing and training.

The expected contract is:

```text
Raw Customer Data
        ↓
Schema Validation
        ↓
Data Cleaning
        ↓
Feature Transformation
        ↓
Train / Validation / Test
        ↓
Model Training
```

Any dataset that fails mandatory schema or quality checks should not proceed to model training.
