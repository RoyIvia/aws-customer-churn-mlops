# ML Problem Definition

## 1. Problem Type

This project is a supervised machine learning problem.

The specific task is binary classification.

The model will predict whether a customer is likely to churn.

## 2. Target Variable

The target variable is:

`Churn`

Possible values:

- Yes
- No

The model will ultimately represent this outcome numerically as a binary target.

## 3. Prediction

For each customer, the model should produce a probability representing the estimated likelihood of churn.

Example:

Customer A:

`Churn probability = 0.82`

Customer B:

`Churn probability = 0.17`

The probability can subsequently be converted into a business decision category using a selected threshold.

## 4. Features

Candidate features include:

- Gender
- SeniorCitizen
- Partner
- Dependents
- Tenure
- PhoneService
- MultipleLines
- InternetService
- OnlineSecurity
- TechSupport
- Contract
- PaperlessBilling
- PaymentMethod
- MonthlyCharges
- TotalCharges

`customerID` will be treated as an identifier rather than a predictive feature.

## 5. Primary Evaluation Objective

The initial primary model metric will be recall.

The reason is that the business wants to identify customers who are likely to churn.

A false negative occurs when:

> The model predicts that a customer is unlikely to churn, but the customer actually churns.

Missing these customers may prevent the business from taking a retention action.

Precision will also be monitored because excessively high false-positive rates could cause customer-success resources to be wasted.

Additional metrics will include:

- F1 score
- ROC-AUC
- PR-AUC
- Accuracy
- Confusion matrix

## 6. Important ML Consideration

Accuracy will not be used as the sole model-selection criterion.

Customer churn datasets may contain class imbalance, meaning that the number of customers who do not churn can substantially exceed the number who churn.

Therefore, the model must be evaluated using metrics appropriate for classification performance and the business objective.

## 7. Initial Modeling Strategy

The project will establish a baseline model before introducing more complex modeling.

Candidate approaches include:

1. Logistic Regression baseline
2. XGBoost
3. PyTorch experiment

XGBoost is expected to be the primary production candidate because the problem involves structured/tabular data.

The final selection will be based on measured performance, operational considerations, interpretability, and business requirements rather than algorithm preference alone.

## 8. Training / Validation / Test Strategy

The dataset will be separated into:

- Training data
- Validation data
- Test data

The training set will be used to fit model parameters.

The validation set will be used for model and hyperparameter selection.

The test set will be reserved for final evaluation.

The test dataset must not be used during model selection.

## 9. Data Leakage

Special attention will be given to preventing data leakage.

Features must represent information that would realistically have been available at the time a churn prediction was generated.

Post-churn information must not be used as a predictive feature.

## 10. Prediction Use Case

The model is intended to support customer-success prioritization.

Example:

| Customer | Churn Probability | Risk |
|---|---:|---|
| C001 | 0.91 | High |
| C002 | 0.64 | Medium |
| C003 | 0.08 | Low |

Risk categories and thresholds will be established after model evaluation rather than arbitrarily assumed in advance.
