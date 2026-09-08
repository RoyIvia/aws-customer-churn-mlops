# AWS Customer Churn Prediction & MLOps Platform

> Production-oriented machine learning platform for predicting customer churn using Amazon SageMaker and AWS-native MLOps services.

## Project Status

🚧 **In Progress**

The project has progressed from architecture and requirements design into implementation of the machine learning pipeline.

Current implementation status:

| Component                        | Status             |
| -------------------------------- | ------------------ |
| Business requirements            | ✅ Complete         |
| ML problem definition            | ✅ Complete         |
| Architecture design              | ✅ Complete         |
| Architecture decisions           | ✅ Complete         |
| Dataset acquisition              | ✅ Complete         |
| Data validation                  | ✅ Complete         |
| Data preprocessing               | ✅ Complete         |
| Local model training             | ✅ Complete         |
| Local model evaluation           | ✅ Complete         |
| SageMaker processing container   | ✅ Complete         |
| Amazon ECR repository/image      | ✅ Complete         |
| SageMaker Pipeline definition    | ✅ Complete         |
| Automated pipeline tests         | ✅ 34 tests passing |
| Pipeline deployment to SageMaker | 🚧 In progress     |
| Pipeline execution               | ⏳ Pending          |
| Model Registry integration       | ⏳ Planned          |
| Model deployment                 | ⏳ Planned          |
| Production monitoring            | ⏳ Planned          |

The current deployment blocker is IAM role resolution during the SageMaker Pipeline `upsert()` operation. The pipeline definition itself compiles successfully.



## Executive Summary

Customer churn is a major business risk for subscription-based organizations because retaining an existing customer is typically preferable to acquiring a replacement after cancellation.

This project designs and implements an AWS-based machine learning platform that predicts customers who are likely to churn so that customer-success or retention teams can intervene before cancellation.

The solution demonstrates the complete lifecycle of a production-oriented ML workload:

* business requirements analysis
* data ingestion
* data validation
* feature preprocessing
* model training
* model evaluation
* automated quality gates
* containerized processing
* pipeline orchestration
* model governance
* deployment architecture
* monitoring
* security
* cost optimization

The project is implemented from both an **MLOps engineering** and **AWS Solutions Architecture** perspective rather than treating model training as an isolated data-science exercise.



## Business Problem

A subscription-based company wants to identify customers who are likely to cancel their services.

Without predictive capabilities, retention teams typically operate reactively and may only engage a customer after cancellation or after clear signs of dissatisfaction have already appeared.

The objective of the platform is therefore to answer:

> **Which currently active customers have the highest probability of churning?**

The prediction can then be consumed by downstream customer-success, CRM, marketing, or retention workflows.

### Business Objective

Develop a machine learning system capable of identifying customers at elevated risk of churn early enough for the business to take preventative action.

### ML Objective

The problem is formulated as a **binary classification problem**:

```text
Churn = Yes → 1
Churn = No  → 0
```

The model outputs the probability that a customer belongs to the churn class.



## Solution

The platform uses Amazon SageMaker to automate the machine learning lifecycle.

At a high level:

```text
Customer Dataset
      │
      ▼
 Amazon S3
      │
      ▼
Data Validation
      │
      ▼
Data Preprocessing
      │
      ▼
Model Training
      │
      ▼
Model Evaluation
      │
      ▼
Quality Gate
      │
      ▼
Model Registry
      │
      ▼
Model Approval
      │
      ▼
Deployment
      │
      ▼
Monitoring
```

Amazon S3 provides durable object storage for the raw dataset, processed datasets, model artifacts, and evaluation outputs.

Amazon SageMaker Pipelines orchestrates the machine learning workflow.

Custom preprocessing workloads are packaged as Docker containers and stored in Amazon Elastic Container Registry.

The model is trained using XGBoost.

Models that satisfy defined evaluation thresholds will eventually be registered in SageMaker Model Registry for controlled versioning and approval before deployment.



## Architecture

The target architecture follows an AWS-native managed-service approach.

```text
                     ┌──────────────────────┐
                     │     Raw Dataset      │
                     └──────────┬───────────┘
                                │
                                ▼
                     ┌──────────────────────┐
                     │      Amazon S3       │
                     │      Raw Zone        │
                     └──────────┬───────────┘
                                │
                                ▼
              ┌─────────────────────────────────┐
              │      Amazon SageMaker           │
              │          Pipelines              │
              └──────────────┬──────────────────┘
                             │
          ┌──────────────────┼───────────────────┐
          │                  │                   │
          ▼                  ▼                   ▼
   Data Validation     Preprocessing         Training
   SageMaker           SageMaker             SageMaker
   Processing          Processing            Training Job
          │                  │                   │
          │                  ▼                   │
          │             Amazon S3               │
          │             Processed Data           │
          │                                      ▼
          │                                Model Artifact
          │                                      │
          └──────────────────┬───────────────────┘
                             ▼
                     Model Evaluation
                             │
                             ▼
                       Quality Gate
                             │
                    ┌────────┴────────┐
                    │                 │
                  PASS              FAIL
                    │                 │
                    ▼                 ▼
             Model Registry      Stop Workflow
                    │
                    ▼
              Manual Approval
                    │
                    ▼
             Model Deployment
                    │
                    ▼
               Monitoring
```

Detailed architecture documentation is maintained under the [`docs/`](docs/) and [`architecture/`](architecture/) directories.



## Technology Stack

### AWS

* Amazon SageMaker
* SageMaker Pipelines
* SageMaker Processing
* SageMaker Training
* Amazon SageMaker Model Registry — planned
* Amazon S3
* Amazon Elastic Container Registry
* AWS Identity and Access Management
* Amazon CloudWatch — planned for production monitoring

### Machine Learning

* XGBoost
* pandas
* scikit-learn
* joblib

### MLOps / Engineering

* Python 3.12
* Docker
* Git
* GitHub
* Bash
* pytest
* SageMaker Python SDK



## Dataset

The project uses the **IBM Telco Customer Churn dataset**.

Dataset characteristics:

```text
Rows:       7,043
Columns:    21
Target:     Churn
```

Class distribution:

| Class    | Customers | Percentage |
| -------- | --------: | ---------: |
| No Churn |     5,174 |     73.46% |
| Churn    |     1,869 |     26.54% |

The dataset is intentionally excluded from Git and stored separately because production ML repositories should separate source code from operational datasets.

The raw dataset is stored in Amazon S3 under:

```text
raw/telco/
```



## Data Validation

A dedicated validation stage executes before preprocessing.

The validation logic checks dataset assumptions before computational resources are used for training.

One issue identified during validation was the `TotalCharges` field.

The dataset contains:

```text
11 non-numeric / whitespace TotalCharges values
```

These records correspond to customers with:

```text
tenure = 0
```

The validation stage ensures these data-quality conditions are explicitly detected rather than silently propagating invalid values through the pipeline.



## Machine Learning Approach

### Data Split

The dataset is split into:

```text
Training:     4,225 records
Validation:   1,409 records
Test:         1,409 records
```

The processed dataset contains approximately:

```text
45 engineered model features
```

plus the target variable.

### Algorithm

The project uses **XGBoost binary classification**.

Core training configuration:

```text
objective:          binary:logistic
eval_metric:        logloss
num_round:          200
max_depth:          4
eta:                0.05
subsample:          0.8
colsample_bytree:   0.8
```

Because the dataset is imbalanced, the model uses:

```text
scale_pos_weight ≈ 2.769
```

derived from the negative-to-positive class ratio in the training dataset.



## Model Evaluation

The model is evaluated using metrics appropriate for an imbalanced binary classification problem.

The primary metrics are:

* Precision
* Recall
* F1 Score
* ROC-AUC
* Accuracy

Recall is particularly important because a false negative represents a customer who is predicted to remain but actually churns.

### Local Baseline Results

#### Validation Set

| Metric    | Result |
| --------- | -----: |
| Accuracy  | 0.7466 |
| Precision | 0.5153 |
| Recall    | 0.7647 |
| F1 Score  | 0.6157 |
| ROC-AUC   | 0.8381 |

#### Test Set

| Metric    | Result |
| --------- | -----: |
| Accuracy  | 0.7537 |
| Precision | 0.5244 |
| Recall    | 0.7754 |
| F1 Score  | 0.6257 |
| ROC-AUC   | 0.8429 |

These results provide the initial model baseline before further optimization.



## Automated Quality Gate

Model promotion is controlled through predefined minimum performance thresholds.

```text
F1 Score >= 0.60
Recall   >= 0.70
ROC-AUC  >= 0.75
```

The SageMaker Pipeline evaluates the model against all three thresholds.

Conceptually:

```text
                  Evaluation Metrics
                         │
                         ▼
              ┌─────────────────────┐
              │    Quality Gate     │
              └──────────┬──────────┘
                         │
          ┌──────────────┴──────────────┐
          │                             │
        PASS                           FAIL
          │                             │
          ▼                             ▼
 Model Registration              No Promotion
```

This prevents model promotion based purely on successful training.



## MLOps Pipeline

The implemented SageMaker Pipeline currently contains five stages:

```text
Validation
    │
    ▼
Preprocessing
    │
    ▼
Training
    │
    ▼
Evaluation
    │
    ▼
QualityGate
```

### 1. Validation

Runs dataset checks before preprocessing.

### 2. Preprocessing

Uses SageMaker Processing with a custom Docker image to:

* clean the dataset
* transform features
* encode categorical variables
* generate train, validation, and test datasets
* persist preprocessing artifacts

### 3. Training

Runs a managed SageMaker XGBoost training job.

### 4. Evaluation

Loads the trained model and validation dataset and generates:

```text
evaluation.json
predictions.csv
```

### 5. Quality Gate

Reads the evaluation metrics using SageMaker Pipeline property files and determines whether model performance satisfies the promotion thresholds.



## Containerization

Custom processing logic is packaged using Docker.

The processing image is stored in Amazon ECR.

```text
633605692302.dkr.ecr.us-east-1.amazonaws.com/aws-customer-churn-processing
```

The pipeline references the container using a pinned image digest rather than relying only on a mutable image tag.

This improves reproducibility because the processing pipeline executes against a specific container image version.



## Amazon S3 Layout

The project separates artifacts by lifecycle stage.

```text
s3://<project-bucket>/
│
├── raw/
│   └── telco/
│
├── processed/
│   └── <pipeline-execution-id>/
│
├── artifacts/
│   └── <pipeline-execution-id>/
│       ├── preprocessing/
│       └── evaluation/
│
└── customer-churn-training/
```

Pipeline execution IDs are incorporated into artifact paths to provide execution isolation and lineage between runs.



## Security

Security design is based on AWS least-privilege principles.

The SageMaker workloads use a dedicated execution role:

```text
SageMakerCustomerChurnExecutionRole
```

The role requires controlled access to:

* project S3 objects
* ECR images
* SageMaker jobs
* CloudWatch Logs
* networking resources required by SageMaker-managed jobs

The role trust relationship permits the SageMaker service to assume the execution role.

### IAM Design

The architecture separates:

```text
Human / CI identity
        │
        │ iam:PassRole
        ▼
SageMaker Execution Role
        │
        ▼
Processing / Training / Pipeline workloads
```

The calling identity should only be allowed to pass approved SageMaker execution roles rather than arbitrary IAM roles.

Further security controls are documented in:

```text
docs/security.md
```



## Monitoring

The target production architecture includes monitoring across both infrastructure and ML performance.

Planned monitoring includes:

### Infrastructure Monitoring

Amazon CloudWatch for:

* SageMaker job failures
* pipeline execution failures
* resource utilization
* endpoint errors
* endpoint latency

### Model Monitoring

Future production monitoring will consider:

* input data drift
* prediction distribution drift
* feature drift
* model quality degradation
* prediction latency

Monitoring design is documented in:

```text
docs/monitoring.md
```



## Cost Optimisation

The architecture deliberately avoids maintaining always-on infrastructure during the training workflow.

The pipeline primarily uses ephemeral managed resources:

```text
Processing Job
      ↓
terminated after completion

Training Job
      ↓
terminated after completion
```

Major cost considerations include:

* SageMaker processing instance runtime
* SageMaker training instance runtime
* deployed inference endpoint runtime
* S3 storage
* ECR image storage
* CloudWatch logging

Future optimisations may include:

* SageMaker Managed Spot Training
* right-sizing training instances
* serverless or asynchronous inference where workload characteristics permit
* S3 lifecycle policies
* automatic deletion of obsolete development artifacts

Additional analysis is maintained in:

```text
docs/cost-optimization.md
```



## Deployment Strategy

The intended model lifecycle is:

```text
Train
  │
  ▼
Evaluate
  │
  ▼
Quality Gate
  │
  ▼
Model Registry
  │
  ▼
Pending Manual Approval
  │
  ▼
Approved
  │
  ▼
Deployment
```

Model Registry integration and endpoint deployment have **not yet been implemented**.

The project will not represent these components as deployed until their implementation and validation are complete.



## Current SageMaker Deployment Status

The SageMaker Pipeline definition currently compiles successfully with the following stages:

```text
Validation
Preprocessing
Training
Evaluation
QualityGate
```

The latest deployment attempt reached:

```text
pipeline.upsert()
```

but stopped before pipeline creation because the SageMaker SDK could not automatically resolve a pipeline IAM role from the local caller identity.

The reported error was:

```text
RoleValidationError:
No IAM role could be resolved from your caller identity
for 'pipeline' workloads.
```

This is currently being resolved by explicitly associating the SageMaker execution role with the pipeline.

No SageMaker Pipeline execution is therefore represented as successfully completed yet.



## Testing

The project contains automated tests for core pipeline and ML functionality.

Current test result:

```text
34 passed
```

This validates the local implementation before AWS pipeline execution.



## Project Structure

```text
aws-customer-churn-mlops/
│
├── architecture/
│   └── README.md
│
├── data/
│   └── raw/
│
├── docker/
│
├── docs/
│   ├── architecture-block-diagram.md
│   ├── architecture-decisions.md
│   ├── business-requirements.md
│   ├── cost-optimization.md
│   ├── data-dictionary.md
│   ├── lessons-learned.md
│   ├── ml-problem-definition.md
│   ├── model-evaluation.md
│   ├── monitoring.md
│   └── security.md
│
├── infrastructure/
│
├── pipelines/
│   └── sagemaker/
│
├── scripts/
│   ├── sagemaker/
│   └── setup-dev.sh
│
├── src/
│   └── pipeline/
│
├── tests/
│
├── .gitignore
├── LICENSE
├── Makefile
├── README.md
└── requirements.txt
```

The raw dataset is intentionally excluded from source control.



## Implementation

The project is being implemented incrementally to preserve clear separation between architecture design, local validation, and AWS deployment.

### Phase 1 — Architecture & Requirements
```
✅ Business requirements
✅ ML problem definition
✅ Architecture design
✅ Architecture decisions
✅ Security design
✅ Cost considerations
```

### Phase 2 — Data & Local ML

```
✅ Dataset acquisition
✅ Data validation
✅ Data preprocessing
✅ Feature transformation
✅ Train/validation/test split
✅ Local XGBoost training
✅ Local evaluation baseline
```
### Phase 3 — Containerized Processing

```
✅ Docker processing environment
✅ Amazon ECR repository
✅ Processing container image
✅ Immutable image digest reference
```
### Phase 4 — SageMaker Pipeline

```
✅ Validation step
✅ Preprocessing step
✅ Training step
✅ Evaluation step
✅ Quality gate
✅ Pipeline definition compilation
🚧 Pipeline IAM configuration
⏳ First successful AWS execution
```
### Phase 5 — Model Governance

```
⏳ SageMaker Model Registry
⏳ Model package group
⏳ Model versioning
⏳ Manual approval workflow
```

### Phase 6 — Deployment

```
⏳ Endpoint configuration
⏳ Model deployment
⏳ Inference testing
```
### Phase 7 — Operations

```
⏳ Production monitoring
⏳ Drift monitoring
⏳ Alerting
⏳ Pipeline automation
```

## Results

The current baseline demonstrates that the model can identify a substantial proportion of customers who churn.

Test performance:

```text
Accuracy:   0.7537
Precision:  0.5244
Recall:     0.7754
F1:         0.6257
ROC-AUC:    0.8429
```

The model satisfies the currently defined local promotion thresholds:

```text
F1       >= 0.60
Recall   >= 0.70
ROC-AUC  >= 0.75
```

These results are currently based on local model validation.

They will be compared with results generated by the SageMaker Pipeline after the first successful cloud execution.

---

## Lessons Learned

Key engineering lessons from the project so far include:

* Building an ML model is only one component of a production ML system.
* Dataset assumptions should be validated before preprocessing and training.
* Class imbalance must influence both model configuration and metric selection.
* Accuracy alone is insufficient for churn prediction.
* Reproducible container images improve consistency between environments.
* Pipeline outputs should be isolated by execution to support lineage.
* IAM design is part of ML system architecture, not an afterthought.
* SageMaker SDK interfaces can differ significantly between major SDK versions.
* Local tests and pipeline-definition validation reduce the cost of discovering problems in AWS.
* A successful pipeline definition does not guarantee successful pipeline deployment; AWS identity, role assumption, and `iam:PassRole` requirements must also be satisfied.

---

## Future Improvements

Planned improvements include:

* Complete the first SageMaker Pipeline execution
* Integrate SageMaker Model Registry
* Implement model package versioning
* Add manual model approval
* Deploy an approved model
* Implement inference testing
* Add automated model monitoring
* Add data and model drift detection
* Add CloudWatch alarms
* Evaluate SageMaker Managed Spot Training
* Introduce automated pipeline triggers
* Add CI/CD integration
* Introduce model explainability where appropriate
* Evaluate alternative models and hyperparameter tuning
* Introduce controlled retraining based on monitored model performance

---

## Repository Philosophy

This repository is intended to demonstrate more than the ability to train a machine learning model.

It documents the decisions required to design an ML workload as an operational AWS system:

```text
Business Requirement
        ↓
Architecture
        ↓
Data Engineering
        ↓
Model Development
        ↓
MLOps
        ↓
Security
        ↓
Governance
        ↓
Deployment
        ↓
Operations
```

The objective is to demonstrate how machine learning workloads should be evaluated from both an **ML engineering** and **cloud architecture** perspective.

---

## Disclaimer

This repository distinguishes between components that were **implemented and validated**, components that are **currently being implemented**, and components that are **architecturally designed but not yet deployed**.

No AWS deployment, pipeline execution, model registration, endpoint deployment, or monitoring capability is represented as completed unless it has actually been implemented and validated.

