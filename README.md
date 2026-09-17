# AWS Customer Churn MLOps Platform

An end-to-end MLOps platform for predicting customer churn using **Amazon SageMaker and XGBoost**, covering the complete model lifecycle from data validation and training to model governance, deployment, monitoring, and retraining.

The project demonstrates how a machine learning workload can be designed as a **governed production system on AWS**, rather than simply training and deploying a model.

## Architecture

![AWS Customer Churn MLOps Architecture](architecture/AWS_Churn_MLOps.png)

```text
Customer Data
      ↓
   Amazon S3
      ↓
Data Validation
      ↓
Preprocessing & Feature Engineering
      ↓
SageMaker XGBoost Training
      ↓
Model Evaluation
      ↓
Automated Quality Gate
      ↓
SageMaker Model Registry
      ↓
Manual Approval
      ↓
Real-Time SageMaker Endpoint
      ↓
Monitoring & Drift Detection
      ↓
Retraining
```

## Business Problem

Customer churn directly affects recurring revenue and customer lifetime value. The objective is to identify customers with an elevated risk of churn so retention teams can prioritize interventions before customers leave.

The project uses the **IBM Telco Customer Churn dataset**, containing **7,043 customer records** with demographic, account, service, contract, billing, and churn information.

The dataset has a **26.54% churn rate**, making class imbalance an important consideration during training and evaluation.

## Solution

The platform implements a complete MLOps workflow:

* **Data validation** checks the raw dataset before processing.
* **Preprocessing** cleans the data and creates reproducible model features.
* **Amazon SageMaker Pipelines** orchestrates the ML workflow.
* **SageMaker managed training** trains an XGBoost classifier.
* **Automated evaluation** calculates Accuracy, Precision, Recall, F1, and ROC-AUC.
* **Quality gates** prevent models below defined performance thresholds from progressing.
* **SageMaker Model Registry** versions successful model candidates.
* **Manual approval** provides a governance boundary before production deployment.
* **SageMaker real-time endpoints** provide managed online inference.
* **CloudWatch and SageMaker Model Monitor** provide operational and model observability.
* **Drift detection and retraining** feed new candidates back through the same governed pipeline.

## SageMaker Pipeline

The ML workflow is implemented as:

```text
Validation
    ↓
Preprocessing
    ↓
Training
    ↓
Evaluation
    ↓
QualityGate
    │
    ├── FAIL → Stop
    │
    └── PASS
           ↓
      RegisterModel
           ↓
   PendingManualApproval
           ↓
      Manual Review
           ↓
        Approved
           ↓
       Deployment
```

This separates **model creation from production authorization**. A successfully trained model cannot automatically deploy itself.

## Model Performance

The XGBoost model achieved:

| Metric    | Validation |   Test |
| --------- | ---------: | -----: |
| Accuracy  |     0.7466 | 0.7537 |
| Precision |     0.5153 | 0.5244 |
| Recall    |     0.7647 | 0.7754 |
| F1        |     0.6157 | 0.6257 |
| ROC-AUC   |     0.8381 | 0.8429 |

Because the business problem places importance on identifying customers who may churn, the pipeline explicitly evaluates **Recall** alongside F1 and ROC-AUC rather than relying on accuracy alone.

### Automated Quality Gate

A model must satisfy all three conditions:

| Metric  | Minimum |
| ------- | ------: |
| F1      |    0.60 |
| Recall  |    0.70 |
| ROC-AUC |    0.75 |

Only models satisfying the quality gate enter the Model Registry promotion path.

## Model Governance

Models passing automated evaluation are registered in the SageMaker Model Registry as:

```text
PendingManualApproval
```

Production promotion is a separate action.

The approval workflow validates the exact **Model Package ARN**, package group, package status, and approval state before changing the model to:

```text
Approved
```

The deployment workflow independently verifies that the selected model is approved before creating production inference resources.

This creates the following governance boundary:

```text
Training → Technical Acceptance → Registration → Human Approval → Deployment
```

## Monitoring & Retraining

The production architecture combines:

**SageMaker Endpoint → Data Capture → Amazon S3 → SageMaker Model Monitor → CloudWatch → EventBridge → Retraining**

Monitoring covers:

* endpoint availability and invocation errors
* inference latency
* production data quality
* feature distribution changes
* data drift
* model quality
* model performance drift

Retraining uses the same SageMaker Pipeline, ensuring new models cannot bypass validation, evaluation, quality gates, registration, and approval.

## Security & Reproducibility

The architecture incorporates:

* dedicated SageMaker execution roles
* controlled `iam:PassRole`
* least-privilege IAM design
* S3 access controls
* encryption at rest and in transit
* ECR-hosted processing containers
* immutable container image digests
* versioned model artifacts
* controlled Model Registry promotion
* explicit production approval
* CloudWatch observability
* CloudTrail auditability
* VPC-based workload isolation

The preprocessing environment is containerized and pinned using an immutable ECR image digest, preventing changes to a mutable container tag from silently altering the pipeline execution environment.

## Testing & CI

The project includes automated tests covering data validation, preprocessing, evaluation logic, pipeline components, and model-governance controls.

Run locally with:

```bash
pytest -q
```

GitHub Actions performs automated repository validation on pushes and pull requests, including Python compilation and the test suite.

AWS deployment is deliberately separated from repository CI so source-code validation does not require production AWS credentials.

## Technology Stack

| Area               | Technology                    |
| ------------------ | ----------------------------- |
| Cloud              | AWS                           |
| ML Platform        | Amazon SageMaker              |
| Algorithm          | XGBoost                       |
| Storage            | Amazon S3                     |
| Containers         | Docker                        |
| Container Registry | Amazon ECR                    |
| ML Orchestration   | SageMaker Pipelines           |
| Model Governance   | SageMaker Model Registry      |
| Inference          | SageMaker Real-Time Endpoints |
| Monitoring         | SageMaker Model Monitor       |
| Observability      | Amazon CloudWatch             |
| Event Automation   | Amazon EventBridge            |
| Security           | AWS IAM                       |
| CI                 | GitHub Actions                |
| Testing            | pytest                        |
| Language           | Python                        |

## Repository Structure

```text
.
├── .github/workflows/       # CI workflows
├── architecture/            # Architecture diagram and design
├── data/                    # Dataset structure
├── docker/                  # Processing container
├── docs/                    # Detailed technical documentation
├── infrastructure/          # Infrastructure configuration
├── pipelines/sagemaker/     # SageMaker Pipeline
├── scripts/sagemaker/       # Operational and deployment tooling
├── src/                     # ML/data-processing source code
├── tests/                   # Automated tests
└── README.md
```

## Documentation

Detailed technical documentation is available in:

* [Architecture](architecture/README.md)
* [Business Requirements](docs/business-requirements.md)
* [Model Governance](docs/model-governance.md)
* [Monitoring & Retraining](docs/monitoring.md)
* [Security Architecture](docs/security.md)
* [Operations Runbook](docs/operations.md)

## Key Design Decisions

The project follows several production MLOps principles:

**Reproducibility over manual experimentation** — preprocessing and training are deterministic pipeline stages.

**Immutable artifacts over mutable dependencies** — processing containers and model versions are explicitly identified.

**Automated validation before promotion** — model candidates must satisfy measurable quality criteria.

**Human authorization before production deployment** — technical success alone does not authorize deployment.

**Separation of training and deployment** — model creation cannot directly modify production.

**Traceability across the model lifecycle** — data processing, training, evaluation, registration, approval, and deployment remain connected through versioned artifacts and AWS service metadata.

**Governed retraining** — retrained models pass through the same controls as the original model.

## Project Outcome

The result is a production-oriented AWS MLOps architecture that treats machine learning as a **managed lifecycle**:

**data → model → evaluation → governance → deployment → monitoring → retraining**


