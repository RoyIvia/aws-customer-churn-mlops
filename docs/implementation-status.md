# Implementation Status

## Project Status

The AWS Customer Churn MLOps project has progressed from local model development to deployment of the orchestration layer on Amazon SageMaker.

The SageMaker Pipeline has been successfully created and deployed in `us-east-1`. End-to-end execution testing is currently in progress.

## Completed Components

### Data Preparation

* Dataset acquisition completed
* Dataset schema and quality validation implemented
* Missing and malformed values handled
* Categorical features encoded
* Numerical features prepared for model training
* Stratified train, validation, and test datasets generated
* Preprocessing artifacts persisted to Amazon S3

### Model Development

* XGBoost binary classification model implemented
* Class imbalance handled using `scale_pos_weight`
* Local model training completed
* Validation and test evaluation completed
* Model quality thresholds defined

Current test-set performance:

| Metric    | Result |
| --------- | -----: |
| Accuracy  | 0.7537 |
| Precision | 0.5244 |
| Recall    | 0.7754 |
| F1 Score  | 0.6257 |
| ROC-AUC   | 0.8429 |

Quality-gate thresholds:

| Metric   | Minimum |
| -------- | ------: |
| F1 Score |    0.60 |
| Recall   |    0.70 |
| ROC-AUC  |    0.75 |

The current model satisfies all defined quality thresholds.

### Automated Testing

The project currently contains 34 automated tests covering validation, preprocessing, training-support logic, evaluation, and pipeline-related functionality.

```text
34 passed
```

### Containerization

A custom processing image has been built and pushed to Amazon Elastic Container Registry (ECR).

Repository:

```text
633605692302.dkr.ecr.us-east-1.amazonaws.com/aws-customer-churn-processing
```

The image is pinned using an immutable SHA-256 digest to improve reproducibility.

### SageMaker Pipeline

The current pipeline implements the following workflow:

```text
Raw Dataset
     |
     v
Validation
     |
     v
Preprocessing
     |
     v
Training
     |
     v
Evaluation
     |
     v
Quality Gate
```

The pipeline has been successfully registered with Amazon SageMaker:

```text
aws-customer-churn-mlops-pipeline
```

The pipeline execution role is:

```text
SageMakerCustomerChurnExecutionRole
```

## AWS Execution Testing

Multiple real SageMaker Pipeline executions have been initiated while validating the AWS runtime environment.

During execution testing, several account-level IAM requirements were identified and resolved.

### IAM Permissions Resolved

The execution role required permission to pass itself to SageMaker jobs created by pipeline steps:

```text
iam:PassRole
```

The role also required:

```text
sagemaker:AddTags
```

for resources created and tagged by SageMaker Pipelines.

Finally, the pipeline execution role required:

```text
sagemaker:CreateProcessingJob
```

to launch the Validation and Preprocessing processing jobs.

These permissions have been added to the SageMaker execution role.

## Current Blocker

IAM authorization is no longer the current execution blocker.

The latest SageMaker Pipeline execution successfully progressed far enough to request creation of the Validation Processing Job.

AWS rejected the job because the account currently has an applied quota of zero for:

```text
ml.m5.large for processing job usage
```

Current quota:

```text
0 instances
```

Required quota:

```text
1 instance
```

A Service Quotas increase has been submitted for the `us-east-1` Region.

Quota code:

```text
L-8541302D
```

Requested value:

```text
1
```

AWS Support case:

```text
178904496400585
```

Current request status:

```text
CASE_OPENED
```

This is an AWS account-level service quota restriction rather than an application, pipeline-definition, or IAM failure.

## Remaining Implementation

Once the SageMaker Processing quota increase is approved, implementation will continue with:

```text
Successful Pipeline Execution
        |
        v
Verify S3 Artifacts
        |
        v
Model Registry
        |
        v
Manual Model Approval
        |
        v
Model Deployment
        |
        v
SageMaker Endpoint
        |
        v
Inference Testing
        |
        v
Model Monitoring
        |
        v
Drift Detection
        |
        v
Retraining Workflow
```

Additional production hardening will include:

* IAM least-privilege refinement
* Encryption controls
* Network isolation considerations
* CloudWatch observability
* Model monitoring
* Cost controls
* Resource cleanup procedures
* Deployment and operational documentation

## Current Overall State

The local ML implementation and SageMaker orchestration layer are complete.

The project is currently in the **AWS execution validation stage**, with successful end-to-end pipeline execution pending approval of the required SageMaker Processing service quota.

