# Lessons Learned

## Project status

The project has progressed from local machine learning development to AWS SageMaker pipeline definition.

Completed implementation areas:

- Business requirements
- Architecture decisions
- Architecture diagram
- Data dictionary
- Local dataset validation
- Train, validation, and test splitting
- Preprocessing pipeline
- Baseline XGBoost model
- Model evaluation
- Local quality gate
- Untouched test-set evaluation
- S3 project storage
- SageMaker execution IAM role
- Custom SageMaker Processing Docker image
- ECR image repository and image push
- SageMaker Pipeline definition
- Validation, preprocessing, training, evaluation, and quality-gate steps

## Local model results

The baseline XGBoost model achieved the following results on the untouched test set:

| Metric | Result |
|---|---:|
| Accuracy | 0.7537 |
| Precision | 0.5244 |
| Recall | 0.7754 |
| F1 score | 0.6257 |
| ROC-AUC | 0.8429 |

The model passed the current acceptance thresholds:

- F1 score >= 0.60
- Recall >= 0.70
- ROC-AUC >= 0.75

## AWS implementation lessons

### 1. SageMaker SDK version matters

The project uses SageMaker Python SDK v3. The SDK v3 package structure and object schemas differ from examples written for earlier SDK versions.

Examples encountered:

- Workflow imports use `sagemaker.core`.
- Pipeline classes are provided through `sagemaker.mlops`.
- `ProcessingOutput` separates the local output path from the S3 output configuration.
- `ProcessingS3Output` requires `s3_upload_mode`.

The implementation must therefore be validated against the installed SDK version rather than copied directly from older examples.

### 2. Processing output configuration

The current SDK v3 structure is:

```python
ProcessingOutput(
    output_name="output-name",
    s3_output=ProcessingS3Output(
        s3_uri=output_uri,
        s3_upload_mode="EndOfJob",
    ),
    local_path="/opt/ml/processing/output",
)

The s3_upload_mode parameter belongs inside ProcessingS3Output, while local_path belongs inside ProcessingOutput.
```
### 3. Pipeline quality gates

The pipeline evaluates the model using a SageMaker Processing step that produces an evaluation.json property file.

The quality gate reads:

f1

recall

roc_auc

The pipeline requires all three metrics to meet their configured thresholds.

### 4. Reproducible artifacts

Pipeline execution-specific S3 prefixes are generated using the SageMaker pipeline execution ID.

This prevents different executions from overwriting each other's:

Processed datasets

Preprocessing artifacts

Evaluation reports

### 5. Containerized processing

The custom Processing image contains the validation, preprocessing, and evaluation code. The image is pinned by its ECR digest in the pipeline configuration.

This provides a reproducible processing environment and avoids depending on the local machine's Python environment.

Current limitation

The pipeline currently defines the quality gate but does not yet register the model in SageMaker Model Registry or deploy an endpoint.

Those will be implemented after the pipeline definition has been successfully compiled and reviewed.
