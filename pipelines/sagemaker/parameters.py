"""SageMaker Pipeline parameters."""

from sagemaker.core.workflow.execution_variables import ExecutionVariables
from sagemaker.core.workflow.functions import Join
from sagemaker.core.workflow.parameters import (
    ParameterFloat,
    ParameterInteger,
    ParameterString,
)


PIPELINE_NAME = "aws-customer-churn-mlops-pipeline"

AWS_REGION = "us-east-1"

PROJECT_NAME = "aws-customer-churn-mlops"

S3_BUCKET = (
    "aws-customer-churn-mlops-633605692302-us-east-1"
)

RAW_DATA_URI = (
    f"s3://{S3_BUCKET}/raw/telco/"
    "WA_Fn-UseC_-Telco-Customer-Churn.csv"
)

PIPELINE_EXECUTION_ID = (
    ExecutionVariables.PIPELINE_EXECUTION_ID
)

PROCESSED_DATA_PREFIX = Join(
    on="/",
    values=[
        f"s3://{S3_BUCKET}/processed",
        PIPELINE_EXECUTION_ID,
    ],
)

TRAIN_DATA_URI = Join(
    on="/",
    values=[
        PROCESSED_DATA_PREFIX,
        "train",
    ],
)

VALIDATION_DATA_URI = Join(
    on="/",
    values=[
        PROCESSED_DATA_PREFIX,
        "validation",
    ],
)

TEST_DATA_URI = Join(
    on="/",
    values=[
        PROCESSED_DATA_PREFIX,
        "test",
    ],
)

ARTIFACTS_PREFIX = Join(
    on="/",
    values=[
        f"s3://{S3_BUCKET}/artifacts",
        PIPELINE_EXECUTION_ID,
    ],
)

PREPROCESSING_ARTIFACTS_PREFIX = Join(
    on="/",
    values=[
        ARTIFACTS_PREFIX,
        "preprocessing",
    ],
)

EVALUATION_ARTIFACTS_PREFIX = Join(
    on="/",
    values=[
        ARTIFACTS_PREFIX,
        "evaluation",
    ],
)

PROCESSING_IMAGE_URI = (
    "633605692302.dkr.ecr.us-east-1.amazonaws.com/"
    "aws-customer-churn-processing@sha256:"
    "6f6b7f5d9ff4a2597fa88d81848f929c95e31b900d71a405012d1c00a5e9c105"
)

TRAINING_IMAGE_URI = (
    "683313688378.dkr.ecr.us-east-1.amazonaws.com/"
    "sagemaker-xgboost:1.7-1"
)

PROCESSING_INSTANCE_TYPE = ParameterString(
    name="ProcessingInstanceType",
    default_value="ml.m5.large",
)

TRAINING_INSTANCE_TYPE = ParameterString(
    name="TrainingInstanceType",
    default_value="ml.m5.large",
)

TRAINING_INSTANCE_COUNT = ParameterInteger(
    name="TrainingInstanceCount",
    default_value=1,
)

F1_THRESHOLD = ParameterFloat(
    name="F1Threshold",
    default_value=0.60,
)

RECALL_THRESHOLD = ParameterFloat(
    name="RecallThreshold",
    default_value=0.70,
)

ROC_AUC_THRESHOLD = ParameterFloat(
    name="RocAucThreshold",
    default_value=0.75,
)

SCALE_POS_WEIGHT = 3104 / 1121

XGBOOST_HYPERPARAMETERS = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "num_round": 200,
    "max_depth": 4,
    "eta": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "scale_pos_weight": SCALE_POS_WEIGHT,
}
