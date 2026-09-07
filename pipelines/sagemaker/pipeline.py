"""SageMaker Pipeline definition."""

from sagemaker.core.workflow.conditions import (
    ConditionGreaterThanOrEqualTo,
)
from sagemaker.core.workflow.functions import JsonGet
from sagemaker.core.workflow.pipeline_context import PipelineSession
from sagemaker.mlops.workflow.condition_step import ConditionStep
from sagemaker.mlops.workflow.pipeline import Pipeline

from pipelines.sagemaker.parameters import (
    AWS_REGION,
    F1_THRESHOLD,
    PIPELINE_NAME,
    PROCESSING_INSTANCE_TYPE,
    RECALL_THRESHOLD,
    ROC_AUC_THRESHOLD,
    TRAINING_INSTANCE_COUNT,
    TRAINING_INSTANCE_TYPE,
)
from pipelines.sagemaker.steps import (
    EVALUATION_REPORT,
    create_evaluation_step,
    create_preprocessing_step,
    create_training_step,
    create_validation_step,
)


SAGEMAKER_EXECUTION_ROLE_ARN = (
    "arn:aws:iam::633605692302:role/"
    "SageMakerCustomerChurnExecutionRole"
)

S3_BUCKET = "aws-customer-churn-mlops-633605692302-us-east-1"


def create_quality_gate(
    evaluation_step,
) -> ConditionStep:
    """Create the model quality gate."""

    return ConditionStep(
        name="QualityGate",
        conditions=[
            ConditionGreaterThanOrEqualTo(
                left=JsonGet(
                    step_name=evaluation_step.name,
                    property_file=EVALUATION_REPORT,
                    json_path="f1",
                ),
                right=F1_THRESHOLD,
            ),
            ConditionGreaterThanOrEqualTo(
                left=JsonGet(
                    step_name=evaluation_step.name,
                    property_file=EVALUATION_REPORT,
                    json_path="recall",
                ),
                right=RECALL_THRESHOLD,
            ),
            ConditionGreaterThanOrEqualTo(
                left=JsonGet(
                    step_name=evaluation_step.name,
                    property_file=EVALUATION_REPORT,
                    json_path="roc_auc",
                ),
                right=ROC_AUC_THRESHOLD,
            ),
        ],
        if_steps=[],
        else_steps=[],
    )


def create_pipeline(
    role: str = SAGEMAKER_EXECUTION_ROLE_ARN,
) -> Pipeline:
    """Create the customer churn SageMaker Pipeline."""

    pipeline_session = PipelineSession(
        default_bucket=S3_BUCKET,
    )

    validation_step = create_validation_step(
        role=role,
        pipeline_session=pipeline_session,
    )

    preprocessing_step = create_preprocessing_step(
        role=role,
        pipeline_session=pipeline_session,
    )

    training_step = create_training_step(
        role=role,
        pipeline_session=pipeline_session,
    )

    evaluation_step = create_evaluation_step(
        role=role,
        pipeline_session=pipeline_session,
        training_step=training_step,
    )

    quality_gate = create_quality_gate(
        evaluation_step=evaluation_step,
    )

    preprocessing_step.add_depends_on(
        [validation_step]
    )

    training_step.add_depends_on(
        [preprocessing_step]
    )

    evaluation_step.add_depends_on(
        [training_step]
    )

    quality_gate.add_depends_on(
        [evaluation_step]
    )

    pipeline = Pipeline(
        name=PIPELINE_NAME,
        parameters=[
            PROCESSING_INSTANCE_TYPE,
            TRAINING_INSTANCE_TYPE,
            TRAINING_INSTANCE_COUNT,
            F1_THRESHOLD,
            RECALL_THRESHOLD,
            ROC_AUC_THRESHOLD,
        ],
        steps=[
            validation_step,
            preprocessing_step,
            training_step,
            evaluation_step,
            quality_gate,
        ],
        sagemaker_session=pipeline_session,
    )

    return pipeline


def main() -> None:
    """Compile and print the SageMaker Pipeline definition."""

    pipeline = create_pipeline()

    definition = pipeline.definition()

    print(
        "SageMaker Pipeline definition generated successfully."
    )
    print()
    print(f"Pipeline name: {PIPELINE_NAME}")
    print(f"AWS Region:    {AWS_REGION}")
    print()
    print(definition)


if __name__ == "__main__":
    main()
