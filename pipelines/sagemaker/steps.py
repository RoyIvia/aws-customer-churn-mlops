"""SageMaker Pipeline processing and training steps."""

from sagemaker.core.processing import ScriptProcessor
from sagemaker.core.shapes import (
    ProcessingInput,
    ProcessingOutput,
    ProcessingS3Input,
    ProcessingS3Output,
)
from sagemaker.core.workflow.pipeline_context import PipelineSession
from sagemaker.core.workflow.properties import PropertyFile
from sagemaker.mlops.workflow.steps import (
    ProcessingStep,
    TrainingStep,
)
from sagemaker.train import ModelTrainer
from sagemaker.train.configs import Compute, InputData

from pipelines.sagemaker.parameters import (
    EVALUATION_ARTIFACTS_PREFIX,
    PREPROCESSING_ARTIFACTS_PREFIX,
    PROCESSING_IMAGE_URI,
    PROCESSING_INSTANCE_TYPE,
    PROCESSED_DATA_PREFIX,
    RAW_DATA_URI,
    TRAIN_DATA_URI,
    TRAINING_IMAGE_URI,
    TRAINING_INSTANCE_COUNT,
    TRAINING_INSTANCE_TYPE,
    VALIDATION_DATA_URI,
)


EVALUATION_REPORT = PropertyFile(
    name="EvaluationReport",
    output_name="evaluation",
    path="evaluation.json",
)


def create_processing_processor(
    role: str,
    pipeline_session: PipelineSession,
) -> ScriptProcessor:
    """Create the shared SageMaker Processing processor."""

    return ScriptProcessor(
        role=role,
        image_uri=PROCESSING_IMAGE_URI,
        command=["python"],
        instance_count=1,
        instance_type=PROCESSING_INSTANCE_TYPE,
        base_job_name="customer-churn-processing",
        sagemaker_session=pipeline_session,
    )


def create_validation_step(
    role: str,
    pipeline_session: PipelineSession,
) -> ProcessingStep:
    """Create the SageMaker dataset validation step."""

    processor = create_processing_processor(
        role=role,
        pipeline_session=pipeline_session,
    )

    step_args = processor.run(
        code="scripts/sagemaker/validate.py",
        inputs=[
            ProcessingInput(
                input_name="raw-data",
                s3_input=ProcessingS3Input(
                    s3_uri=RAW_DATA_URI,
                    s3_data_type="S3Prefix",
                    s3_input_mode="File",
                    local_path="/opt/ml/processing/input",
                ),
            ),
        ],
        arguments=[
            "--input-data",
            "/opt/ml/processing/input/"
            "WA_Fn-UseC_-Telco-Customer-Churn.csv",
        ],
        wait=False,
        logs=False,
    )

    return ProcessingStep(
        name="Validation",
        step_args=step_args,
    )


def create_preprocessing_step(
    role: str,
    pipeline_session: PipelineSession,
) -> ProcessingStep:
    """Create the SageMaker preprocessing step."""

    processor = create_processing_processor(
        role=role,
        pipeline_session=pipeline_session,
    )

    step_args = processor.run(
        code="scripts/sagemaker/preprocess.py",
        inputs=[
            ProcessingInput(
                input_name="raw-data",
                s3_input=ProcessingS3Input(
                    s3_uri=RAW_DATA_URI,
                    s3_data_type="S3Prefix",
                    s3_input_mode="File",
                    local_path="/opt/ml/processing/input",
                ),
            ),
        ],
        outputs=[
            ProcessingOutput(
                output_name="processed-data",
                s3_output=ProcessingS3Output(
                    s3_uri=PROCESSED_DATA_PREFIX,
                    s3_upload_mode="EndOfJob",
                    local_path="/opt/ml/processing/processed",
                ),
            ),
            ProcessingOutput(
                output_name="artifacts",
                s3_output=ProcessingS3Output(
                    s3_uri=PREPROCESSING_ARTIFACTS_PREFIX,
                    s3_upload_mode="EndOfJob",
                    local_path="/opt/ml/processing/artifacts",
                ),
            ),
        ],
        arguments=[
            "--input-data",
            "/opt/ml/processing/input/"
            "WA_Fn-UseC_-Telco-Customer-Churn.csv",
            "--processed-dir",
            "/opt/ml/processing/processed",
            "--preprocessor-output",
            "/opt/ml/processing/artifacts/"
            "preprocessor.joblib",
        ],
        wait=False,
        logs=False,
    )

    return ProcessingStep(
        name="Preprocessing",
        step_args=step_args,
    )


def create_training_step(
    role: str,
    pipeline_session: PipelineSession,
) -> TrainingStep:
    """Create the SageMaker XGBoost training step."""

    trainer = ModelTrainer(
        role=role,
        training_image=TRAINING_IMAGE_URI,
        base_job_name="customer-churn-training",
        sagemaker_session=pipeline_session,
        compute=Compute(
            instance_type=TRAINING_INSTANCE_TYPE,
            instance_count=TRAINING_INSTANCE_COUNT,
        ),
        hyperparameters={
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "num_round": 200,
            "max_depth": 4,
            "eta": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "scale_pos_weight": 3104 / 1121,
        },
    )

    training_step_args = trainer.train(
        input_data_config=[
            InputData(
                channel_name="train",
                data_source=TRAIN_DATA_URI.to_string(),
            ),
        ],
        wait=False,
        logs=False,
    )

    return TrainingStep(
        name="Training",
        step_args=training_step_args,
    )


def create_evaluation_step(
    role: str,
    pipeline_session: PipelineSession,
    training_step: TrainingStep,
) -> ProcessingStep:
    """Create the SageMaker model evaluation step."""

    processor = create_processing_processor(
        role=role,
        pipeline_session=pipeline_session,
    )

    step_args = processor.run(
        code="scripts/sagemaker/evaluate.py",
        inputs=[
            ProcessingInput(
                input_name="model",
                s3_input=ProcessingS3Input(
                    s3_uri=(
                        training_step
                        .properties
                        .ModelArtifacts
                        .S3ModelArtifacts
                    ),
                    s3_data_type="S3Prefix",
                    s3_input_mode="File",
                    local_path="/opt/ml/processing/model",
                ),
            ),
            ProcessingInput(
                input_name="validation-data",
                s3_input=ProcessingS3Input(
                    s3_uri=VALIDATION_DATA_URI,
                    s3_data_type="S3Prefix",
                    s3_input_mode="File",
                    local_path="/opt/ml/processing/validation",
                ),
            ),
        ],
        outputs=[
            ProcessingOutput(
                output_name="evaluation",
                s3_output=ProcessingS3Output(
                    s3_uri=EVALUATION_ARTIFACTS_PREFIX,
                    s3_upload_mode="EndOfJob",
                    local_path="/opt/ml/processing/evaluation",
                ),
            ),
        ],
        arguments=[
            "--model",
            "/opt/ml/processing/model/model.tar.gz",
            "--validation-data",
            "/opt/ml/processing/validation/data.csv",
            "--evaluation-output",
            "/opt/ml/processing/evaluation",
        ],
        wait=False,
        logs=False,
    )

    return ProcessingStep(
        name="Evaluation",
        step_args=step_args,
        property_files=[EVALUATION_REPORT],
    )
