"""Tests for the SageMaker model approval utility."""

import importlib.util
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]

APPROVE_MODEL_PATH = (
    PROJECT_ROOT
    / "scripts"
    / "sagemaker"
    / "approve_model.py"
)

spec = importlib.util.spec_from_file_location(
    "approve_model",
    APPROVE_MODEL_PATH,
)

approve_model = importlib.util.module_from_spec(spec)

assert spec.loader is not None
spec.loader.exec_module(approve_model)


validate_model_package = (
    approve_model.validate_model_package
)

validate_model_package_arn = (
    approve_model.validate_model_package_arn
)


REGION = "us-east-1"

MODEL_PACKAGE_ARN = (
    "arn:aws:sagemaker:us-east-1:633605692302:"
    "model-package/aws-customer-churn-xgboost/1"
)


def test_valid_model_package_arn():
    """A correctly structured model package ARN should pass."""

    validate_model_package_arn(
        MODEL_PACKAGE_ARN,
        REGION,
    )


def test_model_package_arn_wrong_region():
    """A model package ARN from another region should fail."""

    arn = (
        "arn:aws:sagemaker:eu-west-1:633605692302:"
        "model-package/aws-customer-churn-xgboost/1"
    )

    with pytest.raises(
        ValueError,
        match="expected AWS Region",
    ):
        validate_model_package_arn(
            arn,
            REGION,
        )


def test_non_model_package_arn():
    """A non-model-package SageMaker ARN should fail."""

    arn = (
        "arn:aws:sagemaker:us-east-1:633605692302:"
        "endpoint/customer-churn"
    )

    with pytest.raises(
        ValueError,
        match="not a SageMaker model package ARN",
    ):
        validate_model_package_arn(
            arn,
            REGION,
        )


def test_pending_model_is_eligible():
    """A completed pending model should pass validation."""

    model_package = {
        "ModelPackageArn": MODEL_PACKAGE_ARN,
        "ModelPackageGroupName": (
            "aws-customer-churn-xgboost"
        ),
        "ModelPackageVersion": 1,
        "ModelPackageStatus": "Completed",
        "ModelApprovalStatus": (
            "PendingManualApproval"
        ),
    }

    validate_model_package(model_package)


def test_wrong_model_package_group():
    """A package from another group should be rejected."""

    model_package = {
        "ModelPackageArn": MODEL_PACKAGE_ARN,
        "ModelPackageGroupName": "another-model-group",
        "ModelPackageVersion": 1,
        "ModelPackageStatus": "Completed",
        "ModelApprovalStatus": (
            "PendingManualApproval"
        ),
    }

    with pytest.raises(
        ValueError,
        match="unexpected package group",
    ):
        validate_model_package(model_package)


def test_incomplete_model_package():
    """A package still being created should be rejected."""

    model_package = {
        "ModelPackageArn": MODEL_PACKAGE_ARN,
        "ModelPackageGroupName": (
            "aws-customer-churn-xgboost"
        ),
        "ModelPackageVersion": 1,
        "ModelPackageStatus": "InProgress",
        "ModelApprovalStatus": (
            "PendingManualApproval"
        ),
    }

    with pytest.raises(
        ValueError,
        match="not ready for approval",
    ):
        validate_model_package(model_package)


def test_already_approved_model():
    """An already approved package should be rejected."""

    model_package = {
        "ModelPackageArn": MODEL_PACKAGE_ARN,
        "ModelPackageGroupName": (
            "aws-customer-churn-xgboost"
        ),
        "ModelPackageVersion": 1,
        "ModelPackageStatus": "Completed",
        "ModelApprovalStatus": "Approved",
    }

    with pytest.raises(
        ValueError,
        match="already approved",
    ):
        validate_model_package(model_package)


def test_rejected_model_cannot_be_approved():
    """A rejected package should not pass validation."""

    model_package = {
        "ModelPackageArn": MODEL_PACKAGE_ARN,
        "ModelPackageGroupName": (
            "aws-customer-churn-xgboost"
        ),
        "ModelPackageVersion": 1,
        "ModelPackageStatus": "Completed",
        "ModelApprovalStatus": "Rejected",
    }

    with pytest.raises(
        ValueError,
        match="not awaiting manual approval",
    ):
        validate_model_package(model_package)
