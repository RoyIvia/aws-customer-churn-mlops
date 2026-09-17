"""Approve a specific SageMaker Model Registry model package.

This utility provides an explicit human-controlled promotion boundary
between model registration and deployment.

A model package must be specified by its full ARN. The script validates
that the package belongs to the expected model package group and is
currently PendingManualApproval before changing its approval status.
"""

from __future__ import annotations

import argparse
import sys
from typing import Any

import boto3
from botocore.exceptions import BotoCoreError, ClientError


DEFAULT_REGION = "us-east-1"
EXPECTED_MODEL_PACKAGE_GROUP = "aws-customer-churn-xgboost"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Approve a specific SageMaker Model Registry model package."
        )
    )

    parser.add_argument(
        "--model-package-arn",
        required=True,
        help="Full ARN of the SageMaker model package to approve.",
    )

    parser.add_argument(
        "--region",
        default=DEFAULT_REGION,
        help=f"AWS Region. Default: {DEFAULT_REGION}",
    )

    parser.add_argument(
        "--approval-description",
        default="Approved after manual model review.",
        help="Audit description recorded with the approval action.",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Validate the model package but do not change "
            "its approval status."
        ),
    )

    return parser.parse_args()


def validate_model_package_arn(
    model_package_arn: str,
    region: str,
) -> None:
    """Validate the basic structure of the model package ARN."""

    expected_prefix = (
        f"arn:aws:sagemaker:{region}:"
    )

    if not model_package_arn.startswith(expected_prefix):
        raise ValueError(
            "Model package ARN does not belong to the "
            f"expected AWS Region: {region}"
        )

    if ":model-package/" not in model_package_arn:
        raise ValueError(
            "The supplied ARN is not a SageMaker model package ARN."
        )


def get_model_package(
    client: Any,
    model_package_arn: str,
) -> dict[str, Any]:
    """Retrieve model package metadata from SageMaker."""

    response = client.describe_model_package(
        ModelPackageName=model_package_arn,
    )

    return response


def validate_model_package(
    model_package: dict[str, Any],
) -> None:
    """Validate package group and current approval state."""

    package_arn = model_package.get(
        "ModelPackageArn",
        "<unknown>",
    )

    package_group = model_package.get(
        "ModelPackageGroupName",
    )

    approval_status = model_package.get(
        "ModelApprovalStatus",
    )

    package_status = model_package.get(
        "ModelPackageStatus",
    )

    if package_group != EXPECTED_MODEL_PACKAGE_GROUP:
        raise ValueError(
            "Model package belongs to an unexpected package group.\n"
            f"Expected: {EXPECTED_MODEL_PACKAGE_GROUP}\n"
            f"Actual:   {package_group}"
        )

    if package_status != "Completed":
        raise ValueError(
            "Model package is not ready for approval.\n"
            f"ARN:    {package_arn}\n"
            f"Status: {package_status}"
        )

    if approval_status == "Approved":
        raise ValueError(
            "Model package is already approved.\n"
            f"ARN: {package_arn}"
        )

    if approval_status != "PendingManualApproval":
        raise ValueError(
            "Model package is not awaiting manual approval.\n"
            f"ARN:             {package_arn}\n"
            f"Approval status: {approval_status}"
        )


def print_model_summary(
    model_package: dict[str, Any],
) -> None:
    """Print the candidate model metadata being reviewed."""

    print()
    print("Model package review")
    print("--------------------")
    print(
        "ARN:              "
        f"{model_package.get('ModelPackageArn')}"
    )
    print(
        "Package group:    "
        f"{model_package.get('ModelPackageGroupName')}"
    )
    print(
        "Version:          "
        f"{model_package.get('ModelPackageVersion')}"
    )
    print(
        "Package status:   "
        f"{model_package.get('ModelPackageStatus')}"
    )
    print(
        "Approval status:  "
        f"{model_package.get('ModelApprovalStatus')}"
    )
    print()


def approve_model_package(
    client: Any,
    model_package_arn: str,
    approval_description: str,
) -> dict[str, Any]:
    """Approve the specified model package."""

    return client.update_model_package(
        ModelPackageArn=model_package_arn,
        ModelApprovalStatus="Approved",
        ApprovalDescription=approval_description,
    )


def main() -> int:
    """Run the manual model approval workflow."""

    args = parse_args()

    try:
        validate_model_package_arn(
            model_package_arn=args.model_package_arn,
            region=args.region,
        )

        client = boto3.client(
            "sagemaker",
            region_name=args.region,
        )

        model_package = get_model_package(
            client=client,
            model_package_arn=args.model_package_arn,
        )

        print_model_summary(model_package)

        validate_model_package(model_package)

        if args.dry_run:
            print(
                "Dry run successful. "
                "The model package is eligible for approval."
            )
            return 0

        response = approve_model_package(
            client=client,
            model_package_arn=args.model_package_arn,
            approval_description=args.approval_description,
        )

        print("Model package approved successfully.")
        print(
            "Model package ARN: "
            f"{response.get('ModelPackageArn')}"
        )

        return 0

    except (ValueError, BotoCoreError, ClientError) as exc:
        print(
            f"ERROR: {exc}",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
