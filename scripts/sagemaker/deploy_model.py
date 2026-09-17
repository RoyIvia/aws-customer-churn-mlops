"""Deploy an approved SageMaker Model Registry package.

The deployment utility enforces the governance boundary established by
the Model Registry. Only model packages with an Approved status can be
deployed to the real-time inference endpoint.
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from typing import Any

import boto3
from botocore.exceptions import BotoCoreError, ClientError


DEFAULT_REGION = "us-east-1"
DEFAULT_INSTANCE_TYPE = "ml.m5.large"
DEFAULT_ENDPOINT_NAME = "aws-customer-churn-endpoint"
EXPECTED_MODEL_PACKAGE_GROUP = "aws-customer-churn-xgboost"

EXECUTION_ROLE_ARN = (
    "arn:aws:iam::633605692302:role/"
    "SageMakerCustomerChurnExecutionRole"
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Deploy an approved SageMaker Model Registry package."
        )
    )

    parser.add_argument(
        "--model-package-arn",
        required=True,
        help="ARN of the approved SageMaker model package.",
    )

    parser.add_argument(
        "--endpoint-name",
        default=DEFAULT_ENDPOINT_NAME,
        help=(
            "SageMaker endpoint name. "
            f"Default: {DEFAULT_ENDPOINT_NAME}"
        ),
    )

    parser.add_argument(
        "--instance-type",
        default=DEFAULT_INSTANCE_TYPE,
        help=(
            "Real-time inference instance type. "
            f"Default: {DEFAULT_INSTANCE_TYPE}"
        ),
    )

    parser.add_argument(
        "--region",
        default=DEFAULT_REGION,
        help=f"AWS Region. Default: {DEFAULT_REGION}",
    )

    parser.add_argument(
        "--wait",
        action="store_true",
        help="Wait until endpoint deployment completes.",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Validate the approved model package without "
            "creating deployment resources."
        ),
    )

    return parser.parse_args()


def describe_and_validate_package(
    client: Any,
    model_package_arn: str,
) -> dict[str, Any]:
    """Validate that a model package is eligible for deployment."""

    package = client.describe_model_package(
        ModelPackageName=model_package_arn,
    )

    package_group = package.get(
        "ModelPackageGroupName"
    )

    package_status = package.get(
        "ModelPackageStatus"
    )

    approval_status = package.get(
        "ModelApprovalStatus"
    )

    if package_group != EXPECTED_MODEL_PACKAGE_GROUP:
        raise ValueError(
            "Model package belongs to an unexpected package group."
        )

    if package_status != "Completed":
        raise ValueError(
            "Model package creation has not completed."
        )

    if approval_status != "Approved":
        raise ValueError(
            "Deployment denied: model package is not Approved."
        )

    return package


def create_resource_names(
    endpoint_name: str,
) -> tuple[str, str]:
    """Generate unique model and endpoint configuration names."""

    timestamp = datetime.now(
        timezone.utc
    ).strftime("%Y%m%d-%H%M%S")

    model_name = (
        f"{endpoint_name}-model-{timestamp}"
    )

    endpoint_config_name = (
        f"{endpoint_name}-config-{timestamp}"
    )

    return model_name, endpoint_config_name


def create_model(
    client: Any,
    model_name: str,
    model_package_arn: str,
) -> None:
    """Create a deployable SageMaker model."""

    client.create_model(
        ModelName=model_name,
        PrimaryContainer={
            "ModelPackageName": model_package_arn,
        },
        ExecutionRoleArn=EXECUTION_ROLE_ARN,
        Tags=[
            {
                "Key": "Project",
                "Value": "aws-customer-churn-mlops",
            },
            {
                "Key": "ManagedBy",
                "Value": "deploy_model.py",
            },
        ],
    )


def create_endpoint_config(
    client: Any,
    endpoint_config_name: str,
    model_name: str,
    instance_type: str,
) -> None:
    """Create the SageMaker endpoint configuration."""

    client.create_endpoint_config(
        EndpointConfigName=endpoint_config_name,
        ProductionVariants=[
            {
                "VariantName": "AllTraffic",
                "ModelName": model_name,
                "InitialInstanceCount": 1,
                "InstanceType": instance_type,
                "InitialVariantWeight": 1.0,
            }
        ],
        Tags=[
            {
                "Key": "Project",
                "Value": "aws-customer-churn-mlops",
            }
        ],
    )


def endpoint_exists(
    client: Any,
    endpoint_name: str,
) -> bool:
    """Return whether the endpoint already exists."""

    try:
        client.describe_endpoint(
            EndpointName=endpoint_name,
        )
        return True

    except ClientError as exc:
        error_code = (
            exc.response
            .get("Error", {})
            .get("Code")
        )

        if error_code == "ValidationException":
            return False

        raise


def deploy_endpoint(
    client: Any,
    endpoint_name: str,
    endpoint_config_name: str,
) -> str:
    """Create or update the real-time endpoint."""

    if endpoint_exists(
        client,
        endpoint_name,
    ):
        client.update_endpoint(
            EndpointName=endpoint_name,
            EndpointConfigName=endpoint_config_name,
        )

        return "update"

    client.create_endpoint(
        EndpointName=endpoint_name,
        EndpointConfigName=endpoint_config_name,
        Tags=[
            {
                "Key": "Project",
                "Value": "aws-customer-churn-mlops",
            }
        ],
    )

    return "create"


def wait_for_endpoint(
    client: Any,
    endpoint_name: str,
) -> None:
    """Wait until the endpoint reaches InService."""

    print(
        f"Waiting for endpoint {endpoint_name}..."
    )

    while True:
        response = client.describe_endpoint(
            EndpointName=endpoint_name,
        )

        status = response["EndpointStatus"]

        print(f"Endpoint status: {status}")

        if status == "InService":
            return

        if status == "Failed":
            failure_reason = response.get(
                "FailureReason",
                "Unknown failure",
            )

            raise RuntimeError(
                "Endpoint deployment failed: "
                f"{failure_reason}"
            )

        time.sleep(30)


def main() -> int:
    """Execute the governed model deployment workflow."""

    args = parse_args()

    try:
        client = boto3.client(
            "sagemaker",
            region_name=args.region,
        )

        package = describe_and_validate_package(
            client=client,
            model_package_arn=args.model_package_arn,
        )

        print()
        print("Deployment candidate")
        print("--------------------")
        print(
            "Package ARN:     "
            f"{package.get('ModelPackageArn')}"
        )
        print(
            "Package version: "
            f"{package.get('ModelPackageVersion')}"
        )
        print(
            "Approval status: "
            f"{package.get('ModelApprovalStatus')}"
        )
        print(
            "Endpoint:        "
            f"{args.endpoint_name}"
        )
        print(
            "Instance type:   "
            f"{args.instance_type}"
        )
        print()

        if args.dry_run:
            print(
                "Dry run successful. "
                "Model package is eligible for deployment."
            )
            return 0

        model_name, endpoint_config_name = (
            create_resource_names(
                args.endpoint_name
            )
        )

        create_model(
            client=client,
            model_name=model_name,
            model_package_arn=args.model_package_arn,
        )

        create_endpoint_config(
            client=client,
            endpoint_config_name=endpoint_config_name,
            model_name=model_name,
            instance_type=args.instance_type,
        )

        operation = deploy_endpoint(
            client=client,
            endpoint_name=args.endpoint_name,
            endpoint_config_name=endpoint_config_name,
        )

        print(
            f"Endpoint {operation} initiated successfully."
        )
        print(f"Model:           {model_name}")
        print(
            "Endpoint config: "
            f"{endpoint_config_name}"
        )
        print(
            f"Endpoint:        {args.endpoint_name}"
        )

        if args.wait:
            wait_for_endpoint(
                client=client,
                endpoint_name=args.endpoint_name,
            )

            print(
                "Endpoint is InService."
            )

        return 0

    except (
        ValueError,
        RuntimeError,
        BotoCoreError,
        ClientError,
    ) as exc:
        print(
            f"ERROR: {exc}",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
