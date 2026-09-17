"""Delete the customer churn SageMaker endpoint.

This utility exists primarily as a cost-control mechanism for portfolio
and development environments where the real-time endpoint should not
remain provisioned after validation.
"""

from __future__ import annotations

import argparse
import sys

import boto3
from botocore.exceptions import BotoCoreError, ClientError


DEFAULT_REGION = "us-east-1"
DEFAULT_ENDPOINT_NAME = "aws-customer-churn-endpoint"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Delete the SageMaker churn endpoint."
    )

    parser.add_argument(
        "--endpoint-name",
        default=DEFAULT_ENDPOINT_NAME,
    )

    parser.add_argument(
        "--region",
        default=DEFAULT_REGION,
    )

    return parser.parse_args()


def main() -> int:
    """Delete the configured SageMaker endpoint."""

    args = parse_args()

    client = boto3.client(
        "sagemaker",
        region_name=args.region,
    )

    try:
        endpoint = client.describe_endpoint(
            EndpointName=args.endpoint_name,
        )

        endpoint_config_name = endpoint[
            "EndpointConfigName"
        ]

        client.delete_endpoint(
            EndpointName=args.endpoint_name,
        )

        print(
            "Endpoint deletion initiated: "
            f"{args.endpoint_name}"
        )

        print(
            "Associated endpoint configuration: "
            f"{endpoint_config_name}"
        )

        print(
            "After endpoint deletion completes, "
            "the endpoint configuration and model "
            "can also be removed if no longer required."
        )

        return 0

    except ClientError as exc:
        error_code = (
            exc.response
            .get("Error", {})
            .get("Code")
        )

        if error_code == "ValidationException":
            print(
                "Endpoint does not exist: "
                f"{args.endpoint_name}"
            )
            return 0

        print(
            f"ERROR: {exc}",
            file=sys.stderr,
        )
        return 1

    except BotoCoreError as exc:
        print(
            f"ERROR: {exc}",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
